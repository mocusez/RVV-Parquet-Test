#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/compute/api.h>
#include <arrow/compute/expression.h>
#include <arrow/dataset/api.h>
#include <arrow/dataset/file_parquet.h>
#include <arrow/dataset/scanner.h>
#include <parquet/arrow/reader.h>
#include <parquet/file_reader.h>

#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <map>
#include <set>
#include <chrono>
#include <string>
#include <memory>
#include <algorithm>

using namespace arrow;
using namespace arrow::compute;

// TPC-H Query 12 result structure
struct Query12Result {
    std::string l_shipmode;
    int64_t high_line_count;
    int64_t low_line_count;
    
    bool operator<(const Query12Result& other) const {
        // Sort by shipmode in ascending order
        return l_shipmode < other.l_shipmode;
    }
};

// Convert ISO date string to days since epoch
int32_t date_string_to_days(const std::string& date_str) {
    std::tm tm = {};
    std::istringstream ss(date_str);
    ss >> std::get_time(&tm, "%Y-%m-%d");
    
    std::time_t time = std::mktime(&tm);
    // Convert to days since epoch (1970-01-01)
    return static_cast<int32_t>(time / (60 * 60 * 24));
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <orders.parquet> <lineitem.parquet>" << std::endl;
        return 1;
    }

    std::string orders_file = argv[1];
    std::string lineitem_file = argv[2];
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Initialize Arrow's memory pool
    arrow::MemoryPool* pool = arrow::default_memory_pool();
    
    // 1. Process orders table to get order priorities
    auto maybe_orders_file = arrow::io::ReadableFile::Open(orders_file);
    if (!maybe_orders_file.ok()) {
        std::cerr << "Could not open orders file: " << maybe_orders_file.status().ToString() << std::endl;
        return 1;
    }
    std::shared_ptr<arrow::io::ReadableFile> orders_infile = *maybe_orders_file;
    
    auto orders_reader_result = parquet::arrow::OpenFile(orders_infile, pool);
    if (!orders_reader_result.ok()) {
        std::cerr << "Could not open orders parquet file: " << orders_reader_result.status().ToString() << std::endl;
        return 1;
    }
    std::unique_ptr<parquet::arrow::FileReader> orders_reader = std::move(*orders_reader_result);
    
    std::shared_ptr<arrow::Table> orders_table;
    auto status = orders_reader->ReadTable(&orders_table);
    if (!status.ok()) {
        std::cerr << "Could not read orders table: " << status.ToString() << std::endl;
        return 1;
    }
    
    std::cout << "Orders table loaded with " << orders_table->num_rows() << " rows" << std::endl;
    
    // Extract required columns from orders
    auto o_orderkey_col = orders_table->GetColumnByName("o_orderkey");
    auto o_orderpriority_col = orders_table->GetColumnByName("o_orderpriority");
    
    // Create a map of orderkey -> orderpriority
    std::map<int64_t, std::string> order_priorities;
    
    for (int chunk_idx = 0; chunk_idx < o_orderkey_col->num_chunks(); chunk_idx++) {
        auto o_orderkey_array = std::static_pointer_cast<arrow::Int64Array>(o_orderkey_col->chunk(chunk_idx));
        auto o_orderpriority_array = std::static_pointer_cast<arrow::StringArray>(o_orderpriority_col->chunk(chunk_idx));
        
        int64_t num_rows = o_orderkey_array->length();
        for (int64_t i = 0; i < num_rows; i++) {
            if (o_orderkey_array->IsNull(i) || o_orderpriority_array->IsNull(i)) {
                continue;
            }
            
            int64_t orderkey = o_orderkey_array->Value(i);
            std::string priority = o_orderpriority_array->GetString(i);
            
            order_priorities[orderkey] = priority;
        }
    }
    
    std::cout << "Loaded " << order_priorities.size() << " order priorities" << std::endl;
    
    // 2. Process lineitem table
    auto maybe_lineitem_file = arrow::io::ReadableFile::Open(lineitem_file);
    if (!maybe_lineitem_file.ok()) {
        std::cerr << "Could not open lineitem file: " << maybe_lineitem_file.status().ToString() << std::endl;
        return 1;
    }
    std::shared_ptr<arrow::io::ReadableFile> lineitem_infile = *maybe_lineitem_file;
    
    auto lineitem_reader_result = parquet::arrow::OpenFile(lineitem_infile, pool);
    if (!lineitem_reader_result.ok()) {
        std::cerr << "Could not open lineitem parquet file: " << lineitem_reader_result.status().ToString() << std::endl;
        return 1;
    }
    std::unique_ptr<parquet::arrow::FileReader> lineitem_reader = std::move(*lineitem_reader_result);
    
    std::shared_ptr<arrow::Table> lineitem_table;
    status = lineitem_reader->ReadTable(&lineitem_table);
    if (!status.ok()) {
        std::cerr << "Could not read lineitem table: " << status.ToString() << std::endl;
        return 1;
    }
    
    std::cout << "Lineitem table loaded with " << lineitem_table->num_rows() << " rows" << std::endl;
    
    // Extract required columns from lineitem
    auto l_orderkey_col = lineitem_table->GetColumnByName("l_orderkey");
    auto l_shipmode_col = lineitem_table->GetColumnByName("l_shipmode");
    auto l_shipdate_col = lineitem_table->GetColumnByName("l_shipdate");
    auto l_commitdate_col = lineitem_table->GetColumnByName("l_commitdate");
    auto l_receiptdate_col = lineitem_table->GetColumnByName("l_receiptdate");
    
    // Convert date filters to days since epoch
    int32_t start_date = date_string_to_days("1994-01-01");
    int32_t end_date = date_string_to_days("1995-01-01"); // One day after end of 1994
    
    // Map to store results by shipmode
    std::map<std::string, Query12Result> results_by_shipmode;
    std::set<std::string> target_shipmodes = {"MAIL", "SHIP"};
    
    int64_t rows_processed = 0;
    int64_t rows_qualified = 0;
    
    for (int chunk_idx = 0; chunk_idx < l_orderkey_col->num_chunks(); chunk_idx++) {
        auto l_orderkey_array = std::static_pointer_cast<arrow::Int64Array>(l_orderkey_col->chunk(chunk_idx));
        auto l_shipmode_array = std::static_pointer_cast<arrow::StringArray>(l_shipmode_col->chunk(chunk_idx));
        auto l_shipdate_array = std::static_pointer_cast<arrow::Date32Array>(l_shipdate_col->chunk(chunk_idx));
        auto l_commitdate_array = std::static_pointer_cast<arrow::Date32Array>(l_commitdate_col->chunk(chunk_idx));
        auto l_receiptdate_array = std::static_pointer_cast<arrow::Date32Array>(l_receiptdate_col->chunk(chunk_idx));
        
        int64_t num_rows = l_orderkey_array->length();
        
        for (int64_t i = 0; i < num_rows; i++) {
            rows_processed++;
            
            if (l_orderkey_array->IsNull(i) || l_shipmode_array->IsNull(i) || 
                l_shipdate_array->IsNull(i) || l_commitdate_array->IsNull(i) || 
                l_receiptdate_array->IsNull(i)) {
                continue;
            }
            
            std::string shipmode = l_shipmode_array->GetString(i);
            
            // Check if shipmode is in the target set (MAIL or SHIP)
            if (target_shipmodes.find(shipmode) == target_shipmodes.end()) {
                continue;
            }
            
            int32_t shipdate = l_shipdate_array->Value(i);
            int32_t commitdate = l_commitdate_array->Value(i);
            int32_t receiptdate = l_receiptdate_array->Value(i);
            
            // Check filter conditions:
            // l_commitdate < l_receiptdate AND l_shipdate < l_commitdate
            // AND l_receiptdate >= DATE '1994-01-01' AND l_receiptdate < DATE '1995-01-01'
            if (commitdate < receiptdate && shipdate < commitdate && 
                receiptdate >= start_date && receiptdate < end_date) {
                
                int64_t orderkey = l_orderkey_array->Value(i);
                
                // Check if we have the order priority
                auto priority_it = order_priorities.find(orderkey);
                if (priority_it == order_priorities.end()) {
                    continue; // Skip if we don't have the priority
                }
                
                std::string priority = priority_it->second;
                
                // Initialize result entry if needed
                if (results_by_shipmode.find(shipmode) == results_by_shipmode.end()) {
                    Query12Result new_entry;
                    new_entry.l_shipmode = shipmode;
                    new_entry.high_line_count = 0;
                    new_entry.low_line_count = 0;
                    results_by_shipmode[shipmode] = new_entry;
                }
                
                // Update counts based on priority
                if (priority == "1-URGENT" || priority == "2-HIGH") {
                    results_by_shipmode[shipmode].high_line_count++;
                } else {
                    results_by_shipmode[shipmode].low_line_count++;
                }
                
                rows_qualified++;
            }
        }
    }
    
    // Convert map to vector for sorting
    std::vector<Query12Result> results;
    for (const auto& [shipmode, result] : results_by_shipmode) {
        results.push_back(result);
    }
    
    // Sort by shipmode
    std::sort(results.begin(), results.end());
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    // Print results
    std::cout << "\nTPC-H Query 12 Results:" << std::endl;
    std::cout << "----------------------" << std::endl;
    std::cout << std::setw(15) << "L_SHIPMODE" 
              << std::setw(20) << "HIGH_LINE_COUNT"
              << std::setw(20) << "LOW_LINE_COUNT" << std::endl;
    
    for (const auto& result : results) {
        std::cout << std::setw(15) << result.l_shipmode
                  << std::setw(20) << result.high_line_count
                  << std::setw(20) << result.low_line_count << std::endl;
    }
    
    std::cout << "\nQuery executed in " << elapsed.count() << " seconds" << std::endl;
    std::cout << "Processed " << rows_processed << " lineitem rows, " << rows_qualified << " qualified" << std::endl;
    
    return 0;
}