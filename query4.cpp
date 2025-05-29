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

using namespace arrow;
using namespace arrow::compute;

// TPC-H Query 4 result structure
struct OrderPriorityCount {
    std::string priority;
    int64_t count = 0;
    
    bool operator<(const OrderPriorityCount& other) const {
        return priority < other.priority;
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
    
    // 1. Open and read the lineitem Parquet file first to find qualifying line items
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
    auto status = lineitem_reader->ReadTable(&lineitem_table);
    if (!status.ok()) {
        std::cerr << "Could not read lineitem table: " << status.ToString() << std::endl;
        return 1;
    }
    
    std::cout << "Lineitem table loaded with " << lineitem_table->num_rows() << " rows" << std::endl;
    
    // Extract required columns from lineitem
    auto l_orderkey_col = lineitem_table->GetColumnByName("l_orderkey");
    auto l_commitdate_col = lineitem_table->GetColumnByName("l_commitdate");
    auto l_receiptdate_col = lineitem_table->GetColumnByName("l_receiptdate");
    
    // Find all orderkeys where l_commitdate < l_receiptdate
    std::set<int64_t> qualifying_orderkeys;
    
    for (int chunk_idx = 0; chunk_idx < l_orderkey_col->num_chunks(); chunk_idx++) {
        auto l_orderkey_array = std::static_pointer_cast<arrow::Int64Array>(l_orderkey_col->chunk(chunk_idx));
        auto l_commitdate_array = std::static_pointer_cast<arrow::Date32Array>(l_commitdate_col->chunk(chunk_idx));
        auto l_receiptdate_array = std::static_pointer_cast<arrow::Date32Array>(l_receiptdate_col->chunk(chunk_idx));
        
        int64_t num_rows = l_orderkey_array->length();
        for (int64_t i = 0; i < num_rows; i++) {
            if (l_orderkey_array->IsNull(i) || l_commitdate_array->IsNull(i) || l_receiptdate_array->IsNull(i)) {
                continue;
            }
            
            int32_t commit_date = l_commitdate_array->Value(i);
            int32_t receipt_date = l_receiptdate_array->Value(i);
            
            // Check if receipt date is later than commit date
            if (commit_date < receipt_date) {
                qualifying_orderkeys.insert(l_orderkey_array->Value(i));
            }
        }
    }
    
    std::cout << "Found " << qualifying_orderkeys.size() << " qualifying orderkeys" << std::endl;
    
    // 2. Now open and process the orders file
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
    status = orders_reader->ReadTable(&orders_table);
    if (!status.ok()) {
        std::cerr << "Could not read orders table: " << status.ToString() << std::endl;
        return 1;
    }
    
    std::cout << "Orders table loaded with " << orders_table->num_rows() << " rows" << std::endl;
    
    // Extract required columns from orders
    auto o_orderkey_col = orders_table->GetColumnByName("o_orderkey");
    auto o_orderdate_col = orders_table->GetColumnByName("o_orderdate");
    auto o_orderpriority_col = orders_table->GetColumnByName("o_orderpriority");
    
    // Calculate date range filter: Q3/1993 (July 1st to September 30th)
    int32_t start_date = date_string_to_days("1993-07-01");
    int32_t end_date = date_string_to_days("1993-10-01");  // One day after end of Q3
    
    std::cout << "Filtering orders between dates: " << start_date << " and " << end_date << std::endl;
    
    // Count orders by priority
    std::map<std::string, int64_t> priority_counts;
    
    for (int chunk_idx = 0; chunk_idx < o_orderkey_col->num_chunks(); chunk_idx++) {
        auto o_orderkey_array = std::static_pointer_cast<arrow::Int64Array>(o_orderkey_col->chunk(chunk_idx));
        auto o_orderdate_array = std::static_pointer_cast<arrow::Date32Array>(o_orderdate_col->chunk(chunk_idx));
        auto o_orderpriority_array = std::static_pointer_cast<arrow::StringArray>(o_orderpriority_col->chunk(chunk_idx));
        
        int64_t num_rows = o_orderkey_array->length();
        for (int64_t i = 0; i < num_rows; i++) {
            if (o_orderkey_array->IsNull(i) || o_orderdate_array->IsNull(i) || o_orderpriority_array->IsNull(i)) {
                continue;
            }
            
            int32_t order_date = o_orderdate_array->Value(i);
            int64_t order_key = o_orderkey_array->Value(i);
            
            // Check if the order is in Q3/1993
            if (order_date >= start_date && order_date < end_date) {
                // Check if any line item for this order satisfies the condition
                if (qualifying_orderkeys.find(order_key) != qualifying_orderkeys.end()) {
                    std::string priority = o_orderpriority_array->GetString(i);
                    priority_counts[priority]++;
                }
            }
        }
    }
    
    // Sort results by o_orderpriority
    std::vector<OrderPriorityCount> sorted_results;
    for (const auto& pair : priority_counts) {
        OrderPriorityCount result;
        result.priority = pair.first;
        result.count = pair.second;
        sorted_results.push_back(result);
    }
    
    std::sort(sorted_results.begin(), sorted_results.end());
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    // Print results
    std::cout << "\nTPC-H Query 4 Results:" << std::endl;
    std::cout << "----------------------" << std::endl;
    std::cout << std::setw(20) << "O_ORDERPRIORITY" << std::setw(15) << "ORDER_COUNT" << std::endl;
    
    for (const auto& result : sorted_results) {
        std::cout << std::setw(20) << result.priority 
                  << std::setw(15) << result.count << std::endl;
    }
    
    std::cout << "\nQuery executed in " << elapsed.count() << " seconds" << std::endl;
    
    return 0;
}