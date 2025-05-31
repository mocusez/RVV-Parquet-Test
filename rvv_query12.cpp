#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <arrow/compute/api_scalar.h>
#include <arrow/io/api.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/table.h>
#include <arrow/util/thread_pool.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>

#include <riscv_vector.h>

#include <chrono>
#include <iostream>
#include <iomanip>
#include <memory>
#include <unordered_set>
#include <vector>
#include <map>
#include <set>
#include <sstream>
#include <ctime>

using arrow::Status;

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

// Vector implementation for multiple date comparison conditions
void check_shipping_conditions_rvv(
    const int32_t* shipdate,
    const int32_t* commitdate,
    const int32_t* receiptdate,
    const int32_t start_date,
    const int32_t end_date,
    uint8_t* results,
    size_t length) {
    
    size_t i = 0;
    size_t vl;
    
    for (; i < length; i += vl) {
        vl = __riscv_vsetvl_e32m8(length - i);
        
        // Load all date values
        vint32m8_t v_shipdate = __riscv_vle32_v_i32m8(shipdate + i, vl);
        vint32m8_t v_commitdate = __riscv_vle32_v_i32m8(commitdate + i, vl);
        vint32m8_t v_receiptdate = __riscv_vle32_v_i32m8(receiptdate + i, vl);
        
        // Create date constants
        vint32m8_t v_start_date = __riscv_vmv_v_x_i32m8(start_date, vl);
        vint32m8_t v_end_date = __riscv_vmv_v_x_i32m8(end_date, vl);
        
        // Evaluate conditions:
        // 1. commitdate < receiptdate
        vbool4_t condition1 = __riscv_vmslt_vv_i32m8_b4(v_commitdate, v_receiptdate, vl);
        
        // 2. shipdate < commitdate
        vbool4_t condition2 = __riscv_vmslt_vv_i32m8_b4(v_shipdate, v_commitdate, vl);
        
        // 3. receiptdate >= start_date
        vbool4_t condition3 = __riscv_vmsge_vv_i32m8_b4(v_receiptdate, v_start_date, vl);
        
        // 4. receiptdate < end_date
        vbool4_t condition4 = __riscv_vmslt_vv_i32m8_b4(v_receiptdate, v_end_date, vl);
        
        // Combine all conditions
        vbool4_t all_conditions = __riscv_vmand_mm_b4(condition1, condition2, vl);
        all_conditions = __riscv_vmand_mm_b4(all_conditions, condition3, vl);
        all_conditions = __riscv_vmand_mm_b4(all_conditions, condition4, vl);
        
        // Store results in bit mask (1 bit per result)
        for (size_t j = 0; j < vl; j++) {
            size_t idx = i + j;
            size_t byte_index = idx / 8;
            size_t bit_index = idx % 8;
            
            // Check if all conditions are met for this element
            bool all_cond_met = 
                commitdate[idx] < receiptdate[idx] && 
                shipdate[idx] < commitdate[idx] && 
                receiptdate[idx] >= start_date && 
                receiptdate[idx] < end_date;
                
            if (all_cond_met) {
                results[byte_index] |= (1 << bit_index);
            }
        }
    }
}

Status RunQuery12(const std::string& orders_file, const std::string& lineitem_file) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    arrow::MemoryPool* pool = arrow::default_memory_pool();
    auto thread_pool = arrow::internal::GetCpuThreadPool();
    
    std::cout << "Reading input files..." << std::endl;
    
    // 1. Load orders table
    std::shared_ptr<arrow::io::ReadableFile> orders_input;
    ARROW_ASSIGN_OR_RAISE(orders_input, arrow::io::ReadableFile::Open(orders_file, pool));
    
    std::unique_ptr<parquet::arrow::FileReader> orders_reader;
    ARROW_ASSIGN_OR_RAISE(orders_reader, parquet::arrow::OpenFile(orders_input, pool));
    
    std::vector<int> orders_indices;
    const auto& schema = orders_reader->parquet_reader()->metadata()->schema();
    for (const auto& col_name : {"o_orderkey", "o_orderpriority"}) {
        int col_idx = schema->ColumnIndex(col_name);
        if (col_idx >= 0) {
            orders_indices.push_back(col_idx);
        } else {
            return Status::Invalid("Column not found: ", col_name);
        }
    }
    
    std::shared_ptr<arrow::Table> orders_table;
    ARROW_RETURN_NOT_OK(orders_reader->ReadTable(orders_indices, &orders_table));
    
    std::cout << "Loaded ORDERS table with " << orders_table->num_rows() << " rows." << std::endl;
    
    // Create a map of orderkey -> orderpriority
    std::map<int64_t, std::string> order_priorities;
    
    auto o_orderkey_col = orders_table->GetColumnByName("o_orderkey");
    auto o_orderpriority_col = orders_table->GetColumnByName("o_orderpriority");
    
    for (int chunk_idx = 0; chunk_idx < o_orderkey_col->num_chunks(); chunk_idx++) {
        auto o_orderkey_array = std::static_pointer_cast<arrow::Int64Array>(o_orderkey_col->chunk(chunk_idx));
        auto o_orderpriority_array = std::static_pointer_cast<arrow::StringArray>(o_orderpriority_col->chunk(chunk_idx));
        
        int64_t num_rows = o_orderkey_array->length();
        for (int64_t i = 0; i < num_rows; i++) {
            if (!o_orderkey_array->IsNull(i) && !o_orderpriority_array->IsNull(i)) {
                int64_t orderkey = o_orderkey_array->Value(i);
                std::string priority = o_orderpriority_array->GetString(i);
                order_priorities[orderkey] = priority;
            }
        }
    }
    
    std::cout << "Loaded " << order_priorities.size() << " order priorities" << std::endl;
    
    // 2. Load lineitem table
    std::shared_ptr<arrow::io::ReadableFile> lineitem_input;
    ARROW_ASSIGN_OR_RAISE(lineitem_input, arrow::io::ReadableFile::Open(lineitem_file, pool));
    
    std::unique_ptr<parquet::arrow::FileReader> lineitem_reader;
    ARROW_ASSIGN_OR_RAISE(lineitem_reader, parquet::arrow::OpenFile(lineitem_input, pool));
    
    std::vector<int> lineitem_indices;
    const auto& lineitem_schema = lineitem_reader->parquet_reader()->metadata()->schema();
    for (const auto& col_name : {"l_orderkey", "l_shipmode", "l_shipdate", "l_commitdate", "l_receiptdate"}) {
        int col_idx = lineitem_schema->ColumnIndex(col_name);
        if (col_idx >= 0) {
            lineitem_indices.push_back(col_idx);
        } else {
            return Status::Invalid("Column not found in lineitem: ", col_name);
        }
    }
    
    std::shared_ptr<arrow::Table> lineitem_table;
    ARROW_RETURN_NOT_OK(lineitem_reader->ReadTable(lineitem_indices, &lineitem_table));
    
    std::cout << "Loaded LINEITEM table with " << lineitem_table->num_rows() << " rows." << std::endl;
    
    // Setup date filters
    int32_t start_date = date_string_to_days("1994-01-01");
    int32_t end_date = date_string_to_days("1995-01-01");
    
    // Target ship modes
    std::set<std::string> target_shipmodes = {"MAIL", "SHIP"};
    
    // Process lineitem table with RVV acceleration
    std::map<std::string, Query12Result> results_by_shipmode;
    int64_t rows_processed = 0;
    int64_t rows_qualified = 0;
    
    auto l_orderkey_col = lineitem_table->GetColumnByName("l_orderkey");
    auto l_shipmode_col = lineitem_table->GetColumnByName("l_shipmode");
    auto l_shipdate_col = lineitem_table->GetColumnByName("l_shipdate");
    auto l_commitdate_col = lineitem_table->GetColumnByName("l_commitdate");
    auto l_receiptdate_col = lineitem_table->GetColumnByName("l_receiptdate");
    
    for (int chunk_idx = 0; chunk_idx < l_orderkey_col->num_chunks(); chunk_idx++) {
        auto l_orderkey_array = std::static_pointer_cast<arrow::Int64Array>(l_orderkey_col->chunk(chunk_idx));
        auto l_shipmode_array = std::static_pointer_cast<arrow::StringArray>(l_shipmode_col->chunk(chunk_idx));
        auto l_shipdate_array = std::static_pointer_cast<arrow::Date32Array>(l_shipdate_col->chunk(chunk_idx));
        auto l_commitdate_array = std::static_pointer_cast<arrow::Date32Array>(l_commitdate_col->chunk(chunk_idx));
        auto l_receiptdate_array = std::static_pointer_cast<arrow::Date32Array>(l_receiptdate_col->chunk(chunk_idx));
        
        int64_t num_rows = l_orderkey_array->length();
        rows_processed += num_rows;
        
        // Create a bit mask to store results (1 bit per row)
        size_t num_bytes = (num_rows + 7) / 8;
        std::vector<uint8_t> qualified_mask(num_bytes, 0);
        
        // Use RVV to accelerate date comparisons
        check_shipping_conditions_rvv(
            l_shipdate_array->raw_values(),
            l_commitdate_array->raw_values(),
            l_receiptdate_array->raw_values(),
            start_date,
            end_date,
            qualified_mask.data(),
            num_rows
        );
        
        // Process qualified rows
        for (int64_t i = 0; i < num_rows; i++) {
            // Check if this row qualified using the bit mask
            size_t byte_index = i / 8;
            size_t bit_index = i % 8;
            bool qualified = (qualified_mask[byte_index] & (1 << bit_index)) != 0;
            
            if (!qualified) continue;
            
            // Check for null values
            if (l_orderkey_array->IsNull(i) || l_shipmode_array->IsNull(i)) continue;
            
            // Check if shipmode is in target set (MAIL or SHIP)
            std::string shipmode = l_shipmode_array->GetString(i);
            if (target_shipmodes.find(shipmode) == target_shipmodes.end()) continue;
            
            int64_t orderkey = l_orderkey_array->Value(i);
            
            // Check if we have the order priority
            auto priority_it = order_priorities.find(orderkey);
            if (priority_it == order_priorities.end()) continue;
            
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
    std::cout << "\nTPC-H Query 12 Results (RVV-accelerated):" << std::endl;
    std::cout << "---------------------------------------" << std::endl;
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
    
    return Status::OK();
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <orders_parquet> <lineitem_parquet>" << std::endl;
        return 1;
    }
    
    std::string orders_file = argv[1];
    std::string lineitem_file = argv[2];
    
    Status st = RunQuery12(orders_file, lineitem_file);
    
    if (!st.ok()) {
        std::cerr << "Error: " << st.ToString() << std::endl;
        return 1;
    }
    
    return 0;
}