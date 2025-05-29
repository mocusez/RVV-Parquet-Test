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
#include <memory>
#include <unordered_set>
#include <vector>
#include <ctime>

using arrow::Status;

// commitdate < receiptdate
void check_late_delivery_rvv(const int32_t* commitdates, const int32_t* receiptdates,
                            uint8_t* results, size_t length) {
  size_t i = 0;
  
  size_t vl;
  
  for (; i < length; i += vl) {
    vl = __riscv_vsetvl_e32m8(length - i);
    
    vint32m8_t v_commit = __riscv_vle32_v_i32m8(commitdates + i, vl);
    vint32m8_t v_receipt = __riscv_vle32_v_i32m8(receiptdates + i, vl);
    
    vbool4_t v_mask = __riscv_vmslt_vv_i32m8_b4(v_commit, v_receipt, vl);
    
    for (size_t j = 0; j < vl; j++) {
      size_t idx = i + j;
      size_t byte_index = idx / 8;
      size_t bit_index = idx % 8;
      
      if (commitdates[i + j] < receiptdates[i + j]) {
        results[byte_index] |= (1 << bit_index);
      }
    }
  }
}

Status RunQuery4(const std::string& orders_file, const std::string& lineitem_file) {
  arrow::MemoryPool* pool = arrow::default_memory_pool();
  
  auto thread_pool = arrow::internal::GetCpuThreadPool();
  
  std::cout << "Reading input files..." << std::endl;
  
  std::shared_ptr<arrow::io::ReadableFile> orders_input;
  ARROW_ASSIGN_OR_RAISE(orders_input, arrow::io::ReadableFile::Open(orders_file, pool));
  
  std::unique_ptr<parquet::arrow::FileReader> orders_reader;
  ARROW_ASSIGN_OR_RAISE(orders_reader, parquet::arrow::OpenFile(orders_input, pool));
  
  std::vector<int> orders_indices;
  const auto& schema = orders_reader->parquet_reader()->metadata()->schema();
  for (const auto& col_name : {"o_orderkey", "o_orderdate", "o_orderpriority"}) {
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
  
  std::shared_ptr<arrow::io::ReadableFile> lineitem_input;
  ARROW_ASSIGN_OR_RAISE(lineitem_input, arrow::io::ReadableFile::Open(lineitem_file, pool));
  
  std::unique_ptr<parquet::arrow::FileReader> lineitem_reader;
  ARROW_ASSIGN_OR_RAISE(lineitem_reader, parquet::arrow::OpenFile(lineitem_input, pool));
  
  std::vector<int> lineitem_indices;
  const auto& lineitem_schema = lineitem_reader->parquet_reader()->metadata()->schema();
  for (const auto& col_name : {"l_orderkey", "l_commitdate", "l_receiptdate"}) {
    int col_idx = lineitem_schema->ColumnIndex(col_name);
    if (col_idx >= 0) {
      lineitem_indices.push_back(col_idx);
    } else {
      return Status::Invalid("Column not found: ", col_name);
    }
  }
  
  std::shared_ptr<arrow::Table> lineitem_table;
  ARROW_RETURN_NOT_OK(lineitem_reader->ReadTable(lineitem_indices, &lineitem_table));
  auto start_time = std::chrono::high_resolution_clock::now();
  
  auto orderdate_col = orders_table->GetColumnByName("o_orderdate");
  
  std::shared_ptr<arrow::Scalar> start_date, end_date;
  auto date_type = arrow::date32();
  ARROW_ASSIGN_OR_RAISE(start_date, arrow::Scalar::Parse(date_type, "1993-07-01"));
  ARROW_ASSIGN_OR_RAISE(end_date, arrow::Scalar::Parse(date_type, "1993-10-01"));
  
  arrow::compute::CompareOptions ge_options(arrow::compute::CompareOperator::GREATER_EQUAL);
  arrow::Datum filter1_datum;
  ARROW_ASSIGN_OR_RAISE(filter1_datum, 
      arrow::compute::CallFunction("greater_equal", {orderdate_col, start_date}));
  
  arrow::compute::CompareOptions lt_options(arrow::compute::CompareOperator::LESS);
  arrow::Datum filter2_datum;
  ARROW_ASSIGN_OR_RAISE(filter2_datum, 
      arrow::compute::CallFunction("less", {orderdate_col, end_date}));
  
  arrow::Datum combined_filter;
  ARROW_ASSIGN_OR_RAISE(combined_filter, 
      arrow::compute::And(filter1_datum, filter2_datum));
  
  arrow::compute::FilterOptions filter_options;
  arrow::Datum filtered_orders_datum;
  ARROW_ASSIGN_OR_RAISE(filtered_orders_datum, 
      arrow::compute::Filter(orders_table, combined_filter, filter_options));
  
  auto filtered_orders = filtered_orders_datum.table();
  
  std::cout << "Filtered ORDERS table has " << filtered_orders->num_rows() 
            << " rows within date range." << std::endl;
  
  auto commit_col = lineitem_table->GetColumnByName("l_commitdate");
  auto receipt_col = lineitem_table->GetColumnByName("l_receiptdate");
  
  auto commit_array = std::static_pointer_cast<arrow::Date32Array>(commit_col->chunk(0));
  auto receipt_array = std::static_pointer_cast<arrow::Date32Array>(receipt_col->chunk(0));
  
  size_t num_lineitem_rows = lineitem_table->num_rows();
  size_t num_bytes = (num_lineitem_rows + 7) / 8;
  
  std::vector<uint8_t> late_delivery_mask(num_bytes, 0);
  
  check_late_delivery_rvv(
      commit_array->raw_values(), 
      receipt_array->raw_values(),
      late_delivery_mask.data(), 
      num_lineitem_rows);
  
  auto lineitem_keys = std::static_pointer_cast<arrow::Int64Array>(
      lineitem_table->GetColumnByName("l_orderkey")->chunk(0));
  
  std::unordered_set<int64_t> late_order_keys;
  
  for (size_t i = 0; i < num_lineitem_rows; ++i) {
    size_t byte_index = i / 8;
    size_t bit_index = i % 8;
    bool is_late = (late_delivery_mask[byte_index] & (1 << bit_index)) != 0;
    
    if (is_late) {
      late_order_keys.insert(lineitem_keys->Value(i));
    }
  }
  
  std::cout << "Found " << late_order_keys.size() 
            << " orders with late deliveries." << std::endl;
  
  auto order_keys = std::static_pointer_cast<arrow::Int64Array>(
      filtered_orders->GetColumnByName("o_orderkey")->chunk(0));
  auto order_priorities = std::static_pointer_cast<arrow::StringArray>(
      filtered_orders->GetColumnByName("o_orderpriority")->chunk(0));
  
  std::map<std::string, int> priority_counts;
  
  size_t num_filtered_orders = filtered_orders->num_rows();
  for (size_t i = 0; i < num_filtered_orders; ++i) {
    int64_t order_key = order_keys->Value(i);
    
    if (late_order_keys.find(order_key) != late_order_keys.end()) {
      std::string priority = order_priorities->GetString(i);
      priority_counts[priority]++;
    }
  }
  

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end_time - start_time;
  std::cout << "\nQuery executed in " << elapsed.count() << " seconds" << std::endl;
  
  std::cout << "\nTPC-H Query 4 Results (with RVV 1.0 optimization):\n";
  std::cout << "---------------------------------------------\n";
  std::cout << "O_ORDERPRIORITY | ORDER_COUNT\n";
  std::cout << "----------------+-------------\n";
  
  for (const auto& entry : priority_counts) {
    std::cout << entry.first << " | " << entry.second << std::endl;
  }
  
  return Status::OK();
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <orders_parquet> <lineitem_parquet>" << std::endl;
    return 1;
  }
  
  std::string orders_file = argv[1];
  std::string lineitem_file = argv[2];
  
  Status st = RunQuery4(orders_file, lineitem_file);
  
  if (!st.ok()) {
    std::cerr << "Error: " << st.ToString() << std::endl;
    return 1;
  }
  
  return 0;
}