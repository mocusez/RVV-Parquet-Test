#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <arrow/compute/api_scalar.h>
#include <arrow/io/api.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <parquet/arrow/reader.h>

#include <riscv_vector.h>

#include <chrono>
#include <iostream>
#include <memory>
#include <ctime>
#include <iomanip>  // Add this for std::setprecision

using arrow::Status;

// SUM(l_extendedprice * l_discount)
double compute_revenue_rvv(const float* price_data, const float* discount_data, size_t length) {
  double sum = 0.0;
  size_t i = 0;
  
  size_t vl;
  
  vfloat32m8_t v_sum = __riscv_vfmv_v_f_f32m8(0.0f, __riscv_vsetvlmax_e32m8());
  
  for (; i < length; i += vl) {
    vl = __riscv_vsetvl_e32m8(length - i);
    
    vfloat32m8_t v_price = __riscv_vle32_v_f32m8(price_data + i, vl);
    vfloat32m8_t v_discount = __riscv_vle32_v_f32m8(discount_data + i, vl);
    
    vfloat32m8_t v_revenue = __riscv_vfmul_vv_f32m8(v_price, v_discount, vl);

    v_sum = __riscv_vfadd_vv_f32m8(v_sum, v_revenue, vl);
  }
  
  vfloat32m1_t v_reduced = __riscv_vfredusum_vs_f32m8_f32m1(
      v_sum, __riscv_vfmv_v_f_f32m1(0.0f, 1), __riscv_vsetvlmax_e32m8());
  
  float result = __riscv_vfmv_f_s_f32m1_f32(v_reduced);
  sum = static_cast<double>(result);
  
  return sum;
}

Status RunQuery6(const std::string& file_path) {
  auto start_time = std::chrono::high_resolution_clock::now();
  
  arrow::MemoryPool* pool = arrow::default_memory_pool();
  
  std::shared_ptr<arrow::io::ReadableFile> input_file;
  ARROW_ASSIGN_OR_RAISE(input_file, arrow::io::ReadableFile::Open(file_path, pool));
  
  std::unique_ptr<parquet::arrow::FileReader> reader;
  ARROW_ASSIGN_OR_RAISE(reader, parquet::arrow::OpenFile(input_file, pool));
  
  std::vector<int> column_indices;
  const auto& schema = reader->parquet_reader()->metadata()->schema();
  for (const auto& col_name : {"l_shipdate", "l_discount", "l_extendedprice", "l_quantity"}) {
    int col_idx = schema->ColumnIndex(col_name);
    if (col_idx >= 0) {
      column_indices.push_back(col_idx);
    } else {
      return Status::Invalid("Column not found: ", col_name);
    }
  }
  
  std::shared_ptr<arrow::Table> table;
  ARROW_RETURN_NOT_OK(reader->ReadTable(column_indices, &table));
  
  std::cout << "Loaded table with " << table->num_rows() << " rows." << std::endl;
  
  std::cout << "Column data types: "
            << table->GetColumnByName("l_extendedprice")->type()->ToString() << ", "
            << table->GetColumnByName("l_discount")->type()->ToString() << ", "
            << table->GetColumnByName("l_quantity")->type()->ToString() << std::endl;
  
  // 1. l_shipdate >= DATE '1994-01-01'
  auto shipdate_col = table->GetColumnByName("l_shipdate");
  
  std::shared_ptr<arrow::Scalar> start_date, end_date;
  auto date_type = arrow::date32();
  ARROW_ASSIGN_OR_RAISE(start_date, arrow::Scalar::Parse(date_type, "1994-01-01"));
  ARROW_ASSIGN_OR_RAISE(end_date, arrow::Scalar::Parse(date_type, "1995-01-01"));
  
  arrow::compute::CompareOptions ge_options(arrow::compute::CompareOperator::GREATER_EQUAL);
  arrow::Datum filter1_datum;
  ARROW_ASSIGN_OR_RAISE(filter1_datum, 
      arrow::compute::CallFunction("greater_equal", {shipdate_col, start_date}));
  
  // 2. l_shipdate < DATE '1995-01-01'
  arrow::compute::CompareOptions lt_options(arrow::compute::CompareOperator::LESS);
  arrow::Datum filter2_datum;
  ARROW_ASSIGN_OR_RAISE(filter2_datum, 
      arrow::compute::CallFunction("less", {shipdate_col, end_date}));
  
  // 3. l_discount BETWEEN 0.05 AND 0.07
  auto discount_col = table->GetColumnByName("l_discount");
  
  std::shared_ptr<arrow::Scalar> min_discount, max_discount;
  if (discount_col->type()->id() == arrow::Type::DECIMAL128) {
    auto decimal_type = std::static_pointer_cast<arrow::DecimalType>(discount_col->type());
    int32_t scale = decimal_type->scale();
    int32_t precision = decimal_type->precision();
    
    ARROW_ASSIGN_OR_RAISE(auto min_val, 
        arrow::Decimal128::FromReal(0.05, precision, scale));
    ARROW_ASSIGN_OR_RAISE(auto max_val,
        arrow::Decimal128::FromReal(0.07, precision, scale));
    
    min_discount = std::make_shared<arrow::Decimal128Scalar>(min_val, decimal_type);
    max_discount = std::make_shared<arrow::Decimal128Scalar>(max_val, decimal_type);
  } else {
    ARROW_ASSIGN_OR_RAISE(min_discount, arrow::Scalar::Parse(arrow::float64(), "0.05"));
    ARROW_ASSIGN_OR_RAISE(max_discount, arrow::Scalar::Parse(arrow::float64(), "0.07"));
  }
  
  arrow::Datum filter3_datum;
  ARROW_ASSIGN_OR_RAISE(filter3_datum, 
      arrow::compute::CallFunction("greater_equal", {discount_col, min_discount}));
  
  arrow::Datum filter4_datum;
  ARROW_ASSIGN_OR_RAISE(filter4_datum, 
      arrow::compute::CallFunction("less_equal", {discount_col, max_discount}));
  
  auto quantity_col = table->GetColumnByName("l_quantity");
  
  std::shared_ptr<arrow::Scalar> max_quantity;
  if (quantity_col->type()->id() == arrow::Type::DECIMAL128) {
    auto decimal_type = std::static_pointer_cast<arrow::DecimalType>(quantity_col->type());
    int32_t scale = decimal_type->scale();
    int32_t precision = decimal_type->precision();
    
    ARROW_ASSIGN_OR_RAISE(auto qty_val, 
        arrow::Decimal128::FromReal(24.0, precision, scale));
    
    max_quantity = std::make_shared<arrow::Decimal128Scalar>(qty_val, decimal_type);
  } else {
    ARROW_ASSIGN_OR_RAISE(max_quantity, arrow::Scalar::Parse(arrow::float64(), "24"));
  }
  
  arrow::Datum filter5_datum;
  ARROW_ASSIGN_OR_RAISE(filter5_datum, 
      arrow::compute::CallFunction("less", {quantity_col, max_quantity}));
  
  arrow::Datum combined_filter;
  ARROW_ASSIGN_OR_RAISE(combined_filter, 
      arrow::compute::And(filter1_datum, filter2_datum));
  ARROW_ASSIGN_OR_RAISE(combined_filter, 
      arrow::compute::And(combined_filter, filter3_datum));
  ARROW_ASSIGN_OR_RAISE(combined_filter, 
      arrow::compute::And(combined_filter, filter4_datum));
  ARROW_ASSIGN_OR_RAISE(combined_filter, 
      arrow::compute::And(combined_filter, filter5_datum));
  
  arrow::compute::FilterOptions filter_options;
  arrow::Datum filtered_datum;
  ARROW_ASSIGN_OR_RAISE(filtered_datum, 
      arrow::compute::Filter(table, combined_filter, filter_options));
  
  auto filtered_table = filtered_datum.table();
  
  std::cout << "Filtered table has " << filtered_table->num_rows() << " rows." << std::endl;
  
  size_t num_rows = filtered_table->num_rows();
  std::vector<float> price_data(num_rows);
  std::vector<float> discount_data(num_rows);
  
  auto price_col = filtered_table->GetColumnByName("l_extendedprice");
  auto discount_col_filtered = filtered_table->GetColumnByName("l_discount");
  
  auto price_array = std::static_pointer_cast<arrow::Decimal128Array>(price_col->chunk(0));
  auto discount_array = std::static_pointer_cast<arrow::Decimal128Array>(discount_col_filtered->chunk(0));
  
  auto decimal_type = std::static_pointer_cast<arrow::DecimalType>(price_array->type());
  int32_t scale_factor = decimal_type->scale();
  
  for (size_t i = 0; i < num_rows; ++i) {
    arrow::Decimal128 price_val(price_array->Value(i));
    arrow::Decimal128 discount_val(discount_array->Value(i));
    
    price_data[i] = price_val.ToDouble(scale_factor);
    discount_data[i] = discount_val.ToDouble(scale_factor);
  }
  
  double revenue = compute_revenue_rvv(price_data.data(), discount_data.data(), num_rows);
  
  std::cout << "\nTPC-H Query 6 Result (with RVV 1.0 optimization):\n";
  std::cout << "---------------------------------------------\n";
  std::cout << "REVENUE\n";
  std::cout << "-------\n";
  std::cout << std::fixed << std::setprecision(2) << revenue << std::endl;

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end_time - start_time;
  std::cout << "\nQuery executed in " << elapsed.count() << " seconds" << std::endl;
  
  return Status::OK();
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <lineitem parquet_file>" << std::endl;
    return 1;
  }
  
  std::string file_path = argv[1];
  Status st = RunQuery6(file_path);
  
  if (!st.ok()) {
    std::cerr << "Error: " << st.ToString() << std::endl;
    return 1;
  }
  
  return 0;
}