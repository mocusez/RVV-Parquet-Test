#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <arrow/dataset/api.h>
#include <arrow/io/api.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/table.h>
#include <parquet/arrow/reader.h>

#include <riscv_vector.h>

#include <ctime>
#include <iomanip> // For std::setw, std::fixed, std::setprecision
#include <iostream>
#include <map> // For std::map
#include <memory>
#include <vector>

// Calculate sum_disc_price = l_extendedprice * (1 - l_discount)
void compute_disc_price_rvv(const float *extendedprice, const float *discount,
                            float *result, size_t n) {
  size_t i = 0;

  size_t vl;

  for (; i < n; i += vl) {
    vl = __riscv_vsetvl_e32m8(n - i);

    vfloat32m8_t v_price = __riscv_vle32_v_f32m8(extendedprice + i, vl);
    vfloat32m8_t v_discount = __riscv_vle32_v_f32m8(discount + i, vl);

    vfloat32m8_t v_one = __riscv_vfmv_v_f_f32m8(1.0f, vl);
    vfloat32m8_t v_one_minus_discount =
        __riscv_vfsub_vv_f32m8(v_one, v_discount, vl);

    vfloat32m8_t v_disc_price =
        __riscv_vfmul_vv_f32m8(v_price, v_one_minus_discount, vl);

    __riscv_vse32_v_f32m8(result + i, v_disc_price, vl);
  }
}

void compute_charge_rvv(const float *disc_price, const float *tax,
                        float *result, size_t n) {
  size_t i = 0;

  size_t vl;

  for (; i < n; i += vl) {
    vl = __riscv_vsetvl_e32m8(n - i);

    vfloat32m8_t v_disc_price = __riscv_vle32_v_f32m8(disc_price + i, vl);
    vfloat32m8_t v_tax = __riscv_vle32_v_f32m8(tax + i, vl);

    vfloat32m8_t v_one = __riscv_vfmv_v_f_f32m8(1.0f, vl);
    vfloat32m8_t v_one_plus_tax = __riscv_vfadd_vv_f32m8(v_one, v_tax, vl);

    vfloat32m8_t v_charge =
        __riscv_vfmul_vv_f32m8(v_disc_price, v_one_plus_tax, vl);

    __riscv_vse32_v_f32m8(result + i, v_charge, vl);
  }
}

float sum_rvv(const float *data, size_t n, const char *name = "unknown") {
  float sum = 0.0f;
  size_t i = 0;
  size_t vl;

  // Print first few values to verify data
  // std::cout << "Summing " << name << " - First 5 values: ";
  // for (size_t j = 0; j < std::min(n, size_t(5)); j++) {
  //   std::cout << data[j] << " ";
  // }
  // std::cout << std::endl;

  vfloat32m8_t v_sum = __riscv_vfmv_v_f_f32m8(0.0f, __riscv_vsetvlmax_e32m8());

  for (; i < n; i += vl) {
    vl = __riscv_vsetvl_e32m8(n - i);
    vfloat32m8_t v_data = __riscv_vle32_v_f32m8(data + i, vl);
    v_sum = __riscv_vfadd_vv_f32m8(v_sum, v_data, vl);

    // Occasionally print intermediate sum
    // if (i > 0 && i % 1000000 == 0) {
    //   vfloat32m1_t v_temp = __riscv_vfredusum_vs_f32m8_f32m1(v_sum,
    //   __riscv_vfmv_v_f_f32m1(0.0f, 1), __riscv_vsetvlmax_e32m8()); float
    //   temp_sum = __riscv_vfmv_f_s_f32m1_f32(v_temp); std::cout <<
    //   "Intermediate " << name << " sum at " << i << " rows: " << temp_sum <<
    //   std::endl;
    // }
  }

  vfloat32m1_t v_reduced = __riscv_vfredusum_vs_f32m8_f32m1(
      v_sum, __riscv_vfmv_v_f_f32m1(0.0f, 1), __riscv_vsetvlmax_e32m8());
  sum = __riscv_vfmv_f_s_f32m1_f32(v_reduced);

  return sum;
}

arrow::Status RunQuery1RVV(const std::string &file_path) {
  arrow::MemoryPool *pool = arrow::default_memory_pool();

  std::shared_ptr<arrow::io::ReadableFile> input_file;
  ARROW_ASSIGN_OR_RAISE(input_file,
                        arrow::io::ReadableFile::Open(file_path, pool));

  std::unique_ptr<parquet::arrow::FileReader> reader;
  ARROW_ASSIGN_OR_RAISE(reader, parquet::arrow::OpenFile(input_file, pool));

  std::shared_ptr<arrow::Table> table;
  ARROW_RETURN_NOT_OK(reader->ReadTable(&table));

  auto start_time = std::chrono::high_resolution_clock::now();

  // std::cout << "Loaded table with " << table->num_rows() << " rows and "
  //           << table->num_columns() << " columns." << std::endl;

  int l_shipdate_idx = table->schema()->GetFieldIndex("l_shipdate");
  int l_returnflag_idx = table->schema()->GetFieldIndex("l_returnflag");
  int l_linestatus_idx = table->schema()->GetFieldIndex("l_linestatus");
  int l_quantity_idx = table->schema()->GetFieldIndex("l_quantity");
  int l_extendedprice_idx = table->schema()->GetFieldIndex("l_extendedprice");
  int l_discount_idx = table->schema()->GetFieldIndex("l_discount");
  int l_tax_idx = table->schema()->GetFieldIndex("l_tax");

  // l_shipdate <= '1998-09-02'
  auto shipdate_col = table->column(l_shipdate_idx);
  std::shared_ptr<arrow::Scalar> cutoff_date;
  auto date_type = arrow::date32();
  ARROW_ASSIGN_OR_RAISE(cutoff_date,
                        arrow::Scalar::Parse(date_type, "1998-09-02"));

  arrow::compute::ExecContext ctx(pool);
  arrow::compute::FilterOptions filter_options;

  arrow::compute::CompareOptions compare_options(
      arrow::compute::CompareOperator::LESS_EQUAL);
  arrow::Datum filter_datum;
  ARROW_ASSIGN_OR_RAISE(filter_datum,
                        arrow::compute::CallFunction(
                            "less_equal", {shipdate_col, cutoff_date}, &ctx));

  arrow::Datum filtered_datum;
  ARROW_ASSIGN_OR_RAISE(
      filtered_datum,
      arrow::compute::Filter(table, filter_datum, filter_options, &ctx));

  auto filtered_table = filtered_datum.table();

  // std::cout << "Filtered table has " << filtered_table->num_rows() << " rows."
  //           << std::endl;

  const auto &l_extendedprice = filtered_table->column(l_extendedprice_idx);
  const auto &l_discount = filtered_table->column(l_discount_idx);
  const auto &l_tax = filtered_table->column(l_tax_idx);
  const auto &l_quantity = filtered_table->column(l_quantity_idx);

  size_t num_rows = filtered_table->num_rows();

  std::vector<float> price_data(num_rows);
  std::vector<float> discount_data(num_rows);
  std::vector<float> tax_data(num_rows);
  std::vector<float> quantity_data(num_rows);

  // Check first few values
  std::shared_ptr<arrow::Decimal128Array> price_array, discount_array,
      tax_array, quantity_array;

  // Cast to proper array types based on detected types
  if (l_extendedprice->chunk(0)->type_id() == arrow::Type::DECIMAL128) {
    price_array = std::static_pointer_cast<arrow::Decimal128Array>(
        l_extendedprice->chunk(0));
    discount_array =
        std::static_pointer_cast<arrow::Decimal128Array>(l_discount->chunk(0));
    tax_array =
        std::static_pointer_cast<arrow::Decimal128Array>(l_tax->chunk(0));
    quantity_array =
        std::static_pointer_cast<arrow::Decimal128Array>(l_quantity->chunk(0));

    // Extract scale factor for conversion
    auto decimal_type =
        std::static_pointer_cast<arrow::DecimalType>(price_array->type());
    int32_t scale_factor = decimal_type->scale();

    // directly
    for (size_t i = 0; i < num_rows; i++) {
      // Create Decimal128 objects using the direct constructor with bytes
      arrow::Decimal128 price_val(price_array->Value(i));
      arrow::Decimal128 discount_val(discount_array->Value(i));
      arrow::Decimal128 tax_val(tax_array->Value(i));
      arrow::Decimal128 quantity_val(quantity_array->Value(i));

      price_data[i] = price_val.ToDouble(scale_factor);
      discount_data[i] = discount_val.ToDouble(scale_factor);
      tax_data[i] = tax_val.ToDouble(scale_factor);
      quantity_data[i] = quantity_val.ToDouble(scale_factor);
    }
  }

  std::vector<float> disc_price_data(num_rows);
  std::vector<float> charge_data(num_rows);

  compute_disc_price_rvv(price_data.data(), discount_data.data(),
                         disc_price_data.data(), num_rows);
  compute_charge_rvv(disc_price_data.data(), tax_data.data(),
                     charge_data.data(), num_rows);

  const auto &l_returnflag = filtered_table->column(l_returnflag_idx);
  const auto &l_linestatus = filtered_table->column(l_linestatus_idx);

  // We'll use a map with (returnflag, linestatus) as key to store the groups
  std::map<std::pair<std::string, std::string>, std::vector<size_t>> groups;

  // Extract returnflag and linestatus strings for each row
  for (size_t i = 0; i < num_rows; i++) {
    std::string returnflag =
        std::static_pointer_cast<arrow::StringArray>(l_returnflag->chunk(0))
            ->GetString(i);
    std::string linestatus =
        std::static_pointer_cast<arrow::StringArray>(l_linestatus->chunk(0))
            ->GetString(i);

    groups[{returnflag, linestatus}].push_back(i);
  }

  // Sort the keys for ORDER BY l_returnflag, l_linestatus
  std::vector<std::pair<std::string, std::string>> sorted_keys;
  for (const auto &group : groups) {
    sorted_keys.push_back(group.first);
  }
  std::sort(sorted_keys.begin(), sorted_keys.end());

  // Print header in SQL-like format
  std::cout << "\nL_RETURNFLAG | L_LINESTATUS | SUM_QTY | SUM_BASE_PRICE | SUM_DISC_PRICE | SUM_CHARGE | AVG_QTY | AVG_PRICE | AVG_DISC | COUNT_ORDER\n"; 
  std::cout << "------------|-------------|---------|---------------|---------------|-----------|---------|-----------|----------|------------\n";

  // Calculate aggregates for each group using RVV functions
  for (const auto &key : sorted_keys) {
    const auto &indices = groups[key];
    size_t group_size = indices.size();

    // Create temporary vectors for this group
    std::vector<float> group_qty(group_size);
    std::vector<float> group_price(group_size);
    std::vector<float> group_disc(group_size);
    std::vector<float> group_disc_price(group_size);
    std::vector<float> group_charge(group_size);

    // Fill group vectors
    for (size_t i = 0; i < group_size; i++) {
      size_t idx = indices[i];
      group_qty[i] = quantity_data[idx];
      group_price[i] = price_data[idx];
      group_disc[i] = discount_data[idx];
      group_disc_price[i] = disc_price_data[idx];
      group_charge[i] = charge_data[idx];
    }

    std::string qty_name = key.first + key.second + "_qty";
    std::string price_name = key.first + key.second + "_price";
    std::string disc_price_name = key.first + key.second + "_disc_price";
    std::string charge_name = key.first + key.second + "_charge";
    std::string disc_name = key.first + key.second + "_disc";

    // Use RVV to calculate sums for this group
    float sum_qty = sum_rvv(group_qty.data(), group_size, qty_name.c_str());
    float sum_price =
        sum_rvv(group_price.data(), group_size, price_name.c_str());
    float sum_disc_price =
        sum_rvv(group_disc_price.data(), group_size, disc_price_name.c_str());
    float sum_charge =
        sum_rvv(group_charge.data(), group_size, charge_name.c_str());
    float sum_disc = sum_rvv(group_disc.data(), group_size, disc_name.c_str());

    // Calculate averages
    float avg_qty = sum_qty / group_size;
    float avg_price = sum_price / group_size;
    float avg_disc = sum_disc / group_size;

    
    // Print in SQL-like format with proper alignment
    std::cout << std::setw(12) << key.first << " | " << std::setw(11)
              << key.second << " | " << std::setw(7) << std::fixed
              << std::setprecision(2) << sum_qty << " | " << std::setw(13)
              << std::fixed << std::setprecision(2) << sum_price << " | "
              << std::setw(13) << std::fixed << std::setprecision(2)
              << sum_disc_price << " | " << std::setw(9) << std::fixed
              << std::setprecision(2) << sum_charge << " | " << std::setw(7)
              << std::fixed << std::setprecision(2) << avg_qty << " | "
              << std::setw(9) << std::fixed << std::setprecision(2) << avg_price
              << " | " << std::setw(8) << std::fixed << std::setprecision(2)
              << avg_disc << " | " << std::setw(10) << group_size << "\n";
  }
  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end_time - start_time;
  std::cout << "\nQuery executed in " << elapsed.count() << " seconds" << std::endl;

  return arrow::Status::OK();
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <lineitem parquet_file>" << std::endl;
    return 1;
  }

  std::string file_path = argv[1];
  arrow::Status st = RunQuery1RVV(file_path);

  if (!st.ok()) {
    std::cerr << "Error: " << st.ToString() << std::endl;
    return 1;
  }

  return 0;
}