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

#include <riscv_vector.h>

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
#include <ctime>

using namespace arrow;
using namespace arrow::compute;

// TPC-H Query 9 result structure
struct Query9Result {
    std::string nation;
    int32_t o_year;
    double sum_profit;
    
    bool operator<(const Query9Result& other) const {
        if (nation != other.nation) {
            return nation < other.nation;
        }
        // Sort by year in descending order
        return o_year > other.o_year;
    }
};

// Helper structure for grouping
struct NationYearKey {
    std::string nation;
    int32_t year;
    
    bool operator<(const NationYearKey& other) const {
        if (nation != other.nation) {
            return nation < other.nation;
        }
        return year < other.year;
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

// Extract year from days since epoch
int32_t days_to_year(int32_t days) {
    std::time_t time = static_cast<std::time_t>(days) * 24 * 60 * 60;
    std::tm* tm = std::gmtime(&time);
    return tm->tm_year + 1900;
}

// Check if a string contains a substring (case sensitive)
bool contains_substring(const std::string& str, const std::string& substr) {
    return str.find(substr) != std::string::npos;
}

// Calculate extendedprice * (1 - discount) using RVV
void compute_disc_price_rvv(const float* extendedprice, const float* discount,
                            float* result, size_t n) {
    size_t i = 0;
    size_t vl;

    for (; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m8(n - i);

        vfloat32m8_t v_price = __riscv_vle32_v_f32m8(extendedprice + i, vl);
        vfloat32m8_t v_discount = __riscv_vle32_v_f32m8(discount + i, vl);

        // Calculate 1 - discount
        vfloat32m8_t v_one = __riscv_vfmv_v_f_f32m8(1.0f, vl);
        vfloat32m8_t v_one_minus_discount = __riscv_vfsub_vv_f32m8(v_one, v_discount, vl);

        // Calculate extendedprice * (1 - discount)
        vfloat32m8_t v_disc_price = __riscv_vfmul_vv_f32m8(v_price, v_one_minus_discount, vl);

        // Store results
        __riscv_vse32_v_f32m8(result + i, v_disc_price, vl);
    }
}

// Calculate supplycost * quantity using RVV
void compute_supply_cost_rvv(const float* supplycost, const float* quantity,
                            float* result, size_t n) {
    size_t i = 0;
    size_t vl;

    for (; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m8(n - i);

        vfloat32m8_t v_cost = __riscv_vle32_v_f32m8(supplycost + i, vl);
        vfloat32m8_t v_quantity = __riscv_vle32_v_f32m8(quantity + i, vl);

        // Calculate supplycost * quantity
        vfloat32m8_t v_supply_cost = __riscv_vfmul_vv_f32m8(v_cost, v_quantity, vl);

        // Store results
        __riscv_vse32_v_f32m8(result + i, v_supply_cost, vl);
    }
}

// Calculate profit amount = disc_price - supply_cost using RVV
void compute_profit_rvv(const float* disc_price, const float* supply_cost,
                        float* result, size_t n) {
    size_t i = 0;
    size_t vl;

    for (; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m8(n - i);

        vfloat32m8_t v_disc_price = __riscv_vle32_v_f32m8(disc_price + i, vl);
        vfloat32m8_t v_supply_cost = __riscv_vle32_v_f32m8(supply_cost + i, vl);

        // Calculate disc_price - supply_cost
        vfloat32m8_t v_profit = __riscv_vfsub_vv_f32m8(v_disc_price, v_supply_cost, vl);

        // Store results
        __riscv_vse32_v_f32m8(result + i, v_profit, vl);
    }
}

// Process a batch of records to compute profit using RVV
void batch_process_profit(
    const float* price_data, 
    const float* discount_data,
    const float* quantity_data,
    const float* supplycost_data,
    float* profit_data,
    size_t batch_size) {

    // Temporary arrays for intermediate results
    float* disc_price_data = new float[batch_size];
    float* supply_cost_data = new float[batch_size];
    
    // Calculate disc_price = extendedprice * (1 - discount)
    compute_disc_price_rvv(price_data, discount_data, disc_price_data, batch_size);
    
    // Calculate supply_cost = supplycost * quantity
    compute_supply_cost_rvv(supplycost_data, quantity_data, supply_cost_data, batch_size);
    
    // Calculate profit = disc_price - supply_cost
    compute_profit_rvv(disc_price_data, supply_cost_data, profit_data, batch_size);
    
    // Clean up
    delete[] disc_price_data;
    delete[] supply_cost_data;
}

int main(int argc, char** argv) {
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0] 
                  << " <part.parquet> <supplier.parquet> <lineitem.parquet> "
                  << "<partsupp.parquet> <orders.parquet> <nation.parquet>" << std::endl;
        return 1;
    }

    std::string part_file = argv[1];
    std::string supplier_file = argv[2];
    std::string lineitem_file = argv[3];
    std::string partsupp_file = argv[4];
    std::string orders_file = argv[5];
    std::string nation_file = argv[6];
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Initialize Arrow's memory pool
    arrow::MemoryPool* pool = arrow::default_memory_pool();
    
    // 1. Process part table - filter by p_name like '%green%'
    auto maybe_part_file = arrow::io::ReadableFile::Open(part_file);
    if (!maybe_part_file.ok()) {
        std::cerr << "Could not open part file: " << maybe_part_file.status().ToString() << std::endl;
        return 1;
    }
    std::shared_ptr<arrow::io::ReadableFile> part_infile = *maybe_part_file;
    
    auto part_reader_result = parquet::arrow::OpenFile(part_infile, pool);
    if (!part_reader_result.ok()) {
        std::cerr << "Could not open part parquet file: " << part_reader_result.status().ToString() << std::endl;
        return 1;
    }
    std::unique_ptr<parquet::arrow::FileReader> part_reader = std::move(*part_reader_result);
    
    std::shared_ptr<arrow::Table> part_table;
    auto status = part_reader->ReadTable(&part_table);
    if (!status.ok()) {
        std::cerr << "Could not read part table: " << status.ToString() << std::endl;
        return 1;
    }
    
    std::cout << "Part table loaded with " << part_table->num_rows() << " rows" << std::endl;
    
    // Extract required columns from part
    auto p_partkey_col = part_table->GetColumnByName("p_partkey");
    auto p_name_col = part_table->GetColumnByName("p_name");
    
    // Filter parts where p_name like '%green%'
    std::set<int64_t> green_parts;
    
    for (int chunk_idx = 0; chunk_idx < p_partkey_col->num_chunks(); chunk_idx++) {
        auto p_partkey_array = std::static_pointer_cast<arrow::Int64Array>(p_partkey_col->chunk(chunk_idx));
        auto p_name_array = std::static_pointer_cast<arrow::StringArray>(p_name_col->chunk(chunk_idx));
        
        int64_t num_rows = p_partkey_array->length();
        for (int64_t i = 0; i < num_rows; i++) {
            if (p_partkey_array->IsNull(i) || p_name_array->IsNull(i)) {
                continue;
            }
            
            std::string name = p_name_array->GetString(i);
            if (contains_substring(name, "green")) {
                green_parts.insert(p_partkey_array->Value(i));
            }
        }
    }
    
    std::cout << "Found " << green_parts.size() << " parts with 'green' in the name" << std::endl;
    
    // 2. Process nation table to get nation names
    auto maybe_nation_file = arrow::io::ReadableFile::Open(nation_file);
    if (!maybe_nation_file.ok()) {
        std::cerr << "Could not open nation file: " << maybe_nation_file.status().ToString() << std::endl;
        return 1;
    }
    std::shared_ptr<arrow::io::ReadableFile> nation_infile = *maybe_nation_file;
    
    auto nation_reader_result = parquet::arrow::OpenFile(nation_infile, pool);
    if (!nation_reader_result.ok()) {
        std::cerr << "Could not open nation parquet file: " << nation_reader_result.status().ToString() << std::endl;
        return 1;
    }
    std::unique_ptr<parquet::arrow::FileReader> nation_reader = std::move(*nation_reader_result);
    
    std::shared_ptr<arrow::Table> nation_table;
    status = nation_reader->ReadTable(&nation_table);
    if (!status.ok()) {
        std::cerr << "Could not read nation table: " << status.ToString() << std::endl;
        return 1;
    }
    
    std::cout << "Nation table loaded with " << nation_table->num_rows() << " rows" << std::endl;
    
    // Build a map of nationkey to nation name
    std::map<int64_t, std::string> nation_map;
    
    auto n_nationkey_col = nation_table->GetColumnByName("n_nationkey");
    auto n_name_col = nation_table->GetColumnByName("n_name");
    
    for (int chunk_idx = 0; chunk_idx < n_nationkey_col->num_chunks(); chunk_idx++) {
        auto n_nationkey_array = std::static_pointer_cast<arrow::Int64Array>(n_nationkey_col->chunk(chunk_idx));
        auto n_name_array = std::static_pointer_cast<arrow::StringArray>(n_name_col->chunk(chunk_idx));
        
        int64_t num_rows = n_nationkey_array->length();
        for (int64_t i = 0; i < num_rows; i++) {
            if (n_nationkey_array->IsNull(i) || n_name_array->IsNull(i)) {
                continue;
            }
            
            int64_t nationkey = n_nationkey_array->Value(i);
            std::string name = n_name_array->GetString(i);
            nation_map[nationkey] = name;
        }
    }
    
    // 3. Process supplier table to get supplier nation relationships
    auto maybe_supplier_file = arrow::io::ReadableFile::Open(supplier_file);
    if (!maybe_supplier_file.ok()) {
        std::cerr << "Could not open supplier file: " << maybe_supplier_file.status().ToString() << std::endl;
        return 1;
    }
    std::shared_ptr<arrow::io::ReadableFile> supplier_infile = *maybe_supplier_file;
    
    auto supplier_reader_result = parquet::arrow::OpenFile(supplier_infile, pool);
    if (!supplier_reader_result.ok()) {
        std::cerr << "Could not open supplier parquet file: " << supplier_reader_result.status().ToString() << std::endl;
        return 1;
    }
    std::unique_ptr<parquet::arrow::FileReader> supplier_reader = std::move(*supplier_reader_result);
    
    std::shared_ptr<arrow::Table> supplier_table;
    status = supplier_reader->ReadTable(&supplier_table);
    if (!status.ok()) {
        std::cerr << "Could not read supplier table: " << status.ToString() << std::endl;
        return 1;
    }
    
    std::cout << "Supplier table loaded with " << supplier_table->num_rows() << " rows" << std::endl;
    
    // Map suppliers to nations
    std::map<int64_t, int64_t> supplier_nation_map;
    
    auto s_suppkey_col = supplier_table->GetColumnByName("s_suppkey");
    auto s_nationkey_col = supplier_table->GetColumnByName("s_nationkey");
    
    for (int chunk_idx = 0; chunk_idx < s_suppkey_col->num_chunks(); chunk_idx++) {
        auto s_suppkey_array = std::static_pointer_cast<arrow::Int64Array>(s_suppkey_col->chunk(chunk_idx));
        auto s_nationkey_array = std::static_pointer_cast<arrow::Int64Array>(s_nationkey_col->chunk(chunk_idx));
        
        int64_t num_rows = s_suppkey_array->length();
        for (int64_t i = 0; i < num_rows; i++) {
            if (s_suppkey_array->IsNull(i) || s_nationkey_array->IsNull(i)) {
                continue;
            }
            
            int64_t suppkey = s_suppkey_array->Value(i);
            int64_t nationkey = s_nationkey_array->Value(i);
            supplier_nation_map[suppkey] = nationkey;
        }
    }
    
    // 4. Process partsupp table to get supply costs
    auto maybe_partsupp_file = arrow::io::ReadableFile::Open(partsupp_file);
    if (!maybe_partsupp_file.ok()) {
        std::cerr << "Could not open partsupp file: " << maybe_partsupp_file.status().ToString() << std::endl;
        return 1;
    }
    std::shared_ptr<arrow::io::ReadableFile> partsupp_infile = *maybe_partsupp_file;
    
    auto partsupp_reader_result = parquet::arrow::OpenFile(partsupp_infile, pool);
    if (!partsupp_reader_result.ok()) {
        std::cerr << "Could not open partsupp parquet file: " << partsupp_reader_result.status().ToString() << std::endl;
        return 1;
    }
    std::unique_ptr<parquet::arrow::FileReader> partsupp_reader = std::move(*partsupp_reader_result);
    
    std::shared_ptr<arrow::Table> partsupp_table;
    status = partsupp_reader->ReadTable(&partsupp_table);
    if (!status.ok()) {
        std::cerr << "Could not read partsupp table: " << status.ToString() << std::endl;
        return 1;
    }
    
    std::cout << "Partsupp table loaded with " << partsupp_table->num_rows() << " rows" << std::endl;
    
    // Create a composite key for partsupp (partkey, suppkey) -> supplycost
    std::map<std::pair<int64_t, int64_t>, double> partsupp_cost_map;
    
    auto ps_partkey_col = partsupp_table->GetColumnByName("ps_partkey");
    auto ps_suppkey_col = partsupp_table->GetColumnByName("ps_suppkey");
    auto ps_supplycost_col = partsupp_table->GetColumnByName("ps_supplycost");
    
    for (int chunk_idx = 0; chunk_idx < ps_partkey_col->num_chunks(); chunk_idx++) {
        auto ps_partkey_array = std::static_pointer_cast<arrow::Int64Array>(ps_partkey_col->chunk(chunk_idx));
        auto ps_suppkey_array = std::static_pointer_cast<arrow::Int64Array>(ps_suppkey_col->chunk(chunk_idx));
        auto ps_supplycost_chunk = ps_supplycost_col->chunk(chunk_idx);
        
        // Get scale for decimal columns
        int32_t supplycost_scale = 2;
        if (ps_supplycost_chunk->type()->id() == arrow::Type::DECIMAL128) {
            const auto& decimal_type = static_cast<const arrow::Decimal128Type&>(*ps_supplycost_chunk->type());
            supplycost_scale = decimal_type.scale();
        }
        
        auto ps_supplycost_decimal_array = std::static_pointer_cast<arrow::Decimal128Array>(ps_supplycost_chunk);
        
        int64_t num_rows = ps_partkey_array->length();
        for (int64_t i = 0; i < num_rows; i++) {
            if (ps_partkey_array->IsNull(i) || ps_suppkey_array->IsNull(i) || ps_supplycost_decimal_array->IsNull(i)) {
                continue;
            }
            
            int64_t partkey = ps_partkey_array->Value(i);
            int64_t suppkey = ps_suppkey_array->Value(i);
            
            // Skip part-supplier combinations that don't involve "green" parts
            if (green_parts.find(partkey) == green_parts.end()) {
                continue;
            }
            
            arrow::Decimal128 supplycost_val(ps_supplycost_decimal_array->Value(i));
            double supplycost = supplycost_val.ToDouble(supplycost_scale);
            partsupp_cost_map[{partkey, suppkey}] = supplycost;
        }
    }
    
    std::cout << "Found " << partsupp_cost_map.size() << " part-supplier combinations for green parts" << std::endl;
    
    // 5. Process orders table to get order dates
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
    
    // Map orderkey to order year
    std::map<int64_t, int32_t> order_year_map;
    
    auto o_orderkey_col = orders_table->GetColumnByName("o_orderkey");
    auto o_orderdate_col = orders_table->GetColumnByName("o_orderdate");
    
    for (int chunk_idx = 0; chunk_idx < o_orderkey_col->num_chunks(); chunk_idx++) {
        auto o_orderkey_array = std::static_pointer_cast<arrow::Int64Array>(o_orderkey_col->chunk(chunk_idx));
        auto o_orderdate_array = std::static_pointer_cast<arrow::Date32Array>(o_orderdate_col->chunk(chunk_idx));
        
        int64_t num_rows = o_orderkey_array->length();
        for (int64_t i = 0; i < num_rows; i++) {
            if (o_orderkey_array->IsNull(i) || o_orderdate_array->IsNull(i)) {
                continue;
            }
            
            int64_t orderkey = o_orderkey_array->Value(i);
            int32_t orderdate = o_orderdate_array->Value(i);
            
            // Extract year from orderdate
            int32_t year = days_to_year(orderdate);
            order_year_map[orderkey] = year;
        }
    }
    
    // 6. Process lineitem table and compute profits using RVV
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
    auto l_partkey_col = lineitem_table->GetColumnByName("l_partkey");
    auto l_suppkey_col = lineitem_table->GetColumnByName("l_suppkey");
    auto l_quantity_col = lineitem_table->GetColumnByName("l_quantity");
    auto l_extendedprice_col = lineitem_table->GetColumnByName("l_extendedprice");
    auto l_discount_col = lineitem_table->GetColumnByName("l_discount");
    
    // Calculate profits grouped by nation and year
    std::map<NationYearKey, double> profit_by_nation_year;
    int64_t rows_processed = 0;
    int64_t rows_qualified = 0;
    
    // Set batch size for RVV processing
    const size_t BATCH_SIZE = 1024;
    
    // Batch processing structures
    struct BatchInfo {
        std::pair<int64_t, int32_t> nation_year_key; // nationkey, year
        int64_t index;  // Index in the lineitem array
    };
    
    for (int chunk_idx = 0; chunk_idx < l_orderkey_col->num_chunks(); chunk_idx++) {
        auto l_orderkey_array = std::static_pointer_cast<arrow::Int64Array>(l_orderkey_col->chunk(chunk_idx));
        auto l_partkey_array = std::static_pointer_cast<arrow::Int64Array>(l_partkey_col->chunk(chunk_idx));
        auto l_suppkey_array = std::static_pointer_cast<arrow::Int64Array>(l_suppkey_col->chunk(chunk_idx));
        auto l_quantity_chunk = l_quantity_col->chunk(chunk_idx);
        auto l_extendedprice_chunk = l_extendedprice_col->chunk(chunk_idx);
        auto l_discount_chunk = l_discount_col->chunk(chunk_idx);
        
        // Get scale for decimal columns
        int32_t quantity_scale = 2;
        int32_t price_scale = 2;
        int32_t discount_scale = 2;
        
        if (l_quantity_chunk->type()->id() == arrow::Type::DECIMAL128) {
            const auto& decimal_type = static_cast<const arrow::Decimal128Type&>(*l_quantity_chunk->type());
            quantity_scale = decimal_type.scale();
        }
        if (l_extendedprice_chunk->type()->id() == arrow::Type::DECIMAL128) {
            const auto& decimal_type = static_cast<const arrow::Decimal128Type&>(*l_extendedprice_chunk->type());
            price_scale = decimal_type.scale();
        }
        if (l_discount_chunk->type()->id() == arrow::Type::DECIMAL128) {
            const auto& decimal_type = static_cast<const arrow::Decimal128Type&>(*l_discount_chunk->type());
            discount_scale = decimal_type.scale();
        }
        
        auto l_quantity_decimal_array = std::static_pointer_cast<arrow::Decimal128Array>(l_quantity_chunk);
        auto l_extendedprice_decimal_array = std::static_pointer_cast<arrow::Decimal128Array>(l_extendedprice_chunk);
        auto l_discount_decimal_array = std::static_pointer_cast<arrow::Decimal128Array>(l_discount_chunk);
        
        int64_t num_rows = l_orderkey_array->length();
        
        // First pass: identify qualifying rows and collect metadata
        std::vector<BatchInfo> batch_items;
        std::vector<float> price_data;
        std::vector<float> discount_data;
        std::vector<float> quantity_data;
        std::vector<float> supplycost_data;
        std::map<std::pair<int64_t, int32_t>, std::vector<size_t>> nation_year_indices;
        
        for (int64_t i = 0; i < num_rows; i++) {
            rows_processed++;
            
            if (l_orderkey_array->IsNull(i) || l_partkey_array->IsNull(i) || 
                l_suppkey_array->IsNull(i) || l_quantity_decimal_array->IsNull(i) || 
                l_extendedprice_decimal_array->IsNull(i) || l_discount_decimal_array->IsNull(i)) {
                continue;
            }
            
            int64_t partkey = l_partkey_array->Value(i);
            
            // Check if this is a "green" part
            if (green_parts.find(partkey) == green_parts.end()) {
                continue;
            }
            
            int64_t orderkey = l_orderkey_array->Value(i);
            int64_t suppkey = l_suppkey_array->Value(i);
            
            // Skip if we don't have year data for this order
            if (order_year_map.find(orderkey) == order_year_map.end()) {
                continue;
            }
            
            // Skip if we don't have nation data for this supplier
            if (supplier_nation_map.find(suppkey) == supplier_nation_map.end()) {
                continue;
            }
            
            // Skip if we don't have supply cost data
            std::pair<int64_t, int64_t> ps_key = {partkey, suppkey};
            if (partsupp_cost_map.find(ps_key) == partsupp_cost_map.end()) {
                continue;
            }
            
            // Get necessary values
            int64_t nationkey = supplier_nation_map[suppkey];
            int32_t year = order_year_map[orderkey];
            double supplycost = partsupp_cost_map[ps_key];
            
            // Convert decimal values to float
            arrow::Decimal128 quantity_val(l_quantity_decimal_array->Value(i));
            arrow::Decimal128 price_val(l_extendedprice_decimal_array->Value(i));
            arrow::Decimal128 discount_val(l_discount_decimal_array->Value(i));
            
            float quantity = quantity_val.ToDouble(quantity_scale);
            float extendedprice = price_val.ToDouble(price_scale);
            float discount = discount_val.ToDouble(discount_scale);
            
            // Store values for batch processing
            price_data.push_back(extendedprice);
            discount_data.push_back(discount);
            quantity_data.push_back(quantity);
            supplycost_data.push_back(supplycost);
            
            // Keep track of which nation-year each row belongs to
            std::pair<int64_t, int32_t> nation_year_key = {nationkey, year};
            size_t batch_index = price_data.size() - 1;
            nation_year_indices[nation_year_key].push_back(batch_index);
            
            rows_qualified++;
            
            // Process in batches for RVV efficiency
            if (price_data.size() >= BATCH_SIZE) {
                // Allocate array for profit calculation results
                float* profit_results = new float[price_data.size()];
                
                // Process batch using RVV
                batch_process_profit(
                    price_data.data(),
                    discount_data.data(),
                    quantity_data.data(),
                    supplycost_data.data(),
                    profit_results,
                    price_data.size()
                );
                
                // Accumulate profits by nation and year
                for (const auto& [key, indices] : nation_year_indices) {
                    int64_t nationkey = key.first;
                    int32_t year = key.second;
                    std::string nation = nation_map[nationkey];
                    
                    double nation_year_profit = 0.0;
                    for (size_t idx : indices) {
                        nation_year_profit += profit_results[idx];
                    }
                    
                    NationYearKey result_key = {nation, year};
                    profit_by_nation_year[result_key] += nation_year_profit;
                }
                
                // Clean up
                delete[] profit_results;
                price_data.clear();
                discount_data.clear();
                quantity_data.clear();
                supplycost_data.clear();
                nation_year_indices.clear();
            }
        }
        
        // Process any remaining rows
        if (!price_data.empty()) {
            // Allocate array for profit calculation results
            float* profit_results = new float[price_data.size()];
            
            // Process batch using RVV
            batch_process_profit(
                price_data.data(),
                discount_data.data(),
                quantity_data.data(),
                supplycost_data.data(),
                profit_results,
                price_data.size()
            );
            
            // Accumulate profits by nation and year
            for (const auto& [key, indices] : nation_year_indices) {
                int64_t nationkey = key.first;
                int32_t year = key.second;
                std::string nation = nation_map[nationkey];
                
                double nation_year_profit = 0.0;
                for (size_t idx : indices) {
                    nation_year_profit += profit_results[idx];
                }
                
                NationYearKey result_key = {nation, year};
                profit_by_nation_year[result_key] += nation_year_profit;
            }
            
            // Clean up
            delete[] profit_results;
        }
    }
    
    // Convert map to vector for sorting
    std::vector<Query9Result> results;
    for (const auto& [key, profit] : profit_by_nation_year) {
        Query9Result result;
        result.nation = key.nation;
        result.o_year = key.year;
        result.sum_profit = profit;
        results.push_back(result);
    }
    
    // Sort results by nation, o_year DESC
    std::sort(results.begin(), results.end());
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    // Print results
    std::cout << "\nTPC-H Query 9 Results (RVV-accelerated):" << std::endl;
    std::cout << "------------------------------------" << std::endl;
    std::cout << std::setw(25) << "NATION"
              << std::setw(10) << "YEAR"
              << std::setw(20) << "SUM_PROFIT" << std::endl;
    
    for (const auto& result : results) {
        std::cout << std::setw(25) << result.nation
                  << std::setw(10) << result.o_year
                  << std::setw(20) << std::fixed << std::setprecision(2) << result.sum_profit << std::endl;
    }
    
    std::cout << "\nQuery executed in " << elapsed.count() << " seconds" << std::endl;
    std::cout << "Processed " << rows_processed << " lineitem rows, " << rows_qualified << " qualified" << std::endl;
    
    return 0;
}