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
#include <chrono>
#include <string>
#include <memory>

using namespace arrow;
using namespace arrow::compute;

// Convert ISO date string to days since epoch
int32_t date_string_to_days(const std::string& date_str) {
    std::tm tm = {};
    std::istringstream ss(date_str);
    ss >> std::get_time(&tm, "%Y-%m-%d");
    
    std::time_t time = std::mktime(&tm);
    // Convert to days since epoch (1970-01-01)
    return static_cast<int32_t>(time / (60 * 60 * 24));
}

// Helper function to convert Decimal128 to double
double decimal128_bytes_to_double(const uint8_t* raw_value_ptr, int32_t scale) {
    // Create a Decimal128 from the raw bytes
    // The raw_value_ptr contains 16 bytes that represent the Decimal128 value
    
    // Extract low and high bits from the raw bytes (little-endian)
    int64_t low_bits = 0;
    int64_t high_bits = 0;
    memcpy(&low_bits, raw_value_ptr, sizeof(int64_t));
    memcpy(&high_bits, raw_value_ptr + sizeof(int64_t), sizeof(int64_t));
    
    // Create a new Decimal128 directly using the constructor with high and low bits
    arrow::Decimal128 decimal_value(high_bits, low_bits);
    
    // Now convert to double
    double raw_value = static_cast<double>(decimal_value.high_bits()) * std::pow(2.0, 64) + 
                     static_cast<double>(decimal_value.low_bits());
    return raw_value / std::pow(10.0, scale);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <lineitem.parquet>" << std::endl;
        return 1;
    }

    std::string lineitem_file = argv[1];
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Initialize Arrow's memory pool
    arrow::MemoryPool* pool = arrow::default_memory_pool();
    
    // Open the lineitem Parquet file
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
    // std::cout << "Table schema: " << lineitem_table->schema()->ToString() << std::endl;
    
    // Extract required columns
    auto l_shipdate_col = lineitem_table->GetColumnByName("l_shipdate");
    auto l_quantity_col = lineitem_table->GetColumnByName("l_quantity");
    auto l_extendedprice_col = lineitem_table->GetColumnByName("l_extendedprice");
    auto l_discount_col = lineitem_table->GetColumnByName("l_discount");
    
    // Calculate date range for 1994
    int32_t start_date = date_string_to_days("1994-01-01");
    int32_t end_date = date_string_to_days("1995-01-01"); // One day after end of 1994
    
    // std::cout << "Filtering shipments between dates: " << start_date << " and " << end_date << std::endl;
    
    // Process data and calculate revenue
    double total_revenue = 0.0;
    int64_t rows_processed = 0;
    int64_t rows_qualified = 0;
    
    // Discount range: 0.06 - 0.01 to 0.06 + 0.01 (0.05 to 0.07)
    double min_discount = 0.05;
    double max_discount = 0.07;
    
    // Quantity threshold: < 24
    double max_quantity = 24.0;
    
    for (int chunk_idx = 0; chunk_idx < l_shipdate_col->num_chunks(); chunk_idx++) {
        // Get arrays for this chunk
        auto shipdate_chunk = std::static_pointer_cast<arrow::Date32Array>(l_shipdate_col->chunk(chunk_idx));
        auto quantity_chunk = l_quantity_col->chunk(chunk_idx);
        auto extendedprice_chunk = l_extendedprice_col->chunk(chunk_idx);
        auto discount_chunk = l_discount_col->chunk(chunk_idx);
        
        // Get scale for decimal columns
        int32_t quantity_scale = 2; // Default scale for decimal128(15, 2)
        int32_t price_scale = 2;
        int32_t discount_scale = 2;
        
        // Extract scale from decimal type if available
        if (quantity_chunk->type()->id() == arrow::Type::DECIMAL128) {
            const auto& decimal_type = static_cast<const arrow::Decimal128Type&>(*quantity_chunk->type());
            quantity_scale = decimal_type.scale();
        }
        if (extendedprice_chunk->type()->id() == arrow::Type::DECIMAL128) {
            const auto& decimal_type = static_cast<const arrow::Decimal128Type&>(*extendedprice_chunk->type());
            price_scale = decimal_type.scale();
        }
        if (discount_chunk->type()->id() == arrow::Type::DECIMAL128) {
            const auto& decimal_type = static_cast<const arrow::Decimal128Type&>(*discount_chunk->type());
            discount_scale = decimal_type.scale();
        }
        
        // Cast to appropriate array types
        auto quantity_decimal_array = std::static_pointer_cast<arrow::Decimal128Array>(quantity_chunk);
        auto extendedprice_decimal_array = std::static_pointer_cast<arrow::Decimal128Array>(extendedprice_chunk);
        auto discount_decimal_array = std::static_pointer_cast<arrow::Decimal128Array>(discount_chunk);
        
        int64_t num_rows = shipdate_chunk->length();
        
        // Sample values to verify data types
        // if (chunk_idx == 0 && num_rows > 0) {
        //     std::cout << "Sample shipdate: " << shipdate_chunk->Value(0) << std::endl;
        //     std::cout << "Sample quantity: " << decimal128_bytes_to_double(quantity_decimal_array->Value(0), quantity_scale) << std::endl;
        //     std::cout << "Sample extendedprice: " << decimal128_bytes_to_double(extendedprice_decimal_array->Value(0), price_scale) << std::endl;
        //     std::cout << "Sample discount: " << decimal128_bytes_to_double(discount_decimal_array->Value(0), discount_scale) << std::endl;
        // }
        
        for (int64_t i = 0; i < num_rows; i++) {
            rows_processed++;
            
            if (shipdate_chunk->IsNull(i) || 
                quantity_decimal_array->IsNull(i) || 
                extendedprice_decimal_array->IsNull(i) || 
                discount_decimal_array->IsNull(i)) {
                continue;
            }
            
            // Check filter conditions
            int32_t shipdate = shipdate_chunk->Value(i);
            double quantity = decimal128_bytes_to_double(quantity_decimal_array->Value(i), quantity_scale);
            double extendedprice = decimal128_bytes_to_double(extendedprice_decimal_array->Value(i), price_scale);
            double discount = decimal128_bytes_to_double(discount_decimal_array->Value(i), discount_scale);
            
            // Apply WHERE clause filters:
            // l_shipdate >= DATE '1994-01-01' AND l_shipdate < DATE '1995-01-01'
            // AND l_discount BETWEEN 0.05 AND 0.07
            // AND l_quantity < 24
            if (shipdate >= start_date && shipdate < end_date &&
                discount >= min_discount && discount <= max_discount &&
                quantity < max_quantity) {
                
                // Calculate revenue contribution: l_extendedprice * l_discount
                double revenue = extendedprice * discount;
                total_revenue += revenue;
                rows_qualified++;
                
                // Debug output for first few qualifying rows
                // if (rows_qualified <= 5) {
                //     std::cout << "Qualifying row " << rows_qualified << ": "
                //               << "shipdate=" << shipdate
                //               << ", quantity=" << quantity
                //               << ", discount=" << discount
                //               << ", price=" << extendedprice
                //               << ", revenue=" << revenue << std::endl;
                // }
            }
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    // Print results
    std::cout << "\nTPC-H Query 6 Results:" << std::endl;
    std::cout << "----------------------" << std::endl;
    std::cout << std::setw(15) << "REVENUE" << std::endl;
    std::cout << std::setw(15) << std::fixed << std::setprecision(2) << total_revenue << std::endl;
    
    std::cout << "\nQuery executed in " << elapsed.count() << " seconds" << std::endl;
    std::cout << "Processed " << rows_processed << " rows, " << rows_qualified << " qualified" << std::endl;
    
    return 0;
}