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

// TPC-H Query 1 result structure
struct GroupKey {
    std::string returnflag;
    std::string linestatus;
    
    bool operator<(const GroupKey& other) const {
        if (returnflag != other.returnflag) {
            return returnflag < other.returnflag;
        }
        return linestatus < other.linestatus;
    }
};

struct AggregateValues {
    double sum_qty = 0;
    double sum_base_price = 0;
    double sum_disc_price = 0;
    double sum_charge = 0;
    double avg_qty = 0;
    double avg_price = 0;
    double avg_disc = 0;
    double sum_disc = 0;
    int64_t count_order = 0;
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

    std::string input_file = argv[1];
    
    // Initialize Arrow's memory pool
    arrow::MemoryPool* pool = arrow::default_memory_pool();
    
    // Open the Parquet file
    auto maybe_infile = arrow::io::ReadableFile::Open(input_file);
    if (!maybe_infile.ok()) {
        std::cerr << "Could not open file: " << maybe_infile.status().ToString() << std::endl;
        return 1;
    }
    std::shared_ptr<arrow::io::ReadableFile> infile = *maybe_infile;
    
    // Create a ParquetFileReader
    auto reader_result = parquet::arrow::OpenFile(infile, pool);
    if (!reader_result.ok()) {
        std::cerr << "Could not open parquet file: " << reader_result.status().ToString() << std::endl;
        return 1;
    }
    std::unique_ptr<parquet::arrow::FileReader> reader = std::move(*reader_result);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    // Read the entire file as a Table
    std::shared_ptr<arrow::Table> table;
    auto status = reader->ReadTable(&table);
    if (!status.ok()) {
        std::cerr << "Could not read table: " << status.ToString() << std::endl;
        return 1;
    }
    
    // Print table schema to debug column types
    // std::cout << "Table schema: " << table->schema()->ToString() << std::endl;
    
    // Get required columns for Query 1
    std::shared_ptr<arrow::ChunkedArray> shipdate_col = table->GetColumnByName("l_shipdate");
    std::shared_ptr<arrow::ChunkedArray> returnflag_col = table->GetColumnByName("l_returnflag");
    std::shared_ptr<arrow::ChunkedArray> linestatus_col = table->GetColumnByName("l_linestatus");
    std::shared_ptr<arrow::ChunkedArray> quantity_col = table->GetColumnByName("l_quantity");
    std::shared_ptr<arrow::ChunkedArray> extendedprice_col = table->GetColumnByName("l_extendedprice");
    std::shared_ptr<arrow::ChunkedArray> discount_col = table->GetColumnByName("l_discount");
    std::shared_ptr<arrow::ChunkedArray> tax_col = table->GetColumnByName("l_tax");
    
    // Calculate cutoff date: '1998-12-01' - interval '90' day
    int32_t cutoff_days = date_string_to_days("1998-09-02");
    // std::cout << "Cutoff days: " << cutoff_days << std::endl;
    
    // Process data and perform grouping manually
    std::map<GroupKey, AggregateValues> groups;
    
    // Process each chunk
    int64_t total_rows = table->num_rows();
    int64_t rows_processed = 0;
    int64_t rows_accepted = 0;
    
    for (int chunk_idx = 0; chunk_idx < shipdate_col->num_chunks(); chunk_idx++) {
        // Debug output
        // std::cout << "Processing chunk " << chunk_idx << "/" << shipdate_col->num_chunks() << std::endl;
        
        // Get the correct array types based on the schema
        std::shared_ptr<arrow::Array> shipdate_chunk = shipdate_col->chunk(chunk_idx);
        std::shared_ptr<arrow::Array> returnflag_chunk = returnflag_col->chunk(chunk_idx);
        std::shared_ptr<arrow::Array> linestatus_chunk = linestatus_col->chunk(chunk_idx);
        std::shared_ptr<arrow::Array> quantity_chunk = quantity_col->chunk(chunk_idx);
        std::shared_ptr<arrow::Array> extendedprice_chunk = extendedprice_col->chunk(chunk_idx);
        std::shared_ptr<arrow::Array> discount_chunk = discount_col->chunk(chunk_idx);
        std::shared_ptr<arrow::Array> tax_chunk = tax_col->chunk(chunk_idx);
        
        // Check array type for debugging
        // std::cout << "Shipdate array type: " << shipdate_chunk->type()->ToString() << std::endl;
        // std::cout << "Quantity array type: " << quantity_chunk->type()->ToString() << std::endl;
        // std::cout << "ExtendedPrice array type: " << extendedprice_chunk->type()->ToString() << std::endl;
        // std::cout << "Discount array type: " << discount_chunk->type()->ToString() << std::endl;
        // std::cout << "Tax array type: " << tax_chunk->type()->ToString() << std::endl;
        
        // Cast arrays to their specific types
        auto shipdate_array = std::static_pointer_cast<arrow::Date32Array>(shipdate_chunk);
        auto returnflag_array = std::static_pointer_cast<arrow::StringArray>(returnflag_chunk);
        auto linestatus_array = std::static_pointer_cast<arrow::StringArray>(linestatus_chunk);
        
        // Get scale for decimal columns
        int32_t quantity_scale = 2; // Default scale for decimal128(15, 2)
        int32_t price_scale = 2;
        int32_t discount_scale = 2;
        int32_t tax_scale = 2;
        
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
        if (tax_chunk->type()->id() == arrow::Type::DECIMAL128) {
            const auto& decimal_type = static_cast<const arrow::Decimal128Type&>(*tax_chunk->type());
            tax_scale = decimal_type.scale();
        }
        
        // Cast to Decimal128Arrays
        auto quantity_decimal_array = std::static_pointer_cast<arrow::Decimal128Array>(quantity_chunk);
        auto extendedprice_decimal_array = std::static_pointer_cast<arrow::Decimal128Array>(extendedprice_chunk);
        auto discount_decimal_array = std::static_pointer_cast<arrow::Decimal128Array>(discount_chunk);
        auto tax_decimal_array = std::static_pointer_cast<arrow::Decimal128Array>(tax_chunk);
        
        // Handle numeric columns properly for Decimal128 type
        int64_t num_rows = shipdate_array->length();
        
        for (int64_t i = 0; i < num_rows; i++) {
            rows_processed++;
            
            if (shipdate_array->IsNull(i)) continue;
            
            // Filter: l_shipdate <= date '1998-12-01' - interval '90' day
            int32_t shipdate_days = shipdate_array->Value(i);
            
            // Print sample values to debug
            // if (i < 5 && chunk_idx == 0) {
            //     std::cout << "Sample date value: " << shipdate_days << std::endl;
            // }
            
            if (shipdate_days > cutoff_days) continue;
            
            rows_accepted++;
            
            // Group by l_returnflag, l_linestatus
            GroupKey key;
            key.returnflag = returnflag_array->GetString(i);
            key.linestatus = linestatus_array->GetString(i);
            
            // Extract Decimal128 values and convert to double
            // Use GetDecimal128 instead of GetValue for Decimal128 objects
            double quantity = decimal128_bytes_to_double(quantity_decimal_array->Value(i), quantity_scale);
            double extendedprice = decimal128_bytes_to_double(extendedprice_decimal_array->Value(i), price_scale);
            double discount = decimal128_bytes_to_double(discount_decimal_array->Value(i), discount_scale);
            double tax = decimal128_bytes_to_double(tax_decimal_array->Value(i), tax_scale);
            
            // Print sample values in first few records for debugging
            // if (rows_accepted < 5) {
            //     std::cout << "Sample record: qty=" << quantity << ", price=" << extendedprice 
            //               << ", discount=" << discount << ", tax=" << tax << std::endl;
            // }
            
            // Calculate computed values
            double disc_price = extendedprice * (1.0 - discount);
            double charge = disc_price * (1.0 + tax);
            
            // Update aggregates
            AggregateValues& agg = groups[key];
            agg.sum_qty += quantity;
            agg.sum_base_price += extendedprice;
            agg.sum_disc_price += disc_price;
            agg.sum_charge += charge;
            agg.sum_disc += discount;  // Track sum of discounts for avg_disc
            agg.count_order++;
        }
    }
    
    // Calculate averages directly
    for (auto& pair : groups) {
        AggregateValues& agg = pair.second;
        if (agg.count_order > 0) {
            agg.avg_qty = agg.sum_qty / agg.count_order;
            agg.avg_price = agg.sum_base_price / agg.count_order;
            agg.avg_disc = agg.sum_disc / agg.count_order;
        }
    }
    
    // Sort results
    std::vector<std::pair<GroupKey, AggregateValues>> sorted_results;
    for (const auto& pair : groups) {
        sorted_results.push_back(pair);
    }
    
    std::sort(sorted_results.begin(), sorted_results.end(), 
        [](const auto& a, const auto& b) {
            return a.first < b.first;
        });
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    // Print results
    std::cout << "\nTPC-H Query 1 Results:" << std::endl;
    std::cout << "----------------------" << std::endl;
    std::cout << std::setw(12) << "l_returnflag" 
              << std::setw(12) << "l_linestatus"
              << std::setw(15) << "sum_qty"
              << std::setw(15) << "sum_base_price"
              << std::setw(15) << "sum_disc_price"
              << std::setw(15) << "sum_charge"
              << std::setw(15) << "avg_qty"
              << std::setw(15) << "avg_price"
              << std::setw(15) << "avg_disc"
              << std::setw(15) << "count_order" << std::endl;
    
    for (const auto& result : sorted_results) {
        const GroupKey& key = result.first;
        const AggregateValues& agg = result.second;
        
        std::cout << std::setw(12) << key.returnflag
                  << std::setw(12) << key.linestatus
                  << std::setw(15) << std::fixed << std::setprecision(2) << agg.sum_qty
                  << std::setw(15) << std::fixed << std::setprecision(2) << agg.sum_base_price
                  << std::setw(15) << std::fixed << std::setprecision(2) << agg.sum_disc_price
                  << std::setw(15) << std::fixed << std::setprecision(2) << agg.sum_charge
                  << std::setw(15) << std::fixed << std::setprecision(2) << agg.avg_qty
                  << std::setw(15) << std::fixed << std::setprecision(2) << agg.avg_price
                  << std::setw(15) << std::fixed << std::setprecision(6) << agg.avg_disc
                  << std::setw(15) << agg.count_order << std::endl;
    }
    
    std::cout << "\nQuery executed in " << elapsed.count() << " seconds" << std::endl;
    std::cout << "Processed " << total_rows << " rows (" << rows_processed << " examined, " 
              << rows_accepted << " passed filter)" << std::endl;
    
    return 0;
}
