cd build
./query1 ../lineitem.parquet > query1.txt
./query4 ../orders.parquet ../lineitem.parquet > query4.txt
./query6 ../lineitem.parquet > query6.txt
./rvv_query1 ../lineitem.parquet > rvv_query1.txt
./rvv_query4 ../orders.parquet ../lineitem.parquet > rvv_query4.txt
./rvv_query6 ../lineitem.parquet > rvv_query6.txt