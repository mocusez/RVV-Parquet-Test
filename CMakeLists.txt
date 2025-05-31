cmake_minimum_required(VERSION 3.14)
project(ArrowTest CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Arrow library paths
set(ARROW_LIB_DIR "/usr/local/arrow/lib" CACHE PATH "Arrow library directory")
set(ARROW_INCLUDE_DIR "/usr/local/arrow/include" CACHE PATH "Arrow include directory")

include_directories(${ARROW_INCLUDE_DIR})

# RISC-V specific compile options
set(RISCV_OPTS -march=rv64gcv)

# Arrow libraries to link
set(ARROW_LIBS
    ${ARROW_LIB_DIR}/libarrow.so
    ${ARROW_LIB_DIR}/libparquet.so
    ${ARROW_LIB_DIR}/libarrow_dataset.so
)

# Helper function to add executables with consistent settings
function(add_arrow_executable name source)
    add_executable(${name} ${source})
    target_compile_options(${name} PRIVATE ${RISCV_OPTS})
    target_link_libraries(${name} ${ARROW_LIBS})
endfunction()

# Add all the executables
add_arrow_executable(query1 query1.cpp)
add_arrow_executable(rvv_query1 rvv_query1.cpp)
add_arrow_executable(query4 query4.cpp)
add_arrow_executable(rvv_query4 rvv_query4.cpp)
add_arrow_executable(query6 query6.cpp)
add_arrow_executable(rvv_query6 rvv_query6.cpp)
add_arrow_executable(query9 query9.cpp)
add_arrow_executable(rvv_query9 rvv_query9.cpp)
add_arrow_executable(query12 query12.cpp)
add_arrow_executable(rvv_query12 rvv_query12.cpp)