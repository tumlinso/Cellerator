cmake_minimum_required(VERSION 3.24)
project(cellerator_bitmap_retrieval LANGUAGES CXX CUDA)

get_filename_component(CELLERATOR_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE)

add_executable(bitmap_retrieval_bench
    bitmap_retrieval_bench.cu
    "${CELLERATOR_ROOT}/src/indexing/cuda_postings.cu")
target_include_directories(bitmap_retrieval_bench PRIVATE
    "${CELLERATOR_ROOT}/include")
target_compile_options(bitmap_retrieval_bench PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
set_target_properties(bitmap_retrieval_bench PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
    CUDA_ARCHITECTURES 70)
