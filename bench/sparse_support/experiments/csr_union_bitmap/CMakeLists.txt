cmake_minimum_required(VERSION 3.24)
project(csr_union_bitmap LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 70)

find_package(CUDAToolkit REQUIRED)
get_filename_component(CELLERATOR_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE)

add_executable(csr_union_bitmap_bench
  csr_union_bitmap_bench.cu
  "${CELLERATOR_ROOT}/src/compute/support/logical_support.cc"
  "${CELLERATOR_ROOT}/src/compute/indexing/reverse_postings.cc"
  "${CELLERATOR_ROOT}/src/compute/indexing/cuda_postings.cu")
target_include_directories(csr_union_bitmap_bench PRIVATE
  "${CELLERATOR_ROOT}/include")
target_link_libraries(csr_union_bitmap_bench PRIVATE CUDA::cudart)
target_compile_options(csr_union_bitmap_bench PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo;-Xptxas=-v>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>)
