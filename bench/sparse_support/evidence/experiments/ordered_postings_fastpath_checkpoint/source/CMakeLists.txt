cmake_minimum_required(VERSION 3.24)
project(ordered_postings_highq LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 70)
find_package(CUDAToolkit REQUIRED)
set(ROOT /home/tumlinson/Cellerator)
add_executable(csr_union_bitmap_highq
  csr_union_bitmap_bench.cu
  ${ROOT}/src/compute/support/logical_support.cc
  ${ROOT}/src/compute/indexing/reverse_postings.cc
  ${ROOT}/src/compute/indexing/cuda_postings.cu)
target_include_directories(csr_union_bitmap_highq PRIVATE ${ROOT}/include)
target_link_libraries(csr_union_bitmap_highq PRIVATE CUDA::cudart)
target_compile_options(csr_union_bitmap_highq PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>)
