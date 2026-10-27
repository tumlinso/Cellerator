cmake_minimum_required(VERSION 3.24)
project(cellerator_parallel_workset_probe LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
get_filename_component(CELLERATOR_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE)

add_executable(parallel_workset_probe
    parallel_workset_probe.cu
    "${CELLERATOR_ROOT}/src/compute/gating/cuda_worksets.cu"
    "${CELLERATOR_ROOT}/src/compute/gating/condition_program.cc"
    "${CELLERATOR_ROOT}/src/compute/indexing/cuda_postings.cu"
    "${CELLERATOR_ROOT}/src/compute/indexing/reverse_postings.cc"
    "${CELLERATOR_ROOT}/src/compute/support/logical_support.cc")
target_include_directories(parallel_workset_probe PRIVATE
    "${CELLERATOR_ROOT}/include")
target_link_libraries(parallel_workset_probe PRIVATE CUDA::cudart)
set_target_properties(parallel_workset_probe PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES CUDA_ARCHITECTURES 70)
target_compile_options(parallel_workset_probe PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
