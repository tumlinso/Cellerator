cmake_minimum_required(VERSION 3.24)
project(cellerator_shared_condition_probe LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CELLERATOR_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../.." CACHE PATH
    "Path to the Cellerator checkout")

add_executable(shared_condition_probe
    shared_condition_probe.cu
    "${CELLERATOR_ROOT}/src/gating/cuda_worksets.cu"
    "${CELLERATOR_ROOT}/src/gating/condition_program.cc")
target_include_directories(shared_condition_probe PRIVATE
    "${CELLERATOR_ROOT}/include")
target_link_libraries(shared_condition_probe PRIVATE CUDA::cudart)
target_compile_features(shared_condition_probe PRIVATE cxx_std_17)
set_target_properties(shared_condition_probe PROPERTIES
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
target_compile_options(shared_condition_probe PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
