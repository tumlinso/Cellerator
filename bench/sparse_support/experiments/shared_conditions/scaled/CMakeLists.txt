cmake_minimum_required(VERSION 3.24)
project(cellerator_scaled_shared_conditions LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CELLERATOR_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../../.." CACHE PATH
    "Path to the Cellerator checkout")

add_executable(scaled_condition_probe
    scaled_condition_probe.cu
    "${CELLERATOR_ROOT}/src/compute/gating/condition_program.cc"
    "${CELLERATOR_ROOT}/src/compute/gating/cuda_worksets.cu")
target_include_directories(scaled_condition_probe PRIVATE
    "${CELLERATOR_ROOT}/include")
target_link_libraries(scaled_condition_probe PRIVATE CUDA::cudart)
target_compile_features(scaled_condition_probe PRIVATE cxx_std_17)
set_target_properties(scaled_condition_probe PROPERTIES
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
target_compile_options(scaled_condition_probe PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
