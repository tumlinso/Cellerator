cmake_minimum_required(VERSION 3.24)
project(CelleratorEqualRowBatchedCandidate LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CELLERATOR_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../..")
add_library(grouped_linear_batched_candidate STATIC
    src/batched_candidate.cu
    "${CELLERATOR_ROOT}/src/modeling/cuda/runtime.cu")
target_compile_features(grouped_linear_batched_candidate PUBLIC cxx_std_17)
target_include_directories(grouped_linear_batched_candidate PUBLIC
    "${CMAKE_CURRENT_LIST_DIR}/include"
    "${CELLERATOR_ROOT}/include")
target_link_libraries(grouped_linear_batched_candidate PUBLIC
    CUDA::cudart CUDA::cublas)
set_target_properties(grouped_linear_batched_candidate PROPERTIES
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES 70)

add_executable(grouped_linear_batched_candidate_smoke tests/candidate_smoke.cu)
target_compile_features(grouped_linear_batched_candidate_smoke PRIVATE cxx_std_17)
target_link_libraries(grouped_linear_batched_candidate_smoke PRIVATE
    grouped_linear_batched_candidate CUDA::cudart CUDA::cublas)
set_target_properties(grouped_linear_batched_candidate_smoke PROPERTIES
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES 70)

enable_testing()
add_test(NAME grouped_linear_batched_candidate_host_fallback
    COMMAND grouped_linear_batched_candidate_smoke --host-only)
