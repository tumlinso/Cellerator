cmake_minimum_required(VERSION 3.24)
project(CelleratorV100Framework LANGUAGES CXX)

set(CELLERATOR_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../.." CACHE PATH "Captured Cellerator source")
set(PROJECT_SOURCE_DIR "${CELLERATOR_SOURCE_DIR}")
set(CELLERATOR_BUILD_TESTS OFF)
set(CELLERATOR_ENABLE_CONDITIONAL ON)
set(CELLERATOR_ENABLE_FRAMEWORK_CUDA ON)
set(CELLERATOR_ENABLE_FRAMEWORK_CUDA_PYTHON ON)
set(CELLERATOR_BUILD_FRAMEWORK_CUDA_SMOKE ON)
set(CELLERATOR_ENABLE_FRAMEWORK_PARAMETER_TRANSFER ON)
include("${CELLERATOR_SOURCE_DIR}/cmake/CelleratorConditional.cmake")
include("${CELLERATOR_SOURCE_DIR}/cmake/CelleratorFrameworkCUDA.cmake")
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

foreach(_bench IN ITEMS dispatch memory quantization)
  if(_bench STREQUAL "dispatch")
    set(_bench_source dispatch/dispatch_benchmark.cu)
  elseif(_bench STREQUAL "memory")
    set(_bench_source memory/benchmark.cu)
  else()
    set(_bench_source quantization/decode_bench.cu)
  endif()
  add_executable(cellerator_v100_${_bench}_benchmark
    "${CELLERATOR_SOURCE_DIR}/planning/v100-framework-2026-10-29/${_bench_source}")
  target_compile_features(cellerator_v100_${_bench}_benchmark PRIVATE cxx_std_17)
  target_compile_options(cellerator_v100_${_bench}_benchmark PRIVATE -UNDEBUG)
  target_link_libraries(cellerator_v100_${_bench}_benchmark PRIVATE Cellerator::framework_cuda)
endforeach()

add_executable(cellerator_v100_regulatory_benchmark
  "${CELLERATOR_SOURCE_DIR}/planning/v100-framework-2026-10-29/regulatory/regulatory_gate_comparison.cu")
target_compile_features(cellerator_v100_regulatory_benchmark PRIVATE cxx_std_17)
target_compile_options(cellerator_v100_regulatory_benchmark PRIVATE -UNDEBUG)
target_link_libraries(cellerator_v100_regulatory_benchmark PRIVATE Cellerator::framework_cuda)

add_executable(cellerator_v100_grouped_linear_benchmark
  "${CELLERATOR_SOURCE_DIR}/planning/v100-framework-2026-10-29/grouped_linear/benchmark/benchmark.cu")
target_compile_features(cellerator_v100_grouped_linear_benchmark PRIVATE cxx_std_17)
target_compile_options(cellerator_v100_grouped_linear_benchmark PRIVATE -UNDEBUG)
target_link_libraries(cellerator_v100_grouped_linear_benchmark PRIVATE Cellerator::framework_cuda)

add_executable(cellerator_v100_packed_embedding_candidate_smoke
  "${CELLERATOR_SOURCE_DIR}/tests/framework/cuda/packed_embedding_smoke.cu")
target_compile_features(cellerator_v100_packed_embedding_candidate_smoke PRIVATE cxx_std_17)
target_compile_options(cellerator_v100_packed_embedding_candidate_smoke PRIVATE -UNDEBUG)
get_target_property(_cellerator_cuda_sources Cellerator::framework_cuda SOURCES)
list(FILTER _cellerator_cuda_sources EXCLUDE REGEX "/packed_embedding\\.cu$")
add_library(cellerator_framework_cuda_packed_candidate STATIC
  ${_cellerator_cuda_sources}
  "${CELLERATOR_SOURCE_DIR}/src/modeling/cuda/packed_embedding.cu")
add_library(Cellerator::framework_cuda_packed_candidate ALIAS cellerator_framework_cuda_packed_candidate)
set_target_properties(cellerator_framework_cuda_packed_candidate PROPERTIES
  POSITION_INDEPENDENT_CODE ON CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
target_compile_features(cellerator_framework_cuda_packed_candidate PUBLIC cxx_std_17)
target_compile_definitions(cellerator_framework_cuda_packed_candidate PRIVATE CELLERATOR_PACKED_EMBEDDING_WARP_BACKWARD=1)
target_include_directories(cellerator_framework_cuda_packed_candidate PUBLIC
  "${CELLERATOR_SOURCE_DIR}/include")
target_link_libraries(cellerator_framework_cuda_packed_candidate PUBLIC CUDA::cudart CUDA::cublas)
target_link_libraries(cellerator_v100_packed_embedding_candidate_smoke PRIVATE Cellerator::framework_cuda_packed_candidate)

