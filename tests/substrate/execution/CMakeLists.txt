cmake_minimum_required(VERSION 3.24)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(CelleratorPreparedHostChecks LANGUAGES CXX)
  include(CTest)
endif()
get_filename_component(CE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
add_executable(ce_prepared_host host.cc)
target_include_directories(ce_prepared_host PRIVATE "${CE_ROOT}/include")
target_compile_features(ce_prepared_host PRIVATE cxx_std_20)
target_compile_options(ce_prepared_host PRIVATE -Wall -Wextra -Werror -fno-fast-math -ffp-contract=off)
if(TARGET Cellerator::local_differential)
  target_link_libraries(ce_prepared_host PRIVATE Cellerator::local_differential)
else()
  find_path(CE_CUDA_HEADERS cuda_runtime_api.h HINTS
    /opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/include REQUIRED)
  target_include_directories(ce_prepared_host PRIVATE "${CE_CUDA_HEADERS}")
  target_sources(ce_prepared_host PRIVATE
    "${CE_ROOT}/src/compute/operation/native_numeric/local_arithmetic.cc"
    "${CE_ROOT}/src/compute/operation/differential/local_arithmetic.cc"
    "${CE_ROOT}/src/compute/operation/operation_core_v2/schema.cc"
    "${CE_ROOT}/src/execution/program/program_v2.cc")
endif()
add_test(NAME ce_prepared_host COMMAND ce_prepared_host)
