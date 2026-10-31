cmake_minimum_required(VERSION 3.24)
project(CelleratorKDAFp32DispatchGeometry LANGUAGES CXX)

if(NOT DEFINED CELLERATOR_SOURCE_DIR OR NOT DEFINED KDA_CANDIDATE_SOURCE_DIR)
  message(FATAL_ERROR "CELLERATOR_SOURCE_DIR and KDA_CANDIDATE_SOURCE_DIR are required")
endif()

set(PROJECT_SOURCE_DIR "${CELLERATOR_SOURCE_DIR}")
set(CELLERATOR_ENABLE_CONDITIONAL ON)
set(CELLERATOR_ENABLE_FRAMEWORK_CUDA ON)
set(CELLERATOR_BUILD_FRAMEWORK_CUDA_SMOKE OFF)
include("${CELLERATOR_SOURCE_DIR}/cmake/CelleratorConditional.cmake")
include("${CELLERATOR_SOURCE_DIR}/cmake/CelleratorFrameworkCUDA.cmake")

get_target_property(_framework_sources Cellerator::framework_cuda SOURCES)
list(FILTER _framework_sources EXCLUDE REGEX "/dispatch\\.cu$")
add_library(cellerator_kda_dispatch STATIC
  ${_framework_sources}
  "${KDA_CANDIDATE_SOURCE_DIR}/src/modeling/cuda/dispatch.cu")
set_target_properties(cellerator_kda_dispatch PROPERTIES
  POSITION_INDEPENDENT_CODE ON CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
target_compile_features(cellerator_kda_dispatch PUBLIC cxx_std_17)
target_include_directories(cellerator_kda_dispatch PUBLIC "${CELLERATOR_SOURCE_DIR}/include")
target_link_libraries(cellerator_kda_dispatch PUBLIC CUDA::cudart CUDA::cublas)

add_executable(cellerator_kda_dispatch_smoke
  "${CELLERATOR_SOURCE_DIR}/tests/framework/cuda/dispatch_smoke.cu")
target_compile_features(cellerator_kda_dispatch_smoke PRIVATE cxx_std_17)
target_compile_options(cellerator_kda_dispatch_smoke PRIVATE -UNDEBUG)
target_link_libraries(cellerator_kda_dispatch_smoke PRIVATE cellerator_kda_dispatch)

add_executable(cellerator_kda_dispatch_geometry_driver dispatch_geometry_driver.cu)
target_compile_features(cellerator_kda_dispatch_geometry_driver PRIVATE cxx_std_17)
target_compile_options(cellerator_kda_dispatch_geometry_driver PRIVATE -UNDEBUG)
target_link_libraries(cellerator_kda_dispatch_geometry_driver PRIVATE cellerator_kda_dispatch)
