cmake_minimum_required(VERSION 3.24)
project(CelleratorKDAAttentionLaunchGeometry LANGUAGES CXX)

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
list(FILTER _framework_sources EXCLUDE REGEX "/attention\\.cu$")
add_library(cellerator_kda_attention STATIC
  ${_framework_sources}
  "${KDA_CANDIDATE_SOURCE_DIR}/src/modeling/cuda/attention.cu")
set_target_properties(cellerator_kda_attention PROPERTIES
  POSITION_INDEPENDENT_CODE ON CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
target_compile_features(cellerator_kda_attention PUBLIC cxx_std_17)
target_include_directories(cellerator_kda_attention PUBLIC "${CELLERATOR_SOURCE_DIR}/include")
target_link_libraries(cellerator_kda_attention PUBLIC CUDA::cudart CUDA::cublas)

add_executable(cellerator_kda_attention_smoke
  "${CELLERATOR_SOURCE_DIR}/tests/framework/cuda/attention_smoke.cu")
target_compile_features(cellerator_kda_attention_smoke PRIVATE cxx_std_17)
target_compile_options(cellerator_kda_attention_smoke PRIVATE -UNDEBUG)
target_link_libraries(cellerator_kda_attention_smoke PRIVATE cellerator_kda_attention)

add_executable(cellerator_kda_attention_geometry_driver attention_geometry_driver.cu)
target_compile_features(cellerator_kda_attention_geometry_driver PRIVATE cxx_std_17)
target_compile_options(cellerator_kda_attention_geometry_driver PRIVATE -UNDEBUG)
target_link_libraries(cellerator_kda_attention_geometry_driver PRIVATE cellerator_kda_attention)
