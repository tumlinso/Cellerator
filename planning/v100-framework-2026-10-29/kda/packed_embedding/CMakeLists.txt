cmake_minimum_required(VERSION 3.24)
project(CelleratorKDAPackedEmbedding LANGUAGES CXX)

if(NOT DEFINED CELLERATOR_SOURCE_DIR OR NOT DEFINED KDA_CANDIDATE_SOURCE_DIR)
  message(FATAL_ERROR "CELLERATOR_SOURCE_DIR and KDA_CANDIDATE_SOURCE_DIR are required")
endif()

set(PROJECT_SOURCE_DIR "${CELLERATOR_SOURCE_DIR}")
set(CELLERATOR_ENABLE_CONDITIONAL ON)
set(CELLERATOR_ENABLE_FRAMEWORK_CUDA ON)
set(CELLERATOR_BUILD_FRAMEWORK_CUDA_SMOKE OFF)
include("${CELLERATOR_SOURCE_DIR}/cmake/CelleratorConditional.cmake")
include("${CELLERATOR_SOURCE_DIR}/cmake/CelleratorFrameworkCUDA.cmake")

get_target_property(_sources Cellerator::framework_cuda SOURCES)
list(FILTER _sources EXCLUDE REGEX "/packed_embedding\\.cu$")
add_library(cellerator_kda_packed_embedding STATIC
  ${_sources}
  "${KDA_CANDIDATE_SOURCE_DIR}/src/modeling/cuda/packed_embedding.cu")
set_target_properties(cellerator_kda_packed_embedding PROPERTIES
  POSITION_INDEPENDENT_CODE ON CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
target_compile_features(cellerator_kda_packed_embedding PUBLIC cxx_std_17)
target_include_directories(cellerator_kda_packed_embedding PUBLIC
  "${KDA_CANDIDATE_SOURCE_DIR}/include"
  "${CELLERATOR_SOURCE_DIR}/include")
target_link_libraries(cellerator_kda_packed_embedding PUBLIC CUDA::cudart CUDA::cublas)
if(KDA_WARP_BACKWARD)
  target_compile_definitions(cellerator_kda_packed_embedding PRIVATE
    CELLERATOR_PACKED_EMBEDDING_WARP_BACKWARD=1)
endif()

add_executable(cellerator_kda_packed_embedding_smoke
  "${CELLERATOR_SOURCE_DIR}/tests/framework/cuda/packed_embedding_smoke.cu")
target_compile_features(cellerator_kda_packed_embedding_smoke PRIVATE cxx_std_17)
target_compile_options(cellerator_kda_packed_embedding_smoke PRIVATE -UNDEBUG)
target_link_libraries(cellerator_kda_packed_embedding_smoke PRIVATE
  cellerator_kda_packed_embedding)
