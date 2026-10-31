cmake_minimum_required(VERSION 3.24)
project(CelleratorFrontierCandidate LANGUAGES CXX)

if(NOT CELLERATOR_SOURCE)
    get_filename_component(CELLERATOR_SOURCE
        "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE)
endif()
set(CELLERATOR_SOURCE "${CELLERATOR_SOURCE}" CACHE PATH "Cellerator source snapshot")
set(PROJECT_SOURCE_DIR "${CELLERATOR_SOURCE}")
set(CELLERATOR_BUILD_TESTS OFF)
set(CELLERATOR_ENABLE_FRAMEWORK_CUDA ON)
set(CELLERATOR_ENABLE_FRAMEWORK_CUDA_PYTHON OFF)
set(CELLERATOR_BUILD_FRAMEWORK_CUDA_SMOKE ON)
set(CELLERATOR_ENABLE_FRAMEWORK_PARAMETER_TRANSFER OFF)
include("${CELLERATOR_SOURCE}/cmake/CelleratorConditional.cmake")
include("${CELLERATOR_SOURCE}/cmake/CelleratorFrameworkCUDA.cmake")

# Retain all production dependencies and compiler flags while replacing just
# the frontier translation unit with this isolated candidate.
get_target_property(_cellerator_cuda_sources Cellerator::framework_cuda SOURCES)
list(FILTER _cellerator_cuda_sources EXCLUDE REGEX "/frontier\\.cu$")
list(APPEND _cellerator_cuda_sources "${CMAKE_CURRENT_LIST_DIR}/frontier_warp_reservation.cu")
set_property(TARGET cellerator_framework_cuda PROPERTY SOURCES "${_cellerator_cuda_sources}")

add_executable(cellerator_frontier_candidate_smoke
    "${CELLERATOR_SOURCE}/tests/framework/cuda/frontier_smoke.cu")
target_link_libraries(cellerator_frontier_candidate_smoke PRIVATE Cellerator::framework_cuda)
target_compile_features(cellerator_frontier_candidate_smoke PRIVATE cxx_std_17)
target_compile_options(cellerator_frontier_candidate_smoke PRIVATE -UNDEBUG)

add_executable(cellerator_frontier_candidate_benchmark frontier_benchmark.cu)
target_link_libraries(cellerator_frontier_candidate_benchmark PRIVATE Cellerator::framework_cuda)
target_compile_features(cellerator_frontier_candidate_benchmark PRIVATE cxx_std_17)
target_compile_options(cellerator_frontier_candidate_benchmark PRIVATE -UNDEBUG)

add_library(cellerator_frontier_original STATIC frontier_baseline.cu)
target_include_directories(cellerator_frontier_original PRIVATE
    "${CELLERATOR_SOURCE}/include")
target_compile_features(cellerator_frontier_original PRIVATE cxx_std_17)
target_link_libraries(cellerator_frontier_original PRIVATE CUDA::cudart)
target_link_libraries(cellerator_frontier_candidate_benchmark PRIVATE
    cellerator_frontier_original)
