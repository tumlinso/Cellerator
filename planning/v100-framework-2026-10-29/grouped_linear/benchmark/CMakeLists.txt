cmake_minimum_required(VERSION 3.24)
project(CelleratorGroupedLinearBenchmark LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)
set(CELLERATOR_CUDA_LIBRARY
    "/tmp/cellerator-v100-framework-build/libcellerator_framework_cuda.a"
    CACHE FILEPATH "Previously built Cellerator CUDA library")
if(NOT EXISTS "${CELLERATOR_CUDA_LIBRARY}")
  message(FATAL_ERROR "Set CELLERATOR_CUDA_LIBRARY to the SM70 framework archive")
endif()
add_executable(cellerator_grouped_linear_benchmark benchmark.cu)
target_compile_features(cellerator_grouped_linear_benchmark PRIVATE cxx_std_17)
target_include_directories(cellerator_grouped_linear_benchmark PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/../../../../include")
target_link_libraries(cellerator_grouped_linear_benchmark PRIVATE
  "${CELLERATOR_CUDA_LIBRARY}" CUDA::cudart CUDA::cublas)
set_target_properties(cellerator_grouped_linear_benchmark PROPERTIES
  CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES 70)
