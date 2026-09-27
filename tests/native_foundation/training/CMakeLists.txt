cmake_minimum_required(VERSION 3.28)
project(CelleratorML2NativeTraining LANGUAGES CXX CUDA)
include(CTest)
find_package(CUDAToolkit REQUIRED)

get_filename_component(CE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
add_library(ce_ml2_test_runtime STATIC
  "${CE_ROOT}/src/runtime/runtime.cu"
  "${CE_ROOT}/src/runtime/device_descriptor.cu"
  "${CE_ROOT}/src/runtime/memory/allocation.cc"
  "${CE_ROOT}/src/runtime/memory/copy.cu"
  "${CE_ROOT}/src/runtime/memory/generation_marks.cc"
  "${CE_ROOT}/src/runtime/memory/image.cc"
  "${CE_ROOT}/src/runtime/memory/session_memory.cu"
  "${CE_ROOT}/src/runtime/value_readiness.cu")
add_library(Cellerator::runtime ALIAS ce_ml2_test_runtime)
target_include_directories(ce_ml2_test_runtime PUBLIC "${CE_ROOT}/include")
target_compile_features(ce_ml2_test_runtime PUBLIC cxx_std_17)
target_link_libraries(ce_ml2_test_runtime PUBLIC CUDA::cudart PRIVATE CUDA::cublas CUDA::cusparse)
set_target_properties(ce_ml2_test_runtime PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)

add_library(ce_ml2_test_program STATIC "${CE_ROOT}/src/execution/program/program_v2.cc")
add_library(Cellerator::executable_program ALIAS ce_ml2_test_program)
target_include_directories(ce_ml2_test_program PUBLIC "${CE_ROOT}/include")
target_compile_features(ce_ml2_test_program PUBLIC cxx_std_17)

add_subdirectory("${CE_ROOT}/src/compute/operation/indexed_mechanism" "${CMAKE_CURRENT_BINARY_DIR}/indexed_mechanism")
add_executable(ce_ml2_native_training training.cu)
target_link_libraries(ce_ml2_native_training PRIVATE Cellerator::indexed_mechanism CUDA::cudart)
target_compile_features(ce_ml2_native_training PRIVATE cxx_std_20)
set_target_properties(ce_ml2_native_training PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES)
add_test(NAME ce_ml2_native_training COMMAND ce_ml2_native_training)
set_tests_properties(ce_ml2_native_training PROPERTIES LABELS "ml2;cuda;native;sm70" TIMEOUT 180)
get_filename_component(_ml2_cuda_bin "${CMAKE_CUDA_COMPILER}" DIRECTORY)
get_filename_component(_ml2_cuda_root "${_ml2_cuda_bin}" DIRECTORY)
set(_ml2_compute_sanitizer "${_ml2_cuda_root}/compute-sanitizer/compute-sanitizer")
if(EXISTS "${_ml2_compute_sanitizer}")
  add_test(NAME ce_ml2_native_training_memcheck COMMAND "${_ml2_compute_sanitizer}"
    --tool memcheck --error-exitcode 99 $<TARGET_FILE:ce_ml2_native_training>)
  set_tests_properties(ce_ml2_native_training_memcheck PROPERTIES
    LABELS "ml2;cuda;native;sanitizer;sm70" TIMEOUT 300)
endif()
