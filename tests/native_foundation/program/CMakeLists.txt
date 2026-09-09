cmake_minimum_required(VERSION 3.28)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(cellerator_nf1_program LANGUAGES CXX)
  include(CTest)
endif()
get_filename_component(CE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
add_executable(ce_nf1_p01 p01.cc "${CE_ROOT}/src/execution/program/program_v2.cc"
  "${CE_ROOT}/src/execution/program/program_v1_adapter.cc")
target_include_directories(ce_nf1_p01 PRIVATE "${CE_ROOT}/include")
target_compile_features(ce_nf1_p01 PRIVATE cxx_std_17)
add_test(NAME ce_nf1_p01 COMMAND ce_nf1_p01)
add_executable(ce_nf1_program_legacy "${CE_ROOT}/tests/jbc/fragment/canonical_relation_apply_smoke.cc" "${CE_ROOT}/src/execution/program/program_v2.cc")
target_include_directories(ce_nf1_program_legacy PRIVATE "${CE_ROOT}/include")
target_compile_features(ce_nf1_program_legacy PRIVATE cxx_std_17)
target_compile_options(ce_nf1_program_legacy PRIVATE -UNDEBUG)
add_test(NAME ce_nf1_program_legacy COMMAND ce_nf1_program_legacy)

target_sources(ce_nf1_program_legacy PRIVATE
  "${CE_ROOT}/src/compute/operation/operation_core_v2/schema.cc"
  "${CE_ROOT}/src/execution/atom_fragment/compiler_registry_v1.cc"
  "${CE_ROOT}/src/execution/atom_fragment/prepared_atom_fragment_v1.cc"
  "${CE_ROOT}/src/execution/atom_fragment/external_plane_binding_v1.cc"
  "${CE_ROOT}/src/execution/atom_fragment/output_affordance_v1.cc"
  "${CE_ROOT}/src/execution/atom_fragment/canonical_fallback_v1.cc")
target_sources(ce_nf1_program_legacy PRIVATE
  "${CE_ROOT}/src/execution/joint_compiler/persistent_identity_v1.cc"
  "${CE_ROOT}/src/execution/joint_compiler/atom_requirement_v1.cc"
  "${CE_ROOT}/src/execution/joint_compiler/atom_affordance_v1.cc"
  "${CE_ROOT}/src/execution/joint_compiler/external_binding_v1.cc")
add_executable(ce_nf1_p02 p02.cc
  "${CE_ROOT}/src/execution/program/program_v2.cc"
  "${CE_ROOT}/src/execution/joint_compiler/persistent_identity_v1.cc"
  "${CE_ROOT}/src/execution/joint_compiler/external_binding_v1.cc")
target_include_directories(ce_nf1_p02 PRIVATE "${CE_ROOT}/include")
target_compile_features(ce_nf1_p02 PRIVATE cxx_std_17)
add_test(NAME ce_nf1_p02 COMMAND ce_nf1_p02)
option(NF1_PROGRAM_CUDA "Qualify real native session-bound programs" OFF)
if(NF1_PROGRAM_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_executable(ce_nf1_p03 p03.cu
    "${CE_ROOT}/src/execution/program/program_v2.cc"
    "${CE_ROOT}/src/execution/program/session_v2.cc"
    "${CE_ROOT}/src/runtime/runtime.cu"
    "${CE_ROOT}/src/runtime/relation_value_readiness.cu")
  target_include_directories(ce_nf1_p03 PRIVATE "${CE_ROOT}/include")
  target_compile_features(ce_nf1_p03 PRIVATE cxx_std_17)
  set_target_properties(ce_nf1_p03 PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
  target_link_libraries(ce_nf1_p03 PRIVATE CUDA::cudart CUDA::cublas CUDA::cusparse)
  add_test(NAME ce_nf1_p03 COMMAND ce_nf1_p03)
  find_program(NF1_COMPUTE_SANITIZER compute-sanitizer HINTS "${CUDAToolkit_ROOT}/compute-sanitizer" REQUIRED)
  add_test(NAME ce_nf1_p03_memcheck COMMAND "${NF1_COMPUTE_SANITIZER}" --tool memcheck --error-exitcode 99 $<TARGET_FILE:ce_nf1_p03>)
  set_tests_properties(ce_nf1_p03 ce_nf1_p03_memcheck PROPERTIES LABELS "nf1;cuda;session" TIMEOUT 120)
endif()
add_executable(ce_nf1_p04 p04.cc
  "${CE_ROOT}/src/execution/program/program_v2.cc"
  "${CE_ROOT}/src/execution/program/workspace_v2.cc")
target_include_directories(ce_nf1_p04 PRIVATE "${CE_ROOT}/include")
target_compile_features(ce_nf1_p04 PRIVATE cxx_std_17)
add_test(NAME ce_nf1_p04 COMMAND ce_nf1_p04)
add_executable(ce_nf1_p05 p05.cc "${CE_ROOT}/src/execution/program/program_v2.cc")
target_include_directories(ce_nf1_p05 PRIVATE "${CE_ROOT}/include")
target_compile_features(ce_nf1_p05 PRIVATE cxx_std_17)
add_test(NAME ce_nf1_p05 COMMAND ce_nf1_p05)
if(NF1_PROGRAM_CUDA)
  add_executable(ce_nf1_p05_cuda p05.cu
    "${CE_ROOT}/src/execution/program/program_v2.cc"
    "${CE_ROOT}/src/execution/program/session_v2.cc"
    "${CE_ROOT}/src/runtime/runtime.cu"
    "${CE_ROOT}/src/runtime/relation_value_readiness.cu")
  target_include_directories(ce_nf1_p05_cuda PRIVATE "${CE_ROOT}/include")
  target_compile_features(ce_nf1_p05_cuda PRIVATE cxx_std_17)
  set_target_properties(ce_nf1_p05_cuda PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
  target_link_libraries(ce_nf1_p05_cuda PRIVATE CUDA::cudart CUDA::cublas CUDA::cusparse)
  add_test(NAME ce_nf1_p05_cuda COMMAND ce_nf1_p05_cuda)
  add_test(NAME ce_nf1_p05_memcheck COMMAND "${NF1_COMPUTE_SANITIZER}" --tool memcheck --error-exitcode 99 $<TARGET_FILE:ce_nf1_p05_cuda>)
  set_tests_properties(ce_nf1_p05_cuda ce_nf1_p05_memcheck PROPERTIES LABELS "nf1;cuda;poison" TIMEOUT 120)
endif()
