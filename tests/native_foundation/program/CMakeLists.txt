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
