cmake_minimum_required(VERSION 3.18)
project(CelleratorIS1HostFoundation LANGUAGES CXX)
include(CTest)
include(host_contracts.cmake)
get_filename_component(CE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
add_executable(ce_is1_contract_consumer contract_consumer.cc)
target_link_libraries(ce_is1_contract_consumer PRIVATE Cellerator::host_operation_contracts)
add_test(NAME ce_is1_contract_consumer COMMAND ce_is1_contract_consumer)
# These are source qualification consumers, not replacement installed libraries.
add_executable(ce_is1_ordered_incidence
  "${CE_ROOT}/tests/native_foundation/nary/h01.cc"
  "${CE_ROOT}/src/compute/operation/indexed_mechanism/incidence.cc")
target_link_libraries(ce_is1_ordered_incidence PRIVATE Cellerator::host_operation_contracts)
add_test(NAME ce_is1_ordered_incidence COMMAND ce_is1_ordered_incidence)
add_executable(ce_is1_host_relation
  "${CE_ROOT}/tests/native_foundation/numeric/n01.cc"
  "${CE_ROOT}/src/compute/operation/native_numeric/host_relation.cc"
  "${CE_ROOT}/src/compute/operation/relation_semantics.cc")
target_link_libraries(ce_is1_host_relation PRIVATE Cellerator::host_operation_contracts)
add_test(NAME ce_is1_host_relation COMMAND ce_is1_host_relation)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  foreach(t ce_is1_contract_consumer ce_is1_ordered_incidence ce_is1_host_relation)
    target_compile_options(${t} PRIVATE -Wall -Wextra -Werror -pedantic -fno-fast-math -ffp-contract=off)
  endforeach()
endif()
target_sources(ce_is1_contract_consumer PRIVATE "${CE_ROOT}/src/compute/operation/relation_semantics.cc")
