cmake_minimum_required(VERSION 3.20)
project(ce_spine_core_contract LANGUAGES CXX)
get_filename_component(CELLERATOR_SOURCE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
add_executable(ceSpineCoreTest descriptor_test.cc
    "${CELLERATOR_SOURCE_ROOT}/src/compute/operation/relation_semantics.cc")
target_include_directories(ceSpineCoreTest PRIVATE "${CELLERATOR_SOURCE_ROOT}/include")
target_compile_features(ceSpineCoreTest PRIVATE cxx_std_17)
add_executable(ceSpineMathematicalDualityTest mathematical_duality_test.cc)
target_compile_features(ceSpineMathematicalDualityTest PRIVATE cxx_std_17)
