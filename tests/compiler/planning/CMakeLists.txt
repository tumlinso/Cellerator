cmake_minimum_required(VERSION 3.20)
project(cellerator_compiler_planning_tests LANGUAGES CXX)

enable_testing()
get_filename_component(CELLERATOR_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

add_executable(ce_ccp1_e04_001
    adapt_semantic_geometry_acquisition_to_planning_ir_test.cc
    "${CELLERATOR_ROOT}/src/compiler/planning/adapt_semantic_geometry_acquisition_to_planning_ir.cc")
target_include_directories(ce_ccp1_e04_001 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_e04_001 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_e04_001 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_e04_001 COMMAND ce_ccp1_e04_001)

add_executable(ce_ccp1_e04_002
    adapt_decomposition_portfolios_to_planning_ir_test.cc
    "${CELLERATOR_ROOT}/src/compiler/planning/adapt_decomposition_portfolios_to_planning_ir.cc")
target_include_directories(ce_ccp1_e04_002 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_e04_002 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_e04_002 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_e04_002 COMMAND ce_ccp1_e04_002)

add_executable(ce_ccp1_e04_003
    represent_split_axes_fragments_halos_and_replicas_test.cc
    "${CELLERATOR_ROOT}/src/compiler/planning/represent_split_axes_fragments_halos_and_replicas.cc")
target_include_directories(ce_ccp1_e04_003 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_e04_003 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_e04_003 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_e04_003 COMMAND ce_ccp1_e04_003)

add_executable(ce_ccp1_e04_004
    represent_partial_result_trees_and_algebra_selection_test.cc
    "${CELLERATOR_ROOT}/src/compiler/planning/represent_partial_result_trees_and_algebra_selection.cc")
target_include_directories(ce_ccp1_e04_004 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_e04_004 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_e04_004 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_e04_004 COMMAND ce_ccp1_e04_004)

add_executable(ce_ccp1_e04_005
    adapt_multi_extent_direct_binding_and_assembly_fallback_test.cc
    "${CELLERATOR_ROOT}/src/compiler/planning/adapt_multi_extent_direct_binding_and_assembly_fallback.cc"
    "${CELLERATOR_ROOT}/src/execution/joint_compiler/external_binding_v1.cc"
    "${CELLERATOR_ROOT}/src/execution/joint_compiler/persistent_identity_v1.cc")
target_include_directories(ce_ccp1_e04_005 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_e04_005 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_e04_005 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_e04_005 COMMAND ce_ccp1_e04_005)
