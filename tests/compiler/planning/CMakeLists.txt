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

add_executable(ce_ccp1_e04_006
    adapt_candidate_catalog_v3_providers_test.cc
    "${CELLERATOR_ROOT}/src/compiler/planning/adapt_candidate_catalog_v3_providers.cc"
    "${CELLERATOR_ROOT}/src/compute/operation/candidate_catalog/catalog_v3.cc"
    "${CELLERATOR_ROOT}/src/compute/operation/candidate_catalog/inventory_v3.cc")
target_include_directories(ce_ccp1_e04_006 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_e04_006 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_e04_006 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_e04_006 COMMAND ce_ccp1_e04_006)

add_executable(ce_ccp1_e04_007
    implement_custom_candidate_registration_test.cc
    "${CELLERATOR_ROOT}/src/compiler/planning/implement_custom_candidate_registration.cc")
target_include_directories(ce_ccp1_e04_007 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_e04_007 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_e04_007 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_e04_007 COMMAND ce_ccp1_e04_007)

add_executable(ce_ccp1_e04_008
    implement_candidate_inclusion_exclusion_and_forcing_test.cc
    "${CELLERATOR_ROOT}/src/compiler/planning/implement_candidate_inclusion_exclusion_and_forcing.cc")
target_include_directories(ce_ccp1_e04_008 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_e04_008 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_e04_008 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_e04_008 COMMAND ce_ccp1_e04_008)

add_executable(ce_ccp1_e04_009
    adapt_external_global_cost_exchange_test.cc
    "${CELLERATOR_ROOT}/src/compiler/planning/adapt_external_global_cost_exchange.cc"
    "${CELLERATOR_ROOT}/src/planner/external_cost/vector_v1.cc")
target_include_directories(ce_ccp1_e04_009 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_e04_009 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_e04_009 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_e04_009 COMMAND ce_ccp1_e04_009)

add_executable(ce_ccp1_e04_010
    implement_complete_cost_normalization_test.cc
    "${CELLERATOR_ROOT}/src/compiler/planning/implement_complete_cost_normalization.cc")
target_include_directories(ce_ccp1_e04_010 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_e04_010 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_e04_010 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_e04_010 COMMAND ce_ccp1_e04_010)

add_executable(ce_ccp1_e04_011
    adapt_transition_and_connected_operation_costs_test.cc
    "${CELLERATOR_ROOT}/src/compiler/planning/adapt_transition_and_connected_operation_costs.cc")
target_include_directories(ce_ccp1_e04_011 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_e04_011 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_e04_011 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_e04_011 COMMAND ce_ccp1_e04_011)

add_executable(ce_ccp1_e04_012
    implement_planner_portfolio_dispatch_test.cc
    "${CELLERATOR_ROOT}/src/compiler/planning/implement_planner_portfolio_dispatch.cc")
target_include_directories(ce_ccp1_e04_012 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_e04_012 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_e04_012 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_e04_012 COMMAND ce_ccp1_e04_012)
