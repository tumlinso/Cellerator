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

add_executable(cellerator_custom_candidate_registration_test
    custom_candidate_registration_test.cc
    "${CELLERATOR_ROOT}/src/compiler/planning/custom_candidate_registration.cc")
target_include_directories(cellerator_custom_candidate_registration_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_custom_candidate_registration_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_custom_candidate_registration_test PRIVATE -Wall -Wextra -Werror)
endif()
add_executable(ce_ccp1_e04_007 ALIAS cellerator_custom_candidate_registration_test)
add_test(NAME ce_ccp1_e04_007 COMMAND cellerator_custom_candidate_registration_test)

add_executable(cellerator_candidate_inclusion_exclusion_and_forcing_test
    candidate_inclusion_exclusion_and_forcing_test.cc
    "${CELLERATOR_ROOT}/src/compiler/planning/candidate_inclusion_exclusion_and_forcing.cc")
target_include_directories(cellerator_candidate_inclusion_exclusion_and_forcing_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_candidate_inclusion_exclusion_and_forcing_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_candidate_inclusion_exclusion_and_forcing_test PRIVATE -Wall -Wextra -Werror)
endif()
add_executable(ce_ccp1_e04_008 ALIAS cellerator_candidate_inclusion_exclusion_and_forcing_test)
add_test(NAME ce_ccp1_e04_008 COMMAND cellerator_candidate_inclusion_exclusion_and_forcing_test)

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

add_executable(cellerator_complete_cost_normalization_test
    complete_cost_normalization_test.cc
    "${CELLERATOR_ROOT}/src/compiler/planning/complete_cost_normalization.cc")
target_include_directories(cellerator_complete_cost_normalization_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_complete_cost_normalization_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_complete_cost_normalization_test PRIVATE -Wall -Wextra -Werror)
endif()
add_executable(ce_ccp1_e04_010 ALIAS cellerator_complete_cost_normalization_test)
add_test(NAME ce_ccp1_e04_010 COMMAND cellerator_complete_cost_normalization_test)

add_executable(ce_ccp1_e04_011
    adapt_transition_and_connected_operation_costs_test.cc
    "${CELLERATOR_ROOT}/src/compiler/planning/adapt_transition_and_connected_operation_costs.cc")
target_include_directories(ce_ccp1_e04_011 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_e04_011 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_e04_011 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_e04_011 COMMAND ce_ccp1_e04_011)

add_executable(cellerator_planner_portfolio_dispatch_test
    planner_portfolio_dispatch_test.cc
    "${CELLERATOR_ROOT}/src/compiler/planning/planner_portfolio_dispatch.cc")
target_include_directories(cellerator_planner_portfolio_dispatch_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_planner_portfolio_dispatch_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_planner_portfolio_dispatch_test PRIVATE -Wall -Wextra -Werror)
endif()
add_executable(ce_ccp1_e04_012 ALIAS cellerator_planner_portfolio_dispatch_test)
add_test(NAME ce_ccp1_e04_012 COMMAND cellerator_planner_portfolio_dispatch_test)

add_executable(cellerator_profile_family_plan_variants_test
    profile_family_plan_variants_test.cc
    "${CELLERATOR_ROOT}/src/compiler/planning/profile_family_plan_variants.cc")
target_include_directories(cellerator_profile_family_plan_variants_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_profile_family_plan_variants_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_profile_family_plan_variants_test PRIVATE -Wall -Wextra -Werror)
endif()
add_executable(ce_ccp1_e04_013 ALIAS cellerator_profile_family_plan_variants_test)
add_test(NAME ce_ccp1_e04_013 COMMAND cellerator_profile_family_plan_variants_test)

add_executable(cellerator_planning_cache_and_invalidation_test
    planning_cache_and_invalidation_test.cc
    "${CELLERATOR_ROOT}/src/compiler/planning/planning_cache_and_invalidation.cc")
target_include_directories(cellerator_planning_cache_and_invalidation_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_planning_cache_and_invalidation_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_planning_cache_and_invalidation_test PRIVATE -Wall -Wextra -Werror)
endif()
add_executable(ce_ccp1_e04_014 ALIAS cellerator_planning_cache_and_invalidation_test)
add_test(NAME ce_ccp1_e04_014 COMMAND cellerator_planning_cache_and_invalidation_test)

add_executable(ce_ccp1_e04_015
    expose_complete_planning_reports_test.cc
    "${CELLERATOR_ROOT}/src/compiler/planning/expose_complete_planning_reports.cc")
target_include_directories(ce_ccp1_e04_015 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_e04_015 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_e04_015 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_e04_015 COMMAND ce_ccp1_e04_015)

add_executable(ce_ccp1_e04_016
    benchmark_planning_scalability_and_boundedness_test.cc
    "${CELLERATOR_ROOT}/src/compiler/planning/benchmark_planning_scalability_and_boundedness.cc")
target_include_directories(ce_ccp1_e04_016 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_e04_016 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_e04_016 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_e04_016 COMMAND ce_ccp1_e04_016)

set(CELLERATOR_COMPILER_INTERFACE_ROOT "" CACHE PATH
    "Root containing frozen compiler interface headers from a contract-split lane")
if(EXISTS "${CELLERATOR_ROOT}/include/Cellerator/compiler/ir/planning/planning_ir_v1.hh")
    set(CELLERATOR_PLANNING_INTERFACE_INCLUDE "${CELLERATOR_ROOT}/include")
elseif(CELLERATOR_COMPILER_INTERFACE_ROOT AND
       EXISTS "${CELLERATOR_COMPILER_INTERFACE_ROOT}/include/Cellerator/compiler/ir/planning/planning_ir_v1.hh")
    set(CELLERATOR_PLANNING_INTERFACE_INCLUDE
        "${CELLERATOR_COMPILER_INTERFACE_ROOT}/include")
else()
    message(FATAL_ERROR
        "CE-CCP1-I19 planning_ir_v1.hh is required; set CELLERATOR_COMPILER_INTERFACE_ROOT for a contract-split lane build")
endif()

add_executable(cellerator_source_to_plan_pipeline_test
    source_to_plan_pipeline_test.cc
    "${CELLERATOR_ROOT}/src/compiler/planning/source_to_plan_pipeline.cc"
    "${CELLERATOR_ROOT}/src/compiler/planning/adapt_decomposition_portfolios_to_planning_ir.cc"
    "${CELLERATOR_ROOT}/src/compiler/planning/complete_cost_normalization.cc"
    "${CELLERATOR_ROOT}/src/compiler/ir/semantic/source_to_semantic_ir_vertical_slice.cc")
target_include_directories(cellerator_source_to_plan_pipeline_test PRIVATE
    "${CELLERATOR_ROOT}/include" "${CELLERATOR_PLANNING_INTERFACE_INCLUDE}")
target_compile_features(cellerator_source_to_plan_pipeline_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_source_to_plan_pipeline_test PRIVATE -Wall -Wextra -Werror)
endif()
add_executable(ce_ccp1_e04_017 ALIAS cellerator_source_to_plan_pipeline_test)
add_test(NAME ce_ccp1_e04_017 COMMAND cellerator_source_to_plan_pipeline_test)

set(CELLERATOR_DISCOVERY_INTERFACE_ROOT "" CACHE PATH
    "Root containing frozen discovery interface headers from a contract-split lane")
set(CELLERATOR_COMPOSITION_INTERFACE_ROOT "" CACHE PATH
    "Root containing frozen composition interface headers from a contract-split lane")

if(EXISTS "${CELLERATOR_ROOT}/include/Cellerator/compiler/discovery/discovery_v1.hh")
    set(CELLERATOR_DISCOVERY_INTERFACE_INCLUDE "${CELLERATOR_ROOT}/include")
elseif(CELLERATOR_DISCOVERY_INTERFACE_ROOT AND
       EXISTS "${CELLERATOR_DISCOVERY_INTERFACE_ROOT}/include/Cellerator/compiler/discovery/discovery_v1.hh")
    set(CELLERATOR_DISCOVERY_INTERFACE_INCLUDE
        "${CELLERATOR_DISCOVERY_INTERFACE_ROOT}/include")
else()
    message(FATAL_ERROR
        "CE-CCP1-I20 discovery_v1.hh is required; set CELLERATOR_DISCOVERY_INTERFACE_ROOT for a contract-split lane build")
endif()

if(EXISTS "${CELLERATOR_ROOT}/include/Cellerator/compiler/composition/grammar_v1.hh" AND
   EXISTS "${CELLERATOR_ROOT}/include/Cellerator/compiler/program/ruleset_v1.hh")
    set(CELLERATOR_COMPOSITION_INTERFACE_INCLUDE "${CELLERATOR_ROOT}/include")
elseif(CELLERATOR_COMPOSITION_INTERFACE_ROOT AND
       EXISTS "${CELLERATOR_COMPOSITION_INTERFACE_ROOT}/include/Cellerator/compiler/composition/grammar_v1.hh" AND
       EXISTS "${CELLERATOR_COMPOSITION_INTERFACE_ROOT}/include/Cellerator/compiler/program/ruleset_v1.hh")
    set(CELLERATOR_COMPOSITION_INTERFACE_INCLUDE
        "${CELLERATOR_COMPOSITION_INTERFACE_ROOT}/include")
else()
    message(FATAL_ERROR
        "CE-CCP1-I21/I22 composition interfaces are required; set CELLERATOR_COMPOSITION_INTERFACE_ROOT for a contract-split lane build")
endif()

add_executable(cellerator_public_planning_compiler_interface_test
    public_planning_compiler_interface_test.cc
    "${CELLERATOR_ROOT}/src/compiler/planning/public_planning_compiler_interface.cc")
target_include_directories(cellerator_public_planning_compiler_interface_test PRIVATE
    "${CELLERATOR_ROOT}/include"
    "${CELLERATOR_PLANNING_INTERFACE_INCLUDE}"
    "${CELLERATOR_DISCOVERY_INTERFACE_INCLUDE}"
    "${CELLERATOR_COMPOSITION_INTERFACE_INCLUDE}")
target_compile_features(cellerator_public_planning_compiler_interface_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_public_planning_compiler_interface_test PRIVATE -Wall -Wextra -Werror)
endif()
add_executable(ce_ccp1_e04_018 ALIAS cellerator_public_planning_compiler_interface_test)
add_test(NAME ce_ccp1_e04_018 COMMAND cellerator_public_planning_compiler_interface_test)
