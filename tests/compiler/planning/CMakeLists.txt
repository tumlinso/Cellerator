cmake_minimum_required(VERSION 3.20)
project(cellerator_compiler_planning_tests LANGUAGES CXX)

enable_testing()
get_filename_component(CELLERATOR_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

add_executable(cellerator_adapt_semantic_geometry_acquisition_to_planning_ir_test
    adapt_semantic_geometry_acquisition_to_planning_ir_test.cc
    "${CELLERATOR_ROOT}/src/compiler/planning/adapt_semantic_geometry_acquisition_to_planning_ir.cc")
target_include_directories(cellerator_adapt_semantic_geometry_acquisition_to_planning_ir_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_adapt_semantic_geometry_acquisition_to_planning_ir_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_adapt_semantic_geometry_acquisition_to_planning_ir_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_adapt_semantic_geometry_acquisition_to_planning_ir_test COMMAND cellerator_adapt_semantic_geometry_acquisition_to_planning_ir_test)

add_executable(cellerator_adapt_decomposition_portfolios_to_planning_ir_test
    adapt_decomposition_portfolios_to_planning_ir_test.cc
    "${CELLERATOR_ROOT}/src/compiler/planning/adapt_decomposition_portfolios_to_planning_ir.cc")
target_include_directories(cellerator_adapt_decomposition_portfolios_to_planning_ir_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_adapt_decomposition_portfolios_to_planning_ir_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_adapt_decomposition_portfolios_to_planning_ir_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_adapt_decomposition_portfolios_to_planning_ir_test COMMAND cellerator_adapt_decomposition_portfolios_to_planning_ir_test)

add_executable(cellerator_represent_split_axes_fragments_halos_and_replicas_test
    represent_split_axes_fragments_halos_and_replicas_test.cc
    "${CELLERATOR_ROOT}/src/compiler/planning/represent_split_axes_fragments_halos_and_replicas.cc")
target_include_directories(cellerator_represent_split_axes_fragments_halos_and_replicas_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_represent_split_axes_fragments_halos_and_replicas_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_represent_split_axes_fragments_halos_and_replicas_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_represent_split_axes_fragments_halos_and_replicas_test COMMAND cellerator_represent_split_axes_fragments_halos_and_replicas_test)

add_executable(cellerator_represent_partial_result_trees_and_algebra_selection_test
    represent_partial_result_trees_and_algebra_selection_test.cc
    "${CELLERATOR_ROOT}/src/compiler/planning/represent_partial_result_trees_and_algebra_selection.cc")
target_include_directories(cellerator_represent_partial_result_trees_and_algebra_selection_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_represent_partial_result_trees_and_algebra_selection_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_represent_partial_result_trees_and_algebra_selection_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_represent_partial_result_trees_and_algebra_selection_test COMMAND cellerator_represent_partial_result_trees_and_algebra_selection_test)

add_executable(cellerator_adapt_multi_extent_direct_binding_and_assembly_fallback_test
    adapt_multi_extent_direct_binding_and_assembly_fallback_test.cc
    "${CELLERATOR_ROOT}/src/compiler/planning/adapt_multi_extent_direct_binding_and_assembly_fallback.cc"
    "${CELLERATOR_ROOT}/src/execution/joint_compiler/external_binding_v1.cc"
    "${CELLERATOR_ROOT}/src/execution/joint_compiler/persistent_identity_v1.cc")
target_include_directories(cellerator_adapt_multi_extent_direct_binding_and_assembly_fallback_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_adapt_multi_extent_direct_binding_and_assembly_fallback_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_adapt_multi_extent_direct_binding_and_assembly_fallback_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_adapt_multi_extent_direct_binding_and_assembly_fallback_test COMMAND cellerator_adapt_multi_extent_direct_binding_and_assembly_fallback_test)

add_executable(cellerator_adapt_candidate_catalog_v3_providers_test
    adapt_candidate_catalog_v3_providers_test.cc
    "${CELLERATOR_ROOT}/src/compiler/planning/adapt_candidate_catalog_v3_providers.cc"
    "${CELLERATOR_ROOT}/src/compute/operation/candidate_catalog/catalog_v3.cc"
    "${CELLERATOR_ROOT}/src/compute/operation/candidate_catalog/inventory_v3.cc")
target_include_directories(cellerator_adapt_candidate_catalog_v3_providers_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_adapt_candidate_catalog_v3_providers_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_adapt_candidate_catalog_v3_providers_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_adapt_candidate_catalog_v3_providers_test COMMAND cellerator_adapt_candidate_catalog_v3_providers_test)

add_executable(cellerator_custom_candidate_registration_test
    custom_candidate_registration_test.cc
    "${CELLERATOR_ROOT}/src/compiler/planning/custom_candidate_registration.cc")
target_include_directories(cellerator_custom_candidate_registration_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_custom_candidate_registration_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_custom_candidate_registration_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_custom_candidate_registration_test COMMAND cellerator_custom_candidate_registration_test)

add_executable(cellerator_candidate_inclusion_exclusion_and_forcing_test
    candidate_inclusion_exclusion_and_forcing_test.cc
    "${CELLERATOR_ROOT}/src/compiler/planning/candidate_inclusion_exclusion_and_forcing.cc")
target_include_directories(cellerator_candidate_inclusion_exclusion_and_forcing_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_candidate_inclusion_exclusion_and_forcing_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_candidate_inclusion_exclusion_and_forcing_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_candidate_inclusion_exclusion_and_forcing_test COMMAND cellerator_candidate_inclusion_exclusion_and_forcing_test)

add_executable(cellerator_adapt_external_global_cost_exchange_test
    adapt_external_global_cost_exchange_test.cc
    "${CELLERATOR_ROOT}/src/compiler/planning/adapt_external_global_cost_exchange.cc"
    "${CELLERATOR_ROOT}/src/planner/external_cost/vector_v1.cc")
target_include_directories(cellerator_adapt_external_global_cost_exchange_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_adapt_external_global_cost_exchange_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_adapt_external_global_cost_exchange_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_adapt_external_global_cost_exchange_test COMMAND cellerator_adapt_external_global_cost_exchange_test)

add_executable(cellerator_complete_cost_normalization_test
    complete_cost_normalization_test.cc
    "${CELLERATOR_ROOT}/src/compiler/planning/complete_cost_normalization.cc")
target_include_directories(cellerator_complete_cost_normalization_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_complete_cost_normalization_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_complete_cost_normalization_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_complete_cost_normalization_test COMMAND cellerator_complete_cost_normalization_test)

add_executable(cellerator_adapt_transition_and_connected_operation_costs_test
    adapt_transition_and_connected_operation_costs_test.cc
    "${CELLERATOR_ROOT}/src/compiler/planning/adapt_transition_and_connected_operation_costs.cc")
target_include_directories(cellerator_adapt_transition_and_connected_operation_costs_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_adapt_transition_and_connected_operation_costs_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_adapt_transition_and_connected_operation_costs_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_adapt_transition_and_connected_operation_costs_test COMMAND cellerator_adapt_transition_and_connected_operation_costs_test)

add_executable(cellerator_planner_portfolio_dispatch_test
    planner_portfolio_dispatch_test.cc
    "${CELLERATOR_ROOT}/src/compiler/planning/planner_portfolio_dispatch.cc")
target_include_directories(cellerator_planner_portfolio_dispatch_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_planner_portfolio_dispatch_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_planner_portfolio_dispatch_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_planner_portfolio_dispatch_test COMMAND cellerator_planner_portfolio_dispatch_test)

add_executable(cellerator_profile_family_plan_variants_test
    profile_family_plan_variants_test.cc
    "${CELLERATOR_ROOT}/src/compiler/planning/profile_family_plan_variants.cc")
target_include_directories(cellerator_profile_family_plan_variants_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_profile_family_plan_variants_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_profile_family_plan_variants_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_profile_family_plan_variants_test COMMAND cellerator_profile_family_plan_variants_test)

add_executable(cellerator_planning_cache_and_invalidation_test
    planning_cache_and_invalidation_test.cc
    "${CELLERATOR_ROOT}/src/compiler/planning/planning_cache_and_invalidation.cc")
target_include_directories(cellerator_planning_cache_and_invalidation_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_planning_cache_and_invalidation_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_planning_cache_and_invalidation_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_planning_cache_and_invalidation_test COMMAND cellerator_planning_cache_and_invalidation_test)

add_executable(cellerator_expose_complete_planning_reports_test
    expose_complete_planning_reports_test.cc
    "${CELLERATOR_ROOT}/src/compiler/planning/expose_complete_planning_reports.cc")
target_include_directories(cellerator_expose_complete_planning_reports_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_expose_complete_planning_reports_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_expose_complete_planning_reports_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_expose_complete_planning_reports_test COMMAND cellerator_expose_complete_planning_reports_test)

add_executable(cellerator_benchmark_planning_scalability_and_boundedness_test
    benchmark_planning_scalability_and_boundedness_test.cc
    "${CELLERATOR_ROOT}/src/compiler/planning/benchmark_planning_scalability_and_boundedness.cc")
target_include_directories(cellerator_benchmark_planning_scalability_and_boundedness_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_benchmark_planning_scalability_and_boundedness_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_benchmark_planning_scalability_and_boundedness_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_benchmark_planning_scalability_and_boundedness_test COMMAND cellerator_benchmark_planning_scalability_and_boundedness_test)

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
add_test(NAME cellerator_source_to_plan_pipeline_test COMMAND cellerator_source_to_plan_pipeline_test)

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
add_test(NAME cellerator_public_planning_compiler_interface_test COMMAND cellerator_public_planning_compiler_interface_test)
