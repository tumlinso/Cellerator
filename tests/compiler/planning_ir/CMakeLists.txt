cmake_minimum_required(VERSION 3.20)
project(cellerator_compiler_planning_ir_tests LANGUAGES CXX)

enable_testing()
get_filename_component(CELLERATOR_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

add_executable(cellerator_planning_ir_module_and_decision_state_model_test
    planning_ir_module_and_decision_state_model_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/planning/planning_ir_module_and_decision_state_model.cc")
target_include_directories(cellerator_planning_ir_module_and_decision_state_model_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_planning_ir_module_and_decision_state_model_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_planning_ir_module_and_decision_state_model_test PRIVATE -Wall -Wextra -Werror)
endif()
add_executable(ce_ccp1_e01_001 ALIAS cellerator_planning_ir_module_and_decision_state_model_test)
add_test(NAME ce_ccp1_e01_001 COMMAND cellerator_planning_ir_module_and_decision_state_model_test)

add_executable(cellerator_planning_problems_and_operation_scopes_test
    planning_problems_and_operation_scopes_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/planning/planning_problems_and_operation_scopes.cc")
target_include_directories(cellerator_planning_problems_and_operation_scopes_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_planning_problems_and_operation_scopes_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_planning_problems_and_operation_scopes_test PRIVATE -Wall -Wextra -Werror)
endif()
add_executable(ce_ccp1_e01_002 ALIAS cellerator_planning_problems_and_operation_scopes_test)
add_test(NAME ce_ccp1_e01_002 COMMAND cellerator_planning_problems_and_operation_scopes_test)

add_executable(cellerator_candidate_family_and_provider_nodes_test
    candidate_family_and_provider_nodes_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/planning/candidate_family_and_provider_nodes.cc")
target_include_directories(cellerator_candidate_family_and_provider_nodes_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_candidate_family_and_provider_nodes_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_candidate_family_and_provider_nodes_test PRIVATE -Wall -Wextra -Werror)
endif()
add_executable(ce_ccp1_e01_003 ALIAS cellerator_candidate_family_and_provider_nodes_test)
add_test(NAME ce_ccp1_e01_003 COMMAND cellerator_candidate_family_and_provider_nodes_test)

add_executable(cellerator_exact_logical_coverage_nodes_test
    exact_logical_coverage_nodes_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/planning/exact_logical_coverage_nodes.cc")
target_include_directories(cellerator_exact_logical_coverage_nodes_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_exact_logical_coverage_nodes_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_exact_logical_coverage_nodes_test PRIVATE -Wall -Wextra -Werror)
endif()
add_executable(ce_ccp1_e01_004 ALIAS cellerator_exact_logical_coverage_nodes_test)
add_test(NAME ce_ccp1_e01_004 COMMAND cellerator_exact_logical_coverage_nodes_test)

add_executable(cellerator_atom_requirement_and_affordance_nodes_test
    atom_requirement_and_affordance_nodes_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/planning/atom_requirement_and_affordance_nodes.cc"
    "${CELLERATOR_ROOT}/src/execution/joint_compiler/persistent_identity_v1.cc"
    "${CELLERATOR_ROOT}/src/execution/joint_compiler/atom_requirement_v1.cc"
    "${CELLERATOR_ROOT}/src/execution/joint_compiler/atom_affordance_v1.cc")
target_include_directories(cellerator_atom_requirement_and_affordance_nodes_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_atom_requirement_and_affordance_nodes_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_atom_requirement_and_affordance_nodes_test PRIVATE -Wall -Wextra -Werror)
endif()
add_executable(ce_ccp1_e01_005 ALIAS cellerator_atom_requirement_and_affordance_nodes_test)
add_test(NAME ce_ccp1_e01_005 COMMAND cellerator_atom_requirement_and_affordance_nodes_test)

add_executable(cellerator_decomposition_alternative_nodes_test
    decomposition_alternative_nodes_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/planning/decomposition_alternative_nodes.cc")
target_include_directories(cellerator_decomposition_alternative_nodes_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_decomposition_alternative_nodes_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_decomposition_alternative_nodes_test PRIVATE -Wall -Wextra -Werror)
endif()
add_executable(ce_ccp1_e01_006 ALIAS cellerator_decomposition_alternative_nodes_test)
add_test(NAME ce_ccp1_e01_006 COMMAND cellerator_decomposition_alternative_nodes_test)

add_executable(cellerator_partial_result_algebra_nodes_test
    partial_result_algebra_nodes_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/planning/partial_result_algebra_nodes.cc"
    "${CELLERATOR_ROOT}/src/execution/joint_compiler/persistent_identity_v1.cc"
    "${CELLERATOR_ROOT}/src/execution/joint_compiler/partial_result_algebra_v1.cc")
target_include_directories(cellerator_partial_result_algebra_nodes_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_partial_result_algebra_nodes_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_partial_result_algebra_nodes_test PRIVATE -Wall -Wextra -Werror)
endif()
add_executable(ce_ccp1_e01_007 ALIAS cellerator_partial_result_algebra_nodes_test)
add_test(NAME ce_ccp1_e01_007 COMMAND cellerator_partial_result_algebra_nodes_test)

add_executable(cellerator_persistent_order_packing_alternatives_test
    persistent_order_packing_alternatives_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/planning/persistent_order_packing_alternatives.cc")
target_include_directories(cellerator_persistent_order_packing_alternatives_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_persistent_order_packing_alternatives_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_persistent_order_packing_alternatives_test PRIVATE -Wall -Wextra -Werror)
endif()
add_executable(ce_ccp1_e01_008 ALIAS cellerator_persistent_order_packing_alternatives_test)
add_test(NAME ce_ccp1_e01_008 COMMAND cellerator_persistent_order_packing_alternatives_test)
add_executable(cellerator_resource_and_stage_inventory_alternatives_test resource_and_stage_inventory_alternatives_test.cc "${CELLERATOR_ROOT}/src/compiler/ir/planning/resource_and_stage_inventory_alternatives.cc")
target_include_directories(cellerator_resource_and_stage_inventory_alternatives_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_resource_and_stage_inventory_alternatives_test PRIVATE cxx_std_17)
add_executable(ce_ccp1_e01_009 ALIAS cellerator_resource_and_stage_inventory_alternatives_test)
add_test(NAME ce_ccp1_e01_009 COMMAND cellerator_resource_and_stage_inventory_alternatives_test)
add_executable(cellerator_complete_cost_vectors_test complete_cost_vectors_test.cc "${CELLERATOR_ROOT}/src/compiler/ir/planning/complete_cost_vectors.cc")
target_include_directories(cellerator_complete_cost_vectors_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_complete_cost_vectors_test PRIVATE cxx_std_17)
add_executable(ce_ccp1_e01_010 ALIAS cellerator_complete_cost_vectors_test)
add_test(NAME ce_ccp1_e01_010 COMMAND cellerator_complete_cost_vectors_test)
add_executable(cellerator_analytical_measured_and_external_evidence_test analytical_measured_and_external_evidence_test.cc "${CELLERATOR_ROOT}/src/compiler/ir/planning/analytical_measured_and_external_evidence.cc")
target_include_directories(cellerator_analytical_measured_and_external_evidence_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_analytical_measured_and_external_evidence_test PRIVATE cxx_std_17)
add_executable(ce_ccp1_e01_011 ALIAS cellerator_analytical_measured_and_external_evidence_test)
add_test(NAME ce_ccp1_e01_011 COMMAND cellerator_analytical_measured_and_external_evidence_test)
add_executable(cellerator_rejection_and_dominance_explanations_test rejection_and_dominance_explanations_test.cc "${CELLERATOR_ROOT}/src/compiler/ir/planning/rejection_and_dominance_explanations.cc")
target_include_directories(cellerator_rejection_and_dominance_explanations_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_rejection_and_dominance_explanations_test PRIVATE cxx_std_17)
add_executable(ce_ccp1_e01_012 ALIAS cellerator_rejection_and_dominance_explanations_test)
add_test(NAME ce_ccp1_e01_012 COMMAND cellerator_rejection_and_dominance_explanations_test)
add_executable(cellerator_user_edits_and_authority_hierarchy_test user_edits_and_authority_hierarchy_test.cc "${CELLERATOR_ROOT}/src/compiler/ir/planning/user_edits_and_authority_hierarchy.cc")
target_include_directories(cellerator_user_edits_and_authority_hierarchy_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_user_edits_and_authority_hierarchy_test PRIVATE cxx_std_17)
add_executable(ce_ccp1_e01_013 ALIAS cellerator_user_edits_and_authority_hierarchy_test)
add_test(NAME ce_ccp1_e01_013 COMMAND cellerator_user_edits_and_authority_hierarchy_test)
add_executable(cellerator_planning_ir_parser_printer_and_validator_test planning_ir_parser_printer_and_validator_test.cc "${CELLERATOR_ROOT}/src/compiler/ir/planning/planning_ir_parser_printer_and_validator.cc")
target_include_directories(cellerator_planning_ir_parser_printer_and_validator_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_planning_ir_parser_printer_and_validator_test PRIVATE cxx_std_17)
add_executable(ce_ccp1_e01_014 ALIAS cellerator_planning_ir_parser_printer_and_validator_test)
add_test(NAME ce_ccp1_e01_014 COMMAND cellerator_planning_ir_parser_printer_and_validator_test)

add_executable(cellerator_semantic_to_planning_lowering_test
    semantic_to_planning_lowering_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/planning/semantic_to_planning_lowering.cc"
    "${CELLERATOR_ROOT}/src/compiler/ir/planning/planning_problems_and_operation_scopes.cc"
    "${CELLERATOR_ROOT}/src/compiler/ir/semantic/state_and_value_plane_ir_types.cc")
target_include_directories(cellerator_semantic_to_planning_lowering_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_semantic_to_planning_lowering_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_semantic_to_planning_lowering_test PRIVATE -Wall -Wextra -Werror)
endif()
add_executable(ce_ccp1_e01_015 ALIAS cellerator_semantic_to_planning_lowering_test)
add_test(NAME ce_ccp1_e01_015 COMMAND cellerator_semantic_to_planning_lowering_test)

add_executable(cellerator_first_inspectable_candidate_search_space_test
    first_inspectable_candidate_search_space_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/planning/first_inspectable_candidate_search_space.cc"
    "${CELLERATOR_ROOT}/src/compiler/ir/planning/complete_cost_vectors.cc"
    "${CELLERATOR_ROOT}/src/compiler/ir/planning/planning_problems_and_operation_scopes.cc"
    "${CELLERATOR_ROOT}/src/compiler/ir/planning/planning_ir_module_and_decision_state_model.cc")
target_include_directories(cellerator_first_inspectable_candidate_search_space_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_first_inspectable_candidate_search_space_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_first_inspectable_candidate_search_space_test PRIVATE -Wall -Wextra -Werror)
endif()
add_executable(ce_ccp1_e01_016 ALIAS cellerator_first_inspectable_candidate_search_space_test)
add_test(NAME ce_ccp1_e01_016 COMMAND cellerator_first_inspectable_candidate_search_space_test)
