cmake_minimum_required(VERSION 3.20)
project(cellerator_compiler_planning_ir_tests LANGUAGES CXX)

enable_testing()
get_filename_component(CELLERATOR_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

add_executable(ce_ccp1_e01_001
    freeze_planning_ir_module_and_decision_state_model_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/planning/freeze_planning_ir_module_and_decision_state_model.cc")
target_include_directories(ce_ccp1_e01_001 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_e01_001 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_e01_001 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_e01_001 COMMAND ce_ccp1_e01_001)

add_executable(ce_ccp1_e01_002
    implement_planning_problems_and_operation_scopes_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/planning/implement_planning_problems_and_operation_scopes.cc")
target_include_directories(ce_ccp1_e01_002 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_e01_002 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_e01_002 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_e01_002 COMMAND ce_ccp1_e01_002)

add_executable(ce_ccp1_e01_003
    implement_candidate_family_and_provider_nodes_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/planning/implement_candidate_family_and_provider_nodes.cc")
target_include_directories(ce_ccp1_e01_003 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_e01_003 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_e01_003 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_e01_003 COMMAND ce_ccp1_e01_003)

add_executable(ce_ccp1_e01_004
    implement_exact_logical_coverage_nodes_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/planning/implement_exact_logical_coverage_nodes.cc")
target_include_directories(ce_ccp1_e01_004 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_e01_004 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_e01_004 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_e01_004 COMMAND ce_ccp1_e01_004)

add_executable(ce_ccp1_e01_005
    implement_atom_requirement_and_affordance_nodes_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/planning/implement_atom_requirement_and_affordance_nodes.cc"
    "${CELLERATOR_ROOT}/src/execution/joint_compiler/persistent_identity_v1.cc"
    "${CELLERATOR_ROOT}/src/execution/joint_compiler/atom_requirement_v1.cc"
    "${CELLERATOR_ROOT}/src/execution/joint_compiler/atom_affordance_v1.cc")
target_include_directories(ce_ccp1_e01_005 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_e01_005 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_e01_005 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_e01_005 COMMAND ce_ccp1_e01_005)

add_executable(ce_ccp1_e01_006
    implement_decomposition_alternative_nodes_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/planning/implement_decomposition_alternative_nodes.cc")
target_include_directories(ce_ccp1_e01_006 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_e01_006 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_e01_006 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_e01_006 COMMAND ce_ccp1_e01_006)

add_executable(ce_ccp1_e01_007
    implement_partial_result_algebra_nodes_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/planning/implement_partial_result_algebra_nodes.cc"
    "${CELLERATOR_ROOT}/src/execution/joint_compiler/persistent_identity_v1.cc"
    "${CELLERATOR_ROOT}/src/execution/joint_compiler/partial_result_algebra_v1.cc")
target_include_directories(ce_ccp1_e01_007 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_e01_007 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_e01_007 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_e01_007 COMMAND ce_ccp1_e01_007)

add_executable(ce_ccp1_e01_008
    implement_persistent_order_projection_and_packing_altern_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/planning/implement_persistent_order_projection_and_packing_altern.cc")
target_include_directories(ce_ccp1_e01_008 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_e01_008 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_e01_008 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_e01_008 COMMAND ce_ccp1_e01_008)
add_executable(ce_ccp1_e01_009 implement_resource_and_stage_inventory_alternatives_test.cc "${CELLERATOR_ROOT}/src/compiler/ir/planning/implement_resource_and_stage_inventory_alternatives.cc")
target_include_directories(ce_ccp1_e01_009 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_e01_009 PRIVATE cxx_std_17)
add_test(NAME ce_ccp1_e01_009 COMMAND ce_ccp1_e01_009)
add_executable(ce_ccp1_e01_010 implement_complete_cost_vectors_test.cc "${CELLERATOR_ROOT}/src/compiler/ir/planning/implement_complete_cost_vectors.cc")
target_include_directories(ce_ccp1_e01_010 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_e01_010 PRIVATE cxx_std_17)
add_test(NAME ce_ccp1_e01_010 COMMAND ce_ccp1_e01_010)
add_executable(ce_ccp1_e01_011 implement_analytical_measured_and_external_evidence_test.cc "${CELLERATOR_ROOT}/src/compiler/ir/planning/implement_analytical_measured_and_external_evidence.cc")
target_include_directories(ce_ccp1_e01_011 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_e01_011 PRIVATE cxx_std_17)
add_test(NAME ce_ccp1_e01_011 COMMAND ce_ccp1_e01_011)
add_executable(ce_ccp1_e01_012 implement_rejection_and_dominance_explanations_test.cc "${CELLERATOR_ROOT}/src/compiler/ir/planning/implement_rejection_and_dominance_explanations.cc")
target_include_directories(ce_ccp1_e01_012 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_e01_012 PRIVATE cxx_std_17)
add_test(NAME ce_ccp1_e01_012 COMMAND ce_ccp1_e01_012)
add_executable(ce_ccp1_e01_013 implement_user_edits_and_authority_hierarchy_test.cc "${CELLERATOR_ROOT}/src/compiler/ir/planning/implement_user_edits_and_authority_hierarchy.cc")
target_include_directories(ce_ccp1_e01_013 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_e01_013 PRIVATE cxx_std_17)
add_test(NAME ce_ccp1_e01_013 COMMAND ce_ccp1_e01_013)
add_executable(ce_ccp1_e01_014 implement_planning_ir_parser_printer_and_validator_test.cc "${CELLERATOR_ROOT}/src/compiler/ir/planning/implement_planning_ir_parser_printer_and_validator.cc")
target_include_directories(ce_ccp1_e01_014 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_e01_014 PRIVATE cxx_std_17)
add_test(NAME ce_ccp1_e01_014 COMMAND ce_ccp1_e01_014)

add_executable(ce_ccp1_e01_015
    implement_semantic_to_planning_lowering_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/planning/implement_semantic_to_planning_lowering.cc"
    "${CELLERATOR_ROOT}/src/compiler/ir/planning/implement_planning_problems_and_operation_scopes.cc"
    "${CELLERATOR_ROOT}/src/compiler/ir/semantic/implement_state_and_value_plane_ir_types.cc")
target_include_directories(ce_ccp1_e01_015 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_e01_015 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_e01_015 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_e01_015 COMMAND ce_ccp1_e01_015)
