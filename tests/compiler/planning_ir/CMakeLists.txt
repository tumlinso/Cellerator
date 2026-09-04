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
