cmake_minimum_required(VERSION 3.20)
project(cellerator_compiler_semantic_ir_tests LANGUAGES CXX)

enable_testing()

get_filename_component(CELLERATOR_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

add_executable(ce_ccp1_d02_001
    freeze_semantic_ir_module_and_symbol_scopes_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/semantic/freeze_semantic_ir_module_and_symbol_scopes.cc")
target_include_directories(ce_ccp1_d02_001 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_d02_001 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_d02_001 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_d02_001 COMMAND ce_ccp1_d02_001)

add_executable(ce_ccp1_d02_002
    implement_domain_and_axis_ir_types_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/semantic/implement_domain_and_axis_ir_types.cc")
target_include_directories(ce_ccp1_d02_002 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_d02_002 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_d02_002 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_d02_002 COMMAND ce_ccp1_d02_002)

add_executable(ce_ccp1_d02_003
    implement_state_and_value_plane_ir_types_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/semantic/implement_state_and_value_plane_ir_types.cc")
target_include_directories(ce_ccp1_d02_003 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_d02_003 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_d02_003 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_d02_003 COMMAND ce_ccp1_d02_003)

add_executable(ce_ccp1_d02_004
    implement_relation_ir_types_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/semantic/implement_relation_ir_types.cc"
    "${CELLERATOR_ROOT}/src/compiler/ir/semantic/implement_domain_and_axis_ir_types.cc")
target_include_directories(ce_ccp1_d02_004 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_d02_004 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_d02_004 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_d02_004 COMMAND ce_ccp1_d02_004)

add_executable(ce_ccp1_d02_005
    implement_execution_field_operations_and_regions_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/semantic/implement_execution_field_operations_and_regions.cc"
    "${CELLERATOR_ROOT}/src/compiler/ir/semantic/freeze_semantic_ir_module_and_symbol_scopes.cc")
target_include_directories(ce_ccp1_d02_005 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_d02_005 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_d02_005 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_d02_005 COMMAND ce_ccp1_d02_005)
