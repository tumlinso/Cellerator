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

add_executable(ce_ccp1_d02_006
    implement_relation_apply_and_transpose_operations_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/semantic/implement_relation_apply_and_transpose_operations.cc"
    "${CELLERATOR_ROOT}/src/compiler/ir/semantic/implement_relation_ir_types.cc"
    "${CELLERATOR_ROOT}/src/compiler/ir/semantic/implement_state_and_value_plane_ir_types.cc"
    "${CELLERATOR_ROOT}/src/compiler/ir/semantic/implement_domain_and_axis_ir_types.cc")
target_include_directories(ce_ccp1_d02_006 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_d02_006 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_d02_006 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_d02_006 COMMAND ce_ccp1_d02_006)

add_executable(ce_ccp1_d02_007
    implement_contraction_segment_and_normalization_operatio_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/semantic/implement_contraction_segment_and_normalization_operatio.cc")
target_include_directories(ce_ccp1_d02_007 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_d02_007 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_d02_007 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_d02_007 COMMAND ce_ccp1_d02_007)

add_executable(ce_ccp1_d02_008
    implement_edge_map_gate_support_mask_and_sparse_update_o_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/semantic/implement_edge_map_gate_support_mask_and_sparse_update_o.cc"
    "${CELLERATOR_ROOT}/src/compiler/ir/semantic/implement_domain_and_axis_ir_types.cc")
target_include_directories(ce_ccp1_d02_008 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_d02_008 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_d02_008 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_d02_008 COMMAND ce_ccp1_d02_008)

add_executable(ce_ccp1_d02_009
    implement_bundle_chain_moments_hierarchy_and_exchange_op_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/semantic/implement_bundle_chain_moments_hierarchy_and_exchange_op.cc")
target_include_directories(ce_ccp1_d02_009 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_d02_009 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_d02_009 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_d02_009 COMMAND ce_ccp1_d02_009)

add_executable(ce_ccp1_d02_010
    implement_gradient_and_publication_operations_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/semantic/implement_gradient_and_publication_operations.cc"
    "${CELLERATOR_ROOT}/src/compiler/ir/semantic/implement_state_and_value_plane_ir_types.cc")
target_include_directories(ce_ccp1_d02_010 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_d02_010 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_d02_010 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_d02_010 COMMAND ce_ccp1_d02_010)

add_executable(ce_ccp1_d02_011
    implement_native_and_opaque_c_call_operations_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/semantic/implement_native_and_opaque_c_call_operations.cc")
target_include_directories(ce_ccp1_d02_011 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_d02_011 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_d02_011 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_d02_011 COMMAND ce_ccp1_d02_011)

add_executable(ce_ccp1_d02_012
    implement_control_flow_and_loop_semantics_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/semantic/implement_control_flow_and_loop_semantics.cc")
target_include_directories(ce_ccp1_d02_012 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_d02_012 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_d02_012 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_d02_012 COMMAND ce_ccp1_d02_012)

add_executable(ce_ccp1_d02_013
    implement_generation_and_epoch_transition_operations_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/semantic/implement_generation_and_epoch_transition_operations.cc")
target_include_directories(ce_ccp1_d02_013 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_d02_013 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_d02_013 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_d02_013 COMMAND ce_ccp1_d02_013)

add_executable(ce_ccp1_d02_014
    implement_semantic_ir_inlining_and_composition_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/semantic/implement_semantic_ir_inlining_and_composition.cc")
target_include_directories(ce_ccp1_d02_014 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_d02_014 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_d02_014 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_d02_014 COMMAND ce_ccp1_d02_014)

add_executable(ce_ccp1_d02_015
    implement_semantic_canonicalization_and_equivalence_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/semantic/implement_semantic_canonicalization_and_equivalence.cc"
    "${CELLERATOR_ROOT}/src/compiler/ir/semantic/implement_state_and_value_plane_ir_types.cc")
target_include_directories(ce_ccp1_d02_015 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_d02_015 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_d02_015 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_d02_015 COMMAND ce_ccp1_d02_015)
