cmake_minimum_required(VERSION 3.20)
project(cellerator_compiler_semantic_ir_tests LANGUAGES CXX)

enable_testing()

get_filename_component(CELLERATOR_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

add_executable(cellerator_semantic_ir_module_and_symbol_scopes_test
    semantic_ir_module_and_symbol_scopes_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/semantic/semantic_ir_module_and_symbol_scopes.cc")
target_include_directories(cellerator_semantic_ir_module_and_symbol_scopes_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_semantic_ir_module_and_symbol_scopes_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_semantic_ir_module_and_symbol_scopes_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_semantic_ir_module_and_symbol_scopes_test COMMAND cellerator_semantic_ir_module_and_symbol_scopes_test)

add_executable(cellerator_domain_and_axis_ir_types_test
    domain_and_axis_ir_types_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/semantic/domain_and_axis_ir_types.cc")
target_include_directories(cellerator_domain_and_axis_ir_types_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_domain_and_axis_ir_types_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_domain_and_axis_ir_types_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_domain_and_axis_ir_types_test COMMAND cellerator_domain_and_axis_ir_types_test)

add_executable(cellerator_state_and_value_plane_ir_types_test
    state_and_value_plane_ir_types_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/semantic/state_and_value_plane_ir_types.cc")
target_include_directories(cellerator_state_and_value_plane_ir_types_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_state_and_value_plane_ir_types_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_state_and_value_plane_ir_types_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_state_and_value_plane_ir_types_test COMMAND cellerator_state_and_value_plane_ir_types_test)

add_executable(cellerator_relation_ir_types_test
    relation_ir_types_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/semantic/relation_ir_types.cc"
    "${CELLERATOR_ROOT}/src/compiler/ir/semantic/domain_and_axis_ir_types.cc")
target_include_directories(cellerator_relation_ir_types_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_relation_ir_types_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_relation_ir_types_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_relation_ir_types_test COMMAND cellerator_relation_ir_types_test)

add_executable(cellerator_execution_field_regions_test
    execution_field_regions_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/semantic/execution_field_regions.cc"
    "${CELLERATOR_ROOT}/src/compiler/ir/semantic/semantic_ir_module_and_symbol_scopes.cc")
target_include_directories(cellerator_execution_field_regions_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_execution_field_regions_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_execution_field_regions_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_execution_field_regions_test COMMAND cellerator_execution_field_regions_test)

add_executable(cellerator_relation_apply_and_transpose_operations_test
    relation_apply_and_transpose_operations_test.cc
    "${CELLERATOR_ROOT}/src/compute/operation/relation_semantics.cc"
    "${CELLERATOR_ROOT}/src/compiler/ir/semantic/relation_apply_and_transpose_operations.cc"
    "${CELLERATOR_ROOT}/src/compiler/ir/semantic/relation_ir_types.cc"
    "${CELLERATOR_ROOT}/src/compiler/ir/semantic/state_and_value_plane_ir_types.cc"
    "${CELLERATOR_ROOT}/src/compiler/ir/semantic/domain_and_axis_ir_types.cc")
target_include_directories(cellerator_relation_apply_and_transpose_operations_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_relation_apply_and_transpose_operations_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_relation_apply_and_transpose_operations_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_relation_apply_and_transpose_operations_test COMMAND cellerator_relation_apply_and_transpose_operations_test)

add_executable(cellerator_contraction_segment_normalization_test
    contraction_segment_normalization_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/semantic/contraction_segment_normalization.cc")
target_include_directories(cellerator_contraction_segment_normalization_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_contraction_segment_normalization_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_contraction_segment_normalization_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_contraction_segment_normalization_test COMMAND cellerator_contraction_segment_normalization_test)

add_executable(cellerator_edge_map_gate_support_mask_and_sparse_update_o_test
    edge_map_gate_support_mask_and_sparse_update_o_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/semantic/edge_map_gate_support_mask_and_sparse_update_o.cc"
    "${CELLERATOR_ROOT}/src/compiler/ir/semantic/domain_and_axis_ir_types.cc")
target_include_directories(cellerator_edge_map_gate_support_mask_and_sparse_update_o_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_edge_map_gate_support_mask_and_sparse_update_o_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_edge_map_gate_support_mask_and_sparse_update_o_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_edge_map_gate_support_mask_and_sparse_update_o_test COMMAND cellerator_edge_map_gate_support_mask_and_sparse_update_o_test)

add_executable(cellerator_bundle_chain_moments_hierarchy_and_exchange_op_test
    bundle_chain_moments_hierarchy_and_exchange_op_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/semantic/bundle_chain_moments_hierarchy_and_exchange_op.cc")
target_include_directories(cellerator_bundle_chain_moments_hierarchy_and_exchange_op_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_bundle_chain_moments_hierarchy_and_exchange_op_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_bundle_chain_moments_hierarchy_and_exchange_op_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_bundle_chain_moments_hierarchy_and_exchange_op_test COMMAND cellerator_bundle_chain_moments_hierarchy_and_exchange_op_test)

add_executable(cellerator_gradient_and_publication_operations_test
    gradient_and_publication_operations_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/semantic/gradient_and_publication_operations.cc"
    "${CELLERATOR_ROOT}/src/compute/operation/relation_calculus.cc"
    "${CELLERATOR_ROOT}/src/compute/operation/relation_semantics.cc")
target_include_directories(cellerator_gradient_and_publication_operations_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_gradient_and_publication_operations_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_gradient_and_publication_operations_test PRIVATE -UNDEBUG -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_gradient_and_publication_operations_test COMMAND cellerator_gradient_and_publication_operations_test)

add_executable(cellerator_native_and_opaque_c_call_operations_test
    native_and_opaque_c_call_operations_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/semantic/native_and_opaque_c_call_operations.cc")
target_include_directories(cellerator_native_and_opaque_c_call_operations_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_native_and_opaque_c_call_operations_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_native_and_opaque_c_call_operations_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_native_and_opaque_c_call_operations_test COMMAND cellerator_native_and_opaque_c_call_operations_test)

add_executable(cellerator_control_flow_and_loop_semantics_test
    control_flow_and_loop_semantics_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/semantic/control_flow_and_loop_semantics.cc")
target_include_directories(cellerator_control_flow_and_loop_semantics_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_control_flow_and_loop_semantics_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_control_flow_and_loop_semantics_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_control_flow_and_loop_semantics_test COMMAND cellerator_control_flow_and_loop_semantics_test)

add_executable(cellerator_generation_and_epoch_transition_operations_test
    generation_and_epoch_transition_operations_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/semantic/generation_and_epoch_transition_operations.cc")
target_include_directories(cellerator_generation_and_epoch_transition_operations_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_generation_and_epoch_transition_operations_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_generation_and_epoch_transition_operations_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_generation_and_epoch_transition_operations_test COMMAND cellerator_generation_and_epoch_transition_operations_test)

add_executable(cellerator_semantic_ir_inlining_and_composition_test
    semantic_ir_inlining_and_composition_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/semantic/semantic_ir_inlining_and_composition.cc")
target_include_directories(cellerator_semantic_ir_inlining_and_composition_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_semantic_ir_inlining_and_composition_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_semantic_ir_inlining_and_composition_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_semantic_ir_inlining_and_composition_test COMMAND cellerator_semantic_ir_inlining_and_composition_test)

add_executable(cellerator_semantic_canonicalization_and_equivalence_test
    semantic_canonicalization_and_equivalence_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/semantic/semantic_canonicalization_and_equivalence.cc"
    "${CELLERATOR_ROOT}/src/compiler/ir/semantic/state_and_value_plane_ir_types.cc")
target_include_directories(cellerator_semantic_canonicalization_and_equivalence_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_semantic_canonicalization_and_equivalence_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_semantic_canonicalization_and_equivalence_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_semantic_canonicalization_and_equivalence_test COMMAND cellerator_semantic_canonicalization_and_equivalence_test)

add_executable(cellerator_source_to_semantic_ir_vertical_slice_test
    source_to_semantic_ir_vertical_slice_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ir/semantic/source_to_semantic_ir_vertical_slice.cc")
target_include_directories(cellerator_source_to_semantic_ir_vertical_slice_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_source_to_semantic_ir_vertical_slice_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_source_to_semantic_ir_vertical_slice_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_source_to_semantic_ir_vertical_slice_test COMMAND cellerator_source_to_semantic_ir_vertical_slice_test)
