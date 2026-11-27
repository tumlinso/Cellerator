add_library(cellerator_operation_decomposition STATIC
    src/execution/planning/decomposition/dense_width_v1.cc
    src/execution/planning/decomposition/destination_disjoint_v1.cc
    src/execution/planning/decomposition/edge_component_v1.cc
    src/ops/reduction/log_sum_exp_state_v1.cc
    src/ops/reduction/moments_state_v1.cc
    src/execution/planning/decomposition/provider_registry_v1.cc
    src/execution/planning/decomposition/relation_bundle_v1.cc
    src/execution/planning/decomposition/segment_disjoint_v1.cc
    src/execution/planning/decomposition/source_k_v1.cc
    src/execution/planning/decomposition/sparse_update_conflict_v1.cc
    src/execution/planning/decomposition/split_segment_reduce_v1.cc
    src/execution/planning/decomposition/support_contraction_v1.cc
    src/execution/planning/decomposition/support_edge_rectangle_v1.cc
    src/execution/planning/decomposition/support_embedding_v1.cc
    src/execution/planning/decomposition/training_gradient_v1.cc
    src/execution/planning/decomposition/transpose_source_partials_v1.cc
)
add_library(Cellerator::operation_decomposition ALIAS cellerator_operation_decomposition)
target_include_directories(cellerator_operation_decomposition PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_features(cellerator_operation_decomposition PUBLIC cxx_std_20)
