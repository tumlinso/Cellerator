add_library(cellerator_operation_decomposition STATIC
    src/compute/decomposition/dense_width_v1.cc
    src/compute/decomposition/destination_disjoint_v1.cc
    src/compute/decomposition/edge_component_v1.cc
    src/compute/decomposition/log_sum_exp_state_v1.cc
    src/compute/decomposition/moments_state_v1.cc
    src/compute/decomposition/provider_registry_v1.cc
    src/compute/decomposition/relation_bundle_v1.cc
    src/compute/decomposition/segment_disjoint_v1.cc
    src/compute/decomposition/source_k_v1.cc
    src/compute/decomposition/sparse_update_conflict_v1.cc
    src/compute/decomposition/split_segment_reduce_v1.cc
    src/compute/decomposition/support_contraction_v1.cc
    src/compute/decomposition/support_edge_rectangle_v1.cc
    src/compute/decomposition/support_embedding_v1.cc
    src/compute/decomposition/training_gradient_v1.cc
    src/compute/decomposition/transpose_source_partials_v1.cc
)
add_library(Cellerator::operation_decomposition ALIAS cellerator_operation_decomposition)
target_include_directories(cellerator_operation_decomposition PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_features(cellerator_operation_decomposition PUBLIC cxx_std_20)
