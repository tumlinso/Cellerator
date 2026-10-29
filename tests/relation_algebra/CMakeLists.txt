function(cellerator_add_relation_algebra_test target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE
        Cellerator::relation_algebra
        CUDA::cudart
    )
    target_compile_options(${target} PRIVATE -UNDEBUG)
    set_target_properties(${target} PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED YES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
endfunction()

cellerator_add_relation_algebra_test(relation_catalog_integration_test
    tests/relation_algebra/catalog_integration_test.cu)
cellerator_add_relation_algebra_test(relation_contract_test
    tests/relation_algebra/contract_test.cc)
cellerator_add_relation_algebra_test(edge_map_or_gate_test
    tests/relation_algebra/edge_map_or_gate_test.cu)
cellerator_add_relation_algebra_test(relation_interface_test
    tests/relation_algebra/interface_test.cu)
cellerator_add_relation_algebra_test(relation_compatibility_test
    tests/relation_algebra/relation_apply_compatibility_test.cu)
cellerator_add_relation_algebra_test(relation_bundle_test
    tests/relation_algebra/relation_bundle_test.cu)
cellerator_add_relation_algebra_test(segment_normalize_test
    tests/relation_algebra/segment_normalize_test.cu)
cellerator_add_relation_algebra_test(segment_reduce_test
    tests/relation_algebra/segment_reduce_test.cu)

cellerator_add_relation_algebra_test(geometry_sm70_advanced_ops_test
    tests/tensor_core/sm70/advanced_operations_test.cu)
cellerator_add_relation_algebra_test(geometry_sm70_edge_value_gradient_test
    tests/tensor_core/sm70/edge_value_gradient_test.cu)
cellerator_add_relation_algebra_test(geometry_sm70_segment_backward_test
    tests/tensor_core/sm70/segment_backward_integration_test.cu)
cellerator_add_relation_algebra_test(geometry_sm70_cover_normalize_test
    tests/tensor_core/sm70/cover_native_normalize_test.cu)
cellerator_add_relation_algebra_test(geometry_sm70_transpose_apply_test
    tests/tensor_core/sm70/transpose_relation_apply_test.cu)

add_custom_target(biological_relation_tests DEPENDS
    relation_catalog_integration_test
    relation_contract_test
    edge_map_or_gate_test
    relation_interface_test
    relation_compatibility_test
    relation_bundle_test
    segment_normalize_test
    segment_reduce_test
    geometry_sm70_advanced_ops_test
    geometry_sm70_edge_value_gradient_test
    geometry_sm70_segment_backward_test
    geometry_sm70_cover_normalize_test
    geometry_sm70_transpose_apply_test
    biological_relations_example
)
