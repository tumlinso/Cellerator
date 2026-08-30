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

cellerator_add_relation_algebra_test(ceGeoRelationCatalogIntegrationTest
    tests/relation_algebra/catalog_integration_test.cu)
cellerator_add_relation_algebra_test(ceGeoRelationContractTest
    tests/relation_algebra/contract_test.cc)
cellerator_add_relation_algebra_test(ceGeoEdgeMapOrGateTest
    tests/relation_algebra/edge_map_or_gate_test.cu)
cellerator_add_relation_algebra_test(ceGeoRelationInterfaceTest
    tests/relation_algebra/interface_test.cu)
cellerator_add_relation_algebra_test(ceGeoRelationCompatibilityTest
    tests/relation_algebra/relation_apply_compatibility_test.cu)
cellerator_add_relation_algebra_test(ceGeoRelationBundleTest
    tests/relation_algebra/relation_bundle_test.cu)
cellerator_add_relation_algebra_test(ceGeoSegmentNormalizeTest
    tests/relation_algebra/segment_normalize_test.cu)
cellerator_add_relation_algebra_test(ceGeoSegmentReduceTest
    tests/relation_algebra/segment_reduce_test.cu)

cellerator_add_relation_algebra_test(ceGeoSm70AdvancedOpsTest
    tests/tensor_core/sm70/advanced_operations_test.cu)
cellerator_add_relation_algebra_test(ceGeoSm70EdgeValueGradientTest
    tests/tensor_core/sm70/edge_value_gradient_test.cu)
cellerator_add_relation_algebra_test(ceGeoSm70SegmentBackwardTest
    tests/tensor_core/sm70/segment_backward_integration_test.cu)
cellerator_add_relation_algebra_test(ceGeoSm70CoverNormalizeTest
    tests/tensor_core/sm70/cover_native_normalize_test.cu)
cellerator_add_relation_algebra_test(ceGeoSm70TransposeApplyTest
    tests/tensor_core/sm70/transpose_relation_apply_test.cu)

add_custom_target(ceGeoBiologyTests DEPENDS
    ceGeoRelationCatalogIntegrationTest
    ceGeoRelationContractTest
    ceGeoEdgeMapOrGateTest
    ceGeoRelationInterfaceTest
    ceGeoRelationCompatibilityTest
    ceGeoRelationBundleTest
    ceGeoSegmentNormalizeTest
    ceGeoSegmentReduceTest
    ceGeoSm70AdvancedOpsTest
    ceGeoSm70EdgeValueGradientTest
    ceGeoSm70SegmentBackwardTest
    ceGeoSm70CoverNormalizeTest
    ceGeoSm70TransposeApplyTest
    ceGeoBiologicalRelationsExample
)
