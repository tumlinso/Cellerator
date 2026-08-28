add_library(cellerator_planner STATIC
    src/planner/end_to_end_planner.cc
    src/planner/distributed/hierarchy.cc
    src/geometry/optimization/objective_v2.cc
    src/planner/objective_v2_calibration.cc
)
add_library(Cellerator::planner ALIAS cellerator_planner)
target_include_directories(cellerator_planner PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellerator_planner PUBLIC
    Cellerator::operation_core
    CellPack::semantic_geometry
    CellPack::alternating_refinement
)
target_compile_features(cellerator_planner PUBLIC cxx_std_17)

add_library(cellerator_live_planner_inputs STATIC
    bench/ce_live/planner/live_planner_inputs.cc
)
target_include_directories(cellerator_live_planner_inputs PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(cellerator_live_planner_inputs PUBLIC
    Cellerator::planner
)
target_compile_features(cellerator_live_planner_inputs PUBLIC cxx_std_17)

add_library(cellerator_live_runtime_fixture STATIC
    bench/ce_live/runtime_fixture/quantitative_fixture.cc
)
target_include_directories(cellerator_live_runtime_fixture PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(cellerator_live_runtime_fixture PUBLIC
    Cellerator::identity_registry
)
target_compile_features(cellerator_live_runtime_fixture PUBLIC cxx_std_17)

add_library(cellerator_candidate_measurement STATIC
    src/planner/candidate_measurement.cu
)
add_library(Cellerator::candidate_measurement ALIAS
    cellerator_candidate_measurement)
target_include_directories(cellerator_candidate_measurement PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellerator_candidate_measurement PUBLIC
    Cellerator::planner
    Cellerator::operation_core
    CUDA::cudart
)
set_target_properties(cellerator_candidate_measurement PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
add_executable(celleratorPlannerV1Test
    tests/planner/end_to_end_planner_test.cc
)
add_executable(celleratorConnectedOperationPlannerTest
    tests/planner/connected_operation_planner_test.cc
)
add_executable(celleratorObjectiveV2CalibrationTest
    tests/planner/objective_v2_calibration_test.cc
)
add_executable(celleratorHierarchyPartitionBoundaryTest
    tests/distributed/hierarchy_partition_boundary_test.cc
)
add_custom_target(celleratorDistributedHierarchyTest
    DEPENDS celleratorHierarchyPartitionBoundaryTest
)

add_executable(celleratorCandidateMeasurementTest
    tests/planner/candidate_measurement_test.cu
)
add_executable(celleratorProjectionActivationTest
    tests/execution/projection_activation_test.cu
)
add_executable(celleratorBuiltinCatalogTest
    tests/math_core/builtin_catalog_test.cc
)
add_custom_target(celleratorBuiltinCandidateCatalogTest
    DEPENDS celleratorBuiltinCatalogTest
)
add_executable(celleratorCusparseCsrCandidateTest
    tests/math_core/cusparse_csr_candidate_test.cu
)
add_executable(celleratorPreparationFactoryTest
    tests/math_core/preparation_factory_test.cu
)
add_executable(celleratorQuantitativeRelationTest
    tests/live/quantitative_relation_test.cu
)
add_executable(celleratorLivePlannerFeaturesTest
    tests/planner/ce_live_planner_features_test.cc
)
add_executable(celleratorExecutableCoreIntegrationTest
    tests/live/integration/executable_core_integration_test.cc
)
target_link_libraries(celleratorCandidateMeasurementTest PRIVATE
    Cellerator::candidate_measurement
    Cellerator::row_masked_n1_candidate
    Cellerator::csr_fallback_candidate
)
set_target_properties(celleratorCandidateMeasurementTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
target_link_libraries(celleratorProjectionActivationTest PRIVATE
    Cellerator::projection_activation
    CUDA::cudart
)
target_link_libraries(celleratorBuiltinCatalogTest PRIVATE
    Cellerator::builtin_candidate_catalog
)
target_link_libraries(celleratorCusparseCsrCandidateTest PRIVATE
    Cellerator::cusparse_csr_candidate
)
target_link_libraries(celleratorPreparationFactoryTest PRIVATE
    Cellerator::preparation_factory
    Cellerator::projection_activation
    CUDA::cudart
)
target_link_libraries(celleratorQuantitativeRelationTest PRIVATE
    cellerator_live_runtime_fixture
    CUDA::cudart
)
target_link_libraries(celleratorLivePlannerFeaturesTest PRIVATE
    cellerator_live_planner_inputs
)
target_link_libraries(celleratorExecutableCoreIntegrationTest PRIVATE
    Cellerator::executable_core
    Cellerator::planner
    cellerator_live_planner_inputs
    cellerator_live_runtime_fixture
)
set_target_properties(
    celleratorProjectionActivationTest
    celleratorCusparseCsrCandidateTest
    celleratorPreparationFactoryTest
    celleratorQuantitativeRelationTest
    PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED YES
)
set_target_properties(
    celleratorBuiltinCatalogTest
    celleratorLivePlannerFeaturesTest
    celleratorExecutableCoreIntegrationTest
    PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
)
target_link_libraries(celleratorPlannerV1Test PRIVATE Cellerator::planner)
target_link_libraries(celleratorConnectedOperationPlannerTest PRIVATE
    Cellerator::planner
)
target_link_libraries(celleratorObjectiveV2CalibrationTest PRIVATE
    Cellerator::planner
)
target_link_libraries(celleratorHierarchyPartitionBoundaryTest PRIVATE
    Cellerator::planner
)
set_target_properties(celleratorPlannerV1Test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)
set_target_properties(celleratorConnectedOperationPlannerTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)
set_target_properties(celleratorObjectiveV2CalibrationTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)
set_target_properties(celleratorHierarchyPartitionBoundaryTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)

add_executable(celleratorRowMaskedN1CandidateTest
    tests/math_core/row_masked_n1_candidate_test.cu
)

add_executable(celleratorCsrFallbackCandidateTest
    tests/math_core/csr_fallback_candidate_test.cu
)
add_executable(celleratorFeatureMajorSmallNCandidateTest
    tests/math_core/feature_major_small_n_candidate_test.cu
)
# Repository-remap validation name: the existing candidate test exercises
# construction, validation, rebinding, and execution of this projection.
add_custom_target(celleratorFeatureMajorProjectionTest
    DEPENDS celleratorFeatureMajorSmallNCandidateTest
)
add_executable(celleratorTransposeBackwardCandidateTest
    tests/math_core/transpose_backward_candidate_test.cu
)
add_executable(celleratorNativeTrainingSliceTest
    tests/math_core/native_training_slice_test.cu
)
add_executable(celleratorCeArch76CandidateBench
    bench/math/feature_major_candidate_compare_bench.cu
)
add_executable(celleratorCeArch92RealRegimeBench
    bench/architecture_evidence/real_regime_bench.cu
)
add_executable(celleratorValueGenerationReuseTest
    tests/math_core/value_generation_reuse_test.cu
)
target_link_libraries(celleratorValueGenerationReuseTest PRIVATE
    CUDA::cudart
)
set_target_properties(celleratorValueGenerationReuseTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
target_link_libraries(celleratorCsrFallbackCandidateTest PRIVATE
    Cellerator::csr_fallback_candidate
    Cellerator::row_masked_n1_candidate
    Cellerator::planner
)
set_target_properties(celleratorCsrFallbackCandidateTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
target_link_libraries(celleratorFeatureMajorSmallNCandidateTest PRIVATE
    Cellerator::feature_major_small_n_candidate
    Cellerator::row_masked_n1_candidate
    Cellerator::csr_fallback_candidate
    Cellerator::planner
    CellPack::execution_image_v2
    cellerator_math_v1_evidence
)
set_target_properties(celleratorFeatureMajorSmallNCandidateTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
target_link_libraries(celleratorTransposeBackwardCandidateTest PRIVATE
    Cellerator::transpose_backward_candidate
    Cellerator::feature_major_small_n_candidate
    CellPack::execution_image_v2
    cellerator_math_v1_evidence
)
set_target_properties(celleratorTransposeBackwardCandidateTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
target_link_libraries(celleratorNativeTrainingSliceTest PRIVATE
    Cellerator::native_training_slice
    Cellerator::feature_major_projection
    Cellerator::transpose_projection
    CUDA::cudart
)
set_target_properties(celleratorNativeTrainingSliceTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
target_link_libraries(celleratorCeArch76CandidateBench PRIVATE
    Cellerator::feature_major_small_n_candidate
    Cellerator::row_masked_n1_candidate
    Cellerator::csr_fallback_candidate
    cellerator_math_v1_evidence
)
target_link_libraries(celleratorCeArch92RealRegimeBench PRIVATE
    Cellerator::feature_major_small_n_candidate
    Cellerator::row_masked_n1_candidate
    Cellerator::csr_fallback_candidate
    cellerator_math_v1_evidence
)
set_target_properties(celleratorCeArch76CandidateBench PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
set_target_properties(celleratorCeArch92RealRegimeBench PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
target_link_libraries(celleratorRowMaskedN1CandidateTest PRIVATE
    Cellerator::row_masked_n1_candidate
    Cellerator::planner
)
set_target_properties(celleratorRowMaskedN1CandidateTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
