include(tests/planner_targets.cmake)

add_executable(celleratorIdentityRegistryTest
    tests/execution_identity/identity_registry_test.cc
)
target_link_libraries(celleratorIdentityRegistryTest PRIVATE
    Cellerator::identity_registry
)

add_executable(celleratorOperationCoreTest
    tests/math_core/operation_core_test.cc
)
target_link_libraries(celleratorOperationCoreTest PRIVATE
    Cellerator::operation_core
)
set_target_properties(celleratorOperationCoreTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)

# CE-ARCH-60: keep the recovered execution substrate and its concurrency
# contract in the normal build graph. This is the only supported math runtime;
# the former DeviceMathContext island has been retired.
add_executable(celleratorExecutionSessionTest
    tests/runtime/execution_session_test.cu
)
target_link_libraries(celleratorExecutionSessionTest PRIVATE
    Cellerator::runtime
    CUDA::cudart
    CUDA::cublas
    CUDA::cusparse
)
set_target_properties(celleratorExecutionSessionTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)

# CE-LIVE-19: keep the runtime-only value-generation readiness contract in the
# native build and validate same-stream/cross-stream ordering without making
# event state part of persistent biological identity.
add_executable(celleratorValueReadinessTest
    tests/runtime/value_readiness_test.cu
)
target_link_libraries(celleratorValueReadinessTest PRIVATE
    Cellerator::runtime
    CUDA::cudart
)
set_target_properties(celleratorValueReadinessTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)

add_executable(quantizedMatrixTest
    tests/quantized_matrix_test.cu
)
target_include_directories(quantizedMatrixTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(quantizedMatrixTest PRIVATE CUDA::cudart)
set_target_properties(quantizedMatrixTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(quantizedMatrixTest)

add_executable(cellshardAccessAdapterCompileTest
    tests/cellshard_access_adapter_compile_test.cc
)
target_include_directories(cellshardAccessAdapterCompileTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cellshardAccessAdapterCompileTest PRIVATE Cellerator::runtime CellShard::headers)
set_target_properties(cellshardAccessAdapterCompileTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
cellerator_enable_perf(cellshardAccessAdapterCompileTest)

add_executable(trajectoryCompileTest
    tests/trajectory_compile_test.cu
)
target_include_directories(trajectoryCompileTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(trajectoryCompileTest PRIVATE CUDA::cudart)
set_target_properties(trajectoryCompileTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(trajectoryCompileTest)

add_executable(trajectoryRuntimeTest
    tests/trajectory_runtime_test.cu
)
target_include_directories(trajectoryRuntimeTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(trajectoryRuntimeTest PRIVATE CUDA::cudart)
set_target_properties(trajectoryRuntimeTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(trajectoryRuntimeTest)

add_executable(exactSearchRuntimeTest
    tests/exact_search_runtime_test.cu
)
target_include_directories(exactSearchRuntimeTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(exactSearchRuntimeTest PRIVATE cellerator_compute_exact_search CUDA::cudart)
set_target_properties(exactSearchRuntimeTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(exactSearchRuntimeTest)

add_executable(sparseOpsRuntimeTest
    tests/sparse_ops_runtime_test.cu
)
target_include_directories(sparseOpsRuntimeTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(sparseOpsRuntimeTest PRIVATE cellerator_compute_sparse_project cellerator_compute_sparse_ops Cellerator::runtime_fleet CellShard::runtime CUDA::cudart CUDA::cusparse)
set_target_properties(sparseOpsRuntimeTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(sparseOpsRuntimeTest)

add_executable(coreSparseLayoutRuntimeTest
    tests/core_sparse_layout_runtime_test.cu
)
target_include_directories(coreSparseLayoutRuntimeTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(coreSparseLayoutRuntimeTest PRIVATE cellerator_compute_matrix_convert Baseplane::seq CUDA::cudart CUDA::cusparse)
set_target_properties(coreSparseLayoutRuntimeTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(coreSparseLayoutRuntimeTest)

add_executable(datasetRuntimeTest
    tests/dataset_runtime_test.cc
)
target_include_directories(datasetRuntimeTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(datasetRuntimeTest PRIVATE cellerator_compute_dataset CUDA::cudart)
set_target_properties(datasetRuntimeTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
cellerator_enable_perf(datasetRuntimeTest)

add_executable(samplingRuntimeTest
    tests/sampling_runtime_test.cc
)
target_include_directories(samplingRuntimeTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(samplingRuntimeTest PRIVATE cellerator_compute_sampling cellerator_compute_dataset CUDA::cudart)
set_target_properties(samplingRuntimeTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
cellerator_enable_perf(samplingRuntimeTest)

add_executable(samplingMaterializationRuntimeTest
    tests/sampling_materialization_runtime_test.cc
)
target_include_directories(samplingMaterializationRuntimeTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(samplingMaterializationRuntimeTest PRIVATE cellerator_compute_sampling CUDA::cudart)
set_target_properties(samplingMaterializationRuntimeTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
cellerator_enable_perf(samplingMaterializationRuntimeTest)

add_executable(geneSupportBitsetRuntimeTest
    tests/gene_support_bitset_runtime_test.cu
)
target_include_directories(geneSupportBitsetRuntimeTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(geneSupportBitsetRuntimeTest PRIVATE cellerator_compute_gene_support CUDA::cudart)
set_target_properties(geneSupportBitsetRuntimeTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
cellerator_enable_perf(geneSupportBitsetRuntimeTest)

add_executable(geneCandidateDiscoveryRuntimeTest
    tests/gene_candidate_discovery_runtime_test.cu
)
target_include_directories(geneCandidateDiscoveryRuntimeTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(geneCandidateDiscoveryRuntimeTest PRIVATE
    cellerator_compute_gene_candidates
    CUDA::cudart
)
set_target_properties(geneCandidateDiscoveryRuntimeTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
cellerator_enable_perf(geneCandidateDiscoveryRuntimeTest)

add_executable(celleratorPreprocessRuntimeTest
    tests/preprocess/preprocess_runtime_test.cu
)
target_include_directories(celleratorPreprocessRuntimeTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(celleratorPreprocessRuntimeTest PRIVATE Cellerator::preprocess CUDA::cudart)
set_target_properties(celleratorPreprocessRuntimeTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(celleratorPreprocessRuntimeTest)

add_executable(celleratorPreprocessQCMetricsEquivalenceTest
    tests/preprocess/qc_metrics_equivalence_test.cu
)
target_include_directories(celleratorPreprocessQCMetricsEquivalenceTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(celleratorPreprocessQCMetricsEquivalenceTest PRIVATE Cellerator::preprocess CUDA::cudart)
set_target_properties(celleratorPreprocessQCMetricsEquivalenceTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(celleratorPreprocessQCMetricsEquivalenceTest)

add_executable(celleratorPreprocessQCMaskGroupsTest
    tests/preprocess/qc_mask_groups_test.cu
)
target_include_directories(celleratorPreprocessQCMaskGroupsTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(celleratorPreprocessQCMaskGroupsTest PRIVATE Cellerator::preprocess CUDA::cudart)
set_target_properties(celleratorPreprocessQCMaskGroupsTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(celleratorPreprocessQCMaskGroupsTest)

add_executable(celleratorPreprocessQCFleetTest
    tests/preprocess/qc_fleet_test.cu
)
target_include_directories(celleratorPreprocessQCFleetTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(celleratorPreprocessQCFleetTest PRIVATE Cellerator::preprocess CUDA::cudart)
set_target_properties(celleratorPreprocessQCFleetTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(celleratorPreprocessQCFleetTest)

add_executable(celleratorPreprocessCellShardSessionApiTest
    tests/preprocess/cellshard_session_api_test.cu
)
target_include_directories(celleratorPreprocessCellShardSessionApiTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(celleratorPreprocessCellShardSessionApiTest PRIVATE Cellerator::preprocess CUDA::cudart)
set_target_properties(celleratorPreprocessCellShardSessionApiTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(celleratorPreprocessCellShardSessionApiTest)

add_executable(celleratorPreprocessAdapterStagingTest
    tests/preprocess/adapter_staging_test.cc
)
target_include_directories(celleratorPreprocessAdapterStagingTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(celleratorPreprocessAdapterStagingTest PRIVATE Cellerator::preprocess)
set_target_properties(celleratorPreprocessAdapterStagingTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
cellerator_enable_perf(celleratorPreprocessAdapterStagingTest)

add_executable(celleratorPreprocessDoublePreprocessRejectionTest
    tests/preprocess/double_preprocess_rejection_test.cc
)
target_include_directories(celleratorPreprocessDoublePreprocessRejectionTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(celleratorPreprocessDoublePreprocessRejectionTest PRIVATE Cellerator::preprocess)
set_target_properties(celleratorPreprocessDoublePreprocessRejectionTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
cellerator_enable_perf(celleratorPreprocessDoublePreprocessRejectionTest)

add_executable(abiRuntimeTest
    tests/abi_runtime_test.cu
)
target_include_directories(abiRuntimeTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(abiRuntimeTest PRIVATE cellerator_abi CUDA::cudart)
set_target_properties(abiRuntimeTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(abiRuntimeTest)

add_executable(celleratorBiologicalAbiHostTest
    tests/biological_abi/biological_abi_host_test.cc
)
target_link_libraries(celleratorBiologicalAbiHostTest PRIVATE
    Cellerator::biological_abi
    Baseplane::seq
)
set_target_properties(celleratorBiologicalAbiHostTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)

add_executable(celleratorBiologicalAbiCudaCompileTest
    tests/biological_abi/biological_abi_cuda_compile_test.cu
)
target_link_libraries(celleratorBiologicalAbiCudaCompileTest PRIVATE
    Cellerator::biological_abi
    CUDA::cudart
)
set_target_properties(celleratorBiologicalAbiCudaCompileTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)

# CE-ARCH-40: Cellerator-owned preparation and execution over frozen Baseplane
# predicate contracts. Baseplane remains independently buildable and owns no
# Cellerator runtime or numerical policy.
add_executable(celleratorBaseplaneSequenceIntegrationTest
    tests/interop/baseplane/baseplane_integration_test.cu
)
target_link_libraries(celleratorBaseplaneSequenceIntegrationTest PRIVATE
    Cellerator::sequence_integration
)
set_target_properties(celleratorBaseplaneSequenceIntegrationTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
cellerator_enable_perf(celleratorBaseplaneSequenceIntegrationTest)

add_executable(stateReduceRuntimeTest
    tests/state_reduce_runtime_test.cu
)
target_include_directories(stateReduceRuntimeTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(stateReduceRuntimeTest PRIVATE cellerator_state_reduce_cuda cellerator_compute_sparse_project cellerator_compute_sparse_ops Cellerator::runtime_fleet CellShard::runtime CUDA::cudart CUDA::cusparse CUDA::cublas)
set_target_properties(stateReduceRuntimeTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(stateReduceRuntimeTest)

add_executable(developmentalTimeCompileTest
    tests/developmental_time_compile_test.cc
)
target_include_directories(developmentalTimeCompileTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(developmentalTimeCompileTest PRIVATE CellShard::runtime CUDA::cudart)
set_target_properties(developmentalTimeCompileTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
cellerator_enable_perf(developmentalTimeCompileTest)

add_executable(developmentalTimeRuntimeTest
    tests/developmental_time_runtime_test.cu
)
target_include_directories(developmentalTimeRuntimeTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(developmentalTimeRuntimeTest PRIVATE cellerator_developmental_time_cuda cellerator_compute_sparse_project cellerator_compute_sparse_ops Cellerator::runtime_fleet CellShard::runtime CUDA::cudart CUDA::cusparse)
set_target_properties(developmentalTimeRuntimeTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(developmentalTimeRuntimeTest)

add_executable(developmentalTimeTrajectoryRuntimeTest
    tests/developmental_time_trajectory_runtime_test.cu
)
target_include_directories(developmentalTimeTrajectoryRuntimeTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(developmentalTimeTrajectoryRuntimeTest PRIVATE cellerator_developmental_time_trajectory_cuda cellerator_developmental_time_cuda cellerator_compute_sparse_project cellerator_compute_sparse_ops Cellerator::runtime_fleet CellShard::runtime CUDA::cudart CUDA::cusparse)
set_target_properties(developmentalTimeTrajectoryRuntimeTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(developmentalTimeTrajectoryRuntimeTest)
