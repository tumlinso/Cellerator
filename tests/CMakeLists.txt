include(tests/planner_targets.cmake)
include(tests/tensor_core/sm70/CMakeLists.txt)
include(tests/relation_algebra/CMakeLists.txt)

find_program(CELLERATOR_PYTHON3_EXECUTABLE NAMES python3 REQUIRED)
add_custom_target(celleratorNoProductionFileIoCheck ALL
    COMMAND ${CELLERATOR_PYTHON3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_no_production_file_io.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Checking that Cellerator production sources do not own files"
    VERBATIM
)

# CE-GEO-120 foundation reconciliation. These targets exercise the frozen
# portable provider/catalog/CSG1/CPE2/acquisition contracts without launching
# benchmark or empirical-promotion work. The companion runner executes this
# concrete list directly because Cellerator does not use CTest as its primary
# test runner.
function(cellerator_add_ce_geo_foundation_test target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE ${ARGN})
    set_target_properties(${target} PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED YES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
endfunction()

cellerator_add_ce_geo_foundation_test(ceGeoCapabilityTest
    tests/ce_geo/hardware/capability_test.cc
    Cellerator::architecture_provider)
cellerator_add_ce_geo_foundation_test(ceGeoProviderRegistryTest
    tests/ce_geo/hardware/provider_registry_test.cc
    Cellerator::architecture_provider)
cellerator_add_ce_geo_foundation_test(ceGeoProviderContractTest
    "tests/ce_geo/hardware/provider_contract_test.cu;tests/ce_geo/hardware/fake_provider.cc"
    Cellerator::architecture_provider CUDA::cudart)

cellerator_add_ce_geo_foundation_test(ceGeoCatalogContractTest
    tests/ce_geo/catalog/catalog_contract_test.cc
    Cellerator::operation_core)
cellerator_add_ce_geo_foundation_test(ceGeoProjectionReferenceV2Test
    tests/ce_geo/catalog/projection_reference_v2_test.cc
    Cellerator::operation_core)
cellerator_add_ce_geo_foundation_test(ceGeoBuiltinFragmentTest
    tests/ce_geo/catalog/builtin_fragment_test.cu
    Cellerator::builtin_candidate_catalog CUDA::cudart)
cellerator_add_ce_geo_foundation_test(ceGeoRelationCatalogTest
    tests/relation_algebra/catalog_integration_test.cu
    Cellerator::relation_algebra CUDA::cudart)

cellerator_add_ce_geo_foundation_test(ceGeoSemanticCompilerTest
    tests/geometry/ce_geo/semantic_compiler_property_test.cc
    Cellerator::semantic_geometry_v1)
cellerator_add_ce_geo_foundation_test(ceGeoCsg1Test
    tests/geometry/ce_geo/csg1_test.cc
    Cellerator::semantic_geometry_v1)
cellerator_add_ce_geo_foundation_test(ceGeoCsg1SupportSectionsTest
    tests/geometry/ce_geo/csg1_support_sections_test.cc
    Cellerator::semantic_geometry_v1)
cellerator_add_ce_geo_foundation_test(ceGeoCpe2CapabilityManifestTest
    tests/geometry/ce_geo/cpe2_capability_manifest_test.cc
    Cellerator::semantic_geometry_v1 CellPack::execution_image_v2)
cellerator_add_ce_geo_foundation_test(ceGeoPreboundProjectionV2Test
    tests/geometry/ce_geo/prebound_projection_v2_test.cc
    Cellerator::geometry_acquisition)

cellerator_add_ce_geo_foundation_test(ceGeoGeometryAcquisitionContractTest
    tests/ce_geo/persistence/geometry_acquisition_contract_test.cc
    Cellerator::geometry_acquisition)
cellerator_add_ce_geo_foundation_test(ceGeoAcquisitionCostMappingTest
    tests/ce_geo/persistence/acquisition_cost_mapping_test.cc
    Cellerator::geometry_acquisition)

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

# CE-PTR-02/13: compile the explicit memory substrate through its owning
# runtime target and keep both host and CUDA/session contracts in the normal
# integration build.
add_executable(celleratorMemoryContractTest
    tests/memory/memory_contract_test.cc
)
target_link_libraries(celleratorMemoryContractTest PRIVATE
    Cellerator::runtime
)
target_compile_features(celleratorMemoryContractTest PRIVATE cxx_std_17)

add_executable(celleratorSessionMemoryTest
    tests/memory/session_memory_test.cu
)
target_link_libraries(celleratorSessionMemoryTest PRIVATE
    Cellerator::runtime
    CUDA::cudart
)
set_target_properties(celleratorSessionMemoryTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)

# CE-PTR-13: normal-build integration coverage for prepared execution,
# graph capture, concurrent streams, and persisted program replay.
add_executable(celleratorExecutableProgramTest
    tests/execution/program_test.cu
)
target_link_libraries(celleratorExecutableProgramTest PRIVATE
    Cellerator::executable_program
    CUDA::cudart
)
set_target_properties(celleratorExecutableProgramTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)

add_executable(celleratorTrainingProgramTest
    tests/execution/training_program_test.cu
)
target_link_libraries(celleratorTrainingProgramTest PRIVATE
    Cellerator::training_program
    CUDA::cudart
)
set_target_properties(celleratorTrainingProgramTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)

add_executable(celleratorCeLiveConcurrencyTest
    tests/execution/ce_live_concurrency_test.cu
)
target_link_libraries(celleratorCeLiveConcurrencyTest PRIVATE
    Cellerator::training_program
    CUDA::cudart
)
set_target_properties(celleratorCeLiveConcurrencyTest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)

add_executable(celleratorCeLiveProgramReplayTest
    tests/persistence/ce_live_program_replay_test.cu
)
target_link_libraries(celleratorCeLiveProgramReplayTest PRIVATE
    Cellerator::executable_program
    Cellerator::opaque_execution_artifact
    CellShard::inspect
    CUDA::cudart
)
set_target_properties(celleratorCeLiveProgramReplayTest PROPERTIES
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

add_executable(celleratorSparseRowTransformsTest tests/sparse_row_transforms_test.cu)
target_link_libraries(celleratorSparseRowTransformsTest PRIVATE Cellerator::sparse_ops CUDA::cudart)
set_target_properties(celleratorSparseRowTransformsTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(celleratorSparseRowTransformsTest)

add_executable(celleratorSparseColumnMomentsTest tests/sparse_column_moments_test.cu)
target_link_libraries(celleratorSparseColumnMomentsTest PRIVATE Cellerator::sparse_ops CUDA::cudart)
set_target_properties(celleratorSparseColumnMomentsTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(celleratorSparseColumnMomentsTest)

add_executable(coreSparseLayoutRuntimeTest
    tests/core_sparse_layout_runtime_test.cu
)
target_include_directories(coreSparseLayoutRuntimeTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(coreSparseLayoutRuntimeTest PRIVATE cellerator_compute_matrix_convert Baseplane::seq CUDA::cudart CUDA::cusparse)
set_target_properties(coreSparseLayoutRuntimeTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(coreSparseLayoutRuntimeTest)

add_executable(samplingRuntimeTest
    tests/sampling_runtime_test.cc
)
target_include_directories(samplingRuntimeTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(samplingRuntimeTest PRIVATE cellerator_compute_sampling CUDA::cudart)
set_target_properties(samplingRuntimeTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
cellerator_enable_perf(samplingRuntimeTest)

add_executable(samplingMaterializationRuntimeTest
    tests/sampling_materialization_runtime_test.cc
)
target_include_directories(samplingMaterializationRuntimeTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(samplingMaterializationRuntimeTest PRIVATE cellerator_compute_sampling CUDA::cudart)
set_target_properties(samplingMaterializationRuntimeTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
cellerator_enable_perf(samplingMaterializationRuntimeTest)

if(TARGET CellShard::export)
    if(CELLERATOR_CELLSHARD_SOURCE_DIR)
        set(cellerator_cellshard_test_source_dir
            "${CELLERATOR_CELLSHARD_SOURCE_DIR}")
    else()
        set(cellerator_cellshard_test_source_dir
            "${CMAKE_CURRENT_SOURCE_DIR}/../CellShard")
    endif()
    add_executable(cellShardCelleratorSamplingAdapterTest
        ${cellerator_cellshard_test_source_dir}/tests/cellerator_sampling_adapter_test.cc)
    target_link_libraries(cellShardCelleratorSamplingAdapterTest PRIVATE
        CellShard::export
        cellerator_compute_sampling
        CUDA::cudart)
    target_compile_features(cellShardCelleratorSamplingAdapterTest PRIVATE cxx_std_17)

    add_executable(cellShardStorageBoundaryIntegrationTest
        ${cellerator_cellshard_test_source_dir}/tests/cellshard_export_runtime_test.cu)
    target_compile_definitions(cellShardStorageBoundaryIntegrationTest PRIVATE
        CELLSHARD_TEST_CELLERATOR_ADAPTER=1)
    target_include_directories(cellShardStorageBoundaryIntegrationTest PRIVATE
        ${cellerator_cellshard_test_source_dir}
        ${cellerator_cellshard_test_source_dir}/include
        ${cellerator_cellshard_test_source_dir}/src
        ${cellerator_cellshard_test_source_dir}/export)
    target_link_libraries(cellShardStorageBoundaryIntegrationTest PRIVATE
        CellShard::export
        cellerator_compute_sampling
        CUDA::cudart)
    set_target_properties(cellShardStorageBoundaryIntegrationTest PROPERTIES
        CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES
        CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
endif()

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
