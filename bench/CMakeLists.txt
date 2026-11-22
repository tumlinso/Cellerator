include(bench/geometry/packing_targets.cmake)
include(bench/planner_targets.cmake)

add_executable(quantizedMatrixBench
    bench/quantized_matrix_bench.cu
)
target_include_directories(quantizedMatrixBench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(quantizedMatrixBench PRIVATE CUDA::cudart)
set_target_properties(quantizedMatrixBench PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(quantizedMatrixBench)

add_executable(quantizedTransferSpmmBench
    bench/quantized_transfer_spmm_bench.cu
)
target_include_directories(quantizedTransferSpmmBench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(quantizedTransferSpmmBench PRIVATE cellerator_compute_sparse_project cellerator_compute_sparse_ops Cellerator::runtime_fleet Cellerator::memory_cuda CUDA::cudart CUDA::cusparse)
set_target_properties(quantizedTransferSpmmBench PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(quantizedTransferSpmmBench)

add_executable(sparseOpsBench
    bench/sparse_ops_bench.cu
)
target_include_directories(sparseOpsBench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(sparseOpsBench PRIVATE cellerator_compute_sparse_project cellerator_compute_sparse_ops Cellerator::runtime_fleet Cellerator::memory_cuda CUDA::cudart CUDA::cusparse)
set_target_properties(sparseOpsBench PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(sparseOpsBench)

add_executable(csrMultiplyBench bench/csr_multiply_bench.cu)
target_include_directories(csrMultiplyBench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(csrMultiplyBench PRIVATE Cellerator::csr_multiply CUDA::cudart CUDA::cusparse)
set_target_properties(csrMultiplyBench PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(csrMultiplyBench)

add_executable(geneCandidateDiscoveryBench
    bench/gene_candidate_discovery_bench.cu
)
target_include_directories(geneCandidateDiscoveryBench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/bench
)
target_link_libraries(geneCandidateDiscoveryBench PRIVATE
    cellerator_compute_gene_candidates
    CUDA::cudart
)
set_target_properties(geneCandidateDiscoveryBench PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED YES
)
cellerator_enable_perf(geneCandidateDiscoveryBench)

add_executable(celleratorFeatureMajorTilesBench
    bench/feature_major_tiles/feature_major_tiles_bench.cu)
target_link_libraries(celleratorFeatureMajorTilesBench PRIVATE
    Cellerator::native_data_cuda Cellerator::csr_multiply CUDA::cusparse)
set_target_properties(celleratorFeatureMajorTilesBench PROPERTIES
    CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(celleratorFeatureMajorTilesBench)

add_executable(celleratorFeatureMajorDatasetTrainingBench
    bench/feature_major_tiles/dataset_training_bench.cu)
target_link_libraries(celleratorFeatureMajorDatasetTrainingBench PRIVATE
    Cellerator::native_data_cuda Cellerator::csr_multiply CUDA::cusparse)
set_target_properties(celleratorFeatureMajorDatasetTrainingBench PROPERTIES
    CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(celleratorFeatureMajorDatasetTrainingBench)

if(CELLERATOR_BUILD_NATIVE_DATA_BENCHMARKS)
    add_subdirectory(bench/native_data ${CELLERATOR_CHECK_EXCLUDE_FROM_ALL})
endif()
