include(bench/planner_targets.cmake)

add_executable(quantizedMatrixBench
    bench/quantized_matrix_bench.cu
)
target_include_directories(quantizedMatrixBench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(quantizedMatrixBench PRIVATE CUDA::cudart)
set_target_properties(quantizedMatrixBench PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(quantizedMatrixBench)

add_executable(quantizedTransferSpmmBench
    bench/quantized_transfer_spmm_bench.cu
)
target_include_directories(quantizedTransferSpmmBench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(quantizedTransferSpmmBench PRIVATE cellerator_compute_sparse_project cellerator_compute_sparse_ops Cellerator::runtime_fleet CellShard::runtime CUDA::cudart CUDA::cusparse)
set_target_properties(quantizedTransferSpmmBench PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(quantizedTransferSpmmBench)

add_executable(sparseOpsBench
    bench/sparse_ops_bench.cu
)
target_include_directories(sparseOpsBench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(sparseOpsBench PRIVATE cellerator_compute_sparse_project cellerator_compute_sparse_ops Cellerator::runtime_fleet CUDA::cudart CUDA::cusparse)
set_target_properties(sparseOpsBench PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(sparseOpsBench)

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
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
cellerator_enable_perf(geneCandidateDiscoveryBench)
