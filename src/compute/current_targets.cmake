add_library(cellerator_compute_exact_search STATIC
    src/compute/neighbors/exact_search/exact_search.cu
)
target_include_directories(cellerator_compute_exact_search PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cellerator_compute_exact_search PRIVATE CUDA::cudart)
set_target_properties(cellerator_compute_exact_search PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(cellerator_compute_exact_search)

add_library(cellerator_compute_forward_neighbors STATIC
    src/compute/neighbors/forward_neighbors/forward_neighbors.cu
)
add_library(Cellerator::forward_neighbors ALIAS cellerator_compute_forward_neighbors)
target_include_directories(cellerator_compute_forward_neighbors
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(cellerator_compute_forward_neighbors
    PUBLIC
        cellerator_compute_exact_search
        CUDA::cudart
        CUDA::cusparse
)
set_target_properties(cellerator_compute_forward_neighbors PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(cellerator_compute_forward_neighbors)

add_library(cellerator_neighbor_math INTERFACE)
target_link_libraries(cellerator_neighbor_math INTERFACE cellerator_compute_exact_search)

add_library(cellerator_compute_sampling STATIC
    src/compute/dataset/sampling.cc
    src/compute/dataset/sampling_materialization.cc
)
add_library(Cellerator::compute_sampling ALIAS cellerator_compute_sampling)
target_include_directories(cellerator_compute_sampling PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
set_target_properties(cellerator_compute_sampling PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
cellerator_enable_perf(cellerator_compute_sampling)

add_library(cellerator_compute_gene_support STATIC
    src/compute/dataset/gene_support_bitset.cu
)
add_library(Cellerator::compute_gene_support ALIAS cellerator_compute_gene_support)
target_include_directories(cellerator_compute_gene_support PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellerator_compute_gene_support PUBLIC cellerator_compute_sampling CUDA::cudart)
set_target_properties(cellerator_compute_gene_support PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
cellerator_enable_perf(cellerator_compute_gene_support)

add_library(cellerator_compute_gene_candidates STATIC
    src/geometry/candidate_discovery/gene_candidate_discovery.cc
    src/geometry/candidate_discovery/gene_candidate_minhash.cu
    src/geometry/candidate_discovery/gene_candidate_discovery_cuda.cu
)
add_library(Cellerator::compute_gene_candidates ALIAS cellerator_compute_gene_candidates)
target_include_directories(cellerator_compute_gene_candidates PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellerator_compute_gene_candidates PUBLIC
    cellerator_compute_gene_support
    CUDA::cudart
)
set_target_properties(cellerator_compute_gene_candidates PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
cellerator_enable_perf(cellerator_compute_gene_candidates)

add_library(cellerator_compute_sparse_ops STATIC
    src/compute/operators/sparse/kernels/base_sparse.cu
    src/compute/operators/sparse/kernels/dist_sparse.cu
    src/compute/operators/sparse/row_transforms.cu
    src/compute/operators/sparse/column_moments.cu
)
add_library(Cellerator::sparse_ops ALIAS cellerator_compute_sparse_ops)
target_include_directories(cellerator_compute_sparse_ops
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(cellerator_compute_sparse_ops PUBLIC Cellerator::runtime_fleet)
target_link_libraries(cellerator_compute_sparse_ops PRIVATE CUDA::cudart CUDA::cusparse)
set_target_properties(cellerator_compute_sparse_ops PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(cellerator_compute_sparse_ops)

add_library(cellerator_compute_sparse_project STATIC
    src/compute/candidate/sparse/project.cu
)
target_include_directories(cellerator_compute_sparse_project PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cellerator_compute_sparse_project PUBLIC cellerator_compute_sparse_ops Cellerator::runtime_fleet)
target_link_libraries(cellerator_compute_sparse_project PRIVATE CUDA::cudart CUDA::cusparse)
set_target_properties(cellerator_compute_sparse_project PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(cellerator_compute_sparse_project)

add_library(cellerator_sparse_math INTERFACE)
target_link_libraries(cellerator_sparse_math INTERFACE cellerator_compute_sparse_project cellerator_compute_sparse_ops Cellerator::runtime_fleet)
add_library(cellerator_sparse_linalg INTERFACE)
target_link_libraries(cellerator_sparse_linalg INTERFACE cellerator_compute_sparse_project cellerator_compute_sparse_ops Cellerator::runtime_fleet)
add_library(cellerator_sparse_ml INTERFACE)
target_link_libraries(cellerator_sparse_ml INTERFACE cellerator_compute_gene_candidates cellerator_compute_gene_support cellerator_compute_sampling cellerator_compute_sparse_project cellerator_compute_sparse_ops Cellerator::runtime_fleet)
