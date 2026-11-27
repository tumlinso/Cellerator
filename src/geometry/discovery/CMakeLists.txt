add_library(cellerator_compute_sampling STATIC
    src/geometry/discovery/sampling.cc
    src/geometry/discovery/sampling_materialization.cc
)
add_library(Cellerator::compute_sampling ALIAS cellerator_compute_sampling)
target_include_directories(cellerator_compute_sampling PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
set_target_properties(cellerator_compute_sampling PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES)
cellerator_enable_perf(cellerator_compute_sampling)

add_library(cellerator_compute_gene_support STATIC
    src/geometry/discovery/gene_support_bitset.cu
)
add_library(Cellerator::compute_gene_support ALIAS cellerator_compute_gene_support)
target_include_directories(cellerator_compute_gene_support PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellerator_compute_gene_support PUBLIC cellerator_compute_sampling CUDA::cudart)
set_target_properties(cellerator_compute_gene_support PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED YES
)
cellerator_enable_perf(cellerator_compute_gene_support)

add_library(cellerator_compute_gene_candidates STATIC
    src/geometry/discovery/gene_candidate_discovery.cc
    src/geometry/discovery/gene_candidate_minhash.cu
    src/geometry/discovery/gene_candidate_discovery_cuda.cu
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
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED YES
)
cellerator_enable_perf(cellerator_compute_gene_candidates)
