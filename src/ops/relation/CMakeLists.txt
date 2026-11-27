add_library(cellerator_row_masked_n1_candidate STATIC
    src/ops/relation/row_masked_n1_candidate.cu
)
add_library(Cellerator::row_masked_n1_candidate ALIAS
    cellerator_row_masked_n1_candidate)
target_include_directories(cellerator_row_masked_n1_candidate PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellerator_row_masked_n1_candidate PUBLIC
    Cellerator::operation_core
    CellPack::feature_weighted_row_reduction_cuda
    CellPack::persistent_packing_payload
)
set_target_properties(cellerator_row_masked_n1_candidate PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED YES
)

# CE-ARCH-72: operation-core activation of the existing conventional CSR N=1
# execution path. Projection construction remains explicit and caller-owned.
add_library(cellerator_csr_fallback_candidate STATIC
    src/ops/relation/csr_fallback_candidate.cu
)
add_library(Cellerator::csr_fallback_candidate ALIAS
    cellerator_csr_fallback_candidate)
target_include_directories(cellerator_csr_fallback_candidate PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_include_directories(cellerator_csr_fallback_candidate PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(cellerator_csr_fallback_candidate PUBLIC
    Cellerator::operation_core
    CellPack::apply_plan
    CellPack::persistent_packing_payload
    CUDA::cudart
)
target_link_libraries(cellerator_csr_fallback_candidate PRIVATE Cellerator::csr_multiply)
set_target_properties(cellerator_csr_fallback_candidate PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED YES
)

add_library(cellerator_feature_major_small_n_candidate STATIC
    src/ops/relation/feature_major_small_n_candidate.cu
)
add_library(Cellerator::feature_major_small_n_candidate ALIAS
    cellerator_feature_major_small_n_candidate)
target_include_directories(cellerator_feature_major_small_n_candidate PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellerator_feature_major_small_n_candidate PUBLIC
    Cellerator::operation_core
    Cellerator::feature_major_projection
    CUDA::cudart
)
target_link_libraries(cellerator_feature_major_small_n_candidate PRIVATE
    Cellerator::feature_major_tiles_cuda)
set_target_properties(cellerator_feature_major_small_n_candidate PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED YES
)

add_library(cellerator_transpose_backward_candidate STATIC
    src/ops/relation/transpose_backward_candidate.cu
    src/ops/relation/cuda/sm70/transpose/relation_n16.cu
)

add_library(Cellerator::transpose_backward_candidate ALIAS
    cellerator_transpose_backward_candidate)
target_include_directories(cellerator_transpose_backward_candidate PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellerator_transpose_backward_candidate PUBLIC
    Cellerator::operation_core
    Cellerator::transpose_projection
    CUDA::cudart
)
set_target_properties(cellerator_transpose_backward_candidate PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED YES
)
