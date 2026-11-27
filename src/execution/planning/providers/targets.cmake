# CE-GEO source-linked provider and exact physical-projection inventory.
# Semantic compilation remains in Cellerator::semantic_geometry_v1; this
# target owns target refinement, physical realization, and provider discovery.
include(src/execution/planning/providers/nvidia/common_sources.cmake)
include(src/execution/planning/providers/nvidia/sm70_sources.cmake)
add_library(cellerator_architecture_provider STATIC
    src/execution/planning/providers/provider_registry.cc
    ${CELLERATOR_NVIDIA_COMMON_PROVIDER_SOURCES}
    ${CELLERATOR_NVIDIA_SM70_PROVIDER_SOURCES}
)
add_library(Cellerator::architecture_provider ALIAS
    cellerator_architecture_provider)
if(TARGET Cellerator::sparse_support)
    # Cold image construction/validation is host-capable and shares one owner.
    target_link_libraries(cellerator_architecture_provider PUBLIC Cellerator::sparse_support)
endif()
target_include_directories(cellerator_architecture_provider PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_include_directories(cellerator_architecture_provider PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(cellerator_architecture_provider PUBLIC
    Cellerator::operation_core
    Cellerator::runtime
    Cellerator::semantic_geometry_v1
    CellPack::execution_image_v2
    Cellerator::layout Cellerator::semantic_projection_layout
    CUDA::cudart
)
set_target_properties(cellerator_architecture_provider PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED YES
)

# CE-EXOP-276: operation-complete sm_70 fan-in. Producer lanes own these
# mechanisms; this integration point only makes their exact source portfolios
# part of the normal architecture-provider build and publishes one inventory.
target_sources(cellerator_architecture_provider PRIVATE
    src/ops/relation/cuda/sm70/residual/degree_partition.cc
    src/ops/relation/cuda/sm70/residual/feature_major_residual.cu
    src/ops/relation/cuda/sm70/residual/portfolio_v1.cc
    src/ops/relation/cuda/sm70/residual/residual_kernels.cu
    src/ops/relation/cuda/sm70/residual/same_owner_fused.cu
    src/ops/relation/cuda/sm70/residual/separate_residual.cu
    src/ops/relation/cuda/sm70/relation_apply/apply_inventory_v1.cc
    src/ops/relation/cuda/sm70/relation_apply/apply_n16_n32_v1.cu
    src/ops/relation/cuda/sm70/relation_apply/apply_n64_v1.cu
    src/ops/relation/cuda/sm70/relation_apply/apply_ptx_m8n8k4_experiment_v1.cu
    src/ops/relation/cuda/sm70/relation_apply/apply_reference_v1.cc
    src/ops/relation/cuda/sm70/relation_apply/apply_registration_v1.cu
    src/ops/relation/cuda/sm70/relation_apply/apply_wide_panels_v1.cu
    src/ops/relation/cuda/sm70/relation_apply/apply_wmma_shapes_v1.cu
    src/ops/relation/cuda/sm70/contract/contract_catalog_v1.cc
    src/ops/relation/cuda/sm70/contract/exact_validation_v1.cc
    src/ops/relation/cuda/sm70/contract/rectangular_mma_contract_v1.cu
    src/ops/relation/cuda/sm70/contract/sparse_contract_v1.cu
    src/ops/relation/cuda/sm70/edge_value_gradient/direct_edge_gradient_v1.cu
    src/ops/relation/cuda/sm70/transpose/transpose_candidates_v1.cc
    src/ops/relation/cuda/sm70/transpose/transpose_candidates_v1.cu
    src/ops/relation/cuda/sm70/transpose/transpose_cover_v1.cc
    src/ops/relation/cuda/sm70/transpose/transpose_integration_v1.cc
    src/ops/relation/cuda/sm70/transpose/transpose_validation_v1.cc
    src/ops/relation/cuda/sm70/transpose/transpose_value_map_v1.cc
)
