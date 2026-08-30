# CE-ARCH-71: operation-core activation of the preserved CP-BP v1 row-masked
# N=1 direct consumer. This adapter adds no kernel or persistence format.
add_library(cellerator_row_masked_n1_candidate STATIC
    src/compute/candidate/row_masked_n1_candidate.cu
)

# CE-GEO source-linked provider and exact physical-projection inventory.
# Semantic compilation remains in Cellerator::semantic_geometry_v1; this
# target owns target refinement, physical realization, and provider discovery.
include(src/compute/architecture/providers/nvidia/common/CMakeLists.txt)
include(src/compute/architecture/providers/nvidia/sm70/CMakeLists.txt)
add_library(cellerator_architecture_provider STATIC
    src/compute/architecture/provider_registry.cc
    ${CELLERATOR_NVIDIA_COMMON_PROVIDER_SOURCES}
    ${CELLERATOR_NVIDIA_SM70_PROVIDER_SOURCES}
    src/compute/projection/mma_physical_cover_validation.cc
    src/compute/projection/mma_physical_work_layout.cc
    src/compute/projection/mma_residual_builder.cc
    src/compute/projection/physical_mma_hybrid.cc
)
add_library(Cellerator::architecture_provider ALIAS
    cellerator_architecture_provider)
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
    CUDA::cudart
)
set_target_properties(cellerator_architecture_provider PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)

# The public relation algebra remains independent of model, trainer, and
# framework ownership. Its candidate-catalog-v2 declarations are source-linked
# here, while schema-v2-only operations continue to fail closed under the
# frozen operation-core schema v1.
include(src/compute/candidate/segment/CMakeLists.txt)
add_library(cellerator_relation_algebra STATIC
    src/compute/operation/relation_algebra_catalog.cc
    src/compute/operation/relation_algebra_assembly.cc
    src/compute/operation/relation_apply_compatibility.cc
    src/compute/candidate/edge_map_or_gate.cu
    src/compute/candidate/relation_bundle.cc
    ${CELLERATOR_RELATION_SEGMENT_SOURCES}
)
add_library(Cellerator::relation_algebra ALIAS cellerator_relation_algebra)
target_include_directories(cellerator_relation_algebra PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellerator_relation_algebra PUBLIC
    Cellerator::operation_core
    CUDA::cudart
)
set_target_properties(cellerator_relation_algebra PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
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
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
# CE-ARCH-72: operation-core activation of the existing conventional CSR N=1
# execution path. Projection construction remains explicit and caller-owned.
add_library(cellerator_csr_fallback_candidate STATIC
    src/compute/candidate/csr_fallback_candidate.cu
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
set_target_properties(cellerator_csr_fallback_candidate PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)

# CE-ARCH-75: a second projection of the frozen CP-BP semantic structure.
# Construction and value packing are explicit host preparation; prepared
# execution consumes the native feature-major payload directly for N=1..16.
add_library(cellerator_feature_major_projection STATIC
    src/compute/projection/physical_feature_major.cc
)
add_library(Cellerator::feature_major_projection ALIAS
    cellerator_feature_major_projection)
target_include_directories(cellerator_feature_major_projection PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellerator_feature_major_projection PUBLIC
    CellPack::persistent_packing_payload
    Cellerator::biological_abi
)
target_compile_features(cellerator_feature_major_projection PUBLIC cxx_std_17)

add_library(cellerator_transpose_projection STATIC
    src/compute/projection/physical_transpose.cc
)
add_library(Cellerator::transpose_projection ALIAS
    cellerator_transpose_projection)
target_include_directories(cellerator_transpose_projection PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellerator_transpose_projection PUBLIC
    Cellerator::feature_major_projection
    Cellerator::biological_abi
)
target_compile_features(cellerator_transpose_projection PUBLIC cxx_std_17)

add_library(cellerator_feature_major_small_n_candidate STATIC
    src/compute/candidate/feature_major_small_n_candidate.cu
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
set_target_properties(cellerator_feature_major_small_n_candidate PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)

add_library(cellerator_transpose_backward_candidate STATIC
    src/compute/candidate/transpose_backward_candidate.cu
)

add_library(cellerator_native_training_slice STATIC
    src/compute/training/native_training_slice.cu
)
add_library(Cellerator::native_training_slice ALIAS
    cellerator_native_training_slice)
target_include_directories(cellerator_native_training_slice PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellerator_native_training_slice PUBLIC
    Cellerator::feature_major_projection
    Cellerator::transpose_projection
    Cellerator::runtime
    CUDA::cudart
)
set_target_properties(cellerator_native_training_slice PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
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
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)

# CE-LIVE-29: Wave B executable-core fan-in. Leaf tasks own behavior; this
# rendezvous owns their shared native target and test wiring.
add_library(cellerator_projection_activation STATIC
    src/execution/projection_activation.cc
)
add_library(Cellerator::projection_activation ALIAS
    cellerator_projection_activation)
target_include_directories(cellerator_projection_activation PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellerator_projection_activation PUBLIC
    CellPack::execution_image_v2
    Cellerator::feature_major_projection
    Cellerator::transpose_projection
    Cellerator::csr_fallback_candidate
)
target_compile_features(cellerator_projection_activation PUBLIC cxx_std_17)

add_library(cellerator_builtin_candidate_catalog STATIC
    src/compute/operation/builtin_catalog.cc
)
add_library(Cellerator::builtin_candidate_catalog ALIAS
    cellerator_builtin_candidate_catalog)
target_include_directories(cellerator_builtin_candidate_catalog PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellerator_builtin_candidate_catalog PUBLIC
    Cellerator::row_masked_n1_candidate
    Cellerator::csr_fallback_candidate
    Cellerator::feature_major_small_n_candidate
    Cellerator::transpose_backward_candidate
)
target_compile_features(cellerator_builtin_candidate_catalog PUBLIC cxx_std_17)

add_library(cellerator_cusparse_csr_candidate STATIC
    src/compute/candidate/cusparse_csr_candidate.cu
)
add_library(Cellerator::cusparse_csr_candidate ALIAS
    cellerator_cusparse_csr_candidate)
target_include_directories(cellerator_cusparse_csr_candidate PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellerator_cusparse_csr_candidate PUBLIC
    Cellerator::operation_core
    Cellerator::runtime
    CellPack::apply_plan
    CUDA::cudart
    CUDA::cusparse
)
set_target_properties(cellerator_cusparse_csr_candidate PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)

add_library(cellerator_preparation_factory STATIC
    src/compute/operation/preparation_factory.cc
)
add_library(Cellerator::preparation_factory ALIAS
    cellerator_preparation_factory)
target_include_directories(cellerator_preparation_factory PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellerator_preparation_factory PUBLIC
    Cellerator::builtin_candidate_catalog
    Cellerator::runtime
)
target_compile_features(cellerator_preparation_factory PUBLIC cxx_std_17)

add_library(cellerator_executable_core INTERFACE)
add_library(Cellerator::executable_core ALIAS cellerator_executable_core)
target_link_libraries(cellerator_executable_core INTERFACE
    Cellerator::projection_activation
    Cellerator::builtin_candidate_catalog
    Cellerator::cusparse_csr_candidate
    Cellerator::preparation_factory
    Cellerator::native_training_slice
)

# CE-LIVE-43: publish the CE-LIVE-30 host orchestration layer as a normal
# native target so framework adapters consume the frozen executable-program
# contract instead of compiling or reimplementing it privately.
add_library(cellerator_executable_program STATIC
    src/execution/program.cc
)
add_library(Cellerator::executable_program ALIAS
    cellerator_executable_program)
target_include_directories(cellerator_executable_program PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellerator_executable_program PUBLIC
    Cellerator::executable_core
    Cellerator::planner
)
target_compile_features(cellerator_executable_program PUBLIC cxx_std_17)

add_library(cellerator_training_program STATIC
    src/execution/training_program.cu
)
add_library(Cellerator::training_program ALIAS cellerator_training_program)
target_include_directories(cellerator_training_program PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellerator_training_program PUBLIC
    Cellerator::native_training_slice
    Cellerator::feature_major_projection
    Cellerator::transpose_projection
    Cellerator::runtime
    CUDA::cudart
)
set_target_properties(cellerator_training_program PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
