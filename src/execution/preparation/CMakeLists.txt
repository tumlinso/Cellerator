# The preparation and profiling APIs validate the same execution records.
add_library(cellerator_execution_record_validation STATIC
    src/execution/preparation/atom_affordance_v1.cc
    src/execution/preparation/atom_fragment_result_v1.cc
    src/execution/preparation/atom_requirement_v1.cc
    src/execution/preparation/coverage_roles_v1.cc
    src/execution/preparation/decomposition_v1.cc
    src/execution/preparation/logical_coverage_v1.cc
    src/execution/preparation/partial_result_algebra_v1.cc
    src/execution/preparation/persistent_identity_v1.cc
)
add_library(Cellerator::execution_record_validation ALIAS
    cellerator_execution_record_validation)
target_include_directories(cellerator_execution_record_validation PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(cellerator_execution_record_validation PUBLIC cxx_std_20)

# Mutable value generations have a separate owner from semantic geometry.
add_library(cellerator_projection_value_plane_v1 STATIC
    ${PROJECT_SOURCE_DIR}/src/execution/preparation/projection_value_plane/canonical_transfer_v1.cc
    ${PROJECT_SOURCE_DIR}/src/execution/preparation/projection_value_plane/composite_plane_v1.cc
    ${PROJECT_SOURCE_DIR}/src/execution/preparation/projection_value_plane/generation_publication_v1.cc
    ${PROJECT_SOURCE_DIR}/src/execution/preparation/projection_value_plane/validation_v1.cc
    ${PROJECT_SOURCE_DIR}/src/execution/preparation/projection_value_plane/value_plane_v1.cc
    ${PROJECT_SOURCE_DIR}/src/execution/preparation/value_pack_portfolio_v1.cc
)
add_library(Cellerator::projection_value_plane_v1 ALIAS
    cellerator_projection_value_plane_v1)
target_include_directories(cellerator_projection_value_plane_v1 PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellerator_projection_value_plane_v1 PUBLIC
    Cellerator::biological_abi
)
target_compile_features(cellerator_projection_value_plane_v1 PUBLIC cxx_std_20)

add_library(cellerator_opaque_execution_artifact STATIC
    src/execution/preparation/opaque_artifact.cc
)
add_library(Cellerator::opaque_execution_artifact ALIAS
    cellerator_opaque_execution_artifact)
target_include_directories(cellerator_opaque_execution_artifact PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellerator_opaque_execution_artifact PUBLIC
    CellPack::execution_image_v2
)
target_compile_features(cellerator_opaque_execution_artifact PUBLIC cxx_std_20)

add_library(cellerator_execution_projection_binding STATIC
    src/execution/preparation/provider_projection.cc
    src/execution/preparation/activated_projection.cc
)
add_library(Cellerator::execution_projection_binding ALIAS
    cellerator_execution_projection_binding)
target_include_directories(cellerator_execution_projection_binding PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellerator_execution_projection_binding PUBLIC
    CellPack::execution_image_v2
    Cellerator::feature_major_projection
    Cellerator::transpose_projection
    Cellerator::csr_fallback_candidate
)
target_compile_features(cellerator_execution_projection_binding PUBLIC cxx_std_20)

add_library(cellerator_execution_geometry_acquisition STATIC
    src/execution/preparation/geometry_acquisition.cc
    src/execution/preparation/geometry_acquisition_diagnostics.cc
    src/execution/preparation/acquisition/assembly.cc
    src/execution/preparation/acquisition/external_payload.cc
    src/execution/preparation/acquisition/projections.cc
    src/execution/preparation/acquisition/schema.cc
)
add_library(Cellerator::execution_geometry_acquisition ALIAS
    cellerator_execution_geometry_acquisition)
target_include_directories(cellerator_execution_geometry_acquisition PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellerator_execution_geometry_acquisition PUBLIC
    Cellerator::operation_core
    Cellerator::execution_planning
    Cellerator::execution_projection_binding
    Cellerator::semantic_geometry_v1
    CellPack::execution_image_v2
)
target_compile_features(cellerator_execution_geometry_acquisition PUBLIC cxx_std_20)

add_library(cellerator_execution_preparation STATIC
    src/execution/preparation/decomposition/gate_input.cc
    src/execution/preparation/atom_fragment/atom_bound_candidate_v1.cc
    src/execution/preparation/atom_fragment/canonical_fallback_v1.cc
    src/execution/preparation/atom_fragment/external_decomposition_v1.cc
    src/execution/preparation/atom_fragment/external_persistent_order_v1.cc
    src/execution/preparation/atom_fragment/external_plane_binding_v1.cc
    src/execution/preparation/atom_fragment/local_candidate_requirements_v1.cc
    src/execution/preparation/atom_fragment/local_index_builder_v1.cc
    src/execution/preparation/atom_fragment/operation_adapter_v1.cc
    src/execution/preparation/atom_fragment/output_affordance_v1.cc
    src/execution/preparation/atom_fragment/prepared_atom_fragment_v1.cc
    src/execution/preparation/atom_fragment/provider_registry_v1.cc
    src/execution/preparation/atom_fragment/requirements_v1.cc
    src/execution/preparation/atom_plane/active_support_overlay_v1.cc
    src/execution/preparation/atom_plane/dense_result_atom_v1.cc
    src/execution/preparation/atom_plane/external_plane_mapping_v1.cc
    src/execution/preparation/atom_plane/generation_publication_binding_v1.cc
    src/execution/preparation/atom_plane/gradient_plane_v1.cc
    src/execution/preparation/atom_plane/mutable_state_plane_v1.cc
    src/execution/preparation/atom_plane/partial_result_atom_v1.cc
    src/execution/preparation/atom_plane/ready_lease_binding_v1.cc
    src/execution/preparation/atom_plane/relation_value_plane_v1.cc
    src/execution/preparation/atom_plane/structural_plane_binding_v1.cc
    src/execution/preparation/atom_fragment_request_v1.cc
    src/execution/preparation/external_binding_v1.cc
    src/execution/preparation/lowering_resumption_v1.cc
    src/execution/preparation/acquisition/chunk_native_acquisition_v1.cc
    src/ops/relation/multi_extent/relation_apply_v1.cc
)
add_library(Cellerator::execution_preparation ALIAS cellerator_execution_preparation)
set_target_properties(cellerator_execution_preparation PROPERTIES EXPORT_NAME execution_preparation)
target_include_directories(cellerator_execution_preparation PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cellerator_execution_preparation PUBLIC
    Cellerator::builtin_candidate_catalog
    Cellerator::execution_program
    Cellerator::execution_geometry_acquisition
    Cellerator::operation_core
    Cellerator::execution_planning
    Cellerator::execution_record_validation
    Cellerator::operation_decomposition
    Cellerator::profiling_v1
    Cellerator::projection_value_plane_v1
)
target_compile_features(cellerator_execution_preparation PUBLIC cxx_std_20)
