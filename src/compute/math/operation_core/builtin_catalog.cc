#include <Cellerator/compute/math/operation_core/builtin_catalog.hh>

#include <Cellerator/compute/math/operation_core/csr_fallback_candidate.hh>
#include <Cellerator/compute/math/operation_core/feature_major_small_n_candidate.hh>
#include <Cellerator/compute/math/operation_core/row_masked_n1_candidate.hh>
#include <Cellerator/compute/math/operation_core/transpose_backward_candidate.hh>

#include <Cellerator/compute/math/physical_csr.hh>
#include <Cellerator/compute/math/physical_feature_major.hh>
#include <Cellerator/compute/math/physical_transpose.hh>

#include <CellPack/persistent_packing_payload.hh>

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace cellerator::compute::math::core {
namespace {

constexpr std::uint32_t common_projection_requirements =
    catalog_prebound_typed_projection
    | catalog_device_resident_projection
    | catalog_projection_local_values;

const built_in_candidate_descriptor catalog_entries[builtin_candidate_count]{
    {builtin_candidate_catalog_schema_version,
        row_masked_n1_candidate_id,
        "cpbp-cpk1-row-masked-n1",
        &row_masked_n1_candidate,
        &register_row_masked_n1_candidate,
        preparation_family::row_masked_n1,
        catalog_numeric_family::configured_row_masked,
        catalog_output_axis::destination,
        1u,
        operation_kind::weighted_relation_reduce,
        projection_kind::native_row_masked,
        backend_kind::native_direct,
        0u,
        candidate_deterministic | candidate_graph_capture
            | candidate_persistent_preprocessing,
        cellpack::persistent_packing_payload_schema_version,
        1u,
        1u,
        1u,
        sizeof(row_masked_n1_prepared_state),
        0u,
        sizeof(row_masked_n1_prepared_state),
        alignof(row_masked_n1_prepared_state),
        common_projection_requirements,
        true,
        true,
        {}},
    {builtin_candidate_catalog_schema_version,
        csr_fallback_candidate_id,
        "cellerator-csr-fallback-n1-f16-f32",
        &csr_fallback_candidate,
        &register_csr_fallback_candidate,
        preparation_family::csr_n1,
        catalog_numeric_family::f16_sparse_f32_compute,
        catalog_output_axis::destination,
        1u,
        operation_kind::weighted_relation_reduce,
        projection_kind::csr,
        backend_kind::native_direct,
        0u,
        candidate_deterministic | candidate_persistent_preprocessing,
        execution_csr_schema_version,
        1u,
        1u,
        1u,
        sizeof(csr_fallback_prepared_state),
        0u,
        sizeof(csr_fallback_prepared_state),
        alignof(csr_fallback_prepared_state),
        common_projection_requirements | catalog_device_ordinal,
        true,
        true,
        {}},
    {builtin_candidate_catalog_schema_version,
        feature_major_small_n_candidate_id,
        "cpbp-feature-major-small-n-f16-f32",
        &feature_major_small_n_candidate,
        &register_feature_major_small_n_candidate,
        preparation_family::feature_major_small_n,
        catalog_numeric_family::f16_sparse_f32_compute,
        catalog_output_axis::destination,
        2u,
        operation_kind::sparse_dense_multiply,
        projection_kind::native_feature_major,
        backend_kind::native_direct,
        0u,
        candidate_deterministic | candidate_graph_capture
            | candidate_persistent_preprocessing,
        feature_major_projection_schema_version,
        feature_major_projection_variant,
        1u,
        16u,
        sizeof(feature_major_small_n_prepared_state),
        0u,
        sizeof(feature_major_small_n_prepared_state),
        alignof(feature_major_small_n_prepared_state),
        common_projection_requirements | catalog_device_ordinal
            | catalog_dense_column_axis,
        true,
        true,
        {}},
    {builtin_candidate_catalog_schema_version,
        feature_major_cta_medium_n_candidate_id,
        "cpbp-feature-major-cta-medium-n-f16-f32",
        &feature_major_cta_medium_n_candidate,
        &register_feature_major_cta_medium_n_candidate,
        preparation_family::feature_major_cta_medium_n,
        catalog_numeric_family::f16_sparse_f32_compute,
        catalog_output_axis::destination,
        2u,
        operation_kind::sparse_dense_multiply,
        projection_kind::native_feature_major,
        backend_kind::native_direct,
        0u,
        candidate_deterministic | candidate_graph_capture
            | candidate_persistent_preprocessing,
        feature_major_projection_schema_version,
        feature_major_projection_variant,
        feature_major_cta_medium_n_minimum,
        feature_major_cta_medium_n_maximum,
        sizeof(feature_major_small_n_prepared_state),
        0u,
        sizeof(feature_major_small_n_prepared_state),
        alignof(feature_major_small_n_prepared_state),
        common_projection_requirements | catalog_device_ordinal
            | catalog_dense_column_axis,
        true,
        true,
        {}},
    {builtin_candidate_catalog_schema_version,
        transpose_backward_n1_candidate_id,
        "cpbp-transpose-backward-n1-f16-f32",
        &transpose_backward_n1_candidate,
        &register_transpose_backward_n1_candidate,
        preparation_family::transpose_backward_n1,
        catalog_numeric_family::f16_sparse_f32_compute,
        catalog_output_axis::source,
        2u,
        operation_kind::sparse_dense_multiply,
        projection_kind::transpose_or_backward,
        backend_kind::native_direct,
        0u,
        candidate_deterministic | candidate_graph_capture
            | candidate_persistent_preprocessing,
        transpose_projection_schema_version,
        transpose_projection_variant,
        1u,
        1u,
        sizeof(transpose_backward_prepared_state),
        0u,
        sizeof(transpose_backward_prepared_state),
        alignof(transpose_backward_prepared_state),
        common_projection_requirements | catalog_device_ordinal
            | catalog_dense_column_axis | catalog_transpose_value_map,
        true,
        true,
        {}}
};

operation_status invalid_catalog(const char *message) noexcept {
    return {operation_status_code::invalid_argument,
        execution::binding_validation_code::ok, message};
}

bool descriptor_matches_candidate(
    const built_in_candidate_descriptor &descriptor,
    const operation_candidate &candidate) noexcept {
    return descriptor.schema_version
            == builtin_candidate_catalog_schema_version
        && same_stable_id(descriptor.identity, candidate.identity)
        && descriptor.name != nullptr && candidate.name != nullptr
        && std::strcmp(descriptor.name, candidate.name) == 0
        && descriptor.operation == candidate.operation
        && descriptor.projection == candidate.projection
        && descriptor.backend == candidate.backend
        && descriptor.capability_flags == candidate.capability_flags
        && descriptor.persistent_bytes == candidate.persistent_bytes
        && descriptor.transient_bytes == candidate.transient_bytes
        && descriptor.factory != nullptr && descriptor.registration != nullptr
        && descriptor.projection_schema_version != 0u
        && descriptor.projection_variant != 0u
        && descriptor.minimum_dense_width != 0u
        && descriptor.minimum_dense_width <= descriptor.maximum_dense_width
        && descriptor.state_bytes == descriptor.persistent_bytes
        && descriptor.state_alignment != 0u
        && (descriptor.preparation_requirements
            & common_projection_requirements) == common_projection_requirements
        && descriptor.output_axis_count != 0u
        && descriptor.output_overwrite
        && descriptor.activation_requires_measurement;
}

} // namespace

built_in_candidate_catalog_view built_in_candidate_catalog() noexcept {
    return {catalog_entries, builtin_candidate_count};
}

const built_in_candidate_descriptor *find_built_in_candidate(
    stable_id identity) noexcept {
    for (const built_in_candidate_descriptor &entry : catalog_entries)
        if (same_stable_id(entry.identity, identity)) return &entry;
    return nullptr;
}

operation_status validate_built_in_candidate_catalog() noexcept {
    for (std::uint32_t index = 0u; index < builtin_candidate_count; ++index) {
        const built_in_candidate_descriptor &entry = catalog_entries[index];
        if (!descriptor_matches_candidate(entry, entry.factory()))
            return invalid_catalog("built-in candidate descriptor is inconsistent");
        for (std::uint32_t previous = 0u; previous < index; ++previous)
            if (same_stable_id(entry.identity, catalog_entries[previous].identity))
                return invalid_catalog("built-in candidate identity is duplicated");
    }
    return {};
}

operation_status register_built_in_candidate_catalog(
    candidate_registry *registry) noexcept {
    if (registry == nullptr)
        return invalid_catalog("built-in registration requires a registry");
    const operation_status valid = validate_built_in_candidate_catalog();
    if (!valid) return valid;
    if (registry->size > operation_candidate_capacity
        || operation_candidate_capacity - registry->size
            < builtin_candidate_count)
        return {operation_status_code::registry_full,
            execution::binding_validation_code::ok,
            "built-in candidates exceed registry capacity"};

    candidate_registry staged = *registry;
    for (const built_in_candidate_descriptor &entry : catalog_entries) {
        const operation_status registered = entry.registration(&staged);
        if (!registered) return registered;
    }
    *registry = staged;
    return {};
}

} // namespace cellerator::compute::math::core
