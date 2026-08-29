#include <Cellerator/compute/operation/builtin_catalog.hh>

#include <Cellerator/compute/candidate/csr_fallback_candidate.hh>
#include <Cellerator/compute/candidate/feature_major_small_n_candidate.hh>
#include <Cellerator/compute/candidate/row_masked_n1_candidate.hh>
#include <Cellerator/compute/candidate/transpose_backward_candidate.hh>

#include <Cellerator/compute/projection/physical_csr.hh>
#include <Cellerator/compute/projection/physical_feature_major.hh>
#include <Cellerator/compute/projection/physical_transpose.hh>

#include <Cellerator/geometry/persistent_packing_payload.hh>

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace cellerator::compute::math::core {
namespace {

constexpr std::uint32_t common_projection_requirements =
    catalog_prebound_typed_projection
    | catalog_device_resident_projection
    | catalog_projection_local_values;

constexpr stable_id builtin_provider_identity_v2{
    0x63656c6c65726174ull, 0x6f722d636f72652dull};
constexpr stable_id builtin_fragment_identity_v2{
    0x636174616c6f672dull, 0x76322d636f72652dull};
constexpr stable_id row_masked_view_type_v2{
    0x726f772d6d61736bull, 0x65642d766965772dull};
constexpr stable_id csr_view_type_v2{
    0x6373722d76696577ull, 0x2d747970652d7632ull};
constexpr stable_id feature_major_view_type_v2{
    0x666561747572652dull, 0x6d616a6f722d7632ull};
constexpr stable_id transpose_view_type_v2{
    0x7472616e73706f73ull, 0x652d766965772d32ull};

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

stable_id view_type_identity_v2(projection_kind projection) noexcept {
    switch (projection) {
    case projection_kind::native_row_masked:
        return row_masked_view_type_v2;
    case projection_kind::csr:
        return csr_view_type_v2;
    case projection_kind::native_feature_major:
        return feature_major_view_type_v2;
    case projection_kind::transpose_or_backward:
        return transpose_view_type_v2;
    default:
        return {};
    }
}

candidate_descriptor_v2 compatibility_descriptor_v2(
    const built_in_candidate_descriptor &legacy) noexcept {
    candidate_descriptor_v2 descriptor{};
    descriptor.candidate = legacy.factory();
    descriptor.provider_identity = builtin_provider_identity_v2;
    descriptor.projection_contract.view_type =
        view_type_identity_v2(legacy.projection);
    descriptor.projection_contract.abi_major = 1u;
    descriptor.projection_contract.schema_version =
        legacy.projection_schema_version;
    descriptor.projection_contract.variant = legacy.projection_variant;
    descriptor.flags = candidate_descriptor_requires_measurement
        | candidate_descriptor_compatibility;
    if (legacy.projection == projection_kind::csr)
        descriptor.flags |= candidate_descriptor_conventional;
    descriptor.minimum_dense_width = legacy.minimum_dense_width;
    descriptor.maximum_dense_width = legacy.maximum_dense_width;
    descriptor.state_bytes = legacy.state_bytes;
    descriptor.state_alignment = legacy.state_alignment;
    return descriptor;
}

const candidate_descriptor_v2 *compatibility_entries_v2() noexcept {
    static const candidate_descriptor_v2 entries[builtin_candidate_count]{
        compatibility_descriptor_v2(catalog_entries[0]),
        compatibility_descriptor_v2(catalog_entries[1]),
        compatibility_descriptor_v2(catalog_entries[2]),
        compatibility_descriptor_v2(catalog_entries[3]),
        compatibility_descriptor_v2(catalog_entries[4])};
    return entries;
}

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

const candidate_catalog_fragment_v2 &
built_in_candidate_catalog_fragment_v2() noexcept {
    static const candidate_catalog_fragment_v2 fragment{
        candidate_catalog_fragment_schema_version_v2,
        sizeof(candidate_catalog_fragment_v2),
        builtin_fragment_identity_v2,
        builtin_provider_identity_v2,
        "cellerator-core-five-v2-compatibility",
        compatibility_entries_v2(),
        builtin_candidate_count,
        candidate_fragment_builtin | candidate_fragment_compatibility,
        1u,
        {}};
    return fragment;
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
    if (validate_candidate_catalog_fragment_v2(
            built_in_candidate_catalog_fragment_v2())
        != candidate_catalog_status_v2::success)
        return invalid_catalog("built-in catalog-v2 fragment is inconsistent");
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
