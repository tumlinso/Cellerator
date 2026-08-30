#include <Cellerator/execution/projection_activation.hh>

#include <Cellerator/geometry/persistence/execution_capability_manifest_v1.hh>

namespace cellerator::execution {
namespace {

using cellpack::persistence::directory_device_readable;
using cellpack::persistence::execution_projection_kind;
using cellpack::persistence::prebound_projection_view_v1;
using cellpack::persistence::projection_forward_capable;
using cellpack::persistence::projection_transpose_capable;

constexpr bool same_projection(
    const prebound_projection_view_v1 &prebound,
    projection_id expected) noexcept {
    return prebound.descriptor.identity_low == expected.low
        && prebound.descriptor.identity_high == expected.high;
}

constexpr bool device_readable(device_location location) noexcept {
    return valid_location(location)
        && location.residency != residency_kind::host;
}

projection_activation_status validate_common(
    const prebound_projection_view_v1 &prebound,
    const projection_activation_context &context,
    execution_projection_kind kind,
    std::uint32_t schema,
    relation_orientation orientation,
    std::uint32_t capability) noexcept {
    if (!valid_identity(context.structure)
        || !valid_identity(context.projection)
        || !valid_handle(context.runtime_structure)
        || !valid_handle(context.runtime_projection)
        || context.epoch.value == 0u) {
        return {projection_activation_code::invalid_argument,
            "activation needs valid structure, projection, epoch, and handles"};
    }
    if (!device_readable(context.location)
        || (prebound.descriptor.flags & directory_device_readable) == 0u) {
        return {projection_activation_code::location_mismatch,
            "projection payload is not declared device-readable"};
    }
    if (prebound.descriptor.kind != kind)
        return {projection_activation_code::kind_mismatch,
            "CPE2 projection kind does not match typed activation"};
    if (prebound.descriptor.schema_version != schema)
        return {projection_activation_code::schema_mismatch,
            "CPE2 projection schema does not match typed payload"};
    if (prebound.descriptor.orientation
        != static_cast<std::uint16_t>(orientation)) {
        return {projection_activation_code::orientation_mismatch,
            "CPE2 projection orientation does not match requested relation view"};
    }
    if ((prebound.descriptor.flags & capability) == 0u)
        return {projection_activation_code::orientation_mismatch,
            "CPE2 projection lacks the requested orientation capability"};
    if (!same_projection(prebound, context.projection))
        return {projection_activation_code::identity_mismatch,
            "CPE2 projection identity does not match activation context"};
    return {};
}

projection_activation_status invalid_physical_view(
    compute::math::physical_view_status status) noexcept {
    if (status.code == compute::math::physical_view_status_code::incompatible_identity)
        return {projection_activation_code::identity_mismatch, status.message};
    if (status.code == compute::math::physical_view_status_code::insufficient_capacity)
        return {projection_activation_code::size_mismatch, status.message};
    return {projection_activation_code::invalid_projection, status.message};
}

bool no_maps(const prebound_projection_view_v1 &prebound) noexcept {
    return prebound.forward_map == nullptr && prebound.forward_map_bytes == 0u
        && prebound.transpose_map == nullptr
        && prebound.transpose_map_bytes == 0u;
}

constexpr bool same_architecture_identity(
    compute::architecture::architecture_identity_v1 lhs,
    compute::architecture::architecture_identity_v1 rhs) noexcept {
    return compute::architecture::same_architecture_identity_v1(lhs, rhs);
}

const compute::architecture::matrix_engine_capability_v1 *find_capability(
    const compute::architecture::architecture_provider_v1 &provider,
    compute::architecture::architecture_identity_v1 identity) noexcept {
    for (std::uint32_t index = 0u; index < provider.capability_count; ++index) {
        if (same_architecture_identity(provider.capabilities[index].identity,
                identity))
            return provider.capabilities + index;
    }
    return nullptr;
}

} // namespace

projection_activation_status validate_projection_provider_descriptor_v1(
    const projection_provider_descriptor_v1 &provider) noexcept {
    if (provider.schema_version != projection_provider_descriptor_schema_v1
        || provider.record_bytes != sizeof(projection_provider_descriptor_v1)
        || provider.architecture == nullptr || provider.validate_host == nullptr
        || provider.activate_device == nullptr
        || provider.projection_schema_version == 0u
        || provider.required_directory_capability == 0u
        || provider.validated_host_view_bytes == 0u
        || provider.activated_device_view_bytes == 0u
        || !compute::architecture::valid_architecture_identity_v1(
            provider.capability_identity))
        return {projection_activation_code::invalid_argument,
            "projection provider descriptor is incomplete"};
    if (provider.reserved0 != 0u)
        return {projection_activation_code::invalid_argument,
            "projection provider descriptor reserved field is nonzero"};
    for (std::uint32_t value : provider.reserved)
        if (value != 0u)
            return {projection_activation_code::invalid_argument,
                "projection provider descriptor reserved field is nonzero"};
    if (provider.architecture->schema_version
            != compute::architecture::architecture_provider_schema_version_v1
        || provider.architecture->record_bytes
            != sizeof(compute::architecture::architecture_provider_v1)
        || !compute::architecture::valid_architecture_identity_v1(
            provider.architecture->identity)
        || provider.architecture->name == nullptr
        || provider.architecture->capabilities == nullptr
        || provider.architecture->capability_count == 0u)
        return {projection_activation_code::provider_mismatch,
            "projection provider architecture descriptor shape is invalid"};
    const auto *capability = find_capability(
        *provider.architecture, provider.capability_identity);
    if (capability == nullptr
        || compute::architecture::validate_matrix_engine_capability_v1(*capability)
            != compute::architecture::capability_status_v1::success
        || !same_architecture_identity(capability->provider_identity,
            provider.architecture->identity))
        return {projection_activation_code::provider_mismatch,
            "projection provider capability is not source-linked"};
    return {};
}

projection_activation_status validate_and_activate_projection_via_provider_v1(
    const projection_provider_descriptor_v1 &provider,
    const cellpack::persistence::prebound_projection_view_v2 &host_prebound,
    const cellpack::persistence::prebound_projection_view_v2 &device_prebound,
    const projection_activation_context &context,
    void *validated_host_view,
    std::size_t validated_host_view_bytes,
    void *activated_device_view,
    std::size_t activated_device_view_bytes) noexcept {
    const auto descriptor_status =
        validate_projection_provider_descriptor_v1(provider);
    if (!descriptor_status) return descriptor_status;
    if (validated_host_view == nullptr || activated_device_view == nullptr
        || validated_host_view_bytes != provider.validated_host_view_bytes
        || activated_device_view_bytes != provider.activated_device_view_bytes)
        return {projection_activation_code::size_mismatch,
            "projection provider view storage does not match its descriptor"};

    const auto &host = host_prebound.projection_v1;
    const auto &device = device_prebound.projection_v1;
    if (host.descriptor.kind != provider.projection_kind
        || device.descriptor.kind != provider.projection_kind)
        return {projection_activation_code::kind_mismatch,
            "projection provider kind does not match prebound projection"};
    if (host.descriptor.schema_version != provider.projection_schema_version
        || device.descriptor.schema_version
            != provider.projection_schema_version)
        return {projection_activation_code::schema_mismatch,
            "projection provider schema does not match prebound projection"};
    if (host.descriptor.orientation
            != static_cast<std::uint16_t>(provider.orientation)
        || device.descriptor.orientation
            != static_cast<std::uint16_t>(provider.orientation)
        || (host.descriptor.flags & provider.required_directory_capability) == 0u
        || (device.descriptor.flags & provider.required_directory_capability) == 0u)
        return {projection_activation_code::orientation_mismatch,
            "projection provider orientation capability does not match"};
    if (host_prebound.capability == nullptr
        || host_prebound.capability_bytes
            != sizeof(cellpack::persistence::execution_capability_manifest_v1)
        || device_prebound.capability == nullptr
        || device_prebound.capability_bytes != host_prebound.capability_bytes)
        return {projection_activation_code::provider_mismatch,
            "projection provider requires a typed capability manifest"};

    const auto &manifest = *static_cast<const
        cellpack::persistence::execution_capability_manifest_v1 *>(
            host_prebound.capability);
    const auto manifest_status =
        cellpack::persistence::validate_execution_capability_manifest_v1(manifest);
    if (!manifest_status)
        return {projection_activation_code::provider_mismatch,
            manifest_status.message};
    const compute::architecture::architecture_identity_v1 manifest_provider{
        manifest.provider_identity_low, manifest.provider_identity_high};
    const compute::architecture::architecture_identity_v1 manifest_capability{
        manifest.capability_identity_low, manifest.capability_identity_high};
    if (!same_architecture_identity(manifest_provider,
            provider.architecture->identity)
        || !same_architecture_identity(manifest_capability,
            provider.capability_identity))
        return {projection_activation_code::provider_mismatch,
            "projection capability manifest does not match source-linked provider"};

    const auto host_common = validate_common(host, context, provider.projection_kind,
        provider.projection_schema_version, provider.orientation,
        provider.required_directory_capability);
    if (!host_common) return host_common;
    const auto device_common = validate_common(device, context,
        provider.projection_kind, provider.projection_schema_version,
        provider.orientation, provider.required_directory_capability);
    if (!device_common) return device_common;
    if (device.payload_bytes != host.payload_bytes
        || device.forward_map_bytes != host.forward_map_bytes
        || device.transpose_map_bytes != host.transpose_map_bytes
        || device.scheduling_summary_bytes != host.scheduling_summary_bytes)
        return {projection_activation_code::size_mismatch,
            "host and device prebind section sizes differ"};
    const auto validation = provider.validate_host(host_prebound, context,
        validated_host_view, validated_host_view_bytes);
    if (!validation) return validation;
    return provider.activate_device(device_prebound, context,
        validated_host_view, validated_host_view_bytes, activated_device_view,
        activated_device_view_bytes);
}

projection_activation_status activate_row_masked_projection(
    const prebound_projection_view_v1 &prebound,
    const projection_activation_context &context,
    const cellpack::persistent_packing_payload_view &validated_host,
    cellpack::persistent_packing_payload_view *out) noexcept {
    if (out == nullptr)
        return {projection_activation_code::invalid_argument,
            "row-masked activation output is null"};
    const auto common = validate_common(prebound, context,
        execution_projection_kind::native_row_masked,
        cellpack::persistent_packing_payload_schema_version,
        relation_orientation::forward, projection_forward_capable);
    if (!common) return common;
    if (prebound.payload == nullptr || prebound.payload_bytes == 0u
        || validated_host.payload_schema_version
            != cellpack::persistent_packing_payload_schema_version
        || validated_host.payload_kind != cellpack::persistent_packing_payload_kind)
        return {projection_activation_code::invalid_projection,
            "row-masked activation needs a validated CPK1 payload"};
    if (!no_maps(prebound))
        return {projection_activation_code::map_mismatch,
            "CPK1 row-masked activation does not consume external value maps"};
    const auto status = cellpack::rebind_persistent_packing_payload(
        validated_host, prebound.payload, prebound.payload_bytes, out);
    if (!status)
        return {status.code == cellpack::validation_code::insufficient_capacity
                    || status.code == cellpack::validation_code::invalid_matrix_view
                ? projection_activation_code::size_mismatch
                : projection_activation_code::invalid_projection,
            status.message};
    return {};
}

projection_activation_status activate_feature_major_projection(
    const prebound_projection_view_v1 &prebound,
    const projection_activation_context &context,
    const compute::math::feature_major_projection_view &validated_host,
    compute::math::feature_major_projection_view *out) noexcept {
    if (out == nullptr)
        return {projection_activation_code::invalid_argument,
            "feature-major activation output is null"};
    const auto common = validate_common(prebound, context,
        execution_projection_kind::native_feature_major,
        compute::math::feature_major_projection_schema_version,
        relation_orientation::forward, projection_forward_capable);
    if (!common) return common;
    if (!same_identity(validated_host.header.structure_identity, context.structure)
        || !same_identity(validated_host.header.projection_identity,
            context.projection))
        return {projection_activation_code::identity_mismatch,
            "FMP1 identity does not match activation context"};
    if (validated_host.header.structure_epoch != context.epoch.value)
        return {projection_activation_code::stale_structure,
            "FMP1 structure epoch is stale"};
    if (prebound.payload_bytes != validated_host.header.payload_bytes)
        return {projection_activation_code::size_mismatch,
            "FMP1 CPE2 section size differs from validated payload"};
    if (!no_maps(prebound))
        return {projection_activation_code::map_mismatch,
            "FMP1 value positions are internal to its typed payload"};
    const auto status = compute::math::rebind_feature_major_projection(
        validated_host, prebound.payload, prebound.payload_bytes, out);
    if (!status) return invalid_physical_view(status);
    out->runtime_structure = context.runtime_structure;
    out->runtime_projection = context.runtime_projection;
    return {};
}

projection_activation_status activate_transpose_projection(
    const prebound_projection_view_v1 &prebound,
    const transpose_projection_activation_context &context,
    const compute::math::transpose_projection_view &validated_host,
    compute::math::transpose_projection_view *out) noexcept {
    if (out == nullptr || !valid_identity(context.forward_projection)
        || !valid_handle(context.runtime_forward_projection))
        return {projection_activation_code::invalid_argument,
            "transpose activation needs output and forward projection identity"};
    const auto common = validate_common(prebound, context.projection,
        execution_projection_kind::transpose_backward,
        compute::math::transpose_projection_schema_version,
        relation_orientation::transpose, projection_transpose_capable);
    if (!common) return common;
    if (!same_identity(validated_host.header.structure_identity,
            context.projection.structure)
        || !same_identity(validated_host.header.projection_identity,
            context.projection.projection)
        || !same_identity(validated_host.header.forward_projection_identity,
            context.forward_projection))
        return {projection_activation_code::identity_mismatch,
            "CTP1 identities do not match activation context"};
    if (validated_host.header.structure_epoch != context.projection.epoch.value)
        return {projection_activation_code::stale_structure,
            "CTP1 structure epoch is stale"};
    if (prebound.payload_bytes != validated_host.header.payload_bytes)
        return {projection_activation_code::size_mismatch,
            "CTP1 CPE2 section size differs from validated payload"};
    const std::size_t one_map_bytes = static_cast<std::size_t>(
        validated_host.header.nnz_count) * sizeof(std::uint32_t);
    const std::size_t map_bytes = one_map_bytes * 2u;
    if (prebound.transpose_map == nullptr
        || prebound.transpose_map_bytes != map_bytes)
        return {projection_activation_code::map_mismatch,
            "CTP1 requires the explicit logical transpose value-position map"};
    if (prebound.forward_map != nullptr || prebound.forward_map_bytes != 0u)
        return {projection_activation_code::map_mismatch,
            "CTP1 does not accept an implicit forward-map substitution"};
    const auto status = compute::math::rebind_transpose_projection(
        validated_host, prebound.payload, prebound.payload_bytes, out);
    if (!status) return invalid_physical_view(status);
    out->runtime_structure = context.projection.runtime_structure;
    out->runtime_projection = context.projection.runtime_projection;
    out->runtime_forward_projection = context.runtime_forward_projection;
    // CPE2 owns the explicit logical-edge map for the activated image. The
    // payload still owns traversal-local forward positions.
    out->logical_to_transpose = static_cast<const std::uint32_t *>(
        prebound.transpose_map);
    out->transpose_to_logical = reinterpret_cast<const std::uint32_t *>(
        static_cast<const unsigned char *>(prebound.transpose_map)
            + one_map_bytes);
    return {};
}

projection_activation_status activate_csr_projection(
    const prebound_projection_view_v1 &prebound,
    const projection_activation_context &context,
    const compute::math::execution_csr_view &prepared_view,
    compute::math::execution_csr_view *out) noexcept {
    if (out == nullptr)
        return {projection_activation_code::invalid_argument,
            "CSR activation output is null"};
    const auto common = validate_common(prebound, context,
        execution_projection_kind::csr,
        compute::math::execution_csr_schema_version,
        relation_orientation::forward, projection_forward_capable);
    if (!common) return common;
    if (prebound.payload != nullptr || prebound.payload_bytes != 0u
        || !no_maps(prebound))
        return {projection_activation_code::map_mismatch,
            "runtime CSR activation cannot hide a CPE2 conversion or map"};
    if (prepared_view.schema_version != compute::math::execution_csr_schema_version
        || prepared_view.row_offsets == nullptr
        || (prepared_view.nnz_count != 0u
            && (prepared_view.execution_feature_ids == nullptr
                || prepared_view.values == nullptr))
        || prepared_view.value_size_bytes == 0u
        || prepared_view.structure.value == 0u)
        return {projection_activation_code::invalid_projection,
            "CSR activation needs a complete caller-prepared runtime view"};
    *out = prepared_view;
    return {};
}

} // namespace cellerator::execution
