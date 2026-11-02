#pragma once

#include <Cellerator/layout/projection/csr.hh>
#include <Cellerator/layout/projection/feature_major.hh>
#include <Cellerator/layout/projection/transpose.hh>
#include <Cellerator/execution/execution_contract.hh>

#include <cstddef>

namespace cellerator::execution {

// Physical maps describe immutable structure and borrowed payload memory.
// Runtime handles belong to this execution-side association and are never
// serialized with the map bytes.
struct FeatureMajorBinding {
    layout::projection::FeatureMajorViewV1 layout{};
    structure_handle runtime_structure{};
    projection_handle runtime_projection{};
};

struct TransposeBinding {
    layout::projection::TransposeViewV1 layout{};
    structure_handle runtime_structure{};
    projection_handle runtime_projection{};
    projection_handle runtime_forward_projection{};
};

struct FeatureMajorBindingBuildRequest {
    layout::projection::feature_major_projection_build_request layout{};
    structure_handle runtime_structure{};
    projection_handle runtime_projection{};
};

struct TransposeBindingBuildRequest {
    structure::projection_id projection_identity{};
    projection_handle runtime_projection{};
    const FeatureMajorBinding *forward = nullptr;
};

inline layout::projection::physical_view_status
query_feature_major_binding_requirements_host(
    const FeatureMajorBindingBuildRequest &request,
    layout::projection::feature_major_projection_requirements *out) noexcept {
    if (!valid_handle(request.runtime_structure)
        || !valid_handle(request.runtime_projection)) {
        return {layout::projection::physical_view_status_code::invalid_argument,
            "feature-major binding needs valid runtime handles"};
    }
    return layout::projection::query_feature_major_projection_requirements_host(
        request.layout, out);
}

inline layout::projection::physical_view_status build_feature_major_binding_host(
    const FeatureMajorBindingBuildRequest &request,
    const layout::projection::feature_major_projection_buffer &buffer,
    FeatureMajorBinding *out) noexcept {
    if (out == nullptr || !valid_handle(request.runtime_structure)
        || !valid_handle(request.runtime_projection)) {
        return {layout::projection::physical_view_status_code::invalid_argument,
            "feature-major binding needs valid runtime handles and output"};
    }
    FeatureMajorBinding result{};
    const auto status = layout::projection::build_feature_major_projection_host(
        request.layout, buffer, &result.layout);
    if (!status) return status;
    result.runtime_structure = request.runtime_structure;
    result.runtime_projection = request.runtime_projection;
    *out = result;
    return {};
}

inline layout::projection::physical_view_status
validate_feature_major_binding_payload_host(
    const void *payload,
    std::size_t payload_bytes,
    structure::structure_id expected_structure,
    structure::structure_epoch expected_epoch,
    structure::projection_id expected_projection,
    structure_handle runtime_structure,
    projection_handle runtime_projection,
    FeatureMajorBinding *out) noexcept {
    if (out == nullptr || !valid_handle(runtime_structure)
        || !valid_handle(runtime_projection)) {
        return {layout::projection::physical_view_status_code::invalid_argument,
            "feature-major validation needs valid runtime handles and output"};
    }
    FeatureMajorBinding result{};
    const auto status = layout::projection::validate_feature_major_projection_payload_host(
        payload, payload_bytes, expected_structure, expected_epoch,
        expected_projection, &result.layout);
    if (!status) return status;
    result.runtime_structure = runtime_structure;
    result.runtime_projection = runtime_projection;
    *out = result;
    return {};
}

inline layout::projection::physical_view_status rebind_feature_major_binding(
    const FeatureMajorBinding &validated_host,
    const void *new_payload_base,
    std::size_t new_payload_bytes,
    FeatureMajorBinding *out) noexcept {
    if (out == nullptr || !valid_handle(validated_host.runtime_structure)
        || !valid_handle(validated_host.runtime_projection)) {
        return {layout::projection::physical_view_status_code::invalid_argument,
            "feature-major rebind needs a valid binding and output"};
    }
    FeatureMajorBinding result{};
    const auto status = layout::projection::rebind_feature_major_projection(
        validated_host.layout, new_payload_base, new_payload_bytes, &result.layout);
    if (!status) return status;
    result.runtime_structure = validated_host.runtime_structure;
    result.runtime_projection = validated_host.runtime_projection;
    *out = result;
    return {};
}

inline layout::projection::physical_view_status
query_transpose_binding_requirements_host(
    const TransposeBindingBuildRequest &request,
    layout::projection::transpose_projection_requirements *out) noexcept {
    if (out == nullptr || request.forward == nullptr
        || !valid_handle(request.runtime_projection)
        || !valid_handle(request.forward->runtime_structure)
        || !valid_handle(request.forward->runtime_projection)) {
        return {layout::projection::physical_view_status_code::invalid_argument,
            "transpose binding needs a forward binding, runtime handles, and output"};
    }
    layout::projection::transpose_projection_build_request layout_request{};
    layout_request.projection_identity = request.projection_identity;
    layout_request.forward = request.forward->layout;
    return layout::projection::query_transpose_projection_requirements_host(
        layout_request, out);
}

inline layout::projection::physical_view_status build_transpose_binding_host(
    const TransposeBindingBuildRequest &request,
    const layout::projection::transpose_projection_buffer &buffer,
    TransposeBinding *out) noexcept {
    if (out == nullptr || request.forward == nullptr
        || !valid_handle(request.runtime_projection)
        || !valid_handle(request.forward->runtime_structure)
        || !valid_handle(request.forward->runtime_projection)) {
        return {layout::projection::physical_view_status_code::invalid_argument,
            "transpose binding needs a forward binding, runtime handles, and output"};
    }
    layout::projection::transpose_projection_build_request layout_request{};
    layout_request.projection_identity = request.projection_identity;
    layout_request.forward = request.forward->layout;
    TransposeBinding result{};
    const auto status = layout::projection::build_transpose_projection_host(
        layout_request, buffer, &result.layout);
    if (!status) return status;
    result.runtime_structure = request.forward->runtime_structure;
    result.runtime_projection = request.runtime_projection;
    result.runtime_forward_projection = request.forward->runtime_projection;
    *out = result;
    return {};
}

inline layout::projection::physical_view_status
validate_transpose_binding_payload_host(
    const void *payload,
    std::size_t payload_bytes,
    structure::structure_id expected_structure,
    structure::structure_epoch expected_epoch,
    structure::projection_id expected_projection,
    structure::projection_id expected_forward_projection,
    structure_handle runtime_structure,
    projection_handle runtime_projection,
    projection_handle runtime_forward_projection,
    TransposeBinding *out) noexcept {
    if (out == nullptr || !valid_handle(runtime_structure)
        || !valid_handle(runtime_projection)
        || !valid_handle(runtime_forward_projection)) {
        return {layout::projection::physical_view_status_code::invalid_argument,
            "transpose validation needs valid runtime handles and output"};
    }
    TransposeBinding result{};
    const auto status = layout::projection::validate_transpose_projection_payload_host(
        payload, payload_bytes, expected_structure, expected_epoch,
        expected_projection, expected_forward_projection, &result.layout);
    if (!status) return status;
    result.runtime_structure = runtime_structure;
    result.runtime_projection = runtime_projection;
    result.runtime_forward_projection = runtime_forward_projection;
    *out = result;
    return {};
}

inline layout::projection::physical_view_status rebind_transpose_binding(
    const TransposeBinding &validated_host,
    const void *new_payload_base,
    std::size_t new_payload_bytes,
    TransposeBinding *out) noexcept {
    if (out == nullptr || !valid_handle(validated_host.runtime_structure)
        || !valid_handle(validated_host.runtime_projection)
        || !valid_handle(validated_host.runtime_forward_projection)) {
        return {layout::projection::physical_view_status_code::invalid_argument,
            "transpose rebind needs a valid binding and output"};
    }
    TransposeBinding result{};
    const auto status = layout::projection::rebind_transpose_projection(
        validated_host.layout, new_payload_base, new_payload_bytes, &result.layout);
    if (!status) return status;
    result.runtime_structure = validated_host.runtime_structure;
    result.runtime_projection = validated_host.runtime_projection;
    result.runtime_forward_projection = validated_host.runtime_forward_projection;
    *out = result;
    return {};
}

inline value_position_map_view transpose_value_position_map(
    const TransposeBinding &projection,
    device_location location) noexcept {
    return {projection.runtime_structure,
        {projection.layout.header.structure_epoch},
        value_map_direction::transpose, {},
        projection.layout.logical_to_transpose,
        projection.layout.transpose_to_logical,
        location, projection.layout.header.nnz_count};
}

} // namespace cellerator::execution
