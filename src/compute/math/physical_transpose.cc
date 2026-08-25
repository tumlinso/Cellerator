#include <Cellerator/compute/math/physical_transpose.hh>

#include <algorithm>
#include <cstring>
#include <limits>
#include <vector>

namespace cellerator::compute::math {
namespace {

physical_view_status fail(physical_view_status_code code,
    const char *message) noexcept { return {code, message}; }

bool add_size(std::size_t a, std::size_t b, std::size_t *out) noexcept {
    if (b > std::numeric_limits<std::size_t>::max() - a) return false;
    *out = a + b;
    return true;
}

bool align_size(std::size_t value, std::size_t *out) noexcept {
    constexpr std::size_t mask = transpose_projection_alignment - 1u;
    if (value > std::numeric_limits<std::size_t>::max() - mask) return false;
    *out = (value + mask) & ~mask;
    return true;
}

bool append_array(std::size_t count, u64 *offset,
    std::size_t *cursor) noexcept {
    if (!align_size(*cursor, cursor)) return false;
    *offset = *cursor;
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(u32))
        return false;
    return add_size(*cursor, count * sizeof(u32), cursor);
}

bool compute_layout(u32 features, u32 nnz,
    transpose_projection_payload_header *header,
    std::size_t *bytes) noexcept {
    std::size_t cursor = sizeof(transpose_projection_payload_header);
    if (!append_array(static_cast<std::size_t>(features) + 1u,
            &header->feature_offsets_offset, &cursor)
        || !append_array(nnz, &header->execution_row_ids_offset, &cursor)
        || !append_array(nnz, &header->forward_value_positions_offset, &cursor)
        || !append_array(nnz, &header->logical_to_transpose_offset, &cursor)
        || !append_array(nnz, &header->transpose_to_logical_offset, &cursor)
        || !align_size(cursor, &cursor)) return false;
    header->payload_bytes = cursor;
    *bytes = cursor;
    return true;
}

template<typename T>
T *mutable_at(void *base, u64 offset) noexcept {
    return reinterpret_cast<T *>(static_cast<unsigned char *>(base) + offset);
}

template<typename T>
const T *at(const void *base, u64 offset) noexcept {
    return reinterpret_cast<const T *>(
        static_cast<const unsigned char *>(base) + offset);
}

bool range_valid(u64 offset, std::size_t count,
    std::size_t bytes) noexcept {
    return offset <= bytes && count <= (bytes - offset) / sizeof(u32);
}

void set_view(const transpose_projection_payload_header &header,
    const void *base, execution::structure_handle structure,
    execution::projection_handle projection,
    execution::projection_handle forward_projection,
    transpose_projection_view *out) noexcept {
    transpose_projection_view value{};
    value.header = header;
    value.runtime_structure = structure;
    value.runtime_projection = projection;
    value.runtime_forward_projection = forward_projection;
    value.payload_base = base;
    value.feature_offsets = at<u32>(base, header.feature_offsets_offset);
    value.execution_row_ids = at<u32>(base, header.execution_row_ids_offset);
    value.forward_value_positions = at<u32>(
        base, header.forward_value_positions_offset);
    value.logical_to_transpose = at<u32>(
        base, header.logical_to_transpose_offset);
    value.transpose_to_logical = at<u32>(
        base, header.transpose_to_logical_offset);
    *out = value;
}

bool valid_forward(const feature_major_projection_view &view) noexcept {
    const auto &header = view.header;
    return header.schema_version == feature_major_projection_schema_version
        && header.payload_kind == feature_major_projection_payload_kind
        && header.payload_bytes != 0u && view.payload_base != nullptr
        && execution::valid_identity(header.structure_identity)
        && execution::valid_identity(header.projection_identity)
        && header.structure_epoch != 0u
        && execution::valid_handle(view.runtime_structure)
        && execution::valid_handle(view.runtime_projection)
        && view.tile_feature_offsets != nullptr
        && view.execution_feature_ids != nullptr
        && view.participating_row_masks != nullptr
        && view.feature_value_offsets != nullptr
        && view.source_value_positions != nullptr;
}

} // namespace

physical_view_status query_transpose_projection_requirements_host(
    const transpose_projection_build_request &request,
    transpose_projection_requirements *out) noexcept {
    if (out == nullptr || !valid_forward(request.forward)
        || !execution::valid_identity(request.projection_identity)
        || !execution::valid_handle(request.runtime_projection)
        || execution::same_identity(request.projection_identity,
            request.forward.header.projection_identity))
        return fail(physical_view_status_code::invalid_argument,
            "transpose requirements need distinct valid projection identities");
    transpose_projection_payload_header header{};
    std::size_t bytes = 0u;
    if (!compute_layout(request.forward.header.feature_count,
            request.forward.header.nnz_count, &header, &bytes))
        return fail(physical_view_status_code::overflow,
            "transpose payload layout overflows");
    *out = {bytes,
        static_cast<std::size_t>(request.forward.header.feature_count) + 1u,
        request.forward.header.nnz_count};
    return {};
}

physical_view_status build_transpose_projection_host(
    const transpose_projection_build_request &request,
    const transpose_projection_buffer &buffer,
    transpose_projection_view *out) noexcept {
    transpose_projection_requirements required{};
    physical_view_status status =
        query_transpose_projection_requirements_host(request, &required);
    if (!status) return status;
    if (out == nullptr || buffer.payload == nullptr
        || buffer.capacity_bytes < required.payload_bytes)
        return fail(physical_view_status_code::insufficient_capacity,
            "transpose payload buffer is too small");
    std::memset(buffer.payload, 0, required.payload_bytes);
    const auto &forward = request.forward;
    const auto &source = forward.header;
    transpose_projection_payload_header header{};
    std::size_t bytes = 0u;
    if (!compute_layout(source.feature_count, source.nnz_count,
            &header, &bytes))
        return fail(physical_view_status_code::overflow,
            "transpose payload layout changed during construction");
    header.structure_identity = source.structure_identity;
    header.projection_identity = request.projection_identity;
    header.forward_projection_identity = source.projection_identity;
    header.structure_epoch = source.structure_epoch;
    header.source_payload_identity = source.source_payload_identity;
    header.feature_block_geometry_identity =
        source.feature_block_geometry_identity;
    header.ordering_identity = source.ordering_identity;
    header.row_domain_identity = source.row_domain_identity;
    header.feature_axis_fingerprint = source.feature_axis_fingerprint;
    header.global_row_begin = source.global_row_begin;
    header.full_row_count = source.full_row_count;
    header.row_count = source.row_count;
    header.feature_count = source.feature_count;
    header.nnz_count = source.nnz_count;
    header.value_size_bytes = source.value_size_bytes;
    header.feature_axis_fingerprint_version =
        source.feature_axis_fingerprint_version;
    std::memcpy(buffer.payload, &header, sizeof(header));

    auto *offsets = mutable_at<u32>(buffer.payload,
        header.feature_offsets_offset);
    auto *rows = mutable_at<u32>(buffer.payload,
        header.execution_row_ids_offset);
    auto *forward_positions = mutable_at<u32>(buffer.payload,
        header.forward_value_positions_offset);
    auto *logical_to_transpose = mutable_at<u32>(buffer.payload,
        header.logical_to_transpose_offset);
    auto *transpose_to_logical = mutable_at<u32>(buffer.payload,
        header.transpose_to_logical_offset);

    for (u32 tile = 0u; tile < source.tile_count; ++tile)
        for (u32 record = forward.tile_feature_offsets[tile];
             record < forward.tile_feature_offsets[tile + 1u]; ++record)
            offsets[forward.execution_feature_ids[record] + 1u]
                += static_cast<u32>(__builtin_popcount(
                    forward.participating_row_masks[record]));
    for (u32 feature = 0u; feature < source.feature_count; ++feature)
        offsets[feature + 1u] += offsets[feature];
    std::vector<u32> cursors(offsets,
        offsets + static_cast<std::size_t>(source.feature_count));
    for (u32 tile = 0u; tile < source.tile_count; ++tile) {
        for (u32 record = forward.tile_feature_offsets[tile];
             record < forward.tile_feature_offsets[tile + 1u]; ++record) {
            const u32 feature = forward.execution_feature_ids[record];
            const u32 mask = forward.participating_row_masks[record];
            u32 local = 0u;
            for (u32 lane = 0u; lane < source.tile_row_width; ++lane) {
                if ((mask & (1u << lane)) == 0u) continue;
                const u32 forward_position =
                    forward.feature_value_offsets[record] + local++;
                const u32 transpose_position = cursors[feature]++;
                const u32 logical = forward.source_value_positions[
                    forward_position];
                rows[transpose_position] =
                    tile * source.tile_row_width + lane;
                forward_positions[transpose_position] = forward_position;
                transpose_to_logical[transpose_position] = logical;
                logical_to_transpose[logical] = transpose_position;
            }
        }
    }
    return validate_transpose_projection_payload_host(buffer.payload, bytes,
        source.structure_identity, {source.structure_epoch},
        request.projection_identity, source.projection_identity,
        forward.runtime_structure, request.runtime_projection,
        forward.runtime_projection, out);
}

physical_view_status validate_transpose_projection_payload_host(
    const void *payload, std::size_t payload_bytes,
    execution::structure_id expected_structure,
    execution::structure_epoch expected_epoch,
    execution::projection_id expected_projection,
    execution::projection_id expected_forward_projection,
    execution::structure_handle runtime_structure,
    execution::projection_handle runtime_projection,
    execution::projection_handle runtime_forward_projection,
    transpose_projection_view *out) noexcept {
    if (payload == nullptr || out == nullptr
        || payload_bytes < sizeof(transpose_projection_payload_header)
        || !execution::valid_identity(expected_structure)
        || expected_epoch.value == 0u
        || !execution::valid_identity(expected_projection)
        || !execution::valid_identity(expected_forward_projection)
        || !execution::valid_handle(runtime_structure)
        || !execution::valid_handle(runtime_projection)
        || !execution::valid_handle(runtime_forward_projection))
        return fail(physical_view_status_code::invalid_argument,
            "transpose validation arguments are invalid");
    transpose_projection_payload_header header{};
    std::memcpy(&header, payload, sizeof(header));
    if (header.schema_version != transpose_projection_schema_version
        || header.payload_kind != transpose_projection_payload_kind
        || header.header_bytes != sizeof(header)
        || header.alignment != transpose_projection_alignment
        || header.payload_bytes != payload_bytes
        || !execution::same_identity(header.structure_identity,
            expected_structure)
        || header.structure_epoch != expected_epoch.value
        || !execution::same_identity(header.projection_identity,
            expected_projection)
        || !execution::same_identity(header.forward_projection_identity,
            expected_forward_projection)
        || execution::same_identity(header.projection_identity,
            header.forward_projection_identity)
        || header.source_payload_identity == 0u
        || header.feature_block_geometry_identity == 0u
        || header.ordering_identity == 0u
        || header.row_domain_identity == 0u
        || header.feature_axis_fingerprint == 0u
        || header.feature_axis_fingerprint_version == 0u
        || header.value_size_bytes == 0u)
        return fail(physical_view_status_code::incompatible_identity,
            "transpose payload identity or metadata is incompatible");
    const u64 array_offsets[]{header.feature_offsets_offset,
        header.execution_row_ids_offset,
        header.forward_value_positions_offset,
        header.logical_to_transpose_offset,
        header.transpose_to_logical_offset};
    for (u64 offset : array_offsets)
        if ((offset % transpose_projection_alignment) != 0u)
            return fail(physical_view_status_code::invalid_geometry,
                "transpose array offset is misaligned");
    if (!range_valid(header.feature_offsets_offset,
            static_cast<std::size_t>(header.feature_count) + 1u, payload_bytes)
        || !range_valid(header.execution_row_ids_offset,
            header.nnz_count, payload_bytes)
        || !range_valid(header.forward_value_positions_offset,
            header.nnz_count, payload_bytes)
        || !range_valid(header.logical_to_transpose_offset,
            header.nnz_count, payload_bytes)
        || !range_valid(header.transpose_to_logical_offset,
            header.nnz_count, payload_bytes))
        return fail(physical_view_status_code::invalid_geometry,
            "transpose array range exceeds payload");
    transpose_projection_view view{};
    set_view(header, payload, runtime_structure, runtime_projection,
        runtime_forward_projection, &view);
    if (view.feature_offsets[0] != 0u
        || view.feature_offsets[header.feature_count] != header.nnz_count)
        return fail(physical_view_status_code::invalid_geometry,
            "transpose terminal offsets are invalid");
    std::vector<unsigned char> seen_forward(header.nnz_count, 0u);
    std::vector<unsigned char> seen_logical(header.nnz_count, 0u);
    for (u32 feature = 0u; feature < header.feature_count; ++feature) {
        const u32 begin = view.feature_offsets[feature];
        const u32 end = view.feature_offsets[feature + 1u];
        if (end < begin || end > header.nnz_count)
            return fail(physical_view_status_code::invalid_geometry,
                "transpose feature offsets are not monotonic");
        u32 previous_row = 0u;
        for (u32 edge = begin; edge < end; ++edge) {
            const u32 row = view.execution_row_ids[edge];
            const u32 forward = view.forward_value_positions[edge];
            const u32 logical = view.transpose_to_logical[edge];
            if (row >= header.row_count
                || (edge != begin && row <= previous_row)
                || forward >= header.nnz_count || seen_forward[forward] != 0u
                || logical >= header.nnz_count || seen_logical[logical] != 0u
                || view.logical_to_transpose[logical] != edge)
                return fail(physical_view_status_code::invalid_geometry,
                    "transpose edge map is not a bijective ordered traversal");
            previous_row = row;
            seen_forward[forward] = 1u;
            seen_logical[logical] = 1u;
        }
    }
    *out = view;
    return {};
}

physical_view_status rebind_transpose_projection(
    const transpose_projection_view &validated_host_view,
    const void *new_payload_base, std::size_t new_payload_bytes,
    transpose_projection_view *out) noexcept {
    if (out == nullptr || new_payload_base == nullptr
        || validated_host_view.payload_base == nullptr
        || validated_host_view.header.schema_version
            != transpose_projection_schema_version
        || validated_host_view.header.payload_bytes != new_payload_bytes)
        return fail(physical_view_status_code::invalid_argument,
            "transpose rebind requires an equal validated payload");
    set_view(validated_host_view.header, new_payload_base,
        validated_host_view.runtime_structure,
        validated_host_view.runtime_projection,
        validated_host_view.runtime_forward_projection, out);
    return {};
}

execution::value_position_map_view transpose_value_position_map(
    const transpose_projection_view &projection,
    execution::device_location location) noexcept {
    return {projection.runtime_structure,
        {projection.header.structure_epoch},
        execution::value_map_direction::transpose, {},
        projection.logical_to_transpose, projection.transpose_to_logical,
        location, projection.header.nnz_count};
}

} // namespace cellerator::compute::math
