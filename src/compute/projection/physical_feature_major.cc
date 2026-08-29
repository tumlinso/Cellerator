#include <Cellerator/compute/projection/physical_feature_major.hh>

#include <algorithm>
#include <cstring>
#include <limits>
#include <vector>

namespace cellerator::compute::math {
namespace {

physical_view_status fail(
    physical_view_status_code code, const char *message) noexcept {
    return {code, message};
}

bool add_size(std::size_t lhs, std::size_t rhs, std::size_t *out) noexcept {
    if (rhs > std::numeric_limits<std::size_t>::max() - lhs) return false;
    *out = lhs + rhs;
    return true;
}

bool multiply_size(std::size_t lhs, std::size_t rhs, std::size_t *out) noexcept {
    if (lhs != 0u && rhs > std::numeric_limits<std::size_t>::max() / lhs)
        return false;
    *out = lhs * rhs;
    return true;
}

bool align_size(std::size_t cursor, std::size_t alignment,
    std::size_t *out) noexcept {
    const std::size_t mask = alignment - 1u;
    if (cursor > std::numeric_limits<std::size_t>::max() - mask) return false;
    *out = (cursor + mask) & ~mask;
    return true;
}

bool same_location_identity(
    execution::structure_id lhs, execution::structure_id rhs) noexcept {
    return execution::same_identity(lhs, rhs);
}

bool valid_request_identity(
    const feature_major_projection_build_request &request) noexcept {
    return execution::valid_identity(request.structure_identity)
        && execution::valid_handle(request.runtime_structure)
        && request.structure_epoch_value.value != 0u
        && execution::valid_identity(request.projection_identity)
        && execution::valid_handle(request.runtime_projection);
}

struct source_counts {
    u32 feature_records = 0u;
    u32 compact_values = 0u;
};

physical_view_status inspect_source(
    const cellpack::persistent_packing_payload_view &payload,
    source_counts *out) noexcept {
    if (out == nullptr
        || !physical_csr_detail::valid_payload_metadata(payload)) {
        return fail(physical_view_status_code::invalid_argument,
            "feature-major construction requires a validated host CPK1 view");
    }
    const auto &plan = payload.plan;
    const auto &tiles = payload.tiles;
    source_counts counts{};
    for (u32 tile = 0u; tile < tiles.tile_count; ++tile) {
        const u32 descriptor_begin = tiles.tile_block_offsets[tile];
        const u32 descriptor_end = tiles.tile_block_offsets[tile + 1u];
        if (descriptor_end < descriptor_begin
            || descriptor_end > tiles.tile_block_count) {
            return fail(physical_view_status_code::invalid_geometry,
                "CPK1 tile descriptor offsets are invalid");
        }
        const u32 remaining = tiles.row_count - tile * tiles.tile_row_width;
        const u32 lane_count = std::min(tiles.tile_row_width, remaining);
        const u32 valid_rows = lane_count == 32u
            ? std::numeric_limits<u32>::max() : (1u << lane_count) - 1u;
        u32 previous_block = std::numeric_limits<u32>::max();
        for (u32 descriptor = descriptor_begin;
             descriptor < descriptor_end; ++descriptor) {
            const u32 block = tiles.tile_block_ids[descriptor];
            const u32 row_mask = tiles.tile_block_cell_masks[descriptor];
            const u32 entry_begin = tiles.block_row_entry_offsets[descriptor];
            const u32 entry_end = tiles.block_row_entry_offsets[descriptor + 1u];
            if (block >= plan.feature_block_count || row_mask == 0u
                || (row_mask & ~valid_rows) != 0u
                || (descriptor != descriptor_begin && block <= previous_block)
                || entry_end < entry_begin
                || entry_end > tiles.row_block_entry_count
                || entry_end - entry_begin
                    != static_cast<u32>(__builtin_popcount(row_mask))) {
                return fail(physical_view_status_code::invalid_geometry,
                    "CPK1 feature block descriptor is invalid");
            }
            previous_block = block;
            const u32 feature_begin = plan.feature_block_offsets[block];
            const u32 feature_end = plan.feature_block_offsets[block + 1u];
            if (feature_end <= feature_begin || feature_end > plan.feature_count
                || feature_end - feature_begin > 32u) {
                return fail(physical_view_status_code::invalid_geometry,
                    "CPK1 feature block width is invalid");
            }
            const u32 valid_features = feature_end - feature_begin == 32u
                ? std::numeric_limits<u32>::max()
                : (1u << (feature_end - feature_begin)) - 1u;
            u32 feature_union = 0u;
            for (u32 entry = entry_begin; entry < entry_end; ++entry) {
                const u32 mask = tiles.row_block_gene_masks[entry];
                const u32 value_begin = tiles.row_block_value_offsets[entry];
                const u32 value_end = tiles.row_block_value_offsets[entry + 1u];
                const u32 value_count = static_cast<u32>(__builtin_popcount(mask));
                if (mask == 0u || (mask & ~valid_features) != 0u
                    || value_end < value_begin || value_end > tiles.nnz_count
                    || value_end - value_begin != value_count) {
                    return fail(physical_view_status_code::invalid_geometry,
                        "CPK1 row feature/value record is invalid");
                }
                feature_union |= mask;
                if (counts.compact_values > tiles.nnz_count - value_count) {
                    return fail(physical_view_status_code::overflow,
                        "feature-major compact value count overflows");
                }
                counts.compact_values += value_count;
            }
            const u32 record_count =
                static_cast<u32>(__builtin_popcount(feature_union));
            if (counts.feature_records
                    > std::numeric_limits<u32>::max() - record_count) {
                return fail(physical_view_status_code::overflow,
                    "feature-major record count overflows");
            }
            counts.feature_records += record_count;
        }
    }
    if (counts.compact_values != tiles.nnz_count) {
        return fail(physical_view_status_code::invalid_geometry,
            "CPK1 compact values do not span logical nnz");
    }
    *out = counts;
    return {};
}

bool compute_layout(
    u32 tile_count,
    u32 feature_records,
    u32 nnz,
    feature_major_projection_payload_header *header,
    std::size_t *payload_bytes) noexcept {
    if (header == nullptr || payload_bytes == nullptr) return false;
    std::size_t cursor = sizeof(feature_major_projection_payload_header);
    std::size_t array_bytes = 0u;
    if (!align_size(cursor, feature_major_projection_alignment, &cursor)) return false;
    header->tile_feature_offsets_offset = cursor;
    if (!multiply_size(static_cast<std::size_t>(tile_count) + 1u,
            sizeof(u32), &array_bytes)
        || !add_size(cursor, array_bytes, &cursor)
        || !align_size(cursor, feature_major_projection_alignment, &cursor))
        return false;
    header->execution_feature_ids_offset = cursor;
    if (!multiply_size(feature_records, sizeof(u32), &array_bytes)
        || !add_size(cursor, array_bytes, &cursor)
        || !align_size(cursor, feature_major_projection_alignment, &cursor))
        return false;
    header->participating_row_masks_offset = cursor;
    if (!multiply_size(feature_records, sizeof(u32), &array_bytes)
        || !add_size(cursor, array_bytes, &cursor)
        || !align_size(cursor, feature_major_projection_alignment, &cursor))
        return false;
    header->feature_value_offsets_offset = cursor;
    if (!multiply_size(static_cast<std::size_t>(feature_records) + 1u,
            sizeof(u32), &array_bytes)
        || !add_size(cursor, array_bytes, &cursor)
        || !align_size(cursor, feature_major_projection_alignment, &cursor))
        return false;
    header->source_value_positions_offset = cursor;
    if (!multiply_size(nnz, sizeof(u32), &array_bytes)
        || !add_size(cursor, array_bytes, &cursor)
        || !align_size(cursor, feature_major_projection_alignment, &cursor))
        return false;
    header->payload_bytes = cursor;
    *payload_bytes = cursor;
    return true;
}

template<typename T>
T *mutable_at(void *base, u64 offset) noexcept {
    return reinterpret_cast<T *>(
        static_cast<unsigned char *>(base) + offset);
}

template<typename T>
const T *pointer_at(const void *base, u64 offset) noexcept {
    return reinterpret_cast<const T *>(
        static_cast<const unsigned char *>(base) + offset);
}

bool range_valid(u64 offset, std::size_t count,
    std::size_t element_bytes, std::size_t payload_bytes) noexcept {
    if (count != 0u
        && element_bytes > std::numeric_limits<std::size_t>::max() / count)
        return false;
    const std::size_t bytes = count * element_bytes;
    return offset <= payload_bytes && bytes <= payload_bytes - offset;
}

bool offset_aligned(u64 offset) noexcept {
    return offset % feature_major_projection_alignment == 0u;
}

void set_view(const feature_major_projection_payload_header &header,
    const void *base,
    execution::structure_handle runtime_structure,
    execution::projection_handle runtime_projection,
    feature_major_projection_view *out) noexcept {
    feature_major_projection_view view{};
    view.header = header;
    view.runtime_structure = runtime_structure;
    view.runtime_projection = runtime_projection;
    view.payload_base = base;
    view.tile_feature_offsets = pointer_at<u32>(
        base, header.tile_feature_offsets_offset);
    view.execution_feature_ids = pointer_at<u32>(
        base, header.execution_feature_ids_offset);
    view.participating_row_masks = pointer_at<u32>(
        base, header.participating_row_masks_offset);
    view.feature_value_offsets = pointer_at<u32>(
        base, header.feature_value_offsets_offset);
    view.source_value_positions = pointer_at<u32>(
        base, header.source_value_positions_offset);
    *out = view;
}

} // namespace

physical_view_status query_feature_major_projection_requirements_host(
    const feature_major_projection_build_request &request,
    feature_major_projection_requirements *out) noexcept {
    if (out == nullptr || !valid_request_identity(request)) {
        return fail(physical_view_status_code::invalid_argument,
            "feature-major requirements need identities and output");
    }
    source_counts counts{};
    const physical_view_status source = inspect_source(request.source, &counts);
    if (!source) return source;
    feature_major_projection_payload_header header{};
    std::size_t payload_bytes = 0u;
    if (!compute_layout(request.source.tiles.tile_count,
            counts.feature_records, counts.compact_values,
            &header, &payload_bytes)) {
        return fail(physical_view_status_code::overflow,
            "feature-major payload layout overflows");
    }
    feature_major_projection_requirements result{};
    result.payload_bytes = payload_bytes;
    result.tile_feature_offset_count =
        static_cast<std::size_t>(request.source.tiles.tile_count) + 1u;
    result.feature_record_count = counts.feature_records;
    result.feature_value_offset_count =
        static_cast<std::size_t>(counts.feature_records) + 1u;
    result.source_value_position_count = counts.compact_values;
    result.construction_workspace = {0u, 1u,
        {cellerator::memory::domain::host, -1, -1, 0u}};
    *out = result;
    return {};
}

physical_view_status build_feature_major_projection_host(
    const feature_major_projection_build_request &request,
    const feature_major_projection_buffer &buffer,
    feature_major_projection_view *out) noexcept {
    feature_major_projection_requirements required{};
    physical_view_status status =
        query_feature_major_projection_requirements_host(request, &required);
    if (!status) return status;
    if (out == nullptr || buffer.payload == nullptr
        || buffer.capacity_bytes < required.payload_bytes) {
        return fail(physical_view_status_code::insufficient_capacity,
            "feature-major payload buffer is too small");
    }
    std::memset(buffer.payload, 0, required.payload_bytes);
    feature_major_projection_payload_header header{};
    std::size_t computed_bytes = 0u;
    if (!compute_layout(request.source.tiles.tile_count,
            static_cast<u32>(required.feature_record_count),
            static_cast<u32>(required.source_value_position_count),
            &header, &computed_bytes)) {
        return fail(physical_view_status_code::overflow,
            "feature-major payload layout changed during construction");
    }
    const auto &source = request.source;
    const auto &tiles = source.tiles;
    header.structure_identity = request.structure_identity;
    header.projection_identity = request.projection_identity;
    header.structure_epoch = request.structure_epoch_value.value;
    header.source_payload_identity = source.payload_identity;
    header.feature_block_geometry_identity =
        source.plan.feature_block_geometry_identity;
    header.ordering_identity = source.order.ordering_identity;
    header.global_row_begin = tiles.global_row_begin;
    header.row_domain_identity = tiles.row_domain_identity;
    header.feature_axis_fingerprint = tiles.feature_axis_fingerprint;
    header.feature_axis_fingerprint_version =
        tiles.feature_axis_fingerprint_version;
    header.full_row_count = tiles.full_row_count;
    header.row_count = tiles.row_count;
    header.feature_count = tiles.feature_count;
    header.tile_row_width = tiles.tile_row_width;
    header.tile_count = tiles.tile_count;
    header.feature_record_count = static_cast<u32>(required.feature_record_count);
    header.nnz_count = tiles.nnz_count;
    header.value_size_bytes = tiles.value_size_bytes;
    std::memcpy(buffer.payload, &header, sizeof(header));

    auto *tile_offsets = mutable_at<u32>(
        buffer.payload, header.tile_feature_offsets_offset);
    auto *feature_ids = mutable_at<u32>(
        buffer.payload, header.execution_feature_ids_offset);
    auto *row_masks = mutable_at<u32>(
        buffer.payload, header.participating_row_masks_offset);
    auto *value_offsets = mutable_at<u32>(
        buffer.payload, header.feature_value_offsets_offset);
    auto *source_positions = mutable_at<u32>(
        buffer.payload, header.source_value_positions_offset);

    u32 record_cursor = 0u;
    u32 value_cursor = 0u;
    tile_offsets[0] = 0u;
    value_offsets[0] = 0u;
    for (u32 tile = 0u; tile < tiles.tile_count; ++tile) {
        for (u32 descriptor = tiles.tile_block_offsets[tile];
             descriptor < tiles.tile_block_offsets[tile + 1u]; ++descriptor) {
            const u32 block = tiles.tile_block_ids[descriptor];
            const u32 feature_begin = source.plan.feature_block_offsets[block];
            const u32 feature_end = source.plan.feature_block_offsets[block + 1u];
            u32 lane_gene_masks[32]{};
            u32 lane_value_begins[32]{};
            u32 entry = tiles.block_row_entry_offsets[descriptor];
            const u32 descriptor_rows = tiles.tile_block_cell_masks[descriptor];
            for (u32 lane = 0u; lane < tiles.tile_row_width; ++lane) {
                if ((descriptor_rows & (1u << lane)) == 0u) continue;
                lane_gene_masks[lane] = tiles.row_block_gene_masks[entry];
                lane_value_begins[lane] =
                    tiles.row_block_value_offsets[entry];
                ++entry;
            }
            for (u32 local = 0u; local < feature_end - feature_begin; ++local) {
                u32 feature_rows = 0u;
                for (u32 lane = 0u; lane < tiles.tile_row_width; ++lane)
                    if ((lane_gene_masks[lane] & (1u << local)) != 0u)
                        feature_rows |= 1u << lane;
                if (feature_rows == 0u) continue;
                feature_ids[record_cursor] = feature_begin + local;
                row_masks[record_cursor] = feature_rows;
                for (u32 lane = 0u; lane < tiles.tile_row_width; ++lane) {
                    if ((feature_rows & (1u << lane)) == 0u) continue;
                    const u32 lower = local == 0u
                        ? 0u : lane_gene_masks[lane] & ((1u << local) - 1u);
                    source_positions[value_cursor++] = lane_value_begins[lane]
                        + static_cast<u32>(__builtin_popcount(lower));
                }
                value_offsets[++record_cursor] = value_cursor;
            }
        }
        tile_offsets[tile + 1u] = record_cursor;
    }
    if (record_cursor != header.feature_record_count
        || value_cursor != header.nnz_count) {
        return fail(physical_view_status_code::invalid_geometry,
            "feature-major construction did not fill declared geometry");
    }
    return validate_feature_major_projection_payload_host(
        buffer.payload, required.payload_bytes,
        request.structure_identity, request.structure_epoch_value,
        request.projection_identity, request.runtime_structure,
        request.runtime_projection, out);
}

physical_view_status validate_feature_major_projection_payload_host(
    const void *payload,
    std::size_t payload_bytes,
    execution::structure_id expected_structure,
    execution::structure_epoch expected_epoch,
    execution::projection_id expected_projection,
    execution::structure_handle runtime_structure,
    execution::projection_handle runtime_projection,
    feature_major_projection_view *out) noexcept {
    if (payload == nullptr || out == nullptr
        || !execution::valid_identity(expected_structure)
        || expected_epoch.value == 0u
        || !execution::valid_identity(expected_projection)
        || !execution::valid_handle(runtime_structure)
        || !execution::valid_handle(runtime_projection)
        || payload_bytes < sizeof(feature_major_projection_payload_header)) {
        return fail(physical_view_status_code::invalid_argument,
            "feature-major validation arguments are invalid");
    }
    feature_major_projection_payload_header header{};
    std::memcpy(&header, payload, sizeof(header));
    if (header.schema_version != feature_major_projection_schema_version
        || header.payload_kind != feature_major_projection_payload_kind
        || header.header_bytes != sizeof(header)
        || header.alignment != feature_major_projection_alignment
        || header.payload_bytes != payload_bytes
        || !same_location_identity(header.structure_identity, expected_structure)
        || header.structure_epoch != expected_epoch.value
        || !execution::same_identity(
            header.projection_identity, expected_projection)
        || header.source_payload_identity == 0u
        || header.feature_block_geometry_identity == 0u
        || header.ordering_identity == 0u
        || header.row_domain_identity == 0u
        || header.feature_axis_fingerprint == 0u
        || header.feature_axis_fingerprint_version == 0u
        || header.tile_row_width == 0u || header.tile_row_width > 32u
        || header.tile_count != header.row_count / header.tile_row_width
            + (header.row_count % header.tile_row_width != 0u ? 1u : 0u)
        || header.value_size_bytes == 0u) {
        return fail(physical_view_status_code::incompatible_identity,
            "feature-major payload identity or metadata is incompatible");
    }
    const u64 offsets[5] = {header.tile_feature_offsets_offset,
        header.execution_feature_ids_offset,
        header.participating_row_masks_offset,
        header.feature_value_offsets_offset,
        header.source_value_positions_offset};
    for (u64 offset : offsets)
        if (!offset_aligned(offset)) {
            return fail(physical_view_status_code::invalid_geometry,
                "feature-major array offset is misaligned");
        }
    if (!range_valid(header.tile_feature_offsets_offset,
            static_cast<std::size_t>(header.tile_count) + 1u,
            sizeof(u32), payload_bytes)
        || !range_valid(header.execution_feature_ids_offset,
            header.feature_record_count, sizeof(u32), payload_bytes)
        || !range_valid(header.participating_row_masks_offset,
            header.feature_record_count, sizeof(u32), payload_bytes)
        || !range_valid(header.feature_value_offsets_offset,
            static_cast<std::size_t>(header.feature_record_count) + 1u,
            sizeof(u32), payload_bytes)
        || !range_valid(header.source_value_positions_offset,
            header.nnz_count, sizeof(u32), payload_bytes)) {
        return fail(physical_view_status_code::invalid_geometry,
            "feature-major array range exceeds its payload");
    }

    feature_major_projection_view view{};
    set_view(header, payload, runtime_structure, runtime_projection, &view);
    if (view.tile_feature_offsets[0] != 0u
        || view.tile_feature_offsets[header.tile_count]
            != header.feature_record_count
        || view.feature_value_offsets[0] != 0u
        || view.feature_value_offsets[header.feature_record_count]
            != header.nnz_count) {
        return fail(physical_view_status_code::invalid_geometry,
            "feature-major terminal offsets are invalid");
    }
    for (u32 tile = 0u; tile < header.tile_count; ++tile) {
        const u32 begin = view.tile_feature_offsets[tile];
        const u32 end = view.tile_feature_offsets[tile + 1u];
        if (end < begin || end > header.feature_record_count) {
            return fail(physical_view_status_code::invalid_geometry,
                "feature-major tile offsets are not monotonic");
        }
        const u32 remaining = header.row_count - tile * header.tile_row_width;
        const u32 lanes = std::min(header.tile_row_width, remaining);
        const u32 valid_rows = lanes == 32u
            ? std::numeric_limits<u32>::max() : (1u << lanes) - 1u;
        u32 previous_feature = std::numeric_limits<u32>::max();
        for (u32 record = begin; record < end; ++record) {
            const u32 feature = view.execution_feature_ids[record];
            const u32 mask = view.participating_row_masks[record];
            const u32 value_begin = view.feature_value_offsets[record];
            const u32 value_end = view.feature_value_offsets[record + 1u];
            if (feature >= header.feature_count
                || (record != begin && feature <= previous_feature)
                || mask == 0u || (mask & ~valid_rows) != 0u
                || value_end < value_begin || value_end > header.nnz_count
                || value_end - value_begin
                    != static_cast<u32>(__builtin_popcount(mask))) {
                return fail(physical_view_status_code::invalid_geometry,
                    "feature-major record geometry is invalid");
            }
            previous_feature = feature;
        }
    }
    std::vector<unsigned char> seen(header.nnz_count, 0u);
    for (u32 value = 0u; value < header.nnz_count; ++value) {
        const u32 source = view.source_value_positions[value];
        if (source >= header.nnz_count || seen[source] != 0u) {
            return fail(physical_view_status_code::invalid_geometry,
                "feature-major source-value map is not a permutation");
        }
        seen[source] = 1u;
    }
    *out = view;
    return {};
}

physical_view_status rebind_feature_major_projection(
    const feature_major_projection_view &validated_host_view,
    const void *new_payload_base,
    std::size_t new_payload_bytes,
    feature_major_projection_view *out) noexcept {
    const auto &header = validated_host_view.header;
    if (out == nullptr || new_payload_base == nullptr
        || validated_host_view.payload_base == nullptr
        || header.schema_version != feature_major_projection_schema_version
        || header.payload_bytes != new_payload_bytes
        || !execution::valid_handle(validated_host_view.runtime_structure)
        || !execution::valid_handle(validated_host_view.runtime_projection)) {
        return fail(physical_view_status_code::invalid_argument,
            "feature-major rebind requires an equal validated payload");
    }
    set_view(header, new_payload_base,
        validated_host_view.runtime_structure,
        validated_host_view.runtime_projection, out);
    return {};
}

physical_view_status pack_feature_major_values_host(
    const feature_major_projection_view &projection,
    const void *source_values,
    std::size_t source_value_bytes,
    const feature_major_value_buffers &buffers) noexcept {
    const auto &header = projection.header;
    if (projection.payload_base == nullptr
        || header.schema_version != feature_major_projection_schema_version
        || header.value_size_bytes == 0u) {
        return fail(physical_view_status_code::invalid_argument,
            "feature-major value packing requires a validated host projection");
    }
    std::size_t required = 0u;
    if (!multiply_size(header.nnz_count, header.value_size_bytes, &required)) {
        return fail(physical_view_status_code::overflow,
            "feature-major value bytes overflow");
    }
    if ((required != 0u && source_values == nullptr)
        || source_value_bytes < required || buffers.capacity_bytes < required
        || (required != 0u && buffers.values == nullptr)
        || (required != 0u && buffers.values == source_values)) {
        return fail(physical_view_status_code::insufficient_capacity,
            "feature-major value buffers are missing, aliased, or too small");
    }
    const auto *source = static_cast<const unsigned char *>(source_values);
    auto *target = static_cast<unsigned char *>(buffers.values);
    for (u32 value = 0u; value < header.nnz_count; ++value) {
        const u32 source_position = projection.source_value_positions[value];
        std::memcpy(target + static_cast<std::size_t>(value)
                * header.value_size_bytes,
            source + static_cast<std::size_t>(source_position)
                * header.value_size_bytes,
            header.value_size_bytes);
    }
    return {};
}

} // namespace cellerator::compute::math
