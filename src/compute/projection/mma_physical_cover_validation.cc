#include <Cellerator/compute/projection/physical_mma_hybrid.hh>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace cellerator::compute::projection {
namespace {

constexpr std::uint64_t image_alignment = 64u;

bool add_u64(std::uint64_t lhs, std::uint64_t rhs,
    std::uint64_t *out) noexcept {
    if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) return false;
    *out = lhs + rhs;
    return true;
}

bool multiply_u64(std::uint64_t lhs, std::uint64_t rhs,
    std::uint64_t *out) noexcept {
    if (lhs != 0u && rhs > std::numeric_limits<std::uint64_t>::max() / lhs)
        return false;
    *out = lhs * rhs;
    return true;
}

bool zero_words(const std::uint32_t *words, std::size_t count) noexcept {
    for (std::size_t i = 0u; i < count; ++i)
        if (words[i] != 0u) return false;
    return true;
}

bool section_range(std::uint64_t offset, std::uint64_t count,
    std::uint64_t element_bytes, std::uint64_t image_bytes,
    std::uint64_t *end) noexcept {
    if (count == 0u) {
        *end = 0u;
        return offset == 0u;
    }
    std::uint64_t bytes = 0u;
    return offset >= sizeof(physical_mma_hybrid_header_v1)
        && offset % image_alignment == 0u
        && multiply_u64(count, element_bytes, &bytes)
        && add_u64(offset, bytes, end)
        && *end <= image_bytes;
}

template<typename T>
bool read_record(const unsigned char *image, std::uint64_t offset,
    std::uint64_t index, T *out) noexcept {
    std::uint64_t delta = 0u, position = 0u;
    if (!multiply_u64(index, sizeof(T), &delta)
        || !add_u64(offset, delta, &position))
        return false;
    std::memcpy(out, image + position, sizeof(T));
    return true;
}

std::uint32_t population_count(const mma_tile_v1 &tile) noexcept {
    std::uint32_t result = 0u;
    for (std::uint64_t word : tile.occupancy_mask) {
        while (word != 0u) {
            word &= word - 1u;
            ++result;
        }
    }
    return result;
}

bool valid_identity(const math::sparse_structure_identity &identity) noexcept {
    return identity.schema_version == math::sparse_structure_identity_schema_version
        && identity.identity_version != 0u && identity.value != 0u;
}

bool valid_order(const math::feature_order_identity &identity) noexcept {
    return identity.schema_version == math::feature_order_identity_schema_version
        && (identity.kind == math::feature_order_kind::canonical
            || identity.kind == math::feature_order_kind::packed)
        && identity.feature_count != 0u
        && identity.feature_axis_identity_version != 0u
        && identity.feature_axis_identity != 0u;
}

} // namespace

bool validate_physical_mma_hybrid_image_v1(
    const void *data, std::size_t bytes) noexcept {
    if (data == nullptr || bytes < sizeof(physical_mma_hybrid_header_v1))
        return false;
    const auto *image = static_cast<const unsigned char *>(data);
    physical_mma_hybrid_header_v1 header{};
    std::memcpy(&header, image, sizeof(header));
    if (header.schema_version != physical_mma_hybrid_schema_version_v1
        || header.header_bytes != sizeof(header)
        || header.image_bytes != bytes
        || !valid_identity(header.structure_identity)
        || !valid_order(header.source_order)
        || !valid_order(header.destination_order)
        || (header.provider_identity_low == 0u
            && header.provider_identity_high == 0u)
        || (header.capability_identity_low == 0u
            && header.capability_identity_high == 0u)
        || header.logical_edge_count == 0u
        || header.logical_edge_count > std::numeric_limits<std::uint32_t>::max()
        || header.dense_width == 0u
        || !valid_logical_edge_id_width_v1(header.logical_edge_id_width)
        || (header.logical_edge_id_width == logical_edge_id_width_v1::u32
            && header.logical_edge_count
                > std::numeric_limits<std::uint32_t>::max())
        || header.source_group_count == 0u
        || header.destination_group_count == 0u
        || header.schedule_entry_count == 0u
        || header.value_map_count != header.logical_edge_count
        || !zero_words(header.reserved, 8u))
        return false;
    for (std::uint8_t value : header.reserved0)
        if (value != 0u) return false;

    std::uint64_t ends[9]{};
    std::uint64_t residual_row_count = header.residual_region_count;
    std::uint64_t residual_column_count = 0u;
    if (!section_range(header.source_group_offset, header.source_group_count,
            sizeof(physical_group_v1), bytes, &ends[0])
        || !section_range(header.destination_group_offset,
            header.destination_group_count, sizeof(physical_group_v1), bytes,
            &ends[1])
        || !section_range(header.tile_offset, header.tile_count,
            sizeof(mma_tile_v1), bytes, &ends[2])
        || !section_range(header.compact_slot_offset, header.compact_slot_count,
            sizeof(mma_compact_slot_v1), bytes, &ends[3])
        || !section_range(header.residual_region_offset,
            header.residual_region_count, sizeof(residual_region_v1), bytes,
            &ends[4])
        || !section_range(header.schedule_entry_offset,
            header.schedule_entry_count, sizeof(projection_schedule_entry_v1),
            bytes, &ends[7])
        || !section_range(header.value_map_offset, header.value_map_count,
            sizeof(projection_value_map_v1), bytes, &ends[8]))
        return false;

    for (std::uint32_t index = 0u; index < header.source_group_count; ++index) {
        physical_group_v1 group{};
        if (!read_record(image, header.source_group_offset, index, &group)
            || group.group_id != index || group.semantic_component_id == 0u
            || group.member_count == 0u
            || group.member_count > mma_group_extent_limit_v1
            || group.padded_count < group.member_count
            || group.padded_count > mma_group_extent_limit_v1
            || !zero_words(group.reserved, 3u))
            return false;
    }
    for (std::uint32_t index = 0u; index < header.destination_group_count;
        ++index) {
        physical_group_v1 group{};
        if (!read_record(image, header.destination_group_offset, index, &group)
            || group.group_id != index || group.semantic_component_id == 0u
            || group.member_count == 0u
            || group.member_count > mma_group_extent_limit_v1
            || group.padded_count < group.member_count
            || group.padded_count > mma_group_extent_limit_v1
            || !zero_words(group.reserved, 3u))
            return false;
    }

    for (std::uint32_t index = 0u; index < header.tile_count; ++index) {
        mma_tile_v1 tile{};
        std::uint64_t slot_end = 0u;
        if (!read_record(image, header.tile_offset, index, &tile)
            || tile.tile_id != index || tile.semantic_component_id == 0u
            || tile.source_group_index >= header.source_group_count
            || tile.destination_group_index >= header.destination_group_count
            || tile.compact_slot_count == 0u
            || population_count(tile) != tile.compact_slot_count
            || !add_u64(tile.compact_slot_offset, tile.compact_slot_count,
                &slot_end)
            || slot_end > header.compact_slot_count
            || !zero_words(tile.reserved, 3u))
            return false;
        for (std::uint32_t local = 0u; local < tile.compact_slot_count; ++local) {
            mma_compact_slot_v1 slot{};
            if (!read_record(image, header.compact_slot_offset,
                    tile.compact_slot_offset + local, &slot)
                || slot.row >= mma_group_extent_limit_v1
                || slot.column >= mma_group_extent_limit_v1
                || slot.dense_slot != slot.row * mma_group_extent_limit_v1
                    + slot.column
                || !mma_occupancy_bit_v1(tile, slot.row, slot.column)
                || slot.logical_edge_index >= header.value_map_count)
                return false;
            projection_value_map_v1 map{};
            if (!read_record(image, header.value_map_offset,
                    slot.logical_edge_index, &map)
                || map.region_kind != physical_region_kind_v1::mma
                || map.region_index != index
                || map.projection_slot != slot.dense_slot)
                return false;
        }
    }

    for (std::uint32_t index = 0u; index < header.residual_region_count;
        ++index) {
        residual_region_v1 residual{};
        if (!read_record(image, header.residual_region_offset, index, &residual)
            || residual.region_id != index || residual.semantic_component_id == 0u
            || residual.encoding != residual_encoding_v1::row_owned_csr
            || residual.destination_group_index >= header.destination_group_count
            || residual.row_count == 0u || !zero_words(residual.reserved, 3u)
            || !add_u64(residual_row_count, residual.row_count,
                &residual_row_count)
            || !add_u64(residual_column_count, residual.edge_count,
                &residual_column_count))
            return false;
        for (std::uint8_t value : residual.reserved0)
            if (value != 0u) return false;
    }
    if (!section_range(header.residual_row_offset_offset, residual_row_count,
            sizeof(std::uint32_t), bytes, &ends[5])
        || !section_range(header.residual_column_index_offset,
            residual_column_count, sizeof(std::uint32_t), bytes, &ends[6]))
        return false;
    const std::uint64_t offsets[] = {header.source_group_offset,
        header.destination_group_offset, header.tile_offset,
        header.compact_slot_offset, header.residual_region_offset,
        header.residual_row_offset_offset,
        header.residual_column_index_offset, header.schedule_entry_offset,
        header.value_map_offset};
    for (std::size_t left = 0u; left < 9u; ++left) {
        if (ends[left] == 0u) continue;
        for (std::size_t right = 0u; right < left; ++right)
            if (ends[right] != 0u && offsets[left] < ends[right]
                && offsets[right] < ends[left])
                return false;
    }

    for (std::uint32_t index = 0u; index < header.schedule_entry_count; ++index) {
        projection_schedule_entry_v1 entry{};
        if (!read_record(image, header.schedule_entry_offset, index, &entry)
            || entry.destination_group_index >= header.destination_group_count
            || entry.dense_column_count == 0u
            || entry.dense_column_begin > header.dense_width
            || entry.dense_column_count
                > header.dense_width - entry.dense_column_begin
            || !zero_words(entry.reserved, 3u))
            return false;
        if ((entry.kind == schedule_work_kind_v1::mma_tile
                && entry.work_index >= header.tile_count)
            || (entry.kind == schedule_work_kind_v1::residual_region
                && entry.work_index >= header.residual_region_count)
            || (entry.kind != schedule_work_kind_v1::mma_tile
                && entry.kind != schedule_work_kind_v1::residual_region))
            return false;
        for (std::uint8_t value : entry.reserved0)
            if (value != 0u) return false;
    }

    for (std::uint64_t index = 0u; index < header.value_map_count; ++index) {
        projection_value_map_v1 map{};
        if (!read_record(image, header.value_map_offset, index, &map)
            || map.logical_edge_id.width != header.logical_edge_id_width
            || map.logical_edge_id.value >= header.logical_edge_count
            || !valid_physical_region_kind_v1(map.region_kind)
            || (map.region_kind == physical_region_kind_v1::mma
                && map.region_index >= header.tile_count)
            || (map.region_kind == physical_region_kind_v1::residual
                && map.region_index >= header.residual_region_count)
            || !zero_words(map.reserved, 2u))
            return false;
        for (std::uint8_t value : map.logical_edge_id.reserved)
            if (value != 0u) return false;
        for (std::uint8_t value : map.reserved0)
            if (value != 0u) return false;
        for (std::uint64_t prior = 0u; prior < index; ++prior) {
            projection_value_map_v1 other{};
            if (!read_record(image, header.value_map_offset, prior, &other)
                || other.logical_edge_id.value == map.logical_edge_id.value)
                return false;
        }
        if (map.region_kind == physical_region_kind_v1::mma) {
            mma_tile_v1 tile{};
            bool found = false;
            if (!read_record(image, header.tile_offset, map.region_index, &tile))
                return false;
            for (std::uint32_t local = 0u; local < tile.compact_slot_count;
                ++local) {
                mma_compact_slot_v1 slot{};
                if (!read_record(image, header.compact_slot_offset,
                        tile.compact_slot_offset + local, &slot))
                    return false;
                found = found || (slot.logical_edge_index == index
                    && slot.dense_slot == map.projection_slot);
            }
            if (!found) return false;
        } else {
            residual_region_v1 residual{};
            if (!read_record(image, header.residual_region_offset,
                    map.region_index, &residual)
                || map.projection_slot >= residual.edge_count)
                return false;
        }
    }
    return true;
}

} // namespace cellerator::compute::projection
