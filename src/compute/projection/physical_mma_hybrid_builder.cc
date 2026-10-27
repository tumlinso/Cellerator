#include <Cellerator/compute/projection/physical_mma_hybrid.hh>

#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <tuple>
#include <utility>

namespace cellerator::compute::projection {
namespace {

constexpr std::size_t image_alignment = 64u;
constexpr std::uint32_t group_width = mma_group_extent_limit_v1;

bool align_image(std::vector<std::uint8_t>* image) {
    const std::size_t remainder = image->size() % image_alignment;
    if (remainder == 0u) return true;
    const std::size_t padding = image_alignment - remainder;
    if (padding > image->max_size() - image->size()) return false;
    image->resize(image->size() + padding, 0u);
    return true;
}

template<class T>
std::uint64_t append_records(std::vector<std::uint8_t>* image,
                             const std::vector<T>& records) {
    if (records.empty()) return 0u;
    if (!align_image(image)
        || records.size() > (image->max_size() - image->size()) / sizeof(T))
        return std::numeric_limits<std::uint64_t>::max();
    const auto offset = static_cast<std::uint64_t>(image->size());
    const auto old_size = image->size();
    image->resize(old_size + records.size() * sizeof(T));
    std::memcpy(image->data() + old_size, records.data(),
                records.size() * sizeof(T));
    return offset;
}

bool valid_identity(const math::sparse_structure_identity& identity) noexcept {
    return identity.schema_version == math::sparse_structure_identity_schema_version
        && identity.identity_version != 0u && identity.value != 0u;
}

bool valid_order(const math::feature_order_identity& identity,
                 std::uint32_t count) noexcept {
    return identity.schema_version == math::feature_order_identity_schema_version
        && identity.kind == math::feature_order_kind::canonical
        && identity.feature_count == count
        && identity.feature_axis_identity_version != 0u
        && identity.feature_axis_identity != 0u;
}

bool valid_request(const physical_mma_hybrid_build_request_v1& request) noexcept {
    const auto edge_count = request.edge_count;
    if (!valid_identity(request.structure_identity)
        || !valid_order(request.source_order, request.source_count)
        || !valid_order(request.destination_order, request.destination_count)
        || (request.provider_identity_low == 0u
            && request.provider_identity_high == 0u)
        || (request.capability_identity_low == 0u
            && request.capability_identity_high == 0u)
        || edge_count == 0u || edge_count > std::numeric_limits<std::uint32_t>::max()
        || request.source_indices == nullptr || request.destination_indices == nullptr
        || request.source_count == 0u || request.destination_count == 0u
        || request.dense_width == 0u || request.dense_width > 64u)
        return false;
    for (std::size_t i = 0; i < edge_count; ++i) {
        if (request.source_indices[i] >= request.source_count
            || request.destination_indices[i] >= request.destination_count)
            return false;
    }
    return true;
}

} // namespace

physical_image_build_status_v1 build_physical_mma_hybrid_image_v1(
    const physical_mma_hybrid_build_request_v1& request,
    std::vector<std::uint8_t>* output) noexcept {
    if (output == nullptr || !valid_request(request))
        return physical_image_build_status_v1::invalid_argument;
    try {
        const auto edge_count = static_cast<std::size_t>(request.edge_count);
        const std::uint32_t source_group_count =
            (request.source_count - 1u) / group_width + 1u;
        const std::uint32_t destination_group_count =
            (request.destination_count - 1u) / group_width + 1u;

        std::vector<physical_group_v1> source_groups(source_group_count);
        std::vector<physical_group_v1> destination_groups(destination_group_count);
        for (std::uint32_t group = 0u; group < source_group_count; ++group) {
            auto& record = source_groups[group];
            record.group_id = group;
            record.semantic_component_id = 1u;
            record.member_offset = group * group_width;
            record.member_count = static_cast<std::uint16_t>(std::min(
                group_width, request.source_count - group * group_width));
            record.padded_count = group_width;
        }
        for (std::uint32_t group = 0u; group < destination_group_count; ++group) {
            auto& record = destination_groups[group];
            record.group_id = group;
            record.semantic_component_id = 1u;
            record.member_offset = group * group_width;
            record.member_count = static_cast<std::uint16_t>(std::min(
                group_width, request.destination_count - group * group_width));
            record.padded_count = group_width;
        }

        // The first occurrence of an endpoint pair occupies the N64 tile. Any
        // repeated logical relation for that pair remains an explicit residual.
        std::map<std::pair<std::uint32_t, std::uint32_t>, std::size_t> first_pair;
        std::vector<std::uint8_t> residual_edge(edge_count, 0u);
        std::map<std::pair<std::uint32_t, std::uint32_t>, std::uint32_t> tile_ids;
        for (std::size_t edge = 0u; edge < edge_count; ++edge) {
            const auto pair = std::pair{request.source_indices[edge],
                                        request.destination_indices[edge]};
            if (!first_pair.emplace(pair, edge).second) {
                residual_edge[edge] = 1u;
                continue;
            }
            const auto tile_key = std::pair{pair.first / group_width,
                                            pair.second / group_width};
            tile_ids.emplace(tile_key, 0u);
        }
        std::uint32_t tile_id = 0u;
        for (auto& item : tile_ids) item.second = tile_id++;

        std::vector<std::vector<std::size_t>> tile_edges(tile_ids.size());
        std::map<std::uint32_t, std::vector<std::size_t>> residual_by_destination_group;
        for (std::size_t edge = 0u; edge < edge_count; ++edge) {
            const auto source = request.source_indices[edge];
            const auto destination = request.destination_indices[edge];
            if (residual_edge[edge]) {
                residual_by_destination_group[destination / group_width].push_back(edge);
            } else {
                const auto key = std::pair{source / group_width,
                                           destination / group_width};
                tile_edges[tile_ids.at(key)].push_back(edge);
            }
        }

        std::vector<mma_tile_v1> tiles(tile_ids.size());
        std::vector<mma_compact_slot_v1> compact_slots;
        std::vector<projection_value_map_v1> value_maps(edge_count);
        std::vector<std::uint64_t> value_map_index(edge_count,
            std::numeric_limits<std::uint64_t>::max());
        std::uint32_t slot_offset = 0u;
        for (const auto& [key, index] : tile_ids) {
            auto& tile = tiles[index];
            tile.tile_id = index;
            tile.source_group_index = key.first;
            tile.destination_group_index = key.second;
            tile.semantic_component_id = 1u;
            tile.compact_slot_offset = slot_offset;
            tile.value_map_offset = 0u;
            for (const auto edge : tile_edges[index]) {
                const auto row = request.destination_indices[edge] % group_width;
                const auto column = request.source_indices[edge] % group_width;
                const auto dense_slot = row * group_width + column;
                const auto mask_bit = dense_slot;
                tile.occupancy_mask[mask_bit / 64u] |=
                    std::uint64_t{1u} << (mask_bit % 64u);
                mma_compact_slot_v1 slot{};
                slot.row = static_cast<std::uint8_t>(row);
                slot.column = static_cast<std::uint8_t>(column);
                slot.dense_slot = static_cast<std::uint16_t>(dense_slot);
                slot.logical_edge_index = static_cast<std::uint32_t>(edge);
                compact_slots.push_back(slot);
                auto& map = value_maps[edge];
                map.logical_edge_id.value = edge;
                map.region_kind = physical_region_kind_v1::mma;
                map.region_index = index;
                map.projection_slot = dense_slot;
                ++slot_offset;
            }
            tile.compact_slot_count = static_cast<std::uint16_t>(
                slot_offset - tile.compact_slot_offset);
        }

        std::vector<residual_region_v1> residuals;
        std::vector<std::uint32_t> residual_row_offsets;
        std::vector<std::uint32_t> residual_columns;
        // MMA value-map positions are compacted in tile/slot order; rewrite
        // slot references after the physical order has been assigned.
        std::uint32_t next_map = 0u;
        for (std::uint32_t index = 0u; index < tiles.size(); ++index) {
            auto& tile = tiles[index];
            tile.value_map_offset = next_map;
            const auto& edges = tile_edges[index];
            for (std::size_t local = 0u; local < edges.size(); ++local) {
                const auto edge = edges[local];
                value_map_index[edge] = next_map;
                value_maps[edge].logical_edge_id.value = edge;
                compact_slots[tile.compact_slot_offset + local].logical_edge_index = next_map;
                ++next_map;
            }
        }

        for (auto& [destination_group, edges] : residual_by_destination_group) {
            std::sort(edges.begin(), edges.end(), [&](std::size_t left, std::size_t right) {
                return std::tuple{request.destination_indices[left] % group_width,
                                  request.source_indices[left] / group_width,
                                  request.source_indices[left] % group_width,
                                  left}
                    < std::tuple{request.destination_indices[right] % group_width,
                                  request.source_indices[right] / group_width,
                                  request.source_indices[right] % group_width,
                                  right};
            });
            residual_region_v1 region{};
            region.region_id = static_cast<std::uint32_t>(residuals.size());
            region.semantic_component_id = 1u;
            region.destination_group_index = destination_group;
            region.row_offset_index = static_cast<std::uint32_t>(residual_row_offsets.size());
            region.row_count = destination_groups[destination_group].member_count;
            region.column_index_offset = static_cast<std::uint32_t>(residual_columns.size());
            region.edge_count = static_cast<std::uint32_t>(edges.size());
            region.value_map_offset = next_map;
            residual_row_offsets.push_back(0u);
            std::size_t cursor = 0u;
            for (std::uint32_t row = 0u; row < region.row_count; ++row) {
                while (cursor < edges.size()
                    && request.destination_indices[edges[cursor]] % group_width == row) {
                    const auto edge = edges[cursor];
                    const auto source = request.source_indices[edge];
                    residual_columns.push_back(source / group_width * group_width
                        + source % group_width);
                    auto& map = value_maps[edge];
                    map.logical_edge_id.value = edge;
                    map.region_kind = physical_region_kind_v1::residual;
                    map.region_index = region.region_id;
                    map.projection_slot = static_cast<std::uint32_t>(cursor);
                    value_map_index[edge] = next_map++;
                    ++cursor;
                }
                residual_row_offsets.push_back(static_cast<std::uint32_t>(cursor));
            }
            if (cursor != edges.size())
                return physical_image_build_status_v1::invalid_argument;
            residuals.push_back(region);
        }
        if (next_map != edge_count)
            return physical_image_build_status_v1::invalid_argument;
        for (std::size_t i = 0u; i < edge_count; ++i)
            value_maps[i].logical_edge_id.value = i;
        // Image map order is physical order, independent of support ordinal.
        std::vector<projection_value_map_v1> ordered_maps(edge_count);
        for (std::size_t i = 0u; i < edge_count; ++i)
            ordered_maps[value_map_index[i]] = value_maps[i];

        std::vector<projection_schedule_entry_v1> schedule;
        schedule.reserve(tiles.size() + residuals.size());
        for (const auto& tile : tiles) {
            projection_schedule_entry_v1 entry{};
            entry.kind = schedule_work_kind_v1::mma_tile;
            entry.work_index = tile.tile_id;
            entry.destination_group_index = tile.destination_group_index;
            entry.dense_column_count = request.dense_width;
            schedule.push_back(entry);
        }
        for (const auto& region : residuals) {
            projection_schedule_entry_v1 entry{};
            entry.kind = schedule_work_kind_v1::residual_region;
            entry.work_index = region.region_id;
            entry.destination_group_index = region.destination_group_index;
            entry.dense_column_count = request.dense_width;
            schedule.push_back(entry);
        }

        std::vector<std::uint8_t> image(sizeof(physical_mma_hybrid_header_v1), 0u);
        physical_mma_hybrid_header_v1 header{};
        header.structure_identity = request.structure_identity;
        header.source_order = request.source_order;
        header.destination_order = request.destination_order;
        header.provider_identity_low = request.provider_identity_low;
        header.provider_identity_high = request.provider_identity_high;
        header.capability_identity_low = request.capability_identity_low;
        header.capability_identity_high = request.capability_identity_high;
        header.logical_edge_count = edge_count;
        header.dense_width = request.dense_width;
        header.logical_edge_id_width = logical_edge_id_width_v1::u32;
        header.source_group_count = source_group_count;
        header.destination_group_count = destination_group_count;
        header.tile_count = static_cast<std::uint32_t>(tiles.size());
        header.compact_slot_count = static_cast<std::uint32_t>(compact_slots.size());
        header.residual_region_count = static_cast<std::uint32_t>(residuals.size());
        header.schedule_entry_count = static_cast<std::uint32_t>(schedule.size());
        header.value_map_count = edge_count;
        header.source_group_offset = append_records(&image, source_groups);
        header.destination_group_offset = append_records(&image, destination_groups);
        header.tile_offset = append_records(&image, tiles);
        header.compact_slot_offset = append_records(&image, compact_slots);
        header.residual_region_offset = append_records(&image, residuals);
        header.residual_row_offset_offset = append_records(&image, residual_row_offsets);
        header.residual_column_index_offset = append_records(&image, residual_columns);
        header.schedule_entry_offset = append_records(&image, schedule);
        header.value_map_offset = append_records(&image, ordered_maps);
        const std::uint64_t offsets[] = {header.source_group_offset,
            header.destination_group_offset, header.tile_offset,
            header.compact_slot_offset, header.residual_region_offset,
            header.residual_row_offset_offset, header.residual_column_index_offset,
            header.schedule_entry_offset, header.value_map_offset};
        if (std::find(offsets, offsets + 9,
                std::numeric_limits<std::uint64_t>::max()) != offsets + 9)
            return physical_image_build_status_v1::arithmetic_overflow;
        header.image_bytes = image.size();
        std::memcpy(image.data(), &header, sizeof(header));
        if (!validate_physical_mma_hybrid_image_v1(image.data(), image.size()))
            return physical_image_build_status_v1::invalid_argument;
        *output = std::move(image);
        return physical_image_build_status_v1::success;
    } catch (...) {
        return physical_image_build_status_v1::allocation_failure;
    }
}

} // namespace cellerator::compute::projection
