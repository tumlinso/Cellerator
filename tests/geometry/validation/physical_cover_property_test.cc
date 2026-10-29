#include <Cellerator/compute/projection/physical_mma_hybrid.hh>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

namespace projection = cellerator::compute::projection;

namespace cellerator::compute::projection {
bool validate_physical_mma_hybrid_image_v1(
    const void *, std::size_t) noexcept;
}

namespace {

std::size_t align64(std::size_t value) {
    return (value + 63u) & ~std::size_t{63u};
}

template<typename T>
std::size_t append_records(std::vector<unsigned char> *image,
    const T *records, std::size_t count) {
    const std::size_t offset = align64(image->size());
    image->resize(offset + sizeof(T) * count);
    std::memcpy(image->data() + offset, records, sizeof(T) * count);
    return offset;
}

template<typename T>
T load_record(const std::vector<unsigned char> &image,
    std::uint64_t offset, std::uint64_t index = 0u) {
    T result{};
    std::memcpy(&result,
        image.data() + offset + index * sizeof(T), sizeof(result));
    return result;
}

template<typename T>
void store_record(std::vector<unsigned char> *image,
    std::uint64_t offset, const T &record, std::uint64_t index = 0u) {
    std::memcpy(image->data() + offset + index * sizeof(T),
        &record, sizeof(record));
}

struct generated_image {
    std::vector<unsigned char> bytes;
    projection::physical_mma_hybrid_header_v1 header{};
    std::uint32_t mma_edges = 0u;
    std::uint32_t residual_edges = 0u;
};

generated_image make_valid_image(std::mt19937 &random,
    std::uint32_t dense_width) {
    std::uniform_int_distribution<std::uint32_t> edge_distribution(2u, 96u);
    const std::uint32_t edge_count = edge_distribution(random);
    std::uniform_int_distribution<std::uint32_t> mma_distribution(
        1u, std::min(edge_count - 1u, 64u));
    const std::uint32_t mma_edge_count = mma_distribution(random);
    const std::uint32_t residual_edge_count = edge_count - mma_edge_count;

    projection::physical_group_v1 source{};
    source.group_id = 0u;
    source.semantic_component_id = 1u;
    source.member_count = static_cast<std::uint16_t>(1u + random() % 16u);
    source.padded_count = 16u;
    projection::physical_group_v1 destination = source;
    destination.semantic_component_id = 2u;

    std::vector<std::uint32_t> dense_slots(256u);
    std::iota(dense_slots.begin(), dense_slots.end(), 0u);
    std::shuffle(dense_slots.begin(), dense_slots.end(), random);
    dense_slots.resize(mma_edge_count);
    std::sort(dense_slots.begin(), dense_slots.end());
    projection::mma_tile_v1 tile{};
    tile.tile_id = 0u;
    tile.source_group_index = 0u;
    tile.destination_group_index = 0u;
    tile.semantic_component_id = 1u;
    tile.compact_slot_count = static_cast<std::uint16_t>(mma_edge_count);
    std::vector<projection::mma_compact_slot_v1> compact_slots(mma_edge_count);
    for (std::uint32_t index = 0u; index < mma_edge_count; ++index) {
        const std::uint32_t dense_slot = dense_slots[index];
        tile.occupancy_mask[dense_slot / 64u] |=
            std::uint64_t{1u} << (dense_slot % 64u);
        compact_slots[index].row = static_cast<std::uint8_t>(dense_slot / 16u);
        compact_slots[index].column =
            static_cast<std::uint8_t>(dense_slot % 16u);
        compact_slots[index].dense_slot =
            static_cast<std::uint16_t>(dense_slot);
        compact_slots[index].logical_edge_index = index;
    }

    projection::residual_region_v1 residual{};
    residual.region_id = 0u;
    residual.semantic_component_id = 2u;
    residual.destination_group_index = 0u;
    residual.row_count = static_cast<std::uint32_t>(
        std::min<std::uint32_t>(16u, residual_edge_count));
    residual.edge_count = residual_edge_count;
    residual.value_map_offset = mma_edge_count;
    std::vector<std::uint32_t> row_offsets(residual.row_count + 1u, 0u);
    for (std::uint32_t row = 0u; row <= residual.row_count; ++row)
        row_offsets[row] = static_cast<std::uint32_t>(
            (static_cast<std::uint64_t>(row) * residual_edge_count)
            / residual.row_count);
    std::vector<std::uint32_t> columns(residual_edge_count);
    for (std::uint32_t &column : columns) column = random() % 16u;

    projection::projection_schedule_entry_v1 schedules[2]{};
    schedules[0].kind = projection::schedule_work_kind_v1::mma_tile;
    schedules[0].work_index = 0u;
    schedules[0].destination_group_index = 0u;
    schedules[0].dense_column_count = dense_width;
    schedules[1].kind = projection::schedule_work_kind_v1::residual_region;
    schedules[1].work_index = 0u;
    schedules[1].destination_group_index = 0u;
    schedules[1].dense_column_count = dense_width;

    std::vector<std::uint64_t> logical_edges(edge_count);
    std::iota(logical_edges.begin(), logical_edges.end(), 0u);
    std::shuffle(logical_edges.begin(), logical_edges.end(), random);
    const auto width = (random() & 1u) == 0u
        ? projection::logical_edge_id_width_v1::u32
        : projection::logical_edge_id_width_v1::u64;
    std::vector<projection::projection_value_map_v1> maps(edge_count);
    for (std::uint32_t index = 0u; index < edge_count; ++index) {
        maps[index].logical_edge_id.value = logical_edges[index];
        maps[index].logical_edge_id.width = width;
        if (index < mma_edge_count) {
            maps[index].region_kind = projection::physical_region_kind_v1::mma;
            maps[index].region_index = 0u;
            maps[index].projection_slot = compact_slots[index].dense_slot;
        } else {
            maps[index].region_kind =
                projection::physical_region_kind_v1::residual;
            maps[index].region_index = 0u;
            maps[index].projection_slot = index - mma_edge_count;
        }
    }

    generated_image result{};
    result.bytes.resize(sizeof(projection::physical_mma_hybrid_header_v1));
    auto &header = result.header;
    header.structure_identity.identity_version = 1u;
    header.structure_identity.value = 0x101u;
    header.source_order.feature_count = 16u;
    header.source_order.feature_axis_identity_version = 1u;
    header.source_order.feature_axis_identity = 0x201u;
    header.destination_order.feature_count = 16u;
    header.destination_order.feature_axis_identity_version = 1u;
    header.destination_order.feature_axis_identity = 0x301u;
    header.provider_identity_low = 0x401u;
    header.capability_identity_low = 0x501u;
    header.logical_edge_count = edge_count;
    header.dense_width = dense_width;
    header.logical_edge_id_width = width;
    header.source_group_count = 1u;
    header.destination_group_count = 1u;
    header.tile_count = 1u;
    header.compact_slot_count = mma_edge_count;
    header.residual_region_count = 1u;
    header.schedule_entry_count = 2u;
    header.value_map_count = edge_count;
    header.source_group_offset = append_records(&result.bytes, &source, 1u);
    header.destination_group_offset =
        append_records(&result.bytes, &destination, 1u);
    header.tile_offset = append_records(&result.bytes, &tile, 1u);
    header.compact_slot_offset = append_records(
        &result.bytes, compact_slots.data(), compact_slots.size());
    header.residual_region_offset =
        append_records(&result.bytes, &residual, 1u);
    header.residual_row_offset_offset = append_records(
        &result.bytes, row_offsets.data(), row_offsets.size());
    header.residual_column_index_offset =
        append_records(&result.bytes, columns.data(), columns.size());
    header.schedule_entry_offset =
        append_records(&result.bytes, schedules, 2u);
    header.value_map_offset =
        append_records(&result.bytes, maps.data(), maps.size());
    header.image_bytes = result.bytes.size();
    store_record(&result.bytes, 0u, header);
    result.mma_edges = mma_edge_count;
    result.residual_edges = residual_edge_count;
    return result;
}

void require_rejected(const std::vector<unsigned char> &image) {
    assert(!projection::validate_physical_mma_hybrid_image_v1(
        image.data(), image.size()));
}

void independently_verify_exact_ownership(const generated_image &generated) {
    const auto &header = generated.header;
    std::vector<std::uint8_t> owners(header.logical_edge_count, 0u);
    for (std::uint64_t index = 0u; index < header.value_map_count; ++index) {
        const auto map = load_record<projection::projection_value_map_v1>(
            generated.bytes, header.value_map_offset, index);
        assert(map.logical_edge_id.value < owners.size());
        assert(owners[map.logical_edge_id.value] == 0u);
        owners[map.logical_edge_id.value] = map.region_kind
                == projection::physical_region_kind_v1::mma
            ? 1u
            : 2u;
        if (map.region_kind == projection::physical_region_kind_v1::mma) {
            assert(index < generated.mma_edges);
            const auto slot =
                load_record<projection::mma_compact_slot_v1>(generated.bytes,
                    header.compact_slot_offset, index);
            assert(slot.logical_edge_index == index);
            assert(slot.dense_slot == map.projection_slot);
        } else {
            assert(index >= generated.mma_edges);
            assert(map.projection_slot == index - generated.mma_edges);
        }
    }
    assert(std::all_of(owners.begin(), owners.end(),
        [](std::uint8_t owner) { return owner == 1u || owner == 2u; }));
    assert(static_cast<std::uint32_t>(
               std::count(owners.begin(), owners.end(), std::uint8_t{1u}))
        == generated.mma_edges);
    assert(static_cast<std::uint32_t>(
               std::count(owners.begin(), owners.end(), std::uint8_t{2u}))
        == generated.residual_edges);
}

void verify_mutations_rejected(const generated_image &generated) {
    const auto &header = generated.header;

    std::vector<unsigned char> omitted = generated.bytes;
    auto omitted_map = load_record<projection::projection_value_map_v1>(
        omitted, header.value_map_offset, header.value_map_count - 1u);
    omitted_map.logical_edge_id.value = header.logical_edge_count;
    store_record(&omitted, header.value_map_offset, omitted_map,
        header.value_map_count - 1u);
    require_rejected(omitted);

    std::vector<unsigned char> duplicate = generated.bytes;
    auto duplicate_map = load_record<projection::projection_value_map_v1>(
        duplicate, header.value_map_offset, 1u);
    duplicate_map.logical_edge_id.value =
        load_record<projection::projection_value_map_v1>(duplicate,
            header.value_map_offset, 0u).logical_edge_id.value;
    store_record(&duplicate, header.value_map_offset, duplicate_map, 1u);
    require_rejected(duplicate);

    std::vector<unsigned char> bad_padding = generated.bytes;
    auto group = load_record<projection::physical_group_v1>(
        bad_padding, header.source_group_offset);
    group.padded_count = 17u;
    store_record(&bad_padding, header.source_group_offset, group);
    require_rejected(bad_padding);

    std::vector<unsigned char> bad_occupancy = generated.bytes;
    auto tile = load_record<projection::mma_tile_v1>(
        bad_occupancy, header.tile_offset);
    tile.occupancy_mask[0] ^= 1u;
    store_record(&bad_occupancy, header.tile_offset, tile);
    require_rejected(bad_occupancy);

    std::vector<unsigned char> bad_map = generated.bytes;
    auto map = load_record<projection::projection_value_map_v1>(
        bad_map, header.value_map_offset);
    map.projection_slot = 256u;
    store_record(&bad_map, header.value_map_offset, map);
    require_rejected(bad_map);

    std::vector<unsigned char> bad_residual = generated.bytes;
    const std::uint32_t wrong_terminal = generated.residual_edges - 1u;
    store_record(&bad_residual, header.residual_row_offset_offset,
        wrong_terminal,
        load_record<projection::residual_region_v1>(bad_residual,
            header.residual_region_offset).row_count);
    require_rejected(bad_residual);

    std::vector<unsigned char> corrupt_offset = generated.bytes;
    auto corrupt_header = header;
    ++corrupt_header.value_map_offset;
    store_record(&corrupt_offset, 0u, corrupt_header);
    require_rejected(corrupt_offset);

    std::vector<unsigned char> bad_width = generated.bytes;
    auto width_map = load_record<projection::projection_value_map_v1>(
        bad_width, header.value_map_offset);
    width_map.logical_edge_id.width = header.logical_edge_id_width
            == projection::logical_edge_id_width_v1::u32
        ? projection::logical_edge_id_width_v1::u64
        : projection::logical_edge_id_width_v1::u32;
    store_record(&bad_width, header.value_map_offset, width_map);
    require_rejected(bad_width);

    std::vector<unsigned char> bad_schedule = generated.bytes;
    auto schedule = load_record<projection::projection_schedule_entry_v1>(
        bad_schedule, header.schedule_entry_offset);
    schedule.dense_column_begin = header.dense_width;
    schedule.dense_column_count = 1u;
    store_record(&bad_schedule, header.schedule_entry_offset, schedule);
    require_rejected(bad_schedule);
}

} // namespace

int main() {
    constexpr std::uint32_t seed = 0xc0fe103u;
    constexpr std::uint32_t trials = 384u;
    constexpr std::uint32_t widths[]{1u, 15u, 16u, 17u, 31u,
        32u, 63u, 64u, 65u, 127u, 128u, 129u};
    std::mt19937 random(seed);
    for (std::uint32_t iteration = 0u; iteration < trials; ++iteration) {
        const generated_image generated = make_valid_image(
            random, widths[iteration % (sizeof(widths) / sizeof(widths[0]))]);
        assert(projection::validate_physical_mma_hybrid_image_v1(
            generated.bytes.data(), generated.bytes.size()));
        independently_verify_exact_ownership(generated);
        verify_mutations_rejected(generated);
    }
    std::cout << "physical_cover_property_test passed seed=" << seed
              << " trials=" << trials
              << " width_transitions="
              << sizeof(widths) / sizeof(widths[0])
              << " mutations_per_trial=9\n";
    return 0;
}
