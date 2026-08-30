#pragma once

#include <Cellerator/compute/projection/identity.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellerator::compute::projection {

inline constexpr std::uint32_t physical_mma_hybrid_schema_version_v1 = 1u;
inline constexpr std::uint32_t mma_occupancy_mask_word_count_v1 = 4u;
inline constexpr std::uint16_t mma_group_extent_limit_v1 = 16u;
inline constexpr std::uint32_t invalid_physical_index_v1 = ~0u;

enum class logical_edge_id_width_v1 : std::uint8_t {
    u32 = 1u,
    u64 = 2u
};

enum class physical_region_kind_v1 : std::uint8_t {
    mma = 1u,
    residual = 2u
};

enum class residual_encoding_v1 : std::uint8_t {
    row_owned_csr = 1u
};

enum class schedule_work_kind_v1 : std::uint8_t {
    mma_tile = 1u,
    residual_region = 2u
};

// All offsets are byte offsets from the beginning of the containing image.
// No process pointer, stream, mutable value generation, or device allocation
// enters this immutable physical projection.
struct physical_mma_hybrid_header_v1 {
    std::uint32_t schema_version = physical_mma_hybrid_schema_version_v1;
    std::uint32_t header_bytes = sizeof(physical_mma_hybrid_header_v1);
    std::uint64_t image_bytes = 0u;
    math::sparse_structure_identity structure_identity{};
    math::feature_order_identity source_order{};
    math::feature_order_identity destination_order{};
    std::uint64_t provider_identity_low = 0u;
    std::uint64_t provider_identity_high = 0u;
    std::uint64_t capability_identity_low = 0u;
    std::uint64_t capability_identity_high = 0u;
    std::uint64_t logical_edge_count = 0u;
    std::uint32_t dense_width = 0u;
    logical_edge_id_width_v1 logical_edge_id_width =
        logical_edge_id_width_v1::u32;
    std::uint8_t reserved0[3]{};
    std::uint32_t source_group_count = 0u;
    std::uint32_t destination_group_count = 0u;
    std::uint32_t tile_count = 0u;
    std::uint32_t compact_slot_count = 0u;
    std::uint32_t residual_region_count = 0u;
    std::uint32_t schedule_entry_count = 0u;
    std::uint64_t value_map_count = 0u;
    std::uint64_t source_group_offset = 0u;
    std::uint64_t destination_group_offset = 0u;
    std::uint64_t tile_offset = 0u;
    std::uint64_t compact_slot_offset = 0u;
    std::uint64_t residual_region_offset = 0u;
    std::uint64_t residual_row_offset_offset = 0u;
    std::uint64_t residual_column_index_offset = 0u;
    std::uint64_t schedule_entry_offset = 0u;
    std::uint64_t value_map_offset = 0u;
    std::uint32_t reserved[8]{};
};

struct physical_group_v1 {
    std::uint32_t group_id = invalid_physical_index_v1;
    std::uint32_t semantic_component_id = 0u;
    std::uint32_t member_offset = 0u;
    std::uint16_t member_count = 0u;
    std::uint16_t padded_count = 0u;
    std::uint32_t original_group_id = invalid_physical_index_v1;
    std::uint32_t reserved[3]{};
};

// Tile coordinates are projection-local group indices. Occupancy bit r*16+c
// owns a real logical contribution. Clear bits are padding and have no edge ID.
struct mma_tile_v1 {
    std::uint32_t tile_id = invalid_physical_index_v1;
    std::uint32_t source_group_index = invalid_physical_index_v1;
    std::uint32_t destination_group_index = invalid_physical_index_v1;
    std::uint32_t semantic_component_id = 0u;
    std::uint64_t occupancy_mask[mma_occupancy_mask_word_count_v1]{};
    std::uint32_t compact_slot_offset = 0u;
    std::uint16_t compact_slot_count = 0u;
    std::uint16_t reserved0 = 0u;
    std::uint32_t value_map_offset = 0u;
    std::uint32_t reserved[3]{};
};

// Compact slots are dense-tile positions for occupied mask bits only.
struct mma_compact_slot_v1 {
    std::uint8_t row = 0u;
    std::uint8_t column = 0u;
    std::uint16_t dense_slot = 0u;
    std::uint32_t logical_edge_index = invalid_physical_index_v1;
};

// The tag is explicit even when one image selects a uniform width. This makes
// individual records self-describing during independent validation.
struct width_tagged_logical_edge_id_v1 {
    std::uint64_t value = 0u;
    logical_edge_id_width_v1 width = logical_edge_id_width_v1::u32;
    std::uint8_t reserved[7]{};
};

// The first residual realization is row-owned CSR in the same physical order.
// Its row offsets and column indices live in image sections named by the header.
struct residual_region_v1 {
    std::uint32_t region_id = invalid_physical_index_v1;
    std::uint32_t semantic_component_id = 0u;
    residual_encoding_v1 encoding = residual_encoding_v1::row_owned_csr;
    std::uint8_t reserved0[3]{};
    std::uint32_t destination_group_index = invalid_physical_index_v1;
    std::uint32_t row_offset_index = 0u;
    std::uint32_t row_count = 0u;
    std::uint32_t column_index_offset = 0u;
    std::uint32_t edge_count = 0u;
    std::uint32_t value_map_offset = 0u;
    std::uint32_t reserved[3]{};
};

struct projection_schedule_entry_v1 {
    schedule_work_kind_v1 kind = schedule_work_kind_v1::mma_tile;
    std::uint8_t reserved0[3]{};
    std::uint32_t work_index = invalid_physical_index_v1;
    std::uint32_t destination_group_index = invalid_physical_index_v1;
    std::uint32_t dense_column_begin = 0u;
    std::uint32_t dense_column_count = 0u;
    std::uint32_t reserved[3]{};
};

// Projection slots are immutable structure positions. Mutable value planes use
// this map to repack a new generation without rebuilding physical geometry.
struct projection_value_map_v1 {
    width_tagged_logical_edge_id_v1 logical_edge_id{};
    physical_region_kind_v1 region_kind = physical_region_kind_v1::residual;
    std::uint8_t reserved0[3]{};
    std::uint32_t region_index = invalid_physical_index_v1;
    std::uint32_t projection_slot = invalid_physical_index_v1;
    std::uint32_t reserved[2]{};
};

constexpr bool valid_logical_edge_id_width_v1(
    logical_edge_id_width_v1 width) noexcept {
    return width == logical_edge_id_width_v1::u32
        || width == logical_edge_id_width_v1::u64;
}

constexpr bool valid_physical_region_kind_v1(
    physical_region_kind_v1 kind) noexcept {
    return kind == physical_region_kind_v1::mma
        || kind == physical_region_kind_v1::residual;
}

constexpr bool mma_occupancy_bit_v1(
    const mma_tile_v1 &tile,
    std::uint32_t row,
    std::uint32_t column) noexcept {
    if (row >= mma_group_extent_limit_v1
        || column >= mma_group_extent_limit_v1)
        return false;
    const std::uint32_t bit = row * mma_group_extent_limit_v1 + column;
    return (tile.occupancy_mask[bit / 64u] & (std::uint64_t{1u} << (bit % 64u)))
        != 0u;
}

static_assert(sizeof(width_tagged_logical_edge_id_v1) == 16u,
    "width-tagged logical edge IDs have a fixed wire size");
static_assert(std::is_trivially_copyable<physical_mma_hybrid_header_v1>::value,
    "physical projection headers must remain pointer-free");
static_assert(std::is_trivially_copyable<physical_group_v1>::value,
    "physical groups must remain pointer-free");
static_assert(std::is_trivially_copyable<mma_tile_v1>::value,
    "MMA tiles must remain pointer-free");
static_assert(std::is_trivially_copyable<mma_compact_slot_v1>::value,
    "compact slots must remain pointer-free");
static_assert(std::is_trivially_copyable<residual_region_v1>::value,
    "residual descriptors must remain pointer-free");
static_assert(std::is_trivially_copyable<projection_schedule_entry_v1>::value,
    "projection schedules must remain pointer-free");
static_assert(std::is_trivially_copyable<projection_value_map_v1>::value,
    "projection value maps must remain pointer-free");

} // namespace cellerator::compute::projection
