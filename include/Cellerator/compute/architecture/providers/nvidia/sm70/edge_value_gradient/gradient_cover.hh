#pragma once
#include <array>
#include <cstdint>
#include <vector>
namespace cellerator::compute::architecture::providers::nvidia::sm70::edge_value_gradient {
constexpr std::uint32_t absent_slot = 0xffffffffu;
struct gradient_edge { std::uint32_t source, destination, physical; };
struct tile_hint {
    std::array<std::uint32_t,16> sources{}, destinations{};
    std::uint32_t source_count=0, destination_count=0;
};
struct gradient_tile {
    tile_hint gather;
    std::array<std::uint32_t,256> physical_slots;
};
struct gradient_cover {
    std::vector<gradient_tile> tiles;
    std::vector<gradient_edge> residual;
    // Indexed by authoritative physical edge; absent_slot means sparse residual.
    std::vector<std::uint32_t> edge_to_tile_slot;
    std::uint64_t persistent_bytes=0, preparation_byte_bound=0;
};
enum class cover_status { success, invalid_argument, capacity_exceeded };
// Cold expected-linear hash lookup and exact O(E + tile slots) ownership marking.
// Input physical slots must be a permutation of [0,E); duplicate endpoints reject.
// Hints gather arbitrary local IDs. Without hints, only occupied 16x16 groups
// with >=128 supported slots become candidates; all other edges stay residual.
// byte_limit bounds flat temporary tables plus conservative vector capacities.
cover_status prepare_gradient_cover(const gradient_edge *edges, std::uint64_t edge_count,
    std::uint32_t source_count, std::uint32_t destination_count,
    const tile_hint *hints, std::uint64_t hint_count, std::uint64_t byte_limit,
    gradient_cover &output) noexcept;
}
