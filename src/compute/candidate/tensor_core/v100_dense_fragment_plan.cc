#include <Cellerator/compute/candidate/tensor_core/v100_dense_fragment_plan.hh>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace cellerator::compute::math::tensor_core {
namespace {

constexpr std::uint32_t extent = 16u;
constexpr std::uint32_t slots = extent * extent;
constexpr std::uint32_t qualification_nnz = slots / 2u;

bool valid_support(const destination_row_csr_support_view &support) noexcept {
    if (support.destination_offsets == nullptr
        || (support.logical_edge_count != 0u
            && support.source_indices == nullptr)
        || support.destination_count == 0u || support.source_count == 0u
        || support.destination_offsets[0] != 0u
        || support.destination_offsets[support.destination_count]
            != support.logical_edge_count)
        return false;
    for (std::uint32_t row = 0u; row < support.destination_count; ++row) {
        const std::uint64_t begin = support.destination_offsets[row];
        const std::uint64_t end = support.destination_offsets[row + 1u];
        if (begin > end || end > support.logical_edge_count) return false;
        std::uint32_t previous = 0u;
        for (std::uint64_t edge = begin; edge < end; ++edge) {
            const std::uint32_t source = support.source_indices[edge];
            if (source >= support.source_count
                || (edge != begin && source <= previous))
                return false;
            previous = source;
        }
    }
    return true;
}

std::uint64_t tile_count(
    const destination_row_csr_support_view &support) noexcept {
    const std::uint64_t row_tiles =
        (support.destination_count + extent - 1u) / extent;
    const std::uint64_t source_tiles =
        (support.source_count + extent - 1u) / extent;
    return row_tiles * source_tiles;
}

dense_fragment_plan_status classify(
    const destination_row_csr_support_view &support,
    v100_dense_fragment_plan_buffers buffers,
    v100_dense_fragment_plan_requirements *requirements) noexcept {
    if (requirements == nullptr || !valid_support(support))
        return dense_fragment_plan_status::invalid_csr;
    const std::uint64_t source_tiles =
        (support.source_count + extent - 1u) / extent;
    const std::uint64_t total_tiles = tile_count(support);
    if (buffers.tile_nnz == nullptr || buffers.tile_capacity < total_tiles)
        return dense_fragment_plan_status::insufficient_capacity;
    std::fill_n(buffers.tile_nnz, total_tiles, std::uint16_t{0u});
    for (std::uint32_t row = 0u; row < support.destination_count; ++row)
        for (std::uint64_t edge = support.destination_offsets[row];
             edge < support.destination_offsets[row + 1u]; ++edge) {
            const std::uint64_t tile = static_cast<std::uint64_t>(row / extent)
                * source_tiles + support.source_indices[edge] / extent;
            ++buffers.tile_nnz[tile];
        }

    v100_dense_fragment_plan_requirements result{};
    result.tile_count = total_tiles;
    for (std::uint32_t row_base = 0u;
         row_base + extent <= support.destination_count; row_base += extent)
        for (std::uint32_t source_base = 0u;
             source_base + extent <= support.source_count;
             source_base += extent) {
            const std::uint64_t tile =
                static_cast<std::uint64_t>(row_base / extent) * source_tiles
                + source_base / extent;
            const std::uint32_t nnz = buffers.tile_nnz[tile];
            result.maximum_tile_nnz = std::max(result.maximum_tile_nnz, nnz);
            if (nnz >= qualification_nnz) {
                ++result.qualified_fragment_count;
                result.packed_slot_count += slots;
            }
        }
    result.residual_edge_count = support.logical_edge_count;
    *requirements = result;
    return dense_fragment_plan_status::ok;
}

} // namespace

dense_fragment_plan_status query_v100_dense_fragment_plan_host(
    const destination_row_csr_support_view &support,
    v100_dense_fragment_plan_buffers scratch,
    v100_dense_fragment_plan_requirements *requirements) noexcept {
    return classify(support, scratch, requirements);
}

dense_fragment_plan_status build_v100_dense_fragment_plan_host(
    const destination_row_csr_support_view &support,
    v100_dense_fragment_plan_buffers buffers,
    v100_dense_fragment_plan_requirements *requirements) noexcept {
    v100_dense_fragment_plan_requirements result{};
    const dense_fragment_plan_status classified =
        classify(support, buffers, &result);
    if (classified != dense_fragment_plan_status::ok) return classified;
    if (buffers.tile_to_fragment == nullptr
        || buffers.tile_capacity < result.tile_count
        || buffers.fragment_capacity < result.qualified_fragment_count
        || (result.qualified_fragment_count != 0u
            && (buffers.fragment_destination_bases == nullptr
                || buffers.fragment_source_bases == nullptr))
        || buffers.logical_edge_to_fragment_slot == nullptr
        || buffers.logical_edge_capacity < support.logical_edge_count
        || (result.packed_slot_count != 0u
            && buffers.fragment_slot_to_logical_edge == nullptr)
        || buffers.packed_slot_capacity < result.packed_slot_count)
        return dense_fragment_plan_status::insufficient_capacity;

    std::fill_n(buffers.tile_to_fragment, result.tile_count, std::int64_t{-1});
    std::fill_n(buffers.logical_edge_to_fragment_slot,
        support.logical_edge_count, invalid_dense_fragment_position);
    std::fill_n(buffers.fragment_slot_to_logical_edge,
        result.packed_slot_count, invalid_dense_fragment_position);
    const std::uint64_t source_tiles =
        (support.source_count + extent - 1u) / extent;
    std::uint64_t fragment = 0u;
    for (std::uint32_t row_base = 0u;
         row_base + extent <= support.destination_count; row_base += extent)
        for (std::uint32_t source_base = 0u;
             source_base + extent <= support.source_count;
             source_base += extent) {
            const std::uint64_t tile =
                static_cast<std::uint64_t>(row_base / extent) * source_tiles
                + source_base / extent;
            if (buffers.tile_nnz[tile] < qualification_nnz) continue;
            buffers.tile_to_fragment[tile] = static_cast<std::int64_t>(fragment);
            buffers.fragment_destination_bases[fragment] = row_base;
            buffers.fragment_source_bases[fragment] = source_base;
            ++fragment;
        }

    std::uint64_t selected_edges = 0u;
    for (std::uint32_t row = 0u; row < support.destination_count; ++row)
        for (std::uint64_t edge = support.destination_offsets[row];
             edge < support.destination_offsets[row + 1u]; ++edge) {
            const std::uint32_t source = support.source_indices[edge];
            const std::uint64_t tile = static_cast<std::uint64_t>(row / extent)
                * source_tiles + source / extent;
            const std::int64_t owner = buffers.tile_to_fragment[tile];
            if (owner < 0) continue;
            const std::uint64_t slot = static_cast<std::uint64_t>(owner) * slots
                + (row % extent) * extent + source % extent;
            buffers.logical_edge_to_fragment_slot[edge] = slot;
            buffers.fragment_slot_to_logical_edge[slot] = edge;
            ++selected_edges;
        }
    result.residual_edge_count = support.logical_edge_count - selected_edges;
    *requirements = result;
    return dense_fragment_plan_status::ok;
}

} // namespace cellerator::compute::math::tensor_core
