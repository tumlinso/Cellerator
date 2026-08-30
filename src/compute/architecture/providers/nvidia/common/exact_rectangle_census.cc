#include <cstddef>
#include <cstdint>
#include <limits>

namespace cellerator::compute::architecture::providers::nvidia {

// Owner codes are deliberately data-only: zero is unassigned during the
// census, one is MMA, and two is exact residual contribution.
bool build_exact_rectangle_cover_v1(
    std::uint32_t source_count,
    std::uint32_t destination_count,
    const std::uint64_t *destination_offsets,
    const std::uint32_t *source_ids,
    const std::uint64_t *logical_edge_ids,
    std::uint64_t edge_count,
    const std::uint32_t *source_to_group,
    const std::uint8_t *source_local,
    std::uint32_t source_group_count,
    const std::uint32_t *destination_to_group,
    const std::uint8_t *destination_local,
    std::uint32_t destination_group_count,
    const std::uint8_t *mma_selected,
    std::uint64_t rectangle_capacity,
    std::uint64_t *occupancy_masks,
    std::uint64_t occupancy_word_capacity,
    std::uint32_t *unique_occupancy,
    std::uint8_t *logical_edge_owner,
    std::uint64_t logical_edge_owner_capacity,
    std::uint64_t *mma_edge_count,
    std::uint64_t *residual_edge_count) noexcept {
    if (source_count == 0u || destination_count == 0u
        || destination_offsets == nullptr || source_ids == nullptr
        || logical_edge_ids == nullptr || edge_count == 0u
        || source_to_group == nullptr || source_local == nullptr
        || source_group_count == 0u || destination_to_group == nullptr
        || destination_local == nullptr || destination_group_count == 0u
        || mma_selected == nullptr || occupancy_masks == nullptr
        || unique_occupancy == nullptr || logical_edge_owner == nullptr
        || mma_edge_count == nullptr || residual_edge_count == nullptr
        || source_group_count
            > std::numeric_limits<std::uint64_t>::max()
                / destination_group_count)
        return false;
    const std::uint64_t rectangle_count =
        static_cast<std::uint64_t>(source_group_count)
        * destination_group_count;
    if (rectangle_capacity < rectangle_count
        || rectangle_count > std::numeric_limits<std::uint64_t>::max() / 4u
        || occupancy_word_capacity < rectangle_count * 4u
        || logical_edge_owner_capacity < edge_count
        || destination_offsets[0] != 0u
        || destination_offsets[destination_count] != edge_count)
        return false;
    for (std::uint32_t destination = 0u; destination < destination_count;
        ++destination)
        if (destination_offsets[destination]
            > destination_offsets[destination + 1u])
            return false;
    for (std::uint32_t source = 0u; source < source_count; ++source) {
        if (source_to_group[source] >= source_group_count
            || source_local[source] >= 16u)
            return false;
        for (std::uint32_t prior = 0u; prior < source; ++prior)
            if (source_to_group[prior] == source_to_group[source]
                && source_local[prior] == source_local[source])
                return false;
    }
    for (std::uint32_t destination = 0u; destination < destination_count;
        ++destination) {
        if (destination_to_group[destination] >= destination_group_count
            || destination_local[destination] >= 16u)
            return false;
        for (std::uint32_t prior = 0u; prior < destination; ++prior)
            if (destination_to_group[prior] == destination_to_group[destination]
                && destination_local[prior] == destination_local[destination])
                return false;
    }
    for (std::uint64_t rectangle = 0u; rectangle < rectangle_count;
        ++rectangle) {
        if (mma_selected[rectangle] > 1u) return false;
        unique_occupancy[rectangle] = 0u;
        for (std::uint32_t word = 0u; word < 4u; ++word)
            occupancy_masks[rectangle * 4u + word] = 0u;
    }
    for (std::uint64_t edge = 0u; edge < edge_count; ++edge)
        logical_edge_owner[edge] = 0u;

    std::uint64_t mma_count = 0u;
    std::uint64_t residual_count = 0u;
    for (std::uint32_t destination = 0u; destination < destination_count;
        ++destination) {
        const std::uint32_t destination_group =
            destination_to_group[destination];
        const std::uint32_t destination_position =
            destination_local[destination];
        for (std::uint64_t edge = destination_offsets[destination];
            edge < destination_offsets[destination + 1u]; ++edge) {
            const std::uint32_t source = source_ids[edge];
            const std::uint64_t logical_edge = logical_edge_ids[edge];
            if (source >= source_count || logical_edge >= edge_count
                || logical_edge_owner[logical_edge] != 0u)
                return false;
            const std::uint64_t rectangle =
                static_cast<std::uint64_t>(destination_group)
                    * source_group_count + source_to_group[source];
            const std::uint32_t bit = destination_position * 16u
                + source_local[source];
            std::uint64_t &word = occupancy_masks[
                rectangle * 4u + bit / 64u];
            const std::uint64_t mask = std::uint64_t{1u} << (bit % 64u);
            const bool first_coordinate = (word & mask) == 0u;
            if (first_coordinate) {
                word |= mask;
                ++unique_occupancy[rectangle];
            }
            const bool mma_owner = mma_selected[rectangle] != 0u
                && first_coordinate;
            logical_edge_owner[logical_edge] = mma_owner ? 1u : 2u;
            mma_count += mma_owner;
            residual_count += !mma_owner;
        }
    }
    if (mma_count + residual_count != edge_count) return false;
    *mma_edge_count = mma_count;
    *residual_edge_count = residual_count;
    return true;
}

} // namespace cellerator::compute::architecture::providers::nvidia
