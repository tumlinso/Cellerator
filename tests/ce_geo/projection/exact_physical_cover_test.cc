#include <cassert>
#include <cstdint>

namespace cellerator::compute::architecture::providers::nvidia {
bool build_exact_rectangle_cover_v1(
    std::uint32_t, std::uint32_t, const std::uint64_t *,
    const std::uint32_t *, const std::uint64_t *, std::uint64_t,
    const std::uint32_t *, const std::uint8_t *, std::uint32_t,
    const std::uint32_t *, const std::uint8_t *, std::uint32_t,
    const std::uint8_t *, std::uint64_t, std::uint64_t *, std::uint64_t,
    std::uint32_t *, std::uint8_t *, std::uint64_t, std::uint64_t *,
    std::uint64_t *) noexcept;
}

namespace provider = cellerator::compute::architecture::providers::nvidia;

int main() {
    // Destination-major relation. The last edge duplicates coordinate (d0,s0)
    // but has a distinct logical identity, so it remains an exact residual.
    const std::uint64_t offsets[] = {0u, 3u, 5u};
    const std::uint32_t sources[] = {0u, 1u, 0u, 0u, 2u};
    const std::uint64_t logical_edges[] = {2u, 0u, 4u, 1u, 3u};
    const std::uint32_t source_groups[] = {0u, 0u, 0u};
    const std::uint8_t source_local[] = {0u, 1u, 2u};
    const std::uint32_t destination_groups[] = {0u, 1u};
    const std::uint8_t destination_local[] = {0u, 0u};
    const std::uint8_t selected[] = {1u, 0u};
    std::uint64_t masks[8]{};
    std::uint32_t occupancy[2]{};
    std::uint8_t owners[5]{};
    std::uint64_t mma_count = 0u;
    std::uint64_t residual_count = 0u;
    assert(provider::build_exact_rectangle_cover_v1(
        3u, 2u, offsets, sources, logical_edges, 5u,
        source_groups, source_local, 1u, destination_groups,
        destination_local, 2u, selected, 2u, masks, 8u, occupancy,
        owners, 5u, &mma_count, &residual_count));

    assert(occupancy[0] == 2u); // duplicate coordinate counted once
    assert(occupancy[1] == 2u);
    assert((masks[0] & 0x3u) == 0x3u);
    assert((masks[4] & 0x5u) == 0x5u);
    assert(mma_count == 2u);
    assert(residual_count == 3u);
    assert(owners[0] == 1u); // d0,s1
    assert(owners[2] == 1u); // first d0,s0
    assert(owners[4] == 2u); // duplicate d0,s0
    assert(owners[1] == 2u && owners[3] == 2u); // unselected rectangle
    for (std::uint8_t owner : owners)
        assert(owner == 1u || owner == 2u);

    const std::uint64_t duplicate_logical[] = {2u, 0u, 2u, 1u, 3u};
    assert(!provider::build_exact_rectangle_cover_v1(
        3u, 2u, offsets, sources, duplicate_logical, 5u,
        source_groups, source_local, 1u, destination_groups,
        destination_local, 2u, selected, 2u, masks, 8u, occupancy,
        owners, 5u, &mma_count, &residual_count));
    return 0;
}
