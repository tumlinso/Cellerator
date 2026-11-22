#pragma once

#include <Cellerator/tensor/block_masked/structure.hh>

#include <cstdint>
#include <memory>
#include <vector>

namespace cellerator::test {

// The existing tests describe support as lists of rows. Flatten these inputs
// before invoking the native builder.
template<bool Retained = false>
auto make_block_masked_structure(
    std::uint64_t features,
    const std::vector<std::vector<std::uint32_t>>& rows,
    std::uint32_t block_width = 32,
    std::uint32_t tile_width = 32,
    const std::vector<std::uint64_t>& canonical_features = {},
    const std::vector<std::uint64_t>& canonical_rows = {}) {
    namespace bm = tensor::block_masked;
    std::vector<std::uint64_t> offsets{0};
    std::vector<std::uint32_t> indices;
    for (const auto& row : rows) {
        indices.insert(indices.end(), row.begin(), row.end());
        offsets.push_back(indices.size());
    }
    const bm::StructureInput input{
        features, {offsets.data(), rows.size(), indices.data(), indices.size()},
        block_width, tile_width,
        {canonical_features.data(), canonical_features.size()},
        {canonical_rows.data(), canonical_rows.size()}};
    const auto bytes = bm::structure_scratch_bytes(input);
    std::vector<std::uint64_t> scratch(bytes / sizeof(std::uint64_t)
        + (bytes % sizeof(std::uint64_t) != 0));
    if constexpr (Retained) {
        return std::make_shared<const bm::Structure>(input, scratch.data(), bytes);
    } else {
        return bm::Structure(input, scratch.data(), bytes);
    }
}

} // namespace cellerator::test
