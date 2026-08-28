#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

namespace cellerator::compute::math::tensor_core {

inline constexpr std::uint64_t invalid_dense_fragment_position =
    std::numeric_limits<std::uint64_t>::max();

enum class dense_fragment_plan_status : std::uint8_t {
    ok = 0u,
    invalid_argument = 1u,
    invalid_csr = 2u,
    insufficient_capacity = 3u
};

struct destination_row_csr_support_view {
    const std::uint64_t *destination_offsets = nullptr;
    const std::uint32_t *source_indices = nullptr;
    std::uint32_t destination_count = 0u;
    std::uint32_t source_count = 0u;
    std::uint64_t logical_edge_count = 0u;
};

struct v100_dense_fragment_plan_requirements {
    std::uint64_t tile_count = 0u;
    std::uint64_t qualified_fragment_count = 0u;
    std::uint64_t packed_slot_count = 0u;
    std::uint64_t residual_edge_count = 0u;
    std::uint32_t maximum_tile_nnz = 0u;
};

struct v100_dense_fragment_plan_buffers {
    std::uint16_t *tile_nnz = nullptr;
    std::int64_t *tile_to_fragment = nullptr;
    std::uint64_t tile_capacity = 0u;
    std::uint32_t *fragment_destination_bases = nullptr;
    std::uint32_t *fragment_source_bases = nullptr;
    std::uint64_t fragment_capacity = 0u;
    std::uint64_t *logical_edge_to_fragment_slot = nullptr;
    std::uint64_t logical_edge_capacity = 0u;
    std::uint64_t *fragment_slot_to_logical_edge = nullptr;
    std::uint64_t packed_slot_capacity = 0u;
};

dense_fragment_plan_status query_v100_dense_fragment_plan_host(
    const destination_row_csr_support_view &support,
    v100_dense_fragment_plan_buffers scratch,
    v100_dense_fragment_plan_requirements *requirements) noexcept;

dense_fragment_plan_status build_v100_dense_fragment_plan_host(
    const destination_row_csr_support_view &support,
    v100_dense_fragment_plan_buffers buffers,
    v100_dense_fragment_plan_requirements *requirements) noexcept;

} // namespace cellerator::compute::math::tensor_core
