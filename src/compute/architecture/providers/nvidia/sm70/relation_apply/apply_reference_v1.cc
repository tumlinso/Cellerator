#include "Cellerator/compute/architecture/providers/nvidia/sm70/relation_apply/apply_reference_v1.hh"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace cellerator::compute::architecture::nvidia::sm70::relation_apply {

apply_reference_status_v1 apply_dense_tile_reference_v1(
    const apply_reference_request_v1 &request) noexcept {
    if (request.relation_tiles == nullptr
        || request.destination_tile_offsets == nullptr
        || request.tile_source_bases == nullptr || request.dense_rhs == nullptr
        || request.output == nullptr || request.tile_count == 0u
        || request.destination_group_count == 0u
        || request.local_source_count < 16u
        || (request.rows_per_group != 8u
            && request.rows_per_group != 16u
            && request.rows_per_group != 32u)
        || request.dense_width == 0u) {
        return apply_reference_status_v1::invalid_argument;
    }
    const std::uint64_t rows =
        static_cast<std::uint64_t>(request.destination_group_count)
        * request.rows_per_group;
    if (rows > std::numeric_limits<std::uint64_t>::max()
            / request.dense_width) {
        return apply_reference_status_v1::arithmetic_overflow;
    }
    const std::uint64_t output_elements = rows * request.dense_width;
    if (request.output_capacity < output_elements
        || output_elements > std::numeric_limits<std::size_t>::max()) {
        return apply_reference_status_v1::insufficient_capacity;
    }
    std::fill(request.output,
        request.output + static_cast<std::size_t>(output_elements), 0.0f);
    for (std::uint32_t group = 0u;
         group < request.destination_group_count; ++group) {
        const std::uint32_t tile_begin =
            request.destination_tile_offsets[group];
        const std::uint32_t tile_end =
            request.destination_tile_offsets[group + 1u];
        if (tile_begin > tile_end || tile_end > request.tile_count) {
            return apply_reference_status_v1::invalid_offsets;
        }
        for (std::uint32_t tile = tile_begin; tile < tile_end; ++tile) {
            const std::uint32_t source_base = request.tile_source_bases[tile];
            if (source_base > request.local_source_count
                || request.local_source_count - source_base < 16u) {
                return apply_reference_status_v1::invalid_offsets;
            }
            const float *relation = request.relation_tiles
                + static_cast<std::size_t>(tile)
                    * request.rows_per_group * 16u;
            for (std::uint32_t row = 0u; row < request.rows_per_group; ++row) {
                float *destination = request.output
                    + (static_cast<std::size_t>(group)
                            * request.rows_per_group
                        + row) * request.dense_width;
                for (std::uint32_t inner = 0u; inner < 16u; ++inner) {
                    const float weight = relation[
                        static_cast<std::size_t>(row) * 16u + inner];
                    const float *source = request.dense_rhs
                        + static_cast<std::size_t>(source_base + inner)
                            * request.dense_width;
                    for (std::uint32_t column = 0u;
                         column < request.dense_width; ++column) {
                        destination[column] += weight * source[column];
                    }
                }
            }
        }
    }
    return apply_reference_status_v1::success;
}

}  // namespace cellerator::compute::architecture::nvidia::sm70::relation_apply
