#pragma once

#include "Cellerator/compute/operation/relation_bundle/plan.hh"

#include <cstddef>
#include <cstdint>

namespace cellerator::compute::relation_bundle {

struct moments_stats {
    std::uint64_t visited_edges{};
    std::uint32_t logical_launches{};
};

inline moments_stats execute_relation_moment(const member_view& relation,
                                             local_index_type destination_count,
                                             std::uint32_t width,
                                             bool squared_input,
                                             float* output) noexcept {
    const std::size_t count = static_cast<std::size_t>(destination_count) * width;
    for (std::size_t index = 0; index < count; ++index) output[index] = 0.0F;
    moments_stats stats{0, 1};
    for (local_index_type destination = 0; destination < destination_count; ++destination) {
        for (local_index_type edge = relation.destination_offsets[destination];
             edge < relation.destination_offsets[destination + 1]; ++edge) {
            const local_index_type source = relation.source_local[edge];
            for (std::uint32_t feature = 0; feature < width; ++feature) {
                const float input = relation.source_features[
                    static_cast<std::size_t>(source) * width + feature];
                output[static_cast<std::size_t>(destination) * width + feature] +=
                    relation.values[edge] * (squared_input ? input * input : input);
            }
            ++stats.visited_edges;
        }
    }
    return stats;
}

// Profiler-visible paired traversal. The two unfused calls above remain public
// and exact; this composition is experimental and requires measurement.
inline moments_stats execute_relation_moments_pair(const member_view& relation,
                                                   local_index_type destination_count,
                                                   std::uint32_t width,
                                                   float* first,
                                                   float* second) noexcept {
    const std::size_t count = static_cast<std::size_t>(destination_count) * width;
    for (std::size_t index = 0; index < count; ++index) {
        first[index] = 0.0F;
        second[index] = 0.0F;
    }
    moments_stats stats{0, 1};
    for (local_index_type destination = 0; destination < destination_count; ++destination) {
        for (local_index_type edge = relation.destination_offsets[destination];
             edge < relation.destination_offsets[destination + 1]; ++edge) {
            const local_index_type source = relation.source_local[edge];
            for (std::uint32_t feature = 0; feature < width; ++feature) {
                const std::size_t output_index = static_cast<std::size_t>(destination) * width + feature;
                const float input = relation.source_features[
                    static_cast<std::size_t>(source) * width + feature];
                const float weight = relation.values[edge];
                first[output_index] += weight * input;
                second[output_index] += weight * input * input;
            }
            ++stats.visited_edges;
        }
    }
    return stats;
}

}  // namespace cellerator::compute::relation_bundle
