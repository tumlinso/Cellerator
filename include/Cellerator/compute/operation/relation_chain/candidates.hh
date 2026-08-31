#pragma once

#include "Cellerator/compute/operation/relation_chain/plan.hh"

#include <cstddef>
#include <cstdint>

namespace cellerator::compute::relation_chain {

struct chain_stats {
    std::uint64_t visited_edges{};
    std::uint64_t order_transforms{};
    std::uint32_t logical_launches{2};
};

inline void apply_stage(const stage_view& stage,
                        const float* input,
                        std::uint32_t width,
                        float* output,
                        chain_stats& stats) noexcept {
    const std::size_t output_count = static_cast<std::size_t>(stage.destination_axis.local_extent) * width;
    for (std::size_t index = 0; index < output_count; ++index) output[index] = 0.0F;
    for (local_index_type destination = 0;
         destination < stage.destination_axis.local_extent; ++destination) {
        for (local_index_type edge = stage.destination_offsets[destination];
             edge < stage.destination_offsets[destination + 1]; ++edge) {
            const local_index_type source = stage.source_local[edge];
            for (std::uint32_t feature = 0; feature < width; ++feature) {
                output[static_cast<std::size_t>(destination) * width + feature] +=
                    stage.values[edge] * input[static_cast<std::size_t>(source) * width + feature];
            }
            ++stats.visited_edges;
        }
    }
}

inline chain_stats execute_persistent_order(const plan_v2& plan,
                                            const float* input,
                                            float* intermediate,
                                            float* output) noexcept {
    chain_stats stats{};
    apply_stage(plan.first, input, plan.feature_width, intermediate, stats);
    apply_stage(plan.second, intermediate, plan.feature_width, output, stats);
    return stats;
}

// The caller supplies both first-order and second-source-order buffers. The
// explicit linear gather is visible in cost and never canonicalizes implicitly.
inline chain_stats execute_materialized(const plan_v2& plan,
                                        const float* input,
                                        float* first_order,
                                        float* second_source_order,
                                        float* output) noexcept {
    chain_stats stats{};
    apply_stage(plan.first, input, plan.feature_width, first_order, stats);
    for (local_index_type source = 0; source < plan.second.source_axis.local_extent; ++source) {
        const local_index_type first_destination = plan.second_source_to_first_destination[source];
        for (std::uint32_t feature = 0; feature < plan.feature_width; ++feature) {
            second_source_order[static_cast<std::size_t>(source) * plan.feature_width + feature] =
                first_order[static_cast<std::size_t>(first_destination) * plan.feature_width + feature];
        }
    }
    stats.order_transforms = plan.second.source_axis.local_extent;
    apply_stage(plan.second, second_source_order, plan.feature_width, output, stats);
    return stats;
}

}  // namespace cellerator::compute::relation_chain
