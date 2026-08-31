#pragma once

#include "Cellerator/compute/operation/relation_bundle/plan.hh"

#include <cstddef>
#include <cstdint>

namespace cellerator::compute::relation_bundle {

enum class candidate_kind : std::uint8_t {
    sequential,
    grouped_launch,
    shared_destination_owner
};

struct execution_stats {
    std::uint64_t visited_edges{};
    std::uint32_t logical_launches{};
};

inline void initialize_output(const plan_v2& plan, float* output) noexcept {
    if (plan.accumulation == accumulation_policy::assign) {
        const std::size_t count = static_cast<std::size_t>(plan.destination_axis.local_extent) *
                                  plan.feature_width;
        for (std::size_t index = 0; index < count; ++index) output[index] = 0.0F;
    }
}

inline void apply_epilogue(const plan_v2& plan, float* output) noexcept {
    const std::size_t count = static_cast<std::size_t>(plan.destination_axis.local_extent) *
                              plan.feature_width;
    for (std::size_t index = 0; index < count; ++index) {
        if (plan.epilogue == epilogue_kind::bias) output[index] += plan.bias[index % plan.feature_width];
        if (plan.epilogue == epilogue_kind::relu && output[index] < 0.0F) output[index] = 0.0F;
    }
}

inline execution_stats execute_sequential(const plan_v2& plan, float* output) noexcept {
    initialize_output(plan, output);
    execution_stats stats{};
    for (std::uint32_t m = 0; m < plan.member_count; ++m) {
        const member_view& member = plan.members[m];
        ++stats.logical_launches;
        for (local_index_type destination = 0;
             destination < plan.destination_axis.local_extent; ++destination) {
            for (local_index_type edge = member.destination_offsets[destination];
                 edge < member.destination_offsets[destination + 1]; ++edge) {
                const local_index_type source = member.source_local[edge];
                for (std::uint32_t feature = 0; feature < plan.feature_width; ++feature) {
                    output[static_cast<std::size_t>(destination) * plan.feature_width + feature] +=
                        member.values[edge] *
                        member.source_features[static_cast<std::size_t>(source) * plan.feature_width + feature];
                }
                ++stats.visited_edges;
            }
        }
    }
    apply_epilogue(plan, output);
    return stats;
}

// Models one grouped launch: work remains member-independent but launch setup
// and the destination epilogue are shared. No projection is reconstructed.
inline execution_stats execute_grouped_launch(const plan_v2& plan, float* output) noexcept {
    initialize_output(plan, output);
    execution_stats stats{};
    stats.logical_launches = 1;
    for (local_index_type destination = 0;
         destination < plan.destination_axis.local_extent; ++destination) {
        for (std::uint32_t m = 0; m < plan.member_count; ++m) {
            const member_view& member = plan.members[m];
            for (local_index_type edge = member.destination_offsets[destination];
                 edge < member.destination_offsets[destination + 1]; ++edge) {
                const local_index_type source = member.source_local[edge];
                for (std::uint32_t feature = 0; feature < plan.feature_width; ++feature) {
                    output[static_cast<std::size_t>(destination) * plan.feature_width + feature] +=
                        member.values[edge] *
                        member.source_features[static_cast<std::size_t>(source) * plan.feature_width + feature];
                }
                ++stats.visited_edges;
            }
        }
    }
    apply_epilogue(plan, output);
    return stats;
}

}  // namespace cellerator::compute::relation_bundle
