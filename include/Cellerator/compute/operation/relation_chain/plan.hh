#pragma once

#include "Cellerator/compute/operation/relation_bundle/plan.hh"

#include <cstdint>

namespace cellerator::compute::relation_chain {

using relation_bundle::axis_view;
using relation_bundle::identity_type;
using relation_bundle::local_index_type;

struct stage_view {
    identity_type stage_id{};
    identity_type relation_id{};
    identity_type projection_id{};
    identity_type value_generation{};
    axis_view source_axis{};
    axis_view destination_axis{};
    const local_index_type* destination_offsets{};
    const local_index_type* source_local{};
    const float* values{};
    std::uint64_t edge_count{};
};

struct plan_v2 {
    identity_type operation_id{};
    identity_type composition_id{};
    stage_view first{};
    stage_view second{};
    const local_index_type* second_source_to_first_destination{};
    std::uint32_t feature_width{};
};

enum class chain_status : std::uint8_t {
    valid_materialized,
    valid_persistent_order,
    empty_feature_width,
    incompatible_intermediate_domain,
    incompatible_intermediate_extent,
    missing_recovery_map,
    invalid_recovery_map,
    invalid_projection
};

inline bool projection_is_valid(const stage_view& stage) noexcept {
    if (stage.destination_offsets == nullptr || stage.source_local == nullptr || stage.values == nullptr) {
        return false;
    }
    local_index_type prior = 0;
    for (local_index_type destination = 0;
         destination <= stage.destination_axis.local_extent; ++destination) {
        const local_index_type offset = stage.destination_offsets[destination];
        if (offset < prior || static_cast<std::uint64_t>(offset) > stage.edge_count) return false;
        prior = offset;
    }
    if (static_cast<std::uint64_t>(prior) != stage.edge_count) return false;
    for (std::uint64_t edge = 0; edge < stage.edge_count; ++edge) {
        if (stage.source_local[edge] >= stage.source_axis.local_extent) return false;
    }
    return true;
}

inline chain_status validate_plan(const plan_v2& plan) noexcept {
    if (plan.feature_width == 0) return chain_status::empty_feature_width;
    if (!projection_is_valid(plan.first) || !projection_is_valid(plan.second)) {
        return chain_status::invalid_projection;
    }
    if (plan.first.destination_axis.domain_id != plan.second.source_axis.domain_id) {
        return chain_status::incompatible_intermediate_domain;
    }
    if (plan.first.destination_axis.local_extent != plan.second.source_axis.local_extent) {
        return chain_status::incompatible_intermediate_extent;
    }
    if (plan.first.destination_axis.order_id == plan.second.source_axis.order_id) {
        return chain_status::valid_persistent_order;
    }
    if (plan.second_source_to_first_destination == nullptr) return chain_status::missing_recovery_map;
    for (local_index_type source = 0; source < plan.second.source_axis.local_extent; ++source) {
        if (plan.second_source_to_first_destination[source] >= plan.first.destination_axis.local_extent) {
            return chain_status::invalid_recovery_map;
        }
    }
    return chain_status::valid_materialized;
}

}  // namespace cellerator::compute::relation_chain
