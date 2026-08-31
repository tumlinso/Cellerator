#pragma once

#include <cstddef>
#include <cstdint>

namespace cellerator::compute::relation_bundle {

using identity_type = std::uint64_t;
using local_index_type = std::uint32_t;

enum class accumulation_policy : std::uint8_t {
    assign,
    add
};

enum class epilogue_kind : std::uint8_t {
    identity,
    bias,
    relu
};

struct axis_view {
    identity_type domain_id{};
    identity_type order_id{};
    identity_type global_extent{};
    identity_type partition_id{};
    local_index_type local_extent{};
    const identity_type* local_to_global{};
};

// One independently projected relation. Offsets are destination-owned CSR-like
// spans; source_local and values contain exactly edge_count entries.
struct member_view {
    identity_type relation_id{};
    identity_type structure_epoch{};
    identity_type projection_id{};
    identity_type value_generation{};
    axis_view source_axis{};
    const local_index_type* destination_offsets{};
    const local_index_type* source_local{};
    const float* values{};
    const float* source_features{};
    std::uint64_t edge_count{};
};

struct plan_v2 {
    identity_type operation_id{};
    identity_type composition_id{};
    axis_view destination_axis{};
    const member_view* members{};
    std::uint32_t member_count{};
    std::uint32_t feature_width{};
    accumulation_policy accumulation{accumulation_policy::assign};
    epilogue_kind epilogue{epilogue_kind::identity};
    const float* bias{};
};

enum class plan_status : std::uint8_t {
    valid,
    null_members,
    empty_bundle,
    empty_feature_width,
    invalid_destination_axis,
    invalid_member_axis,
    missing_projection,
    missing_values,
    missing_source,
    offset_out_of_range,
    source_out_of_range,
    epilogue_operand_missing
};

inline plan_status validate_plan(const plan_v2& plan) noexcept {
    if (plan.member_count == 0) return plan_status::empty_bundle;
    if (plan.members == nullptr) return plan_status::null_members;
    if (plan.feature_width == 0) return plan_status::empty_feature_width;
    if (plan.destination_axis.local_extent == 0 ||
        plan.destination_axis.global_extent < plan.destination_axis.local_extent) {
        return plan_status::invalid_destination_axis;
    }
    if (plan.epilogue == epilogue_kind::bias && plan.bias == nullptr) {
        return plan_status::epilogue_operand_missing;
    }
    for (std::uint32_t m = 0; m < plan.member_count; ++m) {
        const member_view& member = plan.members[m];
        if (member.source_axis.local_extent == 0 ||
            member.source_axis.global_extent < member.source_axis.local_extent) {
            return plan_status::invalid_member_axis;
        }
        if (member.destination_offsets == nullptr || member.source_local == nullptr) {
            return plan_status::missing_projection;
        }
        if (member.values == nullptr) return plan_status::missing_values;
        if (member.source_features == nullptr) return plan_status::missing_source;
        local_index_type prior = 0;
        for (local_index_type d = 0; d <= plan.destination_axis.local_extent; ++d) {
            const local_index_type offset = member.destination_offsets[d];
            if (offset < prior || static_cast<std::uint64_t>(offset) > member.edge_count) {
                return plan_status::offset_out_of_range;
            }
            prior = offset;
        }
        if (static_cast<std::uint64_t>(prior) != member.edge_count) {
            return plan_status::offset_out_of_range;
        }
        for (std::uint64_t edge = 0; edge < member.edge_count; ++edge) {
            if (member.source_local[edge] >= member.source_axis.local_extent) {
                return plan_status::source_out_of_range;
            }
        }
    }
    return plan_status::valid;
}

inline identity_type stable_composition_id(const plan_v2& plan) noexcept {
    identity_type hash = 1469598103934665603ull;
    const auto mix = [&hash](identity_type value) {
        hash ^= value;
        hash *= 1099511628211ull;
    };
    mix(plan.operation_id);
    mix(plan.destination_axis.domain_id);
    mix(plan.destination_axis.order_id);
    mix(plan.feature_width);
    mix(plan.member_count);
    for (std::uint32_t m = 0; m < plan.member_count; ++m) {
        mix(plan.members[m].relation_id);
        mix(plan.members[m].projection_id);
        mix(plan.members[m].source_axis.domain_id);
        mix(plan.members[m].source_axis.order_id);
    }
    return hash;
}

}  // namespace cellerator::compute::relation_bundle
