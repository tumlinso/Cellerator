#pragma once

#include "Cellerator/compute/operation/relation_bundle/candidates.hh"

#include <cstddef>
#include <cstdint>

namespace cellerator::compute::relation_bundle {

enum class mechanism_kind : std::uint8_t {
    independent_projection,
    grouped_launch,
    destination_owner,
    explicit_order_transform,
    persistent_order,
    paired_traversal
};

struct resource_query {
    std::uint64_t persistent_bytes{};
    std::uint64_t transient_bytes{};
    std::uint32_t logical_launches{};
};

struct candidate_descriptor {
    identity_type candidate_id{};
    identity_type stage_id{};
    const char* stable_name{};
    mechanism_kind mechanism{};
    bool experimental{};
    bool requires_measurement{};
};

inline constexpr candidate_descriptor candidate_catalog[]{
    {0x42554e444c450101ull, 0x42554e444c455301ull,
     "relation_bundle_sequential_v1", mechanism_kind::independent_projection, false, false},
    {0x42554e444c450102ull, 0x42554e444c455302ull,
     "relation_bundle_grouped_launch_v1", mechanism_kind::grouped_launch, true, true},
    {0x42554e444c450103ull, 0x42554e444c455303ull,
     "relation_bundle_shared_destination_owner_v1", mechanism_kind::destination_owner, true, true},
    {0x434841494e000101ull, 0x434841494e005301ull,
     "relation_chain_materialized_two_hop_v1", mechanism_kind::explicit_order_transform, false, false},
    {0x434841494e000102ull, 0x434841494e005302ull,
     "relation_chain_persistent_order_two_hop_v1", mechanism_kind::persistent_order, true, true},
    {0x4d4f4d454e540101ull, 0x4d4f4d454e545301ull,
     "relation_moments_paired_traversal_v1", mechanism_kind::paired_traversal, true, true}};

inline constexpr std::size_t candidate_count = sizeof(candidate_catalog) / sizeof(candidate_catalog[0]);

inline resource_query query_resources(candidate_kind candidate,
                                      const plan_v2& plan) noexcept {
    resource_query query{};
    if (candidate == candidate_kind::sequential) query.logical_launches = plan.member_count;
    if (candidate == candidate_kind::grouped_launch) query.logical_launches = 1;
    if (candidate == candidate_kind::shared_destination_owner) {
        query.logical_launches = 1;
        query.transient_bytes = static_cast<std::uint64_t>(plan.feature_width) * sizeof(float);
    }
    return query;
}

}  // namespace cellerator::compute::relation_bundle
