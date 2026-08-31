#include "Cellerator/compute/operation/relation_bundle/catalog.hh"
#include "Cellerator/compute/operation/relation_bundle/moments.hh"
#include "Cellerator/compute/operation/relation_chain/hierarchy.hh"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

using namespace cellerator::compute::relation_bundle;

int main() {
    constexpr local_index_type destination_count = 4096;
    constexpr local_index_type edges_per_destination = 32;
    constexpr std::uint64_t edge_count =
        static_cast<std::uint64_t>(destination_count) * edges_per_destination;
    std::vector<local_index_type> offsets(destination_count + 1);
    std::vector<local_index_type> sources(edge_count);
    std::vector<float> values(edge_count);
    std::vector<float> input(8192);
    std::vector<identity_type> global(8192);
    for (local_index_type index = 0; index <= destination_count; ++index) {
        offsets[index] = index * edges_per_destination;
    }
    for (std::uint64_t edge = 0; edge < edge_count; ++edge) {
        sources[edge] = static_cast<local_index_type>((edge * 17) % input.size());
        values[edge] = (edge % 5 == 0) ? -0.25F : 0.5F;
    }
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = static_cast<float>(static_cast<int>(index % 13) - 6);
        global[index] = (identity_type{1} << 32) + index * 3;
    }
    const axis_view source{1, 2, (identity_type{1} << 33) + 9, 3,
                           static_cast<local_index_type>(input.size()), global.data()};
    const axis_view destination{4, 5, (identity_type{1} << 33) + 11, 6,
                                destination_count, global.data()};
    const member_view member{7, 8, 9, 10, source, offsets.data(), sources.data(),
                             values.data(), input.data(), edge_count};
    const plan_v2 plan{11, 12, destination, &member, 1, 1};
    assert(validate_plan(plan) == plan_status::valid);
    std::vector<float> sequential(destination_count);
    std::vector<float> grouped(destination_count);
    std::vector<float> owned(destination_count);
    float scratch[1]{};
    const execution_stats a = execute_sequential(plan, sequential.data());
    const execution_stats b = execute_grouped_launch(plan, grouped.data());
    const execution_stats c = execute_shared_destination_owner(plan, owned.data(), scratch);
    assert(a.visited_edges == edge_count && b.visited_edges == edge_count && c.visited_edges == edge_count);
    for (local_index_type index = 0; index < destination_count; ++index) {
        assert(std::abs(sequential[index] - grouped[index]) < 1.0e-6F);
        assert(std::abs(sequential[index] - owned[index]) < 1.0e-6F);
    }
    assert(candidate_count == 6);
    for (std::size_t left = 0; left < candidate_count; ++left) {
        assert(candidate_catalog[left].candidate_id != 0 && candidate_catalog[left].stage_id != 0);
        for (std::size_t right = left + 1; right < candidate_count; ++right) {
            assert(candidate_catalog[left].candidate_id != candidate_catalog[right].candidate_id);
            assert(candidate_catalog[left].stage_id != candidate_catalog[right].stage_id);
        }
    }
    const resource_query owner_resources = query_resources(candidate_kind::shared_destination_owner, plan);
    assert(owner_resources.transient_bytes == sizeof(float) && owner_resources.logical_launches == 1);

    const local_index_type parent_offsets[]{0, 2, 5};
    const float children[]{1.0F, 3.0F, 2.0F, 4.0F, 6.0F};
    float parents[2]{};
    float broadcast[5]{};
    cellerator::compute::relation_chain::hierarchy_pool(
        parent_offsets, 2, 1,
        cellerator::compute::relation_chain::hierarchy_pool_kind::mean,
        children, parents);
    cellerator::compute::relation_chain::hierarchy_broadcast(
        parent_offsets, 2, 1, parents, broadcast);
    assert(parents[0] == 2.0F && parents[1] == 4.0F);
    assert(broadcast[0] == 2.0F && broadcast[1] == 2.0F && broadcast[4] == 4.0F);

    std::vector<local_index_type> bad_offsets = offsets;
    bad_offsets[200] = bad_offsets[199] - 1;
    member_view bad_member = member;
    bad_member.destination_offsets = bad_offsets.data();
    plan_v2 bad_plan = plan;
    bad_plan.members = &bad_member;
    assert(validate_plan(bad_plan) == plan_status::offset_out_of_range);
}
