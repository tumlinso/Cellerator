#include <Cellerator/compute/projection_family/cross_operation_pareto_v1.hh>

#include <array>
#include <cassert>
#include <cstdint>

namespace family = cellerator::compute::projection_family;
namespace execution = cellerator::execution;

namespace {

execution::persistent_axis_identity axis(std::uint64_t base) {
    return {{execution::biological_abi_version,
             execution::serialized_record_kind::persistent_axis_identity,
             sizeof(execution::persistent_axis_identity)},
            {base + 1, base + 2}, {base + 3, base + 4},
            {base + 5, base + 6}, {base + 7, base + 8}};
}

family::view_family_measurement_v1 candidate(
    std::uint64_t identity,
    family::view_family_kind_v1 kind,
    std::uint64_t kernel_ns,
    std::uint64_t persistent_bytes,
    std::uint64_t transient_bytes,
    std::uint64_t launches) {
    family::view_family_measurement_v1 value{};
    value.candidate_identity = {identity, identity + 1};
    value.evidence_identity = {100 + identity, 200 + identity};
    value.family.family_identity = {1, 2};
    value.family.exact_support_identity = {3, 4};
    value.family.structure_identity = {5, 6};
    value.family.structure_epoch = {7};
    value.family.source_axis = axis(10);
    value.family.destination_axis = axis(30);
    value.family.logical_edge_order = {50, 51};
    value.family.logical_edge_count = (std::uint64_t{1} << 32u) + 3u;
    value.kind = kind;
    value.supported_operations = family::support_relation_apply_v1
        | family::support_relation_apply_transpose_v1
        | family::support_contract_on_support_v1;
    value.preparation_ns = 10;
    value.persistent_preprocess_ns = 10;
    value.input_pack_ns = 10;
    value.kernel_ns = kernel_ns;
    value.epilogue_ns = 2;
    value.output_transform_ns = 2;
    value.synchronization_ns = 1;
    value.communication_ns = 1;
    value.persistent_bytes = persistent_bytes;
    value.transient_bytes = transient_bytes;
    value.launch_count = launches;
    value.warmup_count = 5;
    value.repeat_count = 20;
    return value;
}

} // namespace

int main() {
    const std::array<family::view_family_measurement_v1, 4> candidates{{
        candidate(10, family::view_family_kind_v1::specialized, 10, 500, 100, 1),
        candidate(20, family::view_family_kind_v1::generalized, 12, 300, 80, 1),
        candidate(30, family::view_family_kind_v1::specialized, 20, 600, 120, 2),
        candidate(40, family::view_family_kind_v1::generalized, 9, 700, 150, 1)}};
    family::cross_operation_pareto_artifact_v1 artifact{};
    const auto result = family::emit_cross_operation_pareto_v1(
        {1000, 1001}, {2000, 2001}, candidates.data(), candidates.size(), 10,
        &artifact);
    assert(result.emitted());
    assert(artifact.frontier_count == 3);
    assert(artifact.frontier_candidate_indices[0] == 0);
    assert(artifact.frontier_candidate_indices[1] == 1);
    assert(artifact.frontier_candidate_indices[2] == 3);
    assert(artifact.disposition
           == family::promotion_disposition_v1::retain_measured_plurality);
    assert(artifact.family.logical_edge_count
           == (std::uint64_t{1} << 32u) + 3u);

    auto generalized_wins = candidates;
    generalized_wins[1].kernel_ns = 5;
    generalized_wins[1].persistent_bytes = 100;
    generalized_wins[1].transient_bytes = 50;
    assert(family::emit_cross_operation_pareto_v1(
               {1000, 1001}, {2000, 2001}, generalized_wins.data(),
               generalized_wins.size(), 10, &artifact)
               .emitted());
    assert(artifact.frontier_count == 1);
    assert(artifact.frontier_candidate_indices[0] == 1);
    assert(artifact.disposition
           == family::promotion_disposition_v1::promote_generalized_family);

    auto self_certified = candidates;
    self_certified[1].correctness =
        family::correctness_evidence_kind_v1::provider_self_report;
    assert(family::emit_cross_operation_pareto_v1(
               {1000, 1001}, {2000, 2001}, self_certified.data(),
               self_certified.size(), 10, &artifact)
               .code == family::cross_operation_pareto_code_v1::invalid_candidate);

    assert(family::emit_cross_operation_pareto_v1(
               {1000, 1001}, {2000, 2001}, candidates.data(),
               family::max_cross_operation_candidates_v1 + 1u, 10, &artifact)
               .code
           == family::cross_operation_pareto_code_v1::
                  candidate_bound_exceeded);

    std::uint64_t random_state = 0x9e3779b97f4a7c15ULL;
    for (std::uint32_t trial = 0; trial < 200; ++trial) {
        std::array<family::view_family_measurement_v1, 12> random_candidates{};
        for (std::uint32_t index = 0; index < random_candidates.size(); ++index) {
            random_state = random_state * 6364136223846793005ULL + 1ULL;
            random_candidates[index] = candidate(
                1000 + static_cast<std::uint64_t>(trial) * 20 + index,
                (index & 1u) == 0 ? family::view_family_kind_v1::specialized
                                  : family::view_family_kind_v1::generalized,
                1 + ((random_state >> 8u) % 100u),
                1 + ((random_state >> 20u) % 1000u),
                1 + ((random_state >> 32u) % 500u),
                1 + ((random_state >> 44u) % 4u));
        }
        assert(family::emit_cross_operation_pareto_v1(
                   {3000 + trial, 4000 + trial}, {5000 + trial, 6000 + trial},
                   random_candidates.data(), random_candidates.size(), 7,
                   &artifact)
                   .emitted());
        std::array<bool, 12> on_frontier{};
        for (std::uint32_t slot = 0; slot < artifact.frontier_count; ++slot) {
            assert(artifact.frontier_candidate_indices[slot]
                   < random_candidates.size());
            on_frontier[artifact.frontier_candidate_indices[slot]] = true;
        }
        for (std::uint32_t index = 0; index < random_candidates.size(); ++index) {
            family::measured_candidate_metrics_v1 rhs{};
            assert(family::measured_total_ns_v1(
                random_candidates[index], 7, &rhs.total_ns));
            rhs.persistent_bytes = random_candidates[index].persistent_bytes;
            rhs.transient_bytes = random_candidates[index].transient_bytes;
            rhs.launch_count = random_candidates[index].launch_count;
            bool dominated = false;
            for (std::uint32_t other = 0;
                 other < random_candidates.size();
                 ++other) {
                family::measured_candidate_metrics_v1 lhs{};
                assert(family::measured_total_ns_v1(
                    random_candidates[other], 7, &lhs.total_ns));
                lhs.persistent_bytes =
                    random_candidates[other].persistent_bytes;
                lhs.transient_bytes = random_candidates[other].transient_bytes;
                lhs.launch_count = random_candidates[other].launch_count;
                dominated = dominated || (other != index
                    && family::dominates_v1(lhs, rhs));
            }
            assert(on_frontier[index] == !dominated);
        }
    }
}
