#include <Cellerator/interop/baseplane/baseplane_integration.cuh>

#include <cmath>

namespace cellerator::compute::sequence {
namespace {

bool same_device_class(
    const execution::device_performance_class &left,
    const execution::device_performance_class &right) noexcept {
    return left.vendor == right.vendor
        && left.architecture_major == right.architecture_major
        && left.architecture_minor == right.architecture_minor
        && left.build_identity == right.build_identity;
}

bool same_measurement_key(
    const sequence_measurement_key &left,
    const sequence_measurement_key &right) noexcept {
    return left.predicate_semantic_hash == right.predicate_semantic_hash
        && execution::same_identity(left.coordinate_order, right.coordinate_order)
        && execution::same_identity(
            left.regulatory_projection, right.regulatory_projection)
        && same_device_class(left.device, right.device)
        && left.runtime_build_identity == right.runtime_build_identity
        && left.local_base_count == right.local_base_count
        && left.predicate_id == right.predicate_id
        && left.output_flags == right.output_flags;
}

} // namespace

sequence_strategy select_sequence_strategy(
    const sequence_prepare_policy &policy) noexcept {
    if (policy.requested == sequence_strategy::materialize_mask)
        return policy.allow_materialization
            ? sequence_strategy::materialize_mask : sequence_strategy::automatic;
    if (policy.requested == sequence_strategy::materialize_relation)
        return policy.allow_materialization
            ? sequence_strategy::materialize_relation
            : sequence_strategy::automatic;
    if (policy.requested == sequence_strategy::fuse_predicate)
        return policy.allow_fusion
            ? sequence_strategy::fuse_predicate : sequence_strategy::automatic;
    if (policy.allow_fusion != policy.allow_materialization)
        return policy.allow_fusion ? sequence_strategy::fuse_predicate
            : sequence_strategy::materialize_relation;
    return sequence_strategy::automatic;
}

sequence_strategy_decision select_sequence_strategy(
    const sequence_measurement_key &key,
    const sequence_prepare_policy &policy) noexcept {
    sequence_strategy_decision decision{};
    if (policy.requested == sequence_strategy::materialize_mask
        || policy.requested == sequence_strategy::materialize_relation) {
        decision.strategy = policy.allow_materialization
            ? policy.requested : sequence_strategy::automatic;
        decision.empirical_measurement_required = false;
        decision.reason = policy.allow_materialization
            ? policy.requested == sequence_strategy::materialize_relation
                ? "direct relation materialization explicitly requested"
                : "mask materialization explicitly requested"
            : "requested materialization is unavailable";
        return decision;
    }
    if (policy.requested == sequence_strategy::fuse_predicate) {
        decision.strategy = policy.allow_fusion
            ? sequence_strategy::fuse_predicate : sequence_strategy::automatic;
        decision.empirical_measurement_required = false;
        decision.reason = policy.allow_fusion
            ? "fusion explicitly requested" : "requested fusion is unavailable";
        return decision;
    }
    if (!policy.allow_materialization && !policy.allow_fusion) {
        decision.reason = "materialization and fusion are both unavailable";
        return decision;
    }
    if (!policy.allow_materialization || !policy.allow_fusion) {
        decision.strategy = policy.allow_materialization
            ? sequence_strategy::materialize_relation
            : sequence_strategy::fuse_predicate;
        decision.empirical_measurement_required = false;
        decision.reason = "only one capable sequence strategy remains";
        return decision;
    }

    const sequence_strategy_evidence *evidence = policy.evidence;
    const bool valid_policy = std::isfinite(policy.practical_tolerance_percent)
        && policy.practical_tolerance_percent >= 0.0
        && policy.practical_tolerance_percent < 100.0
        && std::isfinite(policy.maximum_spread_percent)
        && policy.maximum_spread_percent >= 0.0;
    const bool valid_evidence = evidence != nullptr && valid_policy
        && same_measurement_key(evidence->key, key)
        && evidence->sample_count >= 3u
        && std::isfinite(evidence->fused_per_use_ns)
        && std::isfinite(evidence->first_materialized_use_ns)
        && std::isfinite(evidence->cached_materialized_use_ns)
        && std::isfinite(evidence->fused_spread_percent)
        && std::isfinite(evidence->materialized_spread_percent)
        && evidence->fused_per_use_ns > 0.0
        && evidence->first_materialized_use_ns > 0.0
        && evidence->cached_materialized_use_ns > 0.0
        && evidence->fused_spread_percent >= 0.0
        && evidence->materialized_spread_percent >= 0.0
        && evidence->fused_spread_percent <= policy.maximum_spread_percent
        && evidence->materialized_spread_percent
            <= policy.maximum_spread_percent;
    if (!valid_evidence) {
        decision.reason = "current comparable measurement is required";
        return decision;
    }

    const std::uint32_t state_count = policy.expected_cell_state_count != 0u
        ? policy.expected_cell_state_count
        : policy.expected_predicate_reuse;
    const double reuse = static_cast<double>(state_count == 0u ? 1u : state_count);
    decision.fused_total_ns = evidence->fused_per_use_ns * reuse;
    decision.materialized_total_ns = evidence->first_materialized_use_ns
        + evidence->cached_materialized_use_ns * (reuse - 1.0);
    const double required_ratio =
        1.0 - policy.practical_tolerance_percent / 100.0;
    decision.strategy = decision.materialized_total_ns
            < decision.fused_total_ns * required_ratio
        ? sequence_strategy::materialize_relation
        : sequence_strategy::fuse_predicate;
    decision.empirical_measurement_required = false;
    decision.reason = decision.strategy == sequence_strategy::materialize_relation
        ? "measured amortized direct-relation cost is lower across cell states"
        : "fusion wins or is within practical tolerance";
    return decision;
}

} // namespace cellerator::compute::sequence
