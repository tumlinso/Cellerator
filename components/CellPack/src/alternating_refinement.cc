#include "CellPack/alternating_refinement.hh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace cellpack {
namespace {

constexpr u64 fnv1a_offset = 1469598103934665603ull;
constexpr u64 fnv1a_prime = 1099511628211ull;
constexpr u64 plan_identity_domain = 0x435042503131504cull;
constexpr u64 controller_identity_domain = 0x4350425031304354ull;

void hash_byte(u64 *hash, unsigned char value) noexcept {
    *hash ^= value;
    *hash *= fnv1a_prime;
}

void hash_u64(u64 *hash, u64 value) noexcept {
    for (u32 byte = 0u; byte < 8u; ++byte) {
        hash_byte(hash, static_cast<unsigned char>(value >> (byte * 8u)));
    }
}

void hash_double(u64 *hash, double value) noexcept {
    static_assert(sizeof(double) == sizeof(u64), "double identity requires 64-bit IEEE storage");
    u64 bits = 0u;
    std::memcpy(&bits, &value, sizeof(bits));
    hash_u64(hash, bits);
}

u64 nonzero_hash(u64 hash) noexcept { return hash == 0u ? 1u : hash; }

bool add_overflows(u64 lhs, u64 rhs, u64 *out) noexcept {
    if (lhs > std::numeric_limits<u64>::max() - rhs) return true;
    *out = lhs + rhs;
    return false;
}

bool valid_phase(alternating_refinement_phase phase) noexcept {
    return phase == alternating_refinement_phase::baseline
        || phase == alternating_refinement_phase::gene_blocks
        || phase == alternating_refinement_phase::cell_order_and_tiles;
}

bool valid_weight(double value) noexcept {
    return std::isfinite(value) && value >= 0.0;
}

validation_result validate_weights(
    const alternating_refinement_objective_weights &weights) noexcept {
    if (!valid_weight(weights.encoded_bytes)
        || !valid_weight(weights.metadata_bytes)
        || !valid_weight(weights.active_block_references)
        || !valid_weight(weights.tile_block_union_references)
        || !valid_weight(weights.padding_slots)
        || !valid_weight(weights.runtime_mean_nanoseconds)
        || !valid_weight(weights.preprocessing_mean_nanoseconds)
        || !valid_weight(weights.forward_mean_nanoseconds)
        || !valid_weight(weights.transpose_mean_nanoseconds)
        || !valid_weight(weights.active_interaction_nanoseconds)
        || !valid_weight(weights.partition_cut_edge_nanoseconds)
        || !valid_weight(weights.bootstrap_mad_nanoseconds)) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "alternating-refinement objective weights must be finite and nonnegative");
    }
    if (weights.encoded_bytes == 0.0 && weights.metadata_bytes == 0.0
        && weights.active_block_references == 0.0
        && weights.tile_block_union_references == 0.0
        && weights.padding_slots == 0.0
        && weights.runtime_mean_nanoseconds == 0.0
        && weights.preprocessing_mean_nanoseconds == 0.0
        && weights.forward_mean_nanoseconds == 0.0
        && weights.transpose_mean_nanoseconds == 0.0
        && weights.active_interaction_nanoseconds == 0.0
        && weights.partition_cut_edge_nanoseconds == 0.0
        && weights.bootstrap_mad_nanoseconds == 0.0) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "alternating-refinement objective has no active term");
    }
    return validation_ok();
}

validation_result validate_config(const alternating_refinement_config &config) {
    if (config.schema_version != alternating_refinement_schema_version) {
        return validation_error(validation_code::unsupported_version,
            config.schema_version, "unsupported alternating-refinement schema");
    }
    if (config.maximum_iterations == 0u || config.maximum_evaluations == 0u) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "alternating-refinement caps must be nonzero");
    }
    if (config.dataset_identity == 0u || config.feature_axis_identity == 0u
        || config.feature_axis_identity_version == 0u
        || config.row_domain_identity == 0u || config.split_identity == 0u) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "alternating-refinement identities must be explicit");
    }
    const bool profile_weighted =
        config.weights.forward_mean_nanoseconds != 0.0
        || config.weights.transpose_mean_nanoseconds != 0.0
        || config.weights.active_interaction_nanoseconds != 0.0
        || config.weights.partition_cut_edge_nanoseconds != 0.0
        || config.weights.bootstrap_mad_nanoseconds != 0.0;
    if (profile_weighted && (config.workload_profile_identity == 0u
            || config.workload_evidence_revision == 0u
            || config.minimum_bootstrap_samples == 0u)) {
        return validation_error(validation_code::invalid_plan_geometry,
            invalid_id,
            "workload-weighted refinement needs profile, evidence, and bootstrap identities");
    }
    if (!std::isfinite(config.absolute_improvement_tolerance)
        || !std::isfinite(config.relative_improvement_tolerance)
        || config.absolute_improvement_tolerance < 0.0
        || config.relative_improvement_tolerance < 0.0) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "alternating-refinement tolerances must be finite and nonnegative");
    }
    return validate_weights(config.weights);
}

validation_result validate_metric_binding(
    const frozen_packing_plan &plan,
    const packing_validation_metrics &metrics,
    const alternating_refinement_config &config,
    const packing_validation_metrics *baseline,
    const char *label) {
    validation_result status = validate_packing_validation_metrics(metrics);
    if (!status) return status;
    if (metrics.dataset_identity != config.dataset_identity
        || metrics.feature_axis_identity != config.feature_axis_identity
        || metrics.row_domain_identity != config.row_domain_identity
        || metrics.split_identity != config.split_identity
        || metrics.feature_count != plan.feature_count()) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            label);
    }
    if ((metrics.available & packing_validation_metric_correctness) == 0u
        || metrics.correctness_mismatches != 0u
        || metrics.correctness_items == 0u) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "alternating-refinement observations require exact nonempty correctness evidence");
    }
    if (baseline != nullptr
        && (metrics.row_count != baseline->row_count
            || metrics.feature_count != baseline->feature_count
            || metrics.nnz_count != baseline->nnz_count
            || metrics.runtime_input_nnz != baseline->runtime_input_nnz
            || metrics.runtime_repeat_count != baseline->runtime_repeat_count
            || metrics.forward_repeat_count != baseline->forward_repeat_count
            || metrics.transpose_repeat_count != baseline->transpose_repeat_count
            || metrics.bootstrap_sample_count
                != baseline->bootstrap_sample_count)) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "alternating-refinement candidate changed an objective denominator");
    }
    const bool profile_weighted =
        config.weights.forward_mean_nanoseconds != 0.0
        || config.weights.transpose_mean_nanoseconds != 0.0
        || config.weights.active_interaction_nanoseconds != 0.0
        || config.weights.partition_cut_edge_nanoseconds != 0.0
        || config.weights.bootstrap_mad_nanoseconds != 0.0;
    if (profile_weighted
        && ((metrics.available
                & packing_validation_metric_workload_profile) == 0u
            || metrics.workload_profile_identity
                != config.workload_profile_identity
            || metrics.workload_evidence_revision
                != config.workload_evidence_revision
            || metrics.bootstrap_sample_count
                < config.minimum_bootstrap_samples)) {
        return validation_error(validation_code::invalid_matrix_view,
            invalid_id,
            "workload profile is unavailable, stale, or under-sampled");
    }
    return validation_ok();
}

validation_result validate_observation(
    const alternating_refinement_observation &observation,
    const alternating_refinement_config &config,
    const packing_validation_metrics *training_baseline,
    const packing_validation_metrics *held_out_baseline,
    bool is_baseline,
    u64 expected_parent,
    u32 expected_iteration) {
    if (observation.schema_version != alternating_refinement_schema_version) {
        return validation_error(validation_code::unsupported_version,
            observation.schema_version, "unsupported refinement observation schema");
    }
    if (!valid_phase(observation.phase)
        || (is_baseline && observation.phase != alternating_refinement_phase::baseline)
        || (!is_baseline && observation.phase == alternating_refinement_phase::baseline)) {
        return validation_error(validation_code::invalid_plan_geometry, observation.iteration,
            "alternating-refinement phase is invalid for this observation");
    }
    if (is_baseline && !observation.evaluation_succeeded) {
        return validation_error(validation_code::invalid_plan_geometry, 0u,
            "alternating-refinement baseline evaluation must succeed");
    }
    if (observation.iteration != expected_iteration
        || observation.candidate_identity == 0u
        || observation.parent_plan_identity != expected_parent) {
        return validation_error(validation_code::invalid_plan_geometry, observation.iteration,
            "alternating-refinement iteration or parent identity is inconsistent");
    }
    if (!observation.evaluation_succeeded) return validation_ok();
    if (observation.plan == nullptr) {
        return validation_error(validation_code::null_pointer, observation.iteration,
            "alternating-refinement observation plan is null");
    }
    validation_result status = observation.plan->validate();
    if (!status) return status;
    const packing_plan_identity &identity = observation.plan->identity();
    if (identity.feature_axis_fingerprint != config.feature_axis_identity
        || identity.feature_axis_fingerprint_version
            != config.feature_axis_identity_version
        || identity.row_domain_identity != config.row_domain_identity) {
        return validation_error(validation_code::invalid_plan_geometry, observation.iteration,
            "alternating-refinement plan identity disagrees with controller domain");
    }
    status = validate_metric_binding(*observation.plan, observation.training,
        config, training_baseline,
        "alternating-refinement training metrics disagree with controller domain");
    if (!status) return status;
    return validate_metric_binding(*observation.plan, observation.held_out,
        config, held_out_baseline,
        "alternating-refinement held-out metrics disagree with controller domain");
}

double improvement_tolerance(
    double current,
    const alternating_refinement_config &config) noexcept {
    return std::max(config.absolute_improvement_tolerance,
        config.relative_improvement_tolerance * std::fabs(current));
}

u64 controller_identity(
    const alternating_refinement_config &config,
    const alternating_refinement_result &result) noexcept {
    u64 hash = fnv1a_offset;
    hash_u64(&hash, controller_identity_domain);
    hash_u64(&hash, alternating_refinement_schema_version);
    hash_u64(&hash, config.dataset_identity);
    hash_u64(&hash, config.feature_axis_identity);
    hash_u64(&hash, config.feature_axis_identity_version);
    hash_u64(&hash, config.row_domain_identity);
    hash_u64(&hash, config.split_identity);
    hash_u64(&hash, config.workload_profile_identity);
    hash_u64(&hash, config.workload_evidence_revision);
    hash_u64(&hash, config.seed);
    hash_u64(&hash, config.maximum_iterations);
    hash_u64(&hash, config.maximum_evaluations);
    hash_u64(&hash, config.maximum_consecutive_rejections);
    hash_u64(&hash, config.maximum_preprocessing_nanoseconds);
    hash_u64(&hash, config.minimum_bootstrap_samples);
    hash_double(&hash, config.absolute_improvement_tolerance);
    hash_double(&hash, config.relative_improvement_tolerance);
    hash_double(&hash, config.weights.encoded_bytes);
    hash_double(&hash, config.weights.metadata_bytes);
    hash_double(&hash, config.weights.active_block_references);
    hash_double(&hash, config.weights.tile_block_union_references);
    hash_double(&hash, config.weights.padding_slots);
    hash_double(&hash, config.weights.runtime_mean_nanoseconds);
    hash_double(&hash, config.weights.preprocessing_mean_nanoseconds);
    hash_double(&hash, config.weights.forward_mean_nanoseconds);
    hash_double(&hash, config.weights.transpose_mean_nanoseconds);
    hash_double(&hash, config.weights.active_interaction_nanoseconds);
    hash_double(&hash, config.weights.partition_cut_edge_nanoseconds);
    hash_double(&hash, config.weights.bootstrap_mad_nanoseconds);
    hash_u64(&hash, result.best_plan_identity);
    hash_u64(&hash, result.event_count);
    for (u32 index = 0u; index < result.event_count; ++index) {
        const alternating_refinement_event &event = result.events[index];
        hash_u64(&hash, event.iteration);
        hash_u64(&hash, static_cast<u64>(event.phase));
        hash_u64(&hash, static_cast<u64>(event.outcome));
        hash_u64(&hash, event.candidate_identity);
        hash_u64(&hash, event.parent_plan_identity);
        hash_u64(&hash, event.candidate_plan_identity);
        hash_double(&hash, event.held_out_objective);
    }
    return nonzero_hash(hash);
}

} // namespace

u64 alternating_refinement_plan_identity(
    const frozen_packing_plan &plan) noexcept {
    u64 hash = fnv1a_offset;
    hash_u64(&hash, plan_identity_domain);
    hash_u64(&hash, frozen_plan_validation_identity_version);
    hash_u64(&hash, plan.semantic_schema_version());
    hash_u64(&hash, plan.row_count());
    hash_u64(&hash, plan.feature_count());
    hash_u64(&hash, plan.feature_block_count());
    hash_u64(&hash, plan.row_group_count());
    hash_u64(&hash, plan.maximum_feature_block_width());
    hash_u64(&hash, plan.row_group_width());
    hash_u64(&hash, plan.feature_block_geometry_identity());
    hash_u64(&hash, plan.identity().feature_axis_fingerprint);
    hash_u64(&hash, plan.identity().feature_axis_fingerprint_version);
    hash_u64(&hash, static_cast<u64>(plan.identity().row_domain_kind));
    hash_u64(&hash, plan.identity().row_domain_identity);
    hash_u64(&hash, plan.identity().evaluation_source_identity);
    hash_u64(&hash, plan.identity().sampling_provenance_identity);
    hash_u64(&hash, static_cast<u64>(plan.objective_kind()));
    hash_u64(&hash, plan.cost_policy_identity());
    for (u32 feature = 0u; feature < plan.feature_count(); ++feature) {
        hash_u64(&hash, plan.feature_permutation()[feature]);
    }
    for (u32 block = 0u; block <= plan.feature_block_count(); ++block) {
        hash_u64(&hash, plan.feature_block_offsets()[block]);
    }
    for (u32 group = 0u; group <= plan.row_group_count(); ++group) {
        hash_u64(&hash, plan.row_group_offsets()[group]);
    }
    return nonzero_hash(hash);
}

validation_result evaluate_alternating_refinement_objective(
    const packing_validation_metrics &metrics,
    const alternating_refinement_objective_weights &weights,
    double *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "alternating-refinement objective output is null");
    }
    validation_result status = validate_weights(weights);
    if (!status) return status;
    status = validate_packing_validation_metrics(metrics);
    if (!status) return status;
    if ((weights.encoded_bytes != 0.0 || weights.metadata_bytes != 0.0
            || weights.padding_slots != 0.0)
        && (metrics.available & packing_validation_metric_storage) == 0u) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "storage-weighted refinement requires storage metrics");
    }
    if (weights.active_block_references != 0.0
        && (metrics.available & packing_validation_metric_records) == 0u) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "record-weighted refinement requires record metrics");
    }
    if (weights.tile_block_union_references != 0.0
        && (metrics.available & packing_validation_metric_tiles) == 0u) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "tile-weighted refinement requires tile metrics");
    }
    if (weights.runtime_mean_nanoseconds != 0.0
        && ((metrics.available & packing_validation_metric_runtime) == 0u
            || metrics.runtime_repeat_count == 0u)) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "runtime-weighted refinement requires measured runtime repeats");
    }
    if (weights.preprocessing_mean_nanoseconds != 0.0
        && ((metrics.available & packing_validation_metric_preprocessing) == 0u
            || metrics.preprocessing_repeat_count == 0u)) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "preprocessing-weighted refinement requires measured preprocessing repeats");
    }
    const bool profile_weighted = weights.forward_mean_nanoseconds != 0.0
        || weights.transpose_mean_nanoseconds != 0.0
        || weights.active_interaction_nanoseconds != 0.0
        || weights.partition_cut_edge_nanoseconds != 0.0
        || weights.bootstrap_mad_nanoseconds != 0.0;
    if (profile_weighted
        && (metrics.available
            & packing_validation_metric_workload_profile) == 0u) {
        return validation_error(validation_code::invalid_matrix_view,
            invalid_id,
            "workload-weighted refinement requires measured workload profiles");
    }
    const double runtime_mean = metrics.runtime_repeat_count == 0u ? 0.0
        : static_cast<double>(metrics.runtime_elapsed_nanoseconds)
            / metrics.runtime_repeat_count;
    const double preprocessing_mean = metrics.preprocessing_repeat_count == 0u ? 0.0
        : static_cast<double>(metrics.preprocessing_elapsed_nanoseconds)
            / metrics.preprocessing_repeat_count;
    const double forward_mean = metrics.forward_repeat_count == 0u ? 0.0
        : static_cast<double>(metrics.forward_elapsed_nanoseconds)
            / metrics.forward_repeat_count;
    const double transpose_mean = metrics.transpose_repeat_count == 0u ? 0.0
        : static_cast<double>(metrics.transpose_elapsed_nanoseconds)
            / metrics.transpose_repeat_count;
    const double result = weights.encoded_bytes * metrics.encoded_bytes
        + weights.metadata_bytes * metrics.metadata_bytes
        + weights.active_block_references * metrics.active_block_references
        + weights.tile_block_union_references * metrics.tile_block_union_references
        + weights.padding_slots * metrics.padding_slots
        + weights.runtime_mean_nanoseconds * runtime_mean
        + weights.preprocessing_mean_nanoseconds * preprocessing_mean
        + weights.forward_mean_nanoseconds * forward_mean
        + weights.transpose_mean_nanoseconds * transpose_mean
        + weights.active_interaction_nanoseconds * metrics.active_interactions
        + weights.partition_cut_edge_nanoseconds * metrics.partition_cut_edges
        + weights.bootstrap_mad_nanoseconds * metrics.bootstrap_mad_nanoseconds;
    if (!std::isfinite(result)) {
        return validation_error(validation_code::integer_overflow, invalid_id,
            "alternating-refinement objective is not finite");
    }
    *out = result;
    return validation_ok();
}

validation_result run_alternating_refinement(
    const alternating_refinement_observation &baseline,
    const alternating_refinement_observation *candidates,
    u32 candidate_count,
    const alternating_refinement_config &config,
    const alternating_refinement_buffers &buffers,
    alternating_refinement_result *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "alternating-refinement result is null");
    }
    validation_result status = validate_config(config);
    if (!status) return status;
    if (candidate_count != 0u && candidates == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "alternating-refinement candidate sequence is null");
    }
    const u32 event_limit = std::min(candidate_count, config.maximum_iterations);
    if (event_limit > buffers.event_capacity
        || (event_limit != 0u && buffers.events == nullptr)) {
        return validation_error(validation_code::insufficient_capacity, event_limit,
            "alternating-refinement event capacity is insufficient");
    }
    status = validate_observation(baseline, config, nullptr, nullptr, true, 0u, 0u);
    if (!status) return status;

    alternating_refinement_result result;
    result.best_plan = baseline.plan;
    result.best_plan_identity = alternating_refinement_plan_identity(*baseline.plan);
    result.best_training = baseline.training;
    result.best_held_out = baseline.held_out;
    result.events = buffers.events;
    status = evaluate_alternating_refinement_objective(
        baseline.training, config.weights, &result.best_training_objective);
    if (!status) return status;
    status = evaluate_alternating_refinement_objective(
        baseline.held_out, config.weights, &result.best_held_out_objective);
    if (!status) return status;
    result.total_preprocessing_nanoseconds = baseline.training.preprocessing_elapsed_nanoseconds;
    if (add_overflows(result.total_preprocessing_nanoseconds,
            baseline.held_out.preprocessing_elapsed_nanoseconds,
            &result.total_preprocessing_nanoseconds)) {
        return validation_error(validation_code::integer_overflow, invalid_id,
            "alternating-refinement preprocessing total overflowed");
    }

    for (u32 index = 0u; index < event_limit; ++index) {
        if (result.evaluated_candidates >= config.maximum_evaluations) {
            result.stop_reason = alternating_refinement_stop_reason::evaluation_cap;
            break;
        }
        const alternating_refinement_observation &candidate = candidates[index];
        const alternating_refinement_phase expected_phase = (index & 1u) == 0u
            ? alternating_refinement_phase::gene_blocks
            : alternating_refinement_phase::cell_order_and_tiles;
        if (candidate.phase != expected_phase) {
            return validation_error(validation_code::invalid_plan_geometry, index,
                "alternating-refinement candidates do not alternate gene and cell phases");
        }
        status = validate_observation(candidate, config,
            &baseline.training, &baseline.held_out, false,
            result.best_plan_identity, index + 1u);
        if (!status) return status;

        alternating_refinement_event &event = buffers.events[index];
        event = {};
        event.iteration = candidate.iteration;
        event.phase = candidate.phase;
        event.candidate_identity = candidate.candidate_identity;
        event.parent_plan_identity = candidate.parent_plan_identity;
        event.previous_best_held_out_objective = result.best_held_out_objective;
        ++result.attempted_iterations;
        ++result.event_count;

        if (!candidate.evaluation_succeeded) {
            event.outcome = alternating_refinement_outcome::rejected_evaluation_error;
            ++result.evaluation_errors;
            ++result.rejected_candidates;
            ++result.consecutive_rejections;
        } else {
            ++result.evaluated_candidates;
            event.candidate_plan_identity = alternating_refinement_plan_identity(*candidate.plan);
            status = evaluate_alternating_refinement_objective(
                candidate.training, config.weights, &event.training_objective);
            if (!status) return status;
            status = evaluate_alternating_refinement_objective(
                candidate.held_out, config.weights, &event.held_out_objective);
            if (!status) return status;
            event.held_out_improvement = result.best_held_out_objective
                - event.held_out_objective;
            const bool accept = event.held_out_improvement
                > improvement_tolerance(result.best_held_out_objective, config);
            if (accept) {
                event.outcome = alternating_refinement_outcome::accepted;
                result.best_plan = candidate.plan;
                result.best_plan_identity = event.candidate_plan_identity;
                result.best_training = candidate.training;
                result.best_held_out = candidate.held_out;
                result.best_training_objective = event.training_objective;
                result.best_held_out_objective = event.held_out_objective;
                ++result.accepted_candidates;
                result.consecutive_rejections = 0u;
            } else {
                event.outcome = alternating_refinement_outcome::rejected_no_improvement;
                ++result.rejected_candidates;
                ++result.consecutive_rejections;
            }
            u64 candidate_preprocessing = 0u;
            if (add_overflows(candidate.training.preprocessing_elapsed_nanoseconds,
                    candidate.held_out.preprocessing_elapsed_nanoseconds,
                    &candidate_preprocessing)
                || add_overflows(result.total_preprocessing_nanoseconds,
                    candidate_preprocessing, &result.total_preprocessing_nanoseconds)) {
                return validation_error(validation_code::integer_overflow, index,
                    "alternating-refinement preprocessing total overflowed");
            }
        }

        if (config.maximum_preprocessing_nanoseconds != 0u
            && result.total_preprocessing_nanoseconds
                >= config.maximum_preprocessing_nanoseconds) {
            result.stop_reason = alternating_refinement_stop_reason::preprocessing_cap;
            break;
        }
        if (config.maximum_consecutive_rejections != 0u
            && result.consecutive_rejections
                >= config.maximum_consecutive_rejections) {
            result.stop_reason = alternating_refinement_stop_reason::convergence;
            break;
        }
        if (index + 1u == config.maximum_iterations
            && candidate_count > config.maximum_iterations) {
            result.stop_reason = alternating_refinement_stop_reason::iteration_cap;
        }
    }
    result.controller_identity = controller_identity(config, result);
    *out = result;
    return validation_ok();
}

} // namespace cellpack
