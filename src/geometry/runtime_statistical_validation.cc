#include "Cellerator/geometry/runtime_statistical_validation.hh"

#include <cmath>

namespace cellpack {
namespace {

struct scalar_accumulator {
    u32 count = 0u;
    double minimum = 0.0;
    double mean = 0.0;
    double maximum = 0.0;
    double squared_deviation = 0.0;

    void add(double value) noexcept {
        if (count == 0u) {
            count = 1u;
            minimum = mean = maximum = value;
            return;
        }
        minimum = value < minimum ? value : minimum;
        maximum = value > maximum ? value : maximum;
        ++count;
        const double delta = value - mean;
        mean += delta / count;
        squared_deviation += delta * (value - mean);
    }

    bootstrap_scalar_summary finish() const noexcept {
        bootstrap_scalar_summary result;
        result.observation_count = count;
        if (count == 0u) return result;
        result.minimum = minimum;
        result.mean = mean;
        result.maximum = maximum;
        result.sample_standard_deviation = count < 2u ? 0.0
            : std::sqrt(squared_deviation / static_cast<double>(count - 1u));
        return result;
    }
};

validation_result validate_refinement(
    const alternating_refinement_result &refinement) {
    if (refinement.schema_version != alternating_refinement_schema_version) {
        return validation_error(validation_code::unsupported_version,
            refinement.schema_version, "runtime stability received an unsupported controller schema");
    }
    if (refinement.controller_identity == 0u || refinement.best_plan_identity == 0u
        || refinement.best_plan == nullptr || refinement.evaluated_candidates == 0u) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "runtime stability requires an executed CP-BP-10 controller result");
    }
    validation_result status = refinement.best_plan->validate();
    if (!status) return status;
    if (alternating_refinement_plan_identity(*refinement.best_plan)
        != refinement.best_plan_identity) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "runtime stability controller plan identity is inconsistent");
    }
    status = validate_packing_validation_metrics(refinement.best_training);
    if (!status) return status;
    status = validate_packing_validation_metrics(refinement.best_held_out);
    if (!status) return status;
    if ((refinement.best_training.available & packing_validation_metric_correctness) == 0u
        || (refinement.best_held_out.available & packing_validation_metric_correctness) == 0u
        || refinement.best_training.correctness_mismatches != 0u
        || refinement.best_held_out.correctness_mismatches != 0u) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "runtime stability controller result lacks exact correctness");
    }
    return validation_ok();
}

validation_result validate_input_binding(
    const validation_identity_view &identities,
    const relearned_plan_runtime_input &input) {
    if (input.bootstrap_provenance == nullptr || input.row_multiplicities == nullptr
        || input.refinement == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "runtime stability input pointers are null");
    }
    validation_result status = validate_validation_bootstrap(
        identities, input.row_multiplicities, *input.bootstrap_provenance);
    if (!status) return status;
    status = validate_refinement(*input.refinement);
    if (!status) return status;
    status = validate_relearned_plan_runtime_observation(input.runtime);
    if (!status) return status;
    const alternating_refinement_result &refinement = *input.refinement;
    if (input.runtime.bootstrap_identity
            != input.bootstrap_provenance->bootstrap_identity
        || input.runtime.controller_identity != refinement.controller_identity
        || input.runtime.plan_identity != refinement.best_plan_identity
        || input.runtime.split_identity != refinement.best_held_out.split_identity
        || input.runtime.dataset_identity != refinement.best_held_out.dataset_identity
        || input.runtime.feature_axis_identity
            != refinement.best_held_out.feature_axis_identity
        || input.runtime.row_domain_identity
            != refinement.best_held_out.row_domain_identity) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "runtime observation disagrees with bootstrap/controller output identities");
    }
    return validation_ok();
}

validation_result compare_mapping(
    const frozen_packing_plan &reference,
    const frozen_packing_plan &candidate,
    u64 *pair_count,
    u64 *agreements,
    u64 *disagreements) noexcept {
    if (reference.feature_count() != candidate.feature_count()
        || reference.identity().feature_axis_fingerprint
            != candidate.identity().feature_axis_fingerprint
        || reference.identity().feature_axis_fingerprint_version
            != candidate.identity().feature_axis_fingerprint_version) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "runtime stability mappings use incompatible canonical feature axes");
    }
    u64 pairs = 0u, same = 0u;
    for (u32 lhs = 0u; lhs < reference.feature_count(); ++lhs) {
        for (u32 rhs = lhs + 1u; rhs < reference.feature_count(); ++rhs) {
            ++pairs;
            const bool reference_together = reference.feature_to_block()[lhs]
                == reference.feature_to_block()[rhs];
            const bool candidate_together = candidate.feature_to_block()[lhs]
                == candidate.feature_to_block()[rhs];
            if (reference_together == candidate_together) ++same;
        }
    }
    *pair_count = pairs;
    *agreements = same;
    *disagreements = pairs - same;
    return validation_ok();
}

} // namespace

validation_result validate_relearned_plan_runtime_observation(
    const relearned_plan_runtime_observation &observation) {
    if (observation.schema_version != runtime_statistical_validation_schema_version) {
        return validation_error(validation_code::unsupported_version,
            observation.schema_version, "unsupported runtime-stability observation schema");
    }
    if (observation.timing_scope != runtime_timing_scope::device_resident_kernel) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "runtime-stability timing scope is unsupported");
    }
    if (observation.controller_identity == 0u || observation.plan_identity == 0u
        || observation.bootstrap_identity == 0u || observation.split_identity == 0u
        || observation.dataset_identity == 0u || observation.feature_axis_identity == 0u
        || observation.row_domain_identity == 0u || observation.ordering_identity == 0u
        || observation.tile_identity == 0u || observation.operation_identity == 0u
        || observation.feature_weight_identity == 0u || observation.hardware_identity == 0u
        || observation.toolchain_identity == 0u) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "runtime-stability observation identities must be explicit");
    }
    if (!observation.observed) {
        if (observation.input_nnz != 0u || observation.input_bytes != 0u
            || observation.elapsed_nanoseconds != 0u
            || observation.correctness_items != 0u
            || observation.correctness_mismatches != 0u
            || observation.warmup_count != 0u || observation.repeat_count != 0u
            || observation.launches_per_repeat != 0u) {
            return validation_error(validation_code::invalid_matrix_view, invalid_id,
                "unobserved runtime packet must retain zero raw denominators");
        }
        return validation_ok();
    }
    if (observation.input_nnz == 0u || observation.input_bytes == 0u
        || observation.elapsed_nanoseconds == 0u || observation.correctness_items == 0u
        || observation.correctness_mismatches > observation.correctness_items
        || observation.warmup_count == 0u || observation.repeat_count == 0u
        || observation.launches_per_repeat == 0u) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "runtime-stability observation lacks raw work/timing/correctness denominators");
    }
    if (observation.correctness_mismatches != 0u) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "runtime-stability observation is not numerically correct");
    }
    return validation_ok();
}

validation_result evaluate_relearned_plan_runtime_stability(
    const validation_identity_view &identities,
    const relearned_plan_runtime_input *inputs,
    u32 input_count,
    const relearned_plan_runtime_buffers &buffers,
    relearned_plan_runtime_stability_summary *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "runtime-stability summary output is null");
    }
    if (input_count == 0u || inputs == nullptr) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "runtime-stability inputs must be nonempty");
    }
    if (input_count > buffers.replicate_capacity || buffers.replicates == nullptr) {
        return validation_error(validation_code::insufficient_capacity, input_count,
            "runtime-stability replicate capacity is insufficient");
    }
    for (u32 index = 0u; index < input_count; ++index) {
        validation_result status = validate_input_binding(identities, inputs[index]);
        if (!status) return status;
        for (u32 prior = 0u; prior < index; ++prior) {
            if (inputs[prior].bootstrap_provenance->bootstrap_identity
                == inputs[index].bootstrap_provenance->bootstrap_identity) {
                return validation_error(validation_code::duplicate_id, index,
                    "runtime-stability bootstrap identities must be unique");
            }
        }
    }

    const relearned_plan_runtime_observation &reference_runtime = inputs[0].runtime;
    const frozen_packing_plan &reference_plan = *inputs[0].refinement->best_plan;
    relearned_plan_runtime_stability_summary result;
    result.repeat_count = input_count;
    result.reference_plan_identity = inputs[0].refinement->best_plan_identity;
    result.dataset_identity = reference_runtime.dataset_identity;
    result.feature_axis_identity = reference_runtime.feature_axis_identity;
    result.row_domain_identity = reference_runtime.row_domain_identity;
    result.split_identity = reference_runtime.split_identity;
    result.operation_identity = reference_runtime.operation_identity;
    result.feature_weight_identity = reference_runtime.feature_weight_identity;
    result.hardware_identity = reference_runtime.hardware_identity;
    result.toolchain_identity = reference_runtime.toolchain_identity;
    result.unit_kind = inputs[0].bootstrap_provenance->unit_kind;
    result.claims_group_generalization = result.unit_kind
        == validation_unit_kind::caller_group_identity;

    scalar_accumulator encoded, metadata, preprocessing, runtime_mean;
    scalar_accumulator nnz_per_second, gigabytes_per_second, mapping_agreement;
    for (u32 index = 0u; index < input_count; ++index) {
        const relearned_plan_runtime_input &input = inputs[index];
        const relearned_plan_runtime_observation &runtime = input.runtime;
        if (runtime.timing_scope != reference_runtime.timing_scope
            || runtime.dataset_identity != result.dataset_identity
            || runtime.feature_axis_identity != result.feature_axis_identity
            || runtime.row_domain_identity != result.row_domain_identity
            || runtime.split_identity != result.split_identity
            || runtime.operation_identity != result.operation_identity
            || runtime.feature_weight_identity != result.feature_weight_identity
            || runtime.hardware_identity != result.hardware_identity
            || runtime.toolchain_identity != result.toolchain_identity
            || input.bootstrap_provenance->unit_kind != result.unit_kind) {
            return validation_error(validation_code::invalid_plan_geometry, index,
                "runtime-stability replicates mix incompatible provenance or timing scopes");
        }

        relearned_plan_runtime_replicate replicate;
        replicate.bootstrap_identity = runtime.bootstrap_identity;
        replicate.controller_identity = input.refinement->controller_identity;
        replicate.plan_identity = input.refinement->best_plan_identity;
        replicate.feature_block_geometry_identity =
            input.refinement->best_plan->feature_block_geometry_identity();
        validation_result status = compare_mapping(reference_plan,
            *input.refinement->best_plan, &replicate.co_membership_pair_count,
            &replicate.co_membership_agreements,
            &replicate.co_membership_disagreements);
        if (!status) return status;
        replicate.exact_label_invariant_mapping =
            replicate.co_membership_disagreements == 0u;
        replicate.runtime = runtime;
        replicate.training = input.refinement->best_training;
        replicate.held_out = input.refinement->best_held_out;
        buffers.replicates[index] = replicate;
        if (replicate.exact_label_invariant_mapping) ++result.exact_mapping_count;

        encoded.add(static_cast<double>(replicate.held_out.encoded_bytes));
        metadata.add(static_cast<double>(replicate.held_out.metadata_bytes));
        if (replicate.held_out.preprocessing_repeat_count != 0u) {
            preprocessing.add(static_cast<double>(
                replicate.held_out.preprocessing_elapsed_nanoseconds)
                / replicate.held_out.preprocessing_repeat_count);
        }
        if (runtime.observed) {
            const double elapsed_mean = static_cast<double>(runtime.elapsed_nanoseconds)
                / runtime.repeat_count;
            runtime_mean.add(elapsed_mean);
            nnz_per_second.add(static_cast<double>(runtime.input_nnz)
                * 1.0e9 / elapsed_mean);
            gigabytes_per_second.add(static_cast<double>(runtime.input_bytes)
                / elapsed_mean);
        }
        if (replicate.co_membership_pair_count != 0u) {
            mapping_agreement.add(static_cast<double>(
                replicate.co_membership_agreements)
                / replicate.co_membership_pair_count);
        }
    }
    result.encoded_bytes = encoded.finish();
    result.metadata_bytes = metadata.finish();
    result.preprocessing_mean_nanoseconds = preprocessing.finish();
    result.runtime_mean_nanoseconds = runtime_mean.finish();
    result.runtime_nnz_per_second = nnz_per_second.finish();
    result.runtime_gigabytes_per_second = gigabytes_per_second.finish();
    result.co_membership_agreement_fraction = mapping_agreement.finish();
    *out = result;
    return validation_ok();
}

} // namespace cellpack
