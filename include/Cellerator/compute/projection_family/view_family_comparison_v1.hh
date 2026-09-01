#pragma once

#include <Cellerator/compute/projection_family/support_family_identity_v1.hh>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace cellerator::compute::projection_family {

enum class view_family_kind_v1 : std::uint8_t {
    specialized = 1,
    generalized = 2,
};

enum class correctness_evidence_kind_v1 : std::uint8_t {
    independent_reference = 1,
    provider_self_report = 2,
};

// Complete measured end-to-end cost record. Per-launch cost includes kernel,
// epilogue, order transforms, synchronization, and communication separately so
// a microkernel-only result cannot enter comparison as complete evidence.
struct view_family_measurement_v1 {
    operation::v2::stable_id candidate_identity{};
    operation::v2::stable_id evidence_identity{};
    support_family_identity_v1 family{};
    view_family_kind_v1 kind = view_family_kind_v1::specialized;
    correctness_evidence_kind_v1 correctness =
        correctness_evidence_kind_v1::independent_reference;
    std::uint16_t reserved0 = 0;
    std::uint32_t supported_operations = 0;
    std::uint64_t preparation_ns = 0;
    std::uint64_t persistent_preprocess_ns = 0;
    std::uint64_t input_pack_ns = 0;
    std::uint64_t kernel_ns = 0;
    std::uint64_t epilogue_ns = 0;
    std::uint64_t output_transform_ns = 0;
    std::uint64_t synchronization_ns = 0;
    std::uint64_t communication_ns = 0;
    std::uint64_t persistent_bytes = 0;
    std::uint64_t transient_bytes = 0;
    std::uint64_t launch_count = 0;
    std::uint64_t warmup_count = 0;
    std::uint64_t repeat_count = 0;
};

enum class view_family_comparison_code_v1 : std::uint32_t {
    compared = 0,
    invalid_left,
    invalid_right,
    family_mismatch,
    operation_mismatch,
    zero_expected_reuse,
    arithmetic_overflow,
};

enum class view_family_winner_v1 : std::uint8_t {
    left = 1,
    right = 2,
    exact_tie = 3,
};

enum class measurement_validation_code_v1 : std::uint32_t {
    valid = 0,
    invalid_candidate_identity,
    invalid_evidence_identity,
    invalid_family,
    invalid_kind,
    non_independent_correctness,
    nonzero_reserved,
    empty_operation_set,
    unknown_operation,
    incomplete_timing,
    zero_launch_count,
    zero_repeat_count,
};

struct measurement_validation_v1 {
    measurement_validation_code_v1 code = measurement_validation_code_v1::valid;
    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == measurement_validation_code_v1::valid;
    }
};

struct view_family_comparison_result_v1 {
    view_family_comparison_code_v1 code =
        view_family_comparison_code_v1::compared;
    view_family_winner_v1 latency_winner = view_family_winner_v1::exact_tie;
    view_family_winner_v1 persistent_memory_winner =
        view_family_winner_v1::exact_tie;
    view_family_winner_v1 transient_memory_winner =
        view_family_winner_v1::exact_tie;
    std::uint8_t reserved = 0;
    std::uint64_t left_total_ns = 0;
    std::uint64_t right_total_ns = 0;
    std::uint64_t expected_reuse = 0;
    [[nodiscard]] constexpr bool compared() const noexcept {
        return code == view_family_comparison_code_v1::compared;
    }
};

[[nodiscard]] constexpr bool valid_view_family_kind_v1(
    view_family_kind_v1 kind) noexcept {
    return kind == view_family_kind_v1::specialized
        || kind == view_family_kind_v1::generalized;
}

[[nodiscard]] constexpr measurement_validation_v1
validate_view_family_measurement_v1(
    const view_family_measurement_v1 &measurement) noexcept {
    if (!operation::v2::valid_stable_id(measurement.candidate_identity)) {
        return {measurement_validation_code_v1::invalid_candidate_identity};
    }
    if (!operation::v2::valid_stable_id(measurement.evidence_identity)) {
        return {measurement_validation_code_v1::invalid_evidence_identity};
    }
    if (!validate_support_family_identity_v1(measurement.family).valid()) {
        return {measurement_validation_code_v1::invalid_family};
    }
    if (!valid_view_family_kind_v1(measurement.kind)) {
        return {measurement_validation_code_v1::invalid_kind};
    }
    if (measurement.correctness
        != correctness_evidence_kind_v1::independent_reference) {
        return {measurement_validation_code_v1::non_independent_correctness};
    }
    if (measurement.reserved0 != 0) {
        return {measurement_validation_code_v1::nonzero_reserved};
    }
    if (measurement.supported_operations == 0) {
        return {measurement_validation_code_v1::empty_operation_set};
    }
    if ((measurement.supported_operations
         & ~known_support_family_operations_v1) != 0) {
        return {measurement_validation_code_v1::unknown_operation};
    }
    if (measurement.kernel_ns == 0) {
        return {measurement_validation_code_v1::incomplete_timing};
    }
    if (measurement.launch_count == 0) {
        return {measurement_validation_code_v1::zero_launch_count};
    }
    if (measurement.repeat_count == 0) {
        return {measurement_validation_code_v1::zero_repeat_count};
    }
    return {};
}

[[nodiscard]] constexpr view_family_winner_v1 lower_wins_v1(
    std::uint64_t left, std::uint64_t right) noexcept {
    return left < right ? view_family_winner_v1::left
        : right < left ? view_family_winner_v1::right
                       : view_family_winner_v1::exact_tie;
}

[[nodiscard]] constexpr bool checked_add_v1(
    std::uint64_t lhs, std::uint64_t rhs, std::uint64_t *output) noexcept {
    if (output == nullptr
        || lhs > std::numeric_limits<std::uint64_t>::max() - rhs) {
        return false;
    }
    *output = lhs + rhs;
    return true;
}

[[nodiscard]] constexpr bool measured_total_ns_v1(
    const view_family_measurement_v1 &measurement,
    std::uint64_t expected_reuse,
    std::uint64_t *total) noexcept {
    if (total == nullptr || expected_reuse == 0) return false;
    std::uint64_t setup = 0;
    if (!checked_add_v1(measurement.preparation_ns,
                        measurement.persistent_preprocess_ns, &setup)
        || !checked_add_v1(setup, measurement.input_pack_ns, &setup)) {
        return false;
    }
    std::uint64_t per_execution = 0;
    if (!checked_add_v1(measurement.kernel_ns, measurement.epilogue_ns,
                        &per_execution)
        || !checked_add_v1(per_execution, measurement.output_transform_ns,
                           &per_execution)
        || !checked_add_v1(per_execution, measurement.synchronization_ns,
                           &per_execution)
        || !checked_add_v1(per_execution, measurement.communication_ns,
                           &per_execution)) {
        return false;
    }
    if (per_execution != 0
        && expected_reuse
               > std::numeric_limits<std::uint64_t>::max() / per_execution) {
        return false;
    }
    return checked_add_v1(setup, per_execution * expected_reuse, total);
}

[[nodiscard]] constexpr view_family_comparison_result_v1
compare_view_families_v1(
    const view_family_measurement_v1 &left,
    const view_family_measurement_v1 &right,
    std::uint64_t expected_reuse) noexcept {
    if (!validate_view_family_measurement_v1(left).valid()) {
        return {view_family_comparison_code_v1::invalid_left};
    }
    if (!validate_view_family_measurement_v1(right).valid()) {
        return {view_family_comparison_code_v1::invalid_right};
    }
    if (!same_support_family_identity_v1(left.family, right.family)) {
        return {view_family_comparison_code_v1::family_mismatch};
    }
    if (left.supported_operations != right.supported_operations) {
        return {view_family_comparison_code_v1::operation_mismatch};
    }
    if (expected_reuse == 0) {
        return {view_family_comparison_code_v1::zero_expected_reuse};
    }
    std::uint64_t left_total = 0;
    std::uint64_t right_total = 0;
    if (!measured_total_ns_v1(left, expected_reuse, &left_total)
        || !measured_total_ns_v1(right, expected_reuse, &right_total)) {
        return {view_family_comparison_code_v1::arithmetic_overflow};
    }
    return {view_family_comparison_code_v1::compared,
            lower_wins_v1(left_total, right_total),
            lower_wins_v1(left.persistent_bytes, right.persistent_bytes),
            lower_wins_v1(left.transient_bytes, right.transient_bytes),
            0, left_total, right_total, expected_reuse};
}

static_assert(std::is_trivially_copyable_v<view_family_measurement_v1>);
static_assert(std::is_trivially_copyable_v<view_family_comparison_result_v1>);

} // namespace cellerator::compute::projection_family
