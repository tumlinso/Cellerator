#pragma once

#include <cstdint>
#include <type_traits>

namespace cellerator::compute::decomposition {

inline constexpr std::uint32_t log_sum_exp_state_schema_version_v1 = 1u;

// A stable merge state for log-sum-exp and softmax normalization.  For a
// nonempty set X it represents max(X) and sum(exp(x - max(X))).  Empty state
// is an exact identity and sample_count preserves empty/singleton semantics.
struct log_sum_exp_state_v1 {
    std::uint32_t schema_version = log_sum_exp_state_schema_version_v1;
    std::uint32_t reserved = 0u;
    double maximum = 0.0;
    double scaled_exponential_sum = 0.0;
    std::uint64_t sample_count = 0u;
};

enum class log_sum_exp_state_validation_code_v1 : std::uint8_t {
    ok = 0u,
    unsupported_schema,
    nonzero_reserved,
    invalid_empty_state,
    invalid_maximum,
    invalid_scaled_sum
};

log_sum_exp_state_v1 empty_log_sum_exp_state_v1() noexcept;
log_sum_exp_state_v1 singleton_log_sum_exp_state_v1(double value) noexcept;
log_sum_exp_state_v1 merge_log_sum_exp_state_v1(
    const log_sum_exp_state_v1 &left,
    const log_sum_exp_state_v1 &right) noexcept;

log_sum_exp_state_validation_code_v1 validate_log_sum_exp_state_v1(
    const log_sum_exp_state_v1 &state) noexcept;

double finalize_log_sum_exp_v1(const log_sum_exp_state_v1 &state) noexcept;
double softmax_probability_v1(
    const log_sum_exp_state_v1 &state, double value) noexcept;

static_assert(std::is_trivially_copyable_v<log_sum_exp_state_v1>);
static_assert(std::is_standard_layout_v<log_sum_exp_state_v1>);

}  // namespace cellerator::compute::decomposition
