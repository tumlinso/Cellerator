#include <Cellerator/compute/decomposition/log_sum_exp_state_v1.hh>

#include <algorithm>
#include <cmath>
#include <limits>

namespace cellerator::compute::decomposition {

log_sum_exp_state_v1 empty_log_sum_exp_state_v1() noexcept {
    return {};
}

log_sum_exp_state_v1 singleton_log_sum_exp_state_v1(double value) noexcept {
    log_sum_exp_state_v1 state{};
    state.maximum = value;
    state.scaled_exponential_sum = 1.0;
    state.sample_count = 1u;
    return state;
}

log_sum_exp_state_v1 merge_log_sum_exp_state_v1(
    const log_sum_exp_state_v1 &left,
    const log_sum_exp_state_v1 &right) noexcept {
    if (left.sample_count == 0u)
        return right;
    if (right.sample_count == 0u)
        return left;

    log_sum_exp_state_v1 result{};
    result.maximum = std::max(left.maximum, right.maximum);
    result.scaled_exponential_sum = left.scaled_exponential_sum
            * std::exp(left.maximum - result.maximum)
        + right.scaled_exponential_sum
            * std::exp(right.maximum - result.maximum);
    if (right.sample_count
        > std::numeric_limits<std::uint64_t>::max() - left.sample_count)
        result.sample_count = std::numeric_limits<std::uint64_t>::max();
    else
        result.sample_count = left.sample_count + right.sample_count;
    return result;
}

log_sum_exp_state_validation_code_v1 validate_log_sum_exp_state_v1(
    const log_sum_exp_state_v1 &state) noexcept {
    using code = log_sum_exp_state_validation_code_v1;
    if (state.schema_version != log_sum_exp_state_schema_version_v1)
        return code::unsupported_schema;
    if (state.reserved != 0u)
        return code::nonzero_reserved;
    if (state.sample_count == 0u) {
        return state.maximum == 0.0 && state.scaled_exponential_sum == 0.0
            ? code::ok : code::invalid_empty_state;
    }
    if (!std::isfinite(state.maximum))
        return code::invalid_maximum;
    if (!std::isfinite(state.scaled_exponential_sum)
        || state.scaled_exponential_sum <= 0.0)
        return code::invalid_scaled_sum;
    return code::ok;
}

double finalize_log_sum_exp_v1(const log_sum_exp_state_v1 &state) noexcept {
    if (state.sample_count == 0u)
        return -std::numeric_limits<double>::infinity();
    return state.maximum + std::log(state.scaled_exponential_sum);
}

double softmax_probability_v1(
    const log_sum_exp_state_v1 &state, double value) noexcept {
    if (state.sample_count == 0u)
        return std::numeric_limits<double>::quiet_NaN();
    return std::exp(value - state.maximum) / state.scaled_exponential_sum;
}

}  // namespace cellerator::compute::decomposition
