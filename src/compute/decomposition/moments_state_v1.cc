#include <Cellerator/compute/decomposition/moments_state_v1.hh>

#include <cmath>
#include <limits>

namespace cellerator::compute::decomposition {

moments_state_v1 empty_moments_state_v1() noexcept {
    return {};
}

moments_state_v1 singleton_moments_state_v1(double value) noexcept {
    moments_state_v1 state{};
    state.sample_count = 1u;
    state.mean = value;
    return state;
}

moments_state_validation_code_v1 validate_moments_state_v1(
    const moments_state_v1 &state) noexcept {
    using code = moments_state_validation_code_v1;
    if (state.schema_version != moments_state_schema_version_v1)
        return code::unsupported_schema;
    if (state.reserved != 0u)
        return code::nonzero_reserved;
    if (state.sample_count == 0u)
        return state.mean == 0.0 && state.m2 == 0.0
            ? code::ok : code::invalid_empty_state;
    if (!std::isfinite(state.mean))
        return code::invalid_mean;
    if (!std::isfinite(state.m2) || state.m2 < 0.0)
        return code::invalid_m2;
    return code::ok;
}

moments_merge_result_v1 merge_moments_state_v1(
    const moments_state_v1 &left,
    const moments_state_v1 &right) noexcept {
    if (validate_moments_state_v1(left)
        != moments_state_validation_code_v1::ok)
        return {{}, moments_merge_code_v1::invalid_left};
    if (validate_moments_state_v1(right)
        != moments_state_validation_code_v1::ok)
        return {{}, moments_merge_code_v1::invalid_right};
    if (left.sample_count == 0u)
        return {right, moments_merge_code_v1::ok};
    if (right.sample_count == 0u)
        return {left, moments_merge_code_v1::ok};
    if (right.sample_count
        > std::numeric_limits<std::uint64_t>::max() - left.sample_count)
        return {{}, moments_merge_code_v1::count_overflow};

    moments_state_v1 result{};
    result.sample_count = left.sample_count + right.sample_count;
    const double left_count = static_cast<double>(left.sample_count);
    const double right_count = static_cast<double>(right.sample_count);
    const double total_count = static_cast<double>(result.sample_count);
    const double delta = right.mean - left.mean;
    result.mean = left.mean + delta * (right_count / total_count);
    result.m2 = left.m2 + right.m2
        + delta * delta * (left_count * right_count / total_count);
    return {result, moments_merge_code_v1::ok};
}

double population_variance_v1(const moments_state_v1 &state) noexcept {
    if (state.sample_count == 0u)
        return std::numeric_limits<double>::quiet_NaN();
    return state.m2 / static_cast<double>(state.sample_count);
}

double sample_variance_v1(const moments_state_v1 &state) noexcept {
    if (state.sample_count < 2u)
        return std::numeric_limits<double>::quiet_NaN();
    return state.m2 / static_cast<double>(state.sample_count - 1u);
}

}  // namespace cellerator::compute::decomposition
