#include <Cellerator/compute/decomposition/moments_state_v1.hh>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>

namespace decomposition = cellerator::compute::decomposition;

namespace {

bool close(double left, double right, double tolerance = 1.0e-12) {
    return std::abs(left - right) <= tolerance;
}

decomposition::moments_state_v1 merge_values(
    const double *values, std::uint64_t count) {
    auto state = decomposition::empty_moments_state_v1();
    for (std::uint64_t index = 0u; index < count; ++index) {
        const auto merged = decomposition::merge_moments_state_v1(state,
            decomposition::singleton_moments_state_v1(values[index]));
        assert(merged);
        state = merged.state;
    }
    return state;
}

}  // namespace

int main() {
    const double values[] = {1.0, 2.0, 3.0, 4.0};
    const auto all = merge_values(values, 4u);
    const auto left = merge_values(values, 2u);
    const auto right = merge_values(values + 2, 2u);
    const auto partitioned =
        decomposition::merge_moments_state_v1(left, right);
    assert(partitioned);
    assert(partitioned.state.sample_count == 4u);
    assert(close(partitioned.state.mean, all.mean));
    assert(close(partitioned.state.m2, all.m2));
    assert(close(partitioned.state.mean, 2.5));
    assert(close(decomposition::population_variance_v1(partitioned.state),
        1.25));
    assert(close(decomposition::sample_variance_v1(partitioned.state),
        5.0 / 3.0));

    const auto empty = decomposition::empty_moments_state_v1();
    assert(std::isnan(decomposition::population_variance_v1(empty)));
    assert(std::isnan(decomposition::sample_variance_v1(
        decomposition::singleton_moments_state_v1(1.0))));
    const auto identity =
        decomposition::merge_moments_state_v1(empty, all);
    assert(identity && identity.state.sample_count == all.sample_count);

    auto invalid = all;
    invalid.m2 = -1.0;
    const auto rejected =
        decomposition::merge_moments_state_v1(invalid, all);
    assert(rejected.code == decomposition::moments_merge_code_v1::invalid_left);

    auto huge = decomposition::singleton_moments_state_v1(0.0);
    huge.sample_count = std::numeric_limits<std::uint64_t>::max();
    const auto overflow = decomposition::merge_moments_state_v1(
        huge, decomposition::singleton_moments_state_v1(0.0));
    assert(overflow.code
        == decomposition::moments_merge_code_v1::count_overflow);
    return 0;
}
