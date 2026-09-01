#include <Cellerator/compute/decomposition/log_sum_exp_state_v1.hh>

#include <cassert>
#include <cmath>
#include <limits>

namespace decomposition = cellerator::compute::decomposition;

namespace {

bool close(double left, double right, double tolerance = 1.0e-12) {
    return std::abs(left - right) <= tolerance;
}

}  // namespace

int main() {
    const auto empty = decomposition::empty_log_sum_exp_state_v1();
    assert(decomposition::validate_log_sum_exp_state_v1(empty)
        == decomposition::log_sum_exp_state_validation_code_v1::ok);
    assert(std::isinf(decomposition::finalize_log_sum_exp_v1(empty)));
    assert(decomposition::finalize_log_sum_exp_v1(empty) < 0.0);
    assert(std::isnan(decomposition::softmax_probability_v1(empty, 0.0)));

    const auto a = decomposition::singleton_log_sum_exp_state_v1(1000.0);
    const auto b = decomposition::singleton_log_sum_exp_state_v1(999.0);
    const auto c = decomposition::singleton_log_sum_exp_state_v1(-1000.0);
    const auto left = decomposition::merge_log_sum_exp_state_v1(
        decomposition::merge_log_sum_exp_state_v1(a, b), c);
    const auto right = decomposition::merge_log_sum_exp_state_v1(
        a, decomposition::merge_log_sum_exp_state_v1(b, c));
    assert(decomposition::validate_log_sum_exp_state_v1(left)
        == decomposition::log_sum_exp_state_validation_code_v1::ok);
    assert(left.sample_count == 3u);
    assert(close(left.maximum, right.maximum));
    assert(close(left.scaled_exponential_sum,
        right.scaled_exponential_sum));
    assert(close(decomposition::finalize_log_sum_exp_v1(left),
        1000.0 + std::log1p(std::exp(-1.0))));

    const double probability_sum =
        decomposition::softmax_probability_v1(left, 1000.0)
        + decomposition::softmax_probability_v1(left, 999.0)
        + decomposition::softmax_probability_v1(left, -1000.0);
    assert(close(probability_sum, 1.0));

    auto invalid = left;
    invalid.scaled_exponential_sum = 0.0;
    assert(decomposition::validate_log_sum_exp_state_v1(invalid)
        == decomposition::
            log_sum_exp_state_validation_code_v1::invalid_scaled_sum);
    invalid = empty;
    invalid.maximum = 1.0;
    assert(decomposition::validate_log_sum_exp_state_v1(invalid)
        == decomposition::
            log_sum_exp_state_validation_code_v1::invalid_empty_state);
    return 0;
}
