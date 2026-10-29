#include "parameter_identity_validation.hh"

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <span>
#include <vector>

namespace ix = cellerator::compute::operation::indexed;
namespace detail = cellerator::compute::operation::indexed::detail;
namespace v2 = cellerator::compute::operation::v2;

detail::parameter_input_validation reference_validate(
    std::span<const ix::identity> ids, std::span<const float> values) {
    if (ids.size() != values.size())
        return detail::parameter_input_validation::invalid_identity_or_value;
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (!v2::valid_stable_id(ids[i]) || !std::isfinite(values[i]))
            return detail::parameter_input_validation::invalid_identity_or_value;
        for (std::size_t j = 0; j < i; ++j)
            if (v2::same_stable_id(ids[i], ids[j]))
                return detail::parameter_input_validation::duplicate_identity;
    }
    return detail::parameter_input_validation::valid;
}

void compare(std::vector<ix::identity> ids, std::vector<float> values) {
    assert(detail::validate_parameter_inputs(ids, values)
        == reference_validate(ids, values));
}

int main() {
    using result = detail::parameter_input_validation;
    compare({{1, 7}, {2, 7}, {3, 7}}, {0.5f, -1.0f, 3.0f});
    compare({{1, 7}, {1, 7}}, {0.5f, -1.0f});
    compare({{1, 7}, {}}, {0.5f, -1.0f});
    compare({{1, 7}, {2, 7}, {3, 7}},
            {0.5f, std::numeric_limits<float>::infinity(), 3.0f});
    compare({{1, 7}, {1, 7}, {}}, {0.5f, -1.0f, 3.0f});
    compare({{}, {1, 7}, {1, 7}}, {0.5f, -1.0f, 3.0f});
    const std::vector<ix::identity> mismatched_ids{{1, 7}};
    const std::vector<float> mismatched_values{};
    assert(detail::validate_parameter_inputs(mismatched_ids, mismatched_values)
        == result::invalid_identity_or_value);

    // Differential coverage on small malformed and valid sequences exercises
    // the original left-to-right error precedence without a CUDA context.
    std::mt19937_64 rng(0x4d454348414e4953ull);
    for (std::size_t trial = 0; trial < 500; ++trial) {
        const std::size_t n = rng() % 24;
        std::vector<ix::identity> ids(n);
        std::vector<float> values(n);
        for (std::size_t i = 0; i < n; ++i) {
            const auto code = rng() % 12;
            ids[i] = code == 0 ? ix::identity{} : ix::identity{code, 9};
            const auto value_code = rng() % 20;
            values[i] = value_code == 0
                ? std::numeric_limits<float>::quiet_NaN()
                : static_cast<float>(value_code);
        }
        compare(std::move(ids), std::move(values));
    }

    for (const std::size_t n : {256u, 1024u, 4096u}) {
        std::vector<ix::identity> ids;
        std::vector<float> values(n, 0.25f);
        ids.reserve(n);
        for (std::size_t i = 0; i < n; ++i)
            ids.push_back({static_cast<std::uint64_t>(i + 1), 19});
        auto start = std::chrono::steady_clock::now();
        const auto old_result = reference_validate(ids, values);
        const auto old_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - start).count();
        start = std::chrono::steady_clock::now();
        const auto new_result = detail::validate_parameter_inputs(ids, values);
        const auto new_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - start).count();
        assert(old_result == result::valid && new_result == old_result);
        std::cout << "scale_count=" << n << " quadratic_reference_us=" << old_us
                  << " sorted_validation_us=" << new_us << '\n';
    }

    constexpr std::size_t large_count = 1'300'000;
    std::vector<ix::identity> ids;
    std::vector<float> values(large_count, 0.25f);
    ids.reserve(large_count);
    for (std::size_t i = 0; i < large_count; ++i)
        ids.push_back({static_cast<std::uint64_t>(i + 1), 23});
    const auto start = std::chrono::steady_clock::now();
    const auto large_result = detail::validate_parameter_inputs(ids, values);
    const auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - start).count();
    assert(large_result == result::valid);
    std::cout << "large_identity_count=" << large_count
              << " validation_us=" << elapsed_us << '\n';
}
