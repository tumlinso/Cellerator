#pragma once

#include <cmath>
#include <cstdint>

namespace cellerator::jbc::verification {

struct numerical_fragment_v1 {
    const std::uint64_t *local_to_canonical = nullptr;
    const double *values = nullptr;
    std::uint64_t value_count = 0u;
};

enum class numerical_code_v1 : std::uint8_t {
    success = 0u,
    invalid_argument,
    canonical_out_of_range,
    duplicate_value,
    missing_value,
    non_finite,
    tolerance_exceeded,
};

struct numerical_status_v1 {
    numerical_code_v1 code = numerical_code_v1::success;
    std::uint64_t index = 0u;
    double maximum_absolute_error = 0.0;
    double maximum_relative_error = 0.0;

    constexpr explicit operator bool() const noexcept {
        return code == numerical_code_v1::success;
    }
};

inline numerical_status_v1 reconstruct_canonical_values_v1(
    const numerical_fragment_v1 *fragments, std::uint64_t fragment_count,
    double *canonical, std::uint8_t *written,
    std::uint64_t canonical_count) noexcept {
    if (fragments == nullptr || fragment_count == 0u || canonical == nullptr ||
        written == nullptr || canonical_count == 0u) {
        return {numerical_code_v1::invalid_argument};
    }
    for (std::uint64_t index = 0u; index < canonical_count; ++index) {
        written[index] = 0u;
    }
    for (std::uint64_t fragment_index = 0u;
         fragment_index < fragment_count; ++fragment_index) {
        const auto &fragment = fragments[fragment_index];
        if (fragment.value_count == 0u || fragment.values == nullptr ||
            fragment.local_to_canonical == nullptr) {
            return {numerical_code_v1::invalid_argument, fragment_index};
        }
        for (std::uint64_t local = 0u; local < fragment.value_count; ++local) {
            const auto destination = fragment.local_to_canonical[local];
            if (destination >= canonical_count) {
                return {numerical_code_v1::canonical_out_of_range, destination};
            }
            if (written[destination] != 0u) {
                return {numerical_code_v1::duplicate_value, destination};
            }
            canonical[destination] = fragment.values[local];
            written[destination] = 1u;
        }
    }
    for (std::uint64_t index = 0u; index < canonical_count; ++index) {
        if (written[index] == 0u) {
            return {numerical_code_v1::missing_value, index};
        }
    }
    return {};
}

inline numerical_status_v1 verify_numerical_values_v1(
    const double *expected, const double *actual, std::uint64_t count,
    double absolute_tolerance, double relative_tolerance) noexcept {
    if (expected == nullptr || actual == nullptr || count == 0u ||
        absolute_tolerance < 0.0 || relative_tolerance < 0.0) {
        return {numerical_code_v1::invalid_argument};
    }
    numerical_status_v1 status{};
    for (std::uint64_t index = 0u; index < count; ++index) {
        if (!std::isfinite(expected[index]) || !std::isfinite(actual[index])) {
            return {numerical_code_v1::non_finite, index,
                status.maximum_absolute_error,
                status.maximum_relative_error};
        }
        const auto absolute = std::abs(expected[index] - actual[index]);
        const auto scale = std::abs(expected[index]);
        const auto relative = scale == 0.0 ? absolute : absolute / scale;
        if (absolute > status.maximum_absolute_error) {
            status.maximum_absolute_error = absolute;
        }
        if (relative > status.maximum_relative_error) {
            status.maximum_relative_error = relative;
        }
        if (absolute > absolute_tolerance && relative > relative_tolerance) {
            status.code = numerical_code_v1::tolerance_exceeded;
            status.index = index;
            return status;
        }
    }
    return status;
}

}  // namespace cellerator::jbc::verification
