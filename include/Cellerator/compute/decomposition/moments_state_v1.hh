#pragma once

#include <cstdint>
#include <type_traits>

namespace cellerator::compute::decomposition {

inline constexpr std::uint32_t moments_state_schema_version_v1 = 1u;

// Chan/Welford merge state.  m2 is the sum of squared deviations from mean,
// not a finalized variance, so independently accumulated partitions can be
// combined without reconstructing their observations.
struct moments_state_v1 {
    std::uint32_t schema_version = moments_state_schema_version_v1;
    std::uint32_t reserved = 0u;
    std::uint64_t sample_count = 0u;
    double mean = 0.0;
    double m2 = 0.0;
};

enum class moments_state_validation_code_v1 : std::uint8_t {
    ok = 0u,
    unsupported_schema,
    nonzero_reserved,
    invalid_empty_state,
    invalid_mean,
    invalid_m2
};

enum class moments_merge_code_v1 : std::uint8_t {
    ok = 0u,
    invalid_left,
    invalid_right,
    count_overflow
};

struct moments_merge_result_v1 {
    moments_state_v1 state{};
    moments_merge_code_v1 code = moments_merge_code_v1::ok;

    constexpr explicit operator bool() const noexcept {
        return code == moments_merge_code_v1::ok;
    }
};

moments_state_v1 empty_moments_state_v1() noexcept;
moments_state_v1 singleton_moments_state_v1(double value) noexcept;
moments_merge_result_v1 merge_moments_state_v1(
    const moments_state_v1 &left,
    const moments_state_v1 &right) noexcept;

moments_state_validation_code_v1 validate_moments_state_v1(
    const moments_state_v1 &state) noexcept;

double population_variance_v1(const moments_state_v1 &state) noexcept;
double sample_variance_v1(const moments_state_v1 &state) noexcept;

static_assert(std::is_trivially_copyable_v<moments_state_v1>);
static_assert(std::is_standard_layout_v<moments_state_v1>);
static_assert(std::is_trivially_copyable_v<moments_merge_result_v1>);

}  // namespace cellerator::compute::decomposition
