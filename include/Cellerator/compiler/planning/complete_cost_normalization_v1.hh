#pragma once

#include <cstdint>
#include <vector>

namespace Cellerator::compiler::planning {

enum class cost_evidence_kind_v1 : std::uint8_t {
    analytical = 1u,
    measured,
    cached,
    external,
};

enum class cost_time_unit_v1 : std::uint8_t {
    nanoseconds = 1u,
    microseconds,
    milliseconds,
};

enum cost_phase_v1 : std::uint32_t {
    cost_phase_preparation_v1 = 1u << 0u,
    cost_phase_movement_v1 = 1u << 1u,
    cost_phase_execution_v1 = 1u << 2u,
    cost_phase_synchronization_v1 = 1u << 3u,
    cost_phase_output_transform_v1 = 1u << 4u,
    cost_phase_external_v1 = 1u << 5u,
};

struct complete_cost_evidence_v1 {
    cost_evidence_kind_v1 kind = cost_evidence_kind_v1::analytical;
    cost_time_unit_v1 unit = cost_time_unit_v1::nanoseconds;
    std::uint32_t phases = 0u;
    double mean = 0.0;
    double p95 = 0.0;
    double confidence = 0.0;
    std::uint64_t recurrence = 1u;
    std::uint64_t amortization_horizon = 1u;
};

enum class complete_cost_normalization_code_v1 : std::uint8_t {
    ok = 0u,
    invalid_evidence,
    incomparable_cost,
    double_counted_phase,
    missing_phase,
    arithmetic_overflow,
};

struct normalized_complete_cost_v1 {
    complete_cost_normalization_code_v1 code =
        complete_cost_normalization_code_v1::invalid_evidence;
    std::uint32_t covered_phases = 0u;
    std::uint32_t missing_phases = 0u;
    double mean_nanoseconds = 0.0;
    double p95_nanoseconds = 0.0;
    double minimum_confidence = 0.0;

    constexpr explicit operator bool() const noexcept {
        return code == complete_cost_normalization_code_v1::ok;
    }
};

[[nodiscard]] normalized_complete_cost_v1 normalize_complete_cost_v1(
    const std::vector<complete_cost_evidence_v1>& evidence,
    std::uint32_t required_phases) noexcept;

}  // namespace Cellerator::compiler::planning
