#include <Cellerator/compiler/planning/implement_complete_cost_normalization_v1.hh>

#include <cassert>
#include <cmath>
#include <vector>

namespace planning = Cellerator::compiler::planning;

int main() {
    const std::uint32_t required = planning::cost_phase_preparation_v1 |
        planning::cost_phase_execution_v1 | planning::cost_phase_external_v1;
    const std::vector<planning::complete_cost_evidence_v1> evidence{
        {planning::cost_evidence_kind_v1::analytical,
         planning::cost_time_unit_v1::microseconds,
         planning::cost_phase_preparation_v1, 100.0, 120.0, 0.8, 1u, 10u},
        {planning::cost_evidence_kind_v1::measured,
         planning::cost_time_unit_v1::nanoseconds,
         planning::cost_phase_execution_v1, 2000.0, 2500.0, 0.95, 3u, 1u},
        {planning::cost_evidence_kind_v1::external,
         planning::cost_time_unit_v1::milliseconds,
         planning::cost_phase_external_v1, 0.5, 0.75, 0.9, 1u, 1u},
    };
    const auto normalized = planning::normalize_complete_cost_v1(evidence, required);
    assert(normalized);
    assert(std::abs(normalized.mean_nanoseconds - 516000.0) < 1.0e-9);
    assert(std::abs(normalized.p95_nanoseconds - 769500.0) < 1.0e-9);
    assert(std::abs(normalized.minimum_confidence - 0.8) < 1.0e-12);

    auto duplicate = evidence;
    duplicate.push_back(evidence[1]);
    assert(planning::normalize_complete_cost_v1(duplicate, required).code ==
        planning::complete_cost_normalization_code_v1::double_counted_phase);

    auto missing = evidence;
    missing.pop_back();
    const auto missing_result = planning::normalize_complete_cost_v1(missing, required);
    assert(missing_result.code == planning::complete_cost_normalization_code_v1::missing_phase);
    assert(missing_result.missing_phases == planning::cost_phase_external_v1);

    auto incomparable = evidence;
    incomparable[0].amortization_horizon = 0u;
    assert(planning::normalize_complete_cost_v1(incomparable, required).code ==
        planning::complete_cost_normalization_code_v1::incomparable_cost);
}
