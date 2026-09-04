#include <Cellerator/compiler/planning/implement_complete_cost_normalization_v1.hh>

#include <algorithm>
#include <cmath>
#include <limits>

namespace Cellerator::compiler::planning {
namespace {

double unit_scale(cost_time_unit_v1 unit) noexcept {
    switch (unit) {
    case cost_time_unit_v1::nanoseconds: return 1.0;
    case cost_time_unit_v1::microseconds: return 1000.0;
    case cost_time_unit_v1::milliseconds: return 1000000.0;
    }
    return 0.0;
}

}  // namespace

normalized_complete_cost_v1 normalize_complete_cost_v1(
    const std::vector<complete_cost_evidence_v1>& evidence,
    std::uint32_t required_phases) noexcept {
    normalized_complete_cost_v1 result{};
    result.minimum_confidence = 1.0;
    if (evidence.empty() || required_phases == 0u) return result;
    for (const auto& item : evidence) {
        const double scale = unit_scale(item.unit);
        if (item.phases == 0u || scale == 0.0 || !std::isfinite(item.mean) ||
            !std::isfinite(item.p95) || !std::isfinite(item.confidence) ||
            item.mean < 0.0 || item.p95 < item.mean || item.confidence < 0.0 ||
            item.confidence > 1.0 || item.recurrence == 0u) {
            result.code = complete_cost_normalization_code_v1::invalid_evidence;
            return result;
        }
        if (item.amortization_horizon == 0u) {
            result.code = complete_cost_normalization_code_v1::incomparable_cost;
            return result;
        }
        if ((result.covered_phases & item.phases) != 0u) {
            result.code = complete_cost_normalization_code_v1::double_counted_phase;
            return result;
        }
        const double multiplier = scale * static_cast<double>(item.recurrence) /
            static_cast<double>(item.amortization_horizon);
        const double mean = item.mean * multiplier;
        const double p95 = item.p95 * multiplier;
        if (!std::isfinite(mean) || !std::isfinite(p95) ||
            !std::isfinite(result.mean_nanoseconds + mean) ||
            !std::isfinite(result.p95_nanoseconds + p95)) {
            result.code = complete_cost_normalization_code_v1::arithmetic_overflow;
            return result;
        }
        result.mean_nanoseconds += mean;
        result.p95_nanoseconds += p95;
        result.minimum_confidence = std::min(result.minimum_confidence, item.confidence);
        result.covered_phases |= item.phases;
    }
    result.missing_phases = required_phases & ~result.covered_phases;
    result.code = result.missing_phases == 0u
        ? complete_cost_normalization_code_v1::ok
        : complete_cost_normalization_code_v1::missing_phase;
    return result;
}

}  // namespace Cellerator::compiler::planning
