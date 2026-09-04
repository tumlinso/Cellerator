#include <Cellerator/compiler/backend/nvptx/compare_nvcc_clang_cuda_and_direct_ptx_routes_v1.hh>

#include <algorithm>

namespace Cellerator::compiler::backend::nvptx {

nvptx_route_comparison_v1 compare_nvptx_routes_v1(
    const std::vector<nvptx_route_measurement_v1>& measurements,
    std::string regime,
    const double required_speedup_for_direct_ptx) {
    nvptx_route_comparison_v1 result;
    result.measurements = measurements;
    result.regime = std::move(regime);
    if (measurements.size() != 3u || result.regime.empty() || required_speedup_for_direct_ptx < 1.0) {
        result.reason = "comparison requires three matched routes, a regime, and a nonnegative hurdle";
        return result;
    }
    for (const auto& measurement : measurements) {
        if (measurement.toolchain_identity.empty() || measurement.compile_nanoseconds == 0u ||
            measurement.object_bytes == 0u || measurement.median_execution_nanoseconds == 0u ||
            !measurement.correctness_passed || !measurement.benchmark_mutex_held ||
            measurement.contaminated) {
            result.reason = "route evidence is incomplete, incorrect, unlocked, or contaminated";
            return result;
        }
    }
    const auto direct_position = std::find_if(measurements.begin(), measurements.end(), [](const auto& value) {
        return value.route == nvptx_route_v1::direct_ptx;
    });
    if (direct_position == measurements.end()) {
        result.reason = "comparison has no direct PTX route";
        return result;
    }
    const auto* direct = &*direct_position;
    const auto strongest_conventional = std::min_element(
        measurements.begin(), measurements.end(), [](const auto& left, const auto& right) {
            const bool left_direct = left.route == nvptx_route_v1::direct_ptx;
            const bool right_direct = right.route == nvptx_route_v1::direct_ptx;
            if (left_direct != right_direct) return !left_direct;
            return left.median_execution_nanoseconds < right.median_execution_nanoseconds;
        });
    const double speedup = static_cast<double>(strongest_conventional->median_execution_nanoseconds) /
        static_cast<double>(direct->median_execution_nanoseconds);
    if (speedup >= required_speedup_for_direct_ptx) {
        result.disposition = nvptx_route_promotion_v1::promoted;
        result.selected_route = nvptx_route_v1::direct_ptx;
        result.reason = "direct PTX cleared the regime-specific execution hurdle";
    } else {
        result.disposition = nvptx_route_promotion_v1::evaluated_not_promoted;
        result.selected_route = strongest_conventional->route;
        result.reason = "direct PTX did not clear the regime-specific execution hurdle";
    }
    return result;
}

}  // namespace Cellerator::compiler::backend::nvptx
