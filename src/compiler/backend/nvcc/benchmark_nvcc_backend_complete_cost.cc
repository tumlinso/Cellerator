#include <Cellerator/compiler/backend/nvcc/benchmark_nvcc_backend_complete_cost_v1.hh>

#include <limits>

namespace cellerator::compiler::backend::nvcc::v1 {
namespace {

bool add(std::uint64_t value, std::uint64_t* total) noexcept {
    if (value > std::numeric_limits<std::uint64_t>::max() - *total) {
        return false;
    }
    *total += value;
    return true;
}

} // namespace

std::optional<nvcc_complete_cost_result> complete_nvcc_cost(
    const nvcc_complete_cost_sample& sample,
    std::uint64_t reuse_count) noexcept {
    if (!sample.exact_output || reuse_count == 0 || sample.launch_count == 0) {
        return std::nullopt;
    }
    nvcc_complete_cost_result result;
    result.reuse_count = reuse_count;
    for (const auto value : {sample.preparation_ns, sample.packing_ns,
                             sample.host_to_device_ns,
                             sample.launch_overhead_ns, sample.kernel_ns,
                             sample.device_to_host_ns}) {
        if (!add(value, &result.warm_ns)) {
            return std::nullopt;
        }
    }
    result.cold_ns = result.warm_ns;
    for (const auto value : {sample.planning_ns, sample.source_emission_ns,
                             sample.nvcc_ns}) {
        if (!add(value, &result.cold_ns)) {
            return std::nullopt;
        }
    }
    result.total_reuse_ns = result.cold_ns;
    for (std::uint64_t reuse = 1; reuse < reuse_count; ++reuse) {
        if (!add(result.warm_ns, &result.total_reuse_ns)) {
            return std::nullopt;
        }
    }
    return result;
}

std::optional<nvcc_candidate_comparison> compare_nvcc_candidates(
    const std::vector<nvcc_complete_cost_sample>& samples,
    std::uint64_t reuse_count) noexcept {
    std::optional<nvcc_candidate_comparison> best;
    for (const auto& sample : samples) {
        const auto cost = complete_nvcc_cost(sample, reuse_count);
        if (!cost) {
            continue;
        }
        if (!best || cost->total_reuse_ns < best->selected_cost.total_reuse_ns) {
            best = nvcc_candidate_comparison{sample.candidate, *cost, false};
        }
    }
    if (best) {
        best->generated_promoted =
            best->selected == nvcc_candidate_kind::generated;
    }
    return best;
}

} // namespace cellerator::compiler::backend::nvcc::v1
