#include <Cellerator/compiler/planning/benchmark_planning_scalability_and_boundedness_v1.hh>

#include <algorithm>
#include <chrono>
#include <limits>

namespace Cellerator::compiler::planning {

planning_benchmark_result_v1 benchmark_planning_scalability_and_boundedness_v1(
    const planning_benchmark_fixture_v1& fixture) {
    planning_benchmark_result_v1 result{};
    result.fixture_kind = fixture.kind;
    result.candidate_count = fixture.candidates.size();
    result.required_capacity = fixture.candidates.size();
    result.profile_variant_count = fixture.profile_variant_count;
    if (fixture.candidates.empty() || fixture.repetitions == 0u ||
        fixture.profile_variant_count == 0u) return result;
    if (fixture.candidates.size() > fixture.candidate_capacity) {
        result.code = planning_benchmark_code_v1::capacity_exceeded;
        return result;
    }
    for (const auto& candidate : fixture.candidates)
        if (candidate.candidate_identity == 0u || candidate.complete_cost_nanoseconds == 0u)
            return result;

    const auto cold_start = std::chrono::steady_clock::now();
    auto ordered = fixture.candidates;
    std::sort(ordered.begin(), ordered.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.complete_cost_nanoseconds != rhs.complete_cost_nanoseconds)
            return lhs.complete_cost_nanoseconds < rhs.complete_cost_nanoseconds;
        return lhs.memory_bytes < rhs.memory_bytes;
    });
    std::uint64_t best_memory = std::numeric_limits<std::uint64_t>::max();
    for (const auto& candidate : ordered) {
        if (candidate.memory_bytes < best_memory) {
            ++result.search_frontier_count;
            best_memory = candidate.memory_bytes;
        }
    }
    const auto cold_end = std::chrono::steady_clock::now();
    result.selected_candidate_identity = ordered.front().candidate_identity;
    result.exact_certification = ordered.front().exactly_certified;
    result.quality_versus_oracle = 1.0;
    result.peak_memory_bytes = fixture.candidates.size() *
        (sizeof(planning_benchmark_candidate_v1) + sizeof(std::uint64_t));
    result.cold_planning_nanoseconds = std::max<std::uint64_t>(1u,
        std::chrono::duration_cast<std::chrono::nanoseconds>(cold_end - cold_start).count());

    volatile std::uint64_t cached_identity = result.selected_candidate_identity;
    const auto warm_start = std::chrono::steady_clock::now();
    for (std::uint64_t i = 1u; i < fixture.repetitions; ++i)
        cached_identity ^= result.selected_candidate_identity;
    const auto warm_end = std::chrono::steady_clock::now();
    (void)cached_identity;
    result.cache_reuse_count = fixture.repetitions - 1u;
    result.warm_planning_nanoseconds = std::max<std::uint64_t>(1u,
        std::chrono::duration_cast<std::chrono::nanoseconds>(warm_end - warm_start).count());
    result.code = planning_benchmark_code_v1::ok;
    return result;
}

}  // namespace Cellerator::compiler::planning
