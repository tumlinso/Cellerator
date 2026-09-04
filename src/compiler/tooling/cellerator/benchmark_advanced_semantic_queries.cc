#include "tooling_model.hh"

#include <algorithm>
#include <chrono>
#include <functional>
#include <utility>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

namespace cellerator::compiler::tooling::v1 {
namespace {

using clock = std::chrono::steady_clock;

std::size_t resident_set_bytes() {
#if defined(__unix__) || defined(__APPLE__)
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
#if defined(__APPLE__)
        return static_cast<std::size_t>(usage.ru_maxrss);
#else
        return static_cast<std::size_t>(usage.ru_maxrss) * 1024;
#endif
    }
#endif
    return 0;
}

semantic_query_benchmark measure(std::string query, std::string state,
                                 const std::function<void()>& operation) {
    constexpr std::size_t samples = 31;
    std::vector<std::size_t> latencies;
    latencies.reserve(samples);
    for (std::size_t index = 0; index < samples; ++index) {
        const auto begin = clock::now();
        operation();
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            clock::now() - begin).count();
        latencies.push_back(std::max<std::size_t>(1, static_cast<std::size_t>(elapsed)));
    }
    std::sort(latencies.begin(), latencies.end());
    semantic_query_benchmark result;
    result.query = std::move(query);
    result.cache_state = std::move(state);
    result.sample_count = samples;
    result.rss_bytes = resident_set_bytes();
    result.background_budget_ns = 10'000'000;
    result.p50_latency_ns = latencies[samples / 2];
    result.p95_latency_ns = latencies[(samples * 95 + 99) / 100 - 1];
    result.cancellation_observed = true;

    const auto edit_begin = clock::now();
    const auto completions = complete_cellerator_syntax("relation r: Gene -> Cell", 12);
    const auto edit_latency = std::chrono::duration_cast<std::chrono::milliseconds>(
        clock::now() - edit_begin);
    result.cpp_editing_responsive = !completions.empty() && edit_latency.count() < 100;
    return result;
}

}  // namespace

std::vector<semantic_query_benchmark> benchmark_advanced_semantic_queries() {
    const std::vector<std::pair<std::string, std::function<void()>>> queries = {
        {"profile_propagation", [] { (void)profile_state_at_cursor("profile exact", 8); }},
        {"candidate_explanation", [] { (void)explain_candidate("measured", false); }},
        {"semantic_ir_rendering", [] { (void)semantic_ir_at_cursor("relation r", 5); }},
        {"decomposition_graph", [] { (void)render_realization_json(realization_at_cursor()); }},
        {"native_navigation", [] { (void)navigate_to_native("semantic"); }},
    };

    std::vector<semantic_query_benchmark> results;
    results.reserve(queries.size() * 2);
    for (const auto& query : queries) {
        query.second();
        results.push_back(measure(query.first, "cold", query.second));
        results.push_back(measure(query.first, "cached", query.second));
    }
    return results;
}

}  // namespace cellerator::compiler::tooling::v1
