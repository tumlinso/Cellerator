#include "src/compiler/tooling/cellerator/tooling_model.hh"

#include <cassert>
#include <set>

int main() {
    using namespace cellerator::compiler::tooling::v1;
    const auto results = benchmark_advanced_semantic_queries();
    assert(results.size() == 10);
    std::set<std::string> queries;
    for (const auto& result : results) {
        queries.insert(result.query);
        assert(result.cache_state == "cold" || result.cache_state == "cached");
        assert(result.sample_count == 31);
        assert(result.p50_latency_ns > 0);
        assert(result.p95_latency_ns >= result.p50_latency_ns);
        assert(result.background_budget_ns == 10'000'000);
        assert(result.p95_latency_ns <= result.background_budget_ns);
        assert(result.cancellation_observed);
        assert(result.cpp_editing_responsive);
    }
    assert(queries == std::set<std::string>({"profile_propagation", "candidate_explanation",
                                             "semantic_ir_rendering", "decomposition_graph",
                                             "native_navigation"}));
}
