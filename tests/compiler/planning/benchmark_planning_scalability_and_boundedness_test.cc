#include <Cellerator/compiler/planning/benchmark_planning_scalability_and_boundedness_v1.hh>

#include <cassert>
#include <iostream>

namespace planning = Cellerator::compiler::planning;

int main() {
    planning::planning_benchmark_fixture_v1 synthetic{};
    synthetic.kind = planning::planning_fixture_kind_v1::synthetic;
    synthetic.candidate_capacity = 128u;
    synthetic.repetitions = 1000u;
    synthetic.profile_variant_count = 4u;
    for (std::uint64_t i = 0u; i < 128u; ++i)
        synthetic.candidates.push_back({i + 1u, 1000u + i * 7u,
            8192u - i * 32u, i == 0u});
    const auto synthetic_result =
        planning::benchmark_planning_scalability_and_boundedness_v1(synthetic);
    assert(synthetic_result);
    assert(synthetic_result.candidate_count == 128u);
    assert(synthetic_result.search_frontier_count > 0u);
    assert(synthetic_result.cold_planning_nanoseconds > 0u);
    assert(synthetic_result.warm_planning_nanoseconds > 0u);
    assert(synthetic_result.cache_reuse_count == 999u);
    assert(synthetic_result.profile_variant_count == 4u);
    assert(synthetic_result.exact_certification);
    assert(synthetic_result.quality_versus_oracle == 1.0);

    auto biological = synthetic;
    biological.kind = planning::planning_fixture_kind_v1::biological;
    biological.candidates.resize(32u);
    biological.candidate_capacity = 32u;
    const auto biological_result =
        planning::benchmark_planning_scalability_and_boundedness_v1(biological);
    assert(biological_result && biological_result.candidate_count == 32u);

    std::cout << "fixture,candidates,frontier,peak_bytes,cold_ns,warm_ns,variants,cache_reuse,quality\n"
              << "synthetic," << synthetic_result.candidate_count << ','
              << synthetic_result.search_frontier_count << ','
              << synthetic_result.peak_memory_bytes << ','
              << synthetic_result.cold_planning_nanoseconds << ','
              << synthetic_result.warm_planning_nanoseconds << ','
              << synthetic_result.profile_variant_count << ','
              << synthetic_result.cache_reuse_count << ','
              << synthetic_result.quality_versus_oracle << '\n'
              << "biological," << biological_result.candidate_count << ','
              << biological_result.search_frontier_count << ','
              << biological_result.peak_memory_bytes << ','
              << biological_result.cold_planning_nanoseconds << ','
              << biological_result.warm_planning_nanoseconds << ','
              << biological_result.profile_variant_count << ','
              << biological_result.cache_reuse_count << ','
              << biological_result.quality_versus_oracle << '\n';

    biological.candidate_capacity = 31u;
    const auto bounded =
        planning::benchmark_planning_scalability_and_boundedness_v1(biological);
    assert(bounded.code == planning::planning_benchmark_code_v1::capacity_exceeded);
    assert(bounded.required_capacity == 32u);
}
