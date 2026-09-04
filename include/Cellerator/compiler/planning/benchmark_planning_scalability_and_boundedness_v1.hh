#pragma once

#include <cstdint>
#include <vector>

namespace Cellerator::compiler::planning {

enum class planning_fixture_kind_v1 : std::uint8_t {
    synthetic = 1u,
    biological,
};

struct planning_benchmark_candidate_v1 {
    std::uint64_t candidate_identity = 0u;
    std::uint64_t complete_cost_nanoseconds = 0u;
    std::uint64_t memory_bytes = 0u;
    bool exactly_certified = false;
};

struct planning_benchmark_fixture_v1 {
    planning_fixture_kind_v1 kind = planning_fixture_kind_v1::synthetic;
    std::vector<planning_benchmark_candidate_v1> candidates;
    std::uint64_t candidate_capacity = 0u;
    std::uint64_t repetitions = 1u;
    std::uint64_t profile_variant_count = 1u;
};

enum class planning_benchmark_code_v1 : std::uint8_t {
    ok = 0u,
    invalid_fixture,
    capacity_exceeded,
};

struct planning_benchmark_result_v1 {
    planning_benchmark_code_v1 code = planning_benchmark_code_v1::invalid_fixture;
    planning_fixture_kind_v1 fixture_kind = planning_fixture_kind_v1::synthetic;
    std::uint64_t candidate_count = 0u;
    std::uint64_t required_capacity = 0u;
    std::uint64_t search_frontier_count = 0u;
    std::uint64_t peak_memory_bytes = 0u;
    std::uint64_t cold_planning_nanoseconds = 0u;
    std::uint64_t warm_planning_nanoseconds = 0u;
    std::uint64_t profile_variant_count = 0u;
    std::uint64_t cache_reuse_count = 0u;
    std::uint64_t selected_candidate_identity = 0u;
    bool exact_certification = false;
    double quality_versus_oracle = 0.0;

    constexpr explicit operator bool() const noexcept {
        return code == planning_benchmark_code_v1::ok;
    }
};

[[nodiscard]] planning_benchmark_result_v1
benchmark_planning_scalability_and_boundedness_v1(
    const planning_benchmark_fixture_v1& fixture);

}  // namespace Cellerator::compiler::planning
