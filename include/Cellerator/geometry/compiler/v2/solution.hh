#pragma once

#include <Cellerator/geometry/compiler/v2/exact_evaluator.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::geometry::compiler::v2 {

enum class optimizer_stage : std::uint8_t {
    portable_semantic_geometry = 1,
    target_specific_cover = 2
};

struct immutable_bytes {
    const void *data = nullptr;
    std::uint64_t bytes = 0;
};

struct solution_candidate {
    stable_identity identity{};
    stable_identity strategy_identity{};
    exact_cost exact_objective{};
    immutable_bytes data{};
    bool experimental = false;
    bool requires_measurement = false;
    std::uint8_t reserved[6]{};
};

struct multi_candidate_solution {
    optimizer_stage stage = optimizer_stage::portable_semantic_geometry;
    std::uint8_t reserved[7]{};
    const solution_candidate *candidates = nullptr;
    std::uint64_t candidate_count = 0;
};

struct optimizer_snapshot {
    std::uint32_t schema_version = 1;
    optimizer_stage stage = optimizer_stage::portable_semantic_geometry;
    std::uint8_t reserved0[3]{};
    stable_identity strategy_identity{};
    stable_identity problem_identity{};
    stable_identity work_window_identity{};
    std::uint64_t iteration = 0;
    std::uint64_t deterministic_seed = 0;
    exact_cost current_objective{};
    immutable_bytes state{};
};

workload_status validate_multi_candidate_solution(
    const multi_candidate_solution &solution) noexcept;
workload_status validate_optimizer_snapshot(
    const optimizer_snapshot &snapshot) noexcept;

static_assert(std::is_trivially_copyable_v<immutable_bytes>);
static_assert(std::is_trivially_copyable_v<solution_candidate>);
static_assert(std::is_trivially_copyable_v<multi_candidate_solution>);
static_assert(std::is_trivially_copyable_v<optimizer_snapshot>);

}  // namespace cellerator::geometry::compiler::v2
