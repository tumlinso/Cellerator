#pragma once

#include <Cellerator/geometry/compiler/v2/work_window.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::geometry::compiler::v2 {

struct exact_cost {
    double predicted_latency_ns = 0;
    double preparation_ns = 0;
    double layout_and_canonicalization_ns = 0;
    double value_update_ns = 0;
    std::uint64_t persistent_bytes = 0;
    std::uint64_t transient_bytes = 0;
};

struct exact_contribution {
    std::uint64_t logical_identity = 0;
    std::uint64_t original_group_id = 0;
    exact_cost cost{};
};

struct exact_evaluation_problem {
    stable_identity semantic_solution{};
    stable_identity skeleton{};
    stable_identity work_window{};
    const exact_contribution *contributions = nullptr;
    std::uint64_t contribution_count = 0;
};

struct exact_evaluation {
    stable_identity semantic_solution{};
    exact_cost total{};
    std::uint64_t evaluated_contributions = 0;
};

struct incremental_exact_state {
    stable_identity semantic_solution{};
    stable_identity work_window{};
    exact_cost total{};
    std::uint64_t evaluated_contributions = 0;
    std::uint64_t generation = 0;
};

struct exact_delta {
    exact_cost removed{};
    exact_cost added{};
    std::uint64_t removed_contributions = 0;
    std::uint64_t added_contributions = 0;
    stable_identity next_work_window{};
};

workload_status evaluate_exact(
    const exact_evaluation_problem &problem, exact_evaluation *result) noexcept;
workload_status initialize_incremental_exact_state(
    const exact_evaluation &evaluation,
    stable_identity work_window,
    incremental_exact_state *state) noexcept;
workload_status apply_exact_delta(
    const exact_delta &delta, incremental_exact_state *state) noexcept;

static_assert(std::is_trivially_copyable_v<exact_cost>);
static_assert(std::is_trivially_copyable_v<exact_contribution>);
static_assert(std::is_trivially_copyable_v<exact_evaluation_problem>);
static_assert(std::is_trivially_copyable_v<exact_evaluation>);
static_assert(std::is_trivially_copyable_v<incremental_exact_state>);

}  // namespace cellerator::geometry::compiler::v2
