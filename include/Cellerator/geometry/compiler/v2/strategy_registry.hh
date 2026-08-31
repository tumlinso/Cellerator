#pragma once

#include <Cellerator/geometry/compiler/v2/work_window.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::geometry::compiler::v2 {

struct semantic_strategy_problem;
struct semantic_strategy_solution;

using semantic_workspace_query = workload_status (*)(
    const semantic_strategy_problem &, std::uint64_t *, std::uint64_t *) noexcept;
using semantic_solve_function = workload_status (*)(const semantic_strategy_problem &,
    void *, std::uint64_t, semantic_strategy_solution *) noexcept;

struct semantic_strategy {
    stable_identity identity{};
    const char *name = nullptr;
    semantic_workspace_query query_workspace = nullptr;
    semantic_solve_function solve = nullptr;
    bool deterministic = true;
    bool experimental = false;
    std::uint8_t reserved[6]{};
};

struct semantic_strategy_registry {
    const semantic_strategy *strategies = nullptr;
    std::uint64_t strategy_count = 0;
};

workload_status validate_semantic_strategy_registry(
    const semantic_strategy_registry &registry) noexcept;

static_assert(std::is_trivially_copyable_v<semantic_strategy>);
static_assert(std::is_trivially_copyable_v<semantic_strategy_registry>);

}  // namespace cellerator::geometry::compiler::v2
