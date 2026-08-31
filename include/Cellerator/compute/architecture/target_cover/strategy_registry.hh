#pragma once

#include <Cellerator/geometry/compiler/v2/workload_profile.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::compute::architecture::target_cover {

using stable_identity = geometry::compiler::v2::stable_identity;
using status = geometry::compiler::v2::workload_status;

struct strategy_problem;
struct strategy_solution;

using workspace_query = status (*)(
    const strategy_problem &, std::uint64_t *, std::uint64_t *) noexcept;
using solve_function = status (*)(
    const strategy_problem &, void *, std::uint64_t, strategy_solution *) noexcept;

struct strategy {
    stable_identity identity{};
    stable_identity provider_identity{};
    const char *name = nullptr;
    workspace_query query_workspace = nullptr;
    solve_function solve = nullptr;
    bool deterministic = true;
    bool experimental = false;
    std::uint8_t reserved[6]{};
};

struct strategy_registry {
    const strategy *strategies = nullptr;
    std::uint64_t strategy_count = 0;
};

status validate_strategy_registry(const strategy_registry &registry) noexcept;

static_assert(std::is_trivially_copyable_v<strategy>);
static_assert(std::is_trivially_copyable_v<strategy_registry>);

}  // namespace cellerator::compute::architecture::target_cover
