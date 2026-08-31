#pragma once

#include <Cellerator/compute/architecture/target_cover/strategy_registry.hh>
#include <Cellerator/geometry/compiler/v2/solution.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::compute::architecture::target_cover {

inline constexpr std::uint32_t target_cover_schema_version = 1u;

enum class region_role : std::uint8_t { pure_sparse = 1, matrix_engine = 2 };
enum class cover_kind : std::uint8_t {
    pure_sparse = 1,
    conservative_hybrid = 2,
    aggressive_hybrid = 3
};

struct semantic_component {
    std::uint64_t component_identity = 0;
    std::uint64_t logical_edge_begin = 0;
    std::uint64_t logical_edge_count = 0;
};

struct strategy_problem {
    std::uint32_t schema_version = target_cover_schema_version;
    std::uint32_t record_bytes = sizeof(strategy_problem);
    stable_identity semantic_geometry_identity{};
    stable_identity provider_identity{};
    const semantic_component *semantic_components = nullptr;
    std::uint64_t semantic_component_count = 0;
    std::uint64_t logical_edge_count = 0;
    geometry::compiler::v2::workload_profile workload{};
};

struct target_region {
    std::uint64_t region_identity = 0;
    std::uint64_t semantic_component_identity = 0;
    stable_identity capability_identity{};
    region_role role = region_role::pure_sparse;
    std::uint8_t reserved[7]{};
    std::uint64_t logical_edge_count = 0;
};

struct ownership_range {
    std::uint64_t logical_edge_begin = 0;
    std::uint64_t logical_edge_count = 0;
    std::uint64_t region_index = 0;
};

struct cover_candidate {
    stable_identity identity{};
    cover_kind kind = cover_kind::pure_sparse;
    bool experimental = false;
    bool requires_measurement = false;
    std::uint8_t reserved[5]{};
    const target_region *regions = nullptr;
    std::uint64_t region_count = 0;
    const ownership_range *ownership = nullptr;
    std::uint64_t ownership_range_count = 0;
    geometry::compiler::v2::exact_cost exact_objective{};
};

struct strategy_solution {
    std::uint32_t schema_version = target_cover_schema_version;
    std::uint32_t record_bytes = sizeof(strategy_solution);
    stable_identity semantic_geometry_identity{};
    stable_identity provider_identity{};
    const cover_candidate *candidates = nullptr;
    std::uint64_t candidate_count = 0;
    std::uint64_t logical_edge_count = 0;
};

status validate_problem(const strategy_problem &problem) noexcept;
status validate_solution(
    const strategy_problem &problem, const strategy_solution &solution) noexcept;

static_assert(std::is_trivially_copyable_v<semantic_component>);
static_assert(std::is_trivially_copyable_v<strategy_problem>);
static_assert(std::is_trivially_copyable_v<target_region>);
static_assert(std::is_trivially_copyable_v<ownership_range>);
static_assert(std::is_trivially_copyable_v<cover_candidate>);
static_assert(std::is_trivially_copyable_v<strategy_solution>);

}  // namespace cellerator::compute::architecture::target_cover
