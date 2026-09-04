#pragma once

#include <cstdint>

namespace Cellerator::compiler::planning {

inline constexpr std::uint32_t decomposition_portfolio_schema_v1 = 1u;

enum class decomposition_provider_kind_v1 : std::uint8_t {
    greedy = 1u,
    multilevel = 2u,
    exact_oracle = 3u,
    bounded_overlap = 4u,
    device_assisted = 5u,
    user_provided = 6u,
};

enum decomposition_provider_capability_v1 : std::uint32_t {
    deterministic_provider_v1 = 1u << 0u,
    exact_coverage_provider_v1 = 1u << 1u,
    incremental_provider_v1 = 1u << 2u,
    multi_operation_provider_v1 = 1u << 3u,
    external_provider_v1 = 1u << 4u,
    experimental_provider_v1 = 1u << 5u,
};

struct decomposition_provider_bounds_v1 {
    std::uint64_t maximum_nodes = 0u;
    std::uint64_t maximum_edges = 0u;
    std::uint64_t maximum_workspace_bytes = 0u;
    std::uint64_t maximum_search_steps = 0u;
};

struct decomposition_provider_descriptor_v1 {
    std::uint32_t schema_version = decomposition_portfolio_schema_v1;
    decomposition_provider_kind_v1 kind = decomposition_provider_kind_v1::greedy;
    std::uint8_t registry_order = 0u;
    std::uint16_t reserved = 0u;
    std::uint64_t provider_identity = 0u;
    std::uint32_t capabilities = 0u;
    std::uint32_t reserved1 = 0u;
    decomposition_provider_bounds_v1 bounds{};
    const char* stable_name = nullptr;
};

struct decomposition_portfolio_view_v1 {
    std::uint32_t schema_version = decomposition_portfolio_schema_v1;
    std::uint32_t provider_count = 0u;
    const decomposition_provider_descriptor_v1* providers = nullptr;
};

enum class decomposition_portfolio_validation_code_v1 : std::uint8_t {
    ok = 0u,
    unsupported_schema,
    missing_providers,
    invalid_provider,
    duplicate_provider,
    unbounded_provider,
    hidden_selection_policy,
};

[[nodiscard]] decomposition_portfolio_view_v1
built_in_decomposition_portfolio_v1() noexcept;

[[nodiscard]] decomposition_portfolio_validation_code_v1
validate_decomposition_portfolio_v1(
    const decomposition_portfolio_view_v1& portfolio) noexcept;

[[nodiscard]] const decomposition_provider_descriptor_v1*
find_decomposition_provider_v1(
    const decomposition_portfolio_view_v1& portfolio,
    decomposition_provider_kind_v1 kind) noexcept;

}  // namespace Cellerator::compiler::planning
