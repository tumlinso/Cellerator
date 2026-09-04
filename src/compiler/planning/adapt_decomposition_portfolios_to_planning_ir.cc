#include <Cellerator/compiler/planning/adapt_decomposition_portfolios_to_planning_ir_v1.hh>

#include <array>
#include <cstring>

namespace Cellerator::compiler::planning {
namespace {

constexpr std::uint64_t unbounded = UINT64_MAX;

constexpr std::array<decomposition_provider_descriptor_v1, 6> providers{{
    {1u, decomposition_provider_kind_v1::greedy, 0u, 0u, 0x4752454544590001ULL,
     deterministic_provider_v1 | incremental_provider_v1 | multi_operation_provider_v1,
     0u, {unbounded, unbounded, 1ULL << 34u, unbounded}, "joint-greedy"},
    {1u, decomposition_provider_kind_v1::multilevel, 1u, 0u, 0x4d554c5449000001ULL,
     deterministic_provider_v1 | incremental_provider_v1 | multi_operation_provider_v1,
     0u, {unbounded, unbounded, 1ULL << 34u, unbounded}, "sparse-multilevel"},
    {1u, decomposition_provider_kind_v1::exact_oracle, 2u, 0u, 0x4f5241434c450001ULL,
     deterministic_provider_v1 | exact_coverage_provider_v1 | multi_operation_provider_v1,
     0u, {64u, 4096u, 1ULL << 32u, 1ULL << 40u}, "exact-oracle"},
    {1u, decomposition_provider_kind_v1::bounded_overlap, 3u, 0u, 0x4f5645524c415001ULL,
     deterministic_provider_v1 | exact_coverage_provider_v1 | multi_operation_provider_v1,
     0u, {unbounded, unbounded, 1ULL << 34u, unbounded}, "bounded-overlap"},
    {1u, decomposition_provider_kind_v1::device_assisted, 4u, 0u, 0x4445564943450001ULL,
     exact_coverage_provider_v1 | multi_operation_provider_v1 | experimental_provider_v1,
     0u, {unbounded, unbounded, 1ULL << 34u, unbounded}, "device-assisted"},
    {1u, decomposition_provider_kind_v1::user_provided, 5u, 0u, 0x5553455200000001ULL,
     exact_coverage_provider_v1 | external_provider_v1,
     0u, {unbounded, unbounded, 1ULL << 34u, unbounded}, "user-provided"},
}};

constexpr bool valid_kind(decomposition_provider_kind_v1 kind) noexcept {
    return kind >= decomposition_provider_kind_v1::greedy &&
        kind <= decomposition_provider_kind_v1::user_provided;
}

}  // namespace

decomposition_portfolio_view_v1 built_in_decomposition_portfolio_v1() noexcept {
    return {decomposition_portfolio_schema_v1,
            static_cast<std::uint32_t>(providers.size()), providers.data()};
}

decomposition_portfolio_validation_code_v1 validate_decomposition_portfolio_v1(
    const decomposition_portfolio_view_v1& portfolio) noexcept {
    if (portfolio.schema_version != decomposition_portfolio_schema_v1) {
        return decomposition_portfolio_validation_code_v1::unsupported_schema;
    }
    if (portfolio.providers == nullptr || portfolio.provider_count == 0u) {
        return decomposition_portfolio_validation_code_v1::missing_providers;
    }
    for (std::uint32_t i = 0u; i < portfolio.provider_count; ++i) {
        const auto& provider = portfolio.providers[i];
        if (provider.schema_version != decomposition_portfolio_schema_v1 ||
            !valid_kind(provider.kind) || provider.provider_identity == 0u ||
            provider.stable_name == nullptr || provider.stable_name[0] == '\0' ||
            provider.registry_order != i) {
            return decomposition_portfolio_validation_code_v1::invalid_provider;
        }
        if (provider.bounds.maximum_nodes == 0u ||
            provider.bounds.maximum_edges == 0u ||
            provider.bounds.maximum_workspace_bytes == 0u ||
            provider.bounds.maximum_search_steps == 0u) {
            return decomposition_portfolio_validation_code_v1::unbounded_provider;
        }
        for (std::uint32_t j = 0u; j < i; ++j) {
            if (portfolio.providers[j].provider_identity == provider.provider_identity ||
                portfolio.providers[j].kind == provider.kind ||
                std::strcmp(portfolio.providers[j].stable_name, provider.stable_name) == 0) {
                return decomposition_portfolio_validation_code_v1::duplicate_provider;
            }
        }
    }
    return decomposition_portfolio_validation_code_v1::ok;
}

const decomposition_provider_descriptor_v1* find_decomposition_provider_v1(
    const decomposition_portfolio_view_v1& portfolio,
    decomposition_provider_kind_v1 kind) noexcept {
    if (validate_decomposition_portfolio_v1(portfolio) !=
        decomposition_portfolio_validation_code_v1::ok) {
        return nullptr;
    }
    for (std::uint32_t i = 0u; i < portfolio.provider_count; ++i) {
        if (portfolio.providers[i].kind == kind) return &portfolio.providers[i];
    }
    return nullptr;
}

}  // namespace Cellerator::compiler::planning
