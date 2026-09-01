#pragma once

#include <Cellerator/compute/decomposition/vocabulary_v1.hh>
#include <Cellerator/compute/operation/operation_core_v2/schema.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::compute::decomposition {

inline constexpr std::uint32_t provider_registry_schema_version_v1 = 1u;
inline constexpr std::uint64_t builtin_provider_count_v1 = 11u;

enum class decomposition_provider_kind_v1 : std::uint8_t {
    destination_disjoint = 1u,
    source_k = 2u,
    dense_width = 3u,
    edge_component = 4u,
    relation_bundle_type = 5u,
    transpose_source_partial = 6u,
    support_axis = 7u,
    support_edge_rectangle = 8u,
    support_embedding = 9u,
    segment_disjoint = 10u,
    split_segment_reduce = 11u
};

enum class provider_partial_mode_v1 : std::uint8_t {
    never = 1u,
    always = 2u,
    instance_dependent = 3u
};

enum class provider_instance_validation_code_v1 : std::uint8_t {
    ok = 0u,
    missing_instance,
    missing_workspace,
    invalid_instance
};

struct provider_validation_workspace_v1 {
    void *data = nullptr;
    std::uint64_t byte_count = 0u;
};

using provider_validate_instance_fn_v1 =
    provider_instance_validation_code_v1 (*)(
        const void *, provider_validation_workspace_v1) noexcept;

struct decomposition_provider_v1 {
    operation::v2::stable_id provider_identity{};
    operation::v2::stable_id independent_validation_identity{};
    decomposition_provider_kind_v1 kind =
        decomposition_provider_kind_v1::destination_disjoint;
    operation::v2::operation_kind operation =
        operation::v2::operation_kind::relation_apply;
    split_axis_kind_v1 primary_split_axis = split_axis_kind_v1::none;
    provider_partial_mode_v1 partial_mode = provider_partial_mode_v1::never;
    bool unsplit_fallback_available = true;
    bool requires_exact_coverage = true;
    std::uint16_t reserved = 0u;
    std::uint64_t validation_revision = 0u;
    provider_validate_instance_fn_v1 validate_instance = nullptr;
};

struct decomposition_provider_registry_v1 {
    std::uint32_t schema_version = provider_registry_schema_version_v1;
    std::uint32_t reserved = 0u;
    const decomposition_provider_v1 *providers = nullptr;
    std::uint64_t provider_count = 0u;
};

enum class provider_registry_validation_code_v1 : std::uint8_t {
    ok = 0u,
    unsupported_schema,
    nonzero_reserved,
    missing_providers,
    invalid_provider_count,
    invalid_provider_identity,
    validation_identity_alias,
    provider_order_mismatch,
    invalid_kind,
    invalid_operation,
    invalid_split_axis,
    invalid_partial_mode,
    missing_unsplit_fallback,
    missing_exact_coverage,
    missing_validation_revision,
    missing_validator
};

struct provider_registry_validation_result_v1 {
    provider_registry_validation_code_v1 code =
        provider_registry_validation_code_v1::ok;
    std::uint64_t provider_index = 0u;

    constexpr explicit operator bool() const noexcept {
        return code == provider_registry_validation_code_v1::ok;
    }
};

enum class provider_lookup_code_v1 : std::uint8_t {
    found = 0u,
    no_candidate = 1u,
    invalid_registry = 2u
};

struct provider_lookup_result_v1 {
    const decomposition_provider_v1 *provider = nullptr;
    provider_lookup_code_v1 code = provider_lookup_code_v1::no_candidate;

    constexpr explicit operator bool() const noexcept {
        return code == provider_lookup_code_v1::found;
    }
};

decomposition_provider_registry_v1 builtin_decomposition_providers_v1() noexcept;
provider_registry_validation_result_v1 validate_provider_registry_v1(
    const decomposition_provider_registry_v1 &registry) noexcept;
provider_lookup_result_v1 find_decomposition_provider_v1(
    const decomposition_provider_registry_v1 &registry,
    decomposition_provider_kind_v1 kind) noexcept;

static_assert(std::is_trivially_copyable_v<provider_validation_workspace_v1>);
static_assert(std::is_trivially_copyable_v<decomposition_provider_v1>);
static_assert(std::is_standard_layout_v<decomposition_provider_v1>);
static_assert(std::is_trivially_copyable_v<decomposition_provider_registry_v1>);
static_assert(std::is_trivially_copyable_v<provider_lookup_result_v1>);

}  // namespace cellerator::compute::decomposition
