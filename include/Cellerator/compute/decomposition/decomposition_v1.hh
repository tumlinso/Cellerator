#pragma once

#include <Cellerator/compute/decomposition/partial_result_algebra_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::compute::decomposition {

inline constexpr std::uint32_t decomposition_schema_version_v1 = 1u;

enum class split_axis_v1 : std::uint8_t {
    none = 0u,
    source_axis = 1u,
    destination_axis = 2u,
    relation_edges = 3u,
    semantic_components = 4u,
    segments = 5u,
    modules = 6u,
    extents = 7u
};

enum decomposition_flag_v1 : std::uint32_t {
    legal_alternative_v1 = 1u << 0u,
    produces_partial_result_v1 = 1u << 1u,
    requires_replication_v1 = 1u << 2u,
    requires_halo_v1 = 1u << 3u,
    complete_unsplit_fallback_v1 = 1u << 4u
};

inline constexpr std::uint32_t known_decomposition_flags_v1 =
    legal_alternative_v1 | produces_partial_result_v1
    | requires_replication_v1 | requires_halo_v1
    | complete_unsplit_fallback_v1;

struct decomposition_alternative_v1 {
    execution::joint_compiler::persistent_identity_v1 alternative_identity{};
    execution::joint_compiler::persistent_identity_v1 candidate_family{};
    split_axis_v1 split_axis = split_axis_v1::none;
    std::uint8_t reserved0[3]{};
    std::uint32_t flags = 0u;
    const execution::joint_compiler::persistent_identity_v1
        *required_input_coverages = nullptr;
    std::uint64_t required_input_coverage_count = 0u;
    execution::joint_compiler::persistent_identity_v1 output_coverage{};
    const execution::joint_compiler::persistent_identity_v1
        *replication_coverages = nullptr;
    std::uint64_t replication_coverage_count = 0u;
    const execution::joint_compiler::persistent_identity_v1
        *halo_coverages = nullptr;
    std::uint64_t halo_coverage_count = 0u;
    execution::order_id input_order{};
    execution::order_id output_order{};
    execution::joint_compiler::persistent_identity_v1 partial_algebra{};
    operation::v2::numerical_policy numerical{};
};

struct decomposition_portfolio_v1 {
    std::uint32_t schema_version = decomposition_schema_version_v1;
    std::uint32_t record_bytes = sizeof(decomposition_portfolio_v1);
    execution::joint_compiler::persistent_identity_v1 portfolio_identity{};
    const decomposition_alternative_v1 *alternatives = nullptr;
    std::uint64_t alternative_count = 0u;
};

enum class decomposition_validation_code_v1 : std::uint8_t {
    ok = 0u,
    unsupported_schema = 1u,
    invalid_record_bytes = 2u,
    invalid_portfolio_identity = 3u,
    missing_alternatives = 4u,
    invalid_alternative_identity = 5u,
    duplicate_or_unordered_alternative = 6u,
    invalid_candidate_family = 7u,
    invalid_split_axis = 8u,
    nonzero_reserved = 9u,
    unknown_flag = 10u,
    alternative_not_legal = 11u,
    invalid_fallback = 12u,
    duplicate_fallback = 13u,
    missing_fallback = 14u,
    invalid_input_coverage = 15u,
    duplicate_or_unordered_input_coverage = 16u,
    invalid_output_coverage = 17u,
    invalid_replication_coverage = 18u,
    invalid_replication_flag = 19u,
    invalid_halo_coverage = 20u,
    invalid_halo_flag = 21u,
    invalid_order = 22u,
    invalid_partial_algebra = 23u,
    unexpected_partial_algebra = 24u,
    invalid_numerical_policy = 25u
};

struct decomposition_validation_result_v1 {
    decomposition_validation_code_v1 code =
        decomposition_validation_code_v1::ok;
    std::uint64_t alternative_index = 0u;
    std::uint64_t element_index = 0u;

    constexpr explicit operator bool() const noexcept {
        return code == decomposition_validation_code_v1::ok;
    }
};

decomposition_validation_result_v1 validate_decomposition_portfolio_v1(
    const decomposition_portfolio_v1 &portfolio) noexcept;

static_assert(std::is_standard_layout_v<decomposition_alternative_v1>);
static_assert(std::is_trivially_copyable_v<decomposition_alternative_v1>);
static_assert(std::is_standard_layout_v<decomposition_portfolio_v1>);
static_assert(std::is_trivially_copyable_v<decomposition_portfolio_v1>);

}  // namespace cellerator::compute::decomposition
