#pragma once

#include <Cellerator/compute/decomposition/decomposition_v1.hh>
#include <Cellerator/execution/joint_compiler/atom_affordance_v1.hh>
#include <Cellerator/execution/joint_compiler/atom_fragment_result_v1.hh>
#include <Cellerator/profiling/partition_export.h>

#include <cstdint>
#include <type_traits>

namespace cellerator::profiling::joint_compiler {

inline constexpr std::uint32_t execution_export_schema_version_v2 = 2u;

enum class correctness_compatibility_v2 : std::uint8_t {
    unverified = 1u,
    verified_compatible = 2u,
    verified_incompatible = 3u
};

enum class performance_freshness_v2 : std::uint8_t {
    analytical_only = 1u,
    current = 2u,
    stale = 3u
};

struct performance_freshness_record_v2 {
    performance_freshness_v2 status =
        performance_freshness_v2::analytical_only;
    std::uint8_t reserved[7]{};
    execution::joint_compiler::persistent_identity_v1 evidence_identity{};
    execution::joint_compiler::persistent_identity_v1 device_performance_identity{};
    execution::joint_compiler::persistent_identity_v1 build_identity{};
    std::uint64_t evidence_revision = 0u;
};

struct atom_execution_stage_v2 {
    execution::joint_compiler::persistent_identity_v1 stage_identity{};
    execution::joint_compiler::persistent_identity_v1 candidate_identity{};
    execution::joint_compiler::persistent_identity_v1 input_coverage{};
    execution::joint_compiler::persistent_identity_v1 output_coverage{};
    const std::uint32_t *dependencies = nullptr;
    std::uint32_t dependency_count = 0u;
    std::uint32_t launch_count = 0u;
};

struct execution_export_v2 {
    std::uint32_t schema_version = execution_export_schema_version_v2;
    std::uint32_t record_bytes = sizeof(execution_export_v2);
    execution::joint_compiler::persistent_identity_v1 export_identity{};
    generic_execution_export_v1 compatibility_v1{};
    const execution::joint_compiler::logical_coverage_view_v1
        *exact_coverages = nullptr;
    std::uint64_t exact_coverage_count = 0u;
    const compute::decomposition::decomposition_portfolio_v1 *decomposition =
        nullptr;
    const execution::joint_compiler::atom_requirement_v1 *requirements = nullptr;
    std::uint64_t requirement_count = 0u;
    const execution::joint_compiler::atom_affordance_v1 *affordances = nullptr;
    std::uint64_t affordance_count = 0u;
    const compute::decomposition::partial_result_algebra_v1
        *partial_algebras = nullptr;
    std::uint64_t partial_algebra_count = 0u;
    const execution::order_id *persistent_orders = nullptr;
    std::uint64_t persistent_order_count = 0u;
    const execution::joint_compiler::atom_fragment_result_v1
        *candidate_frontier = nullptr;
    const atom_execution_stage_v2 *stages = nullptr;
    std::uint64_t stage_count = 0u;
    execution::joint_compiler::fragment_complete_cost_v1 complete_cost{};
    correctness_compatibility_v2 correctness =
        correctness_compatibility_v2::unverified;
    std::uint8_t reserved[7]{};
    execution::joint_compiler::persistent_identity_v1 correctness_receipt{};
    performance_freshness_record_v2 performance{};
};

enum class execution_export_validation_code_v2 : std::uint8_t {
    ok = 0u,
    unsupported_schema = 1u,
    invalid_record_bytes = 2u,
    nonzero_reserved = 3u,
    invalid_export_identity = 4u,
    invalid_v1_compatibility = 5u,
    missing_coverages = 6u,
    invalid_coverage = 7u,
    duplicate_or_unordered_coverage = 8u,
    invalid_decomposition = 9u,
    missing_requirements = 10u,
    invalid_requirement = 11u,
    duplicate_or_unordered_requirement = 12u,
    missing_affordances = 13u,
    invalid_affordance = 14u,
    duplicate_or_unordered_affordance = 15u,
    inconsistent_partial_algebra_pointer = 16u,
    invalid_partial_algebra = 17u,
    duplicate_or_unordered_partial_algebra = 18u,
    missing_orders = 19u,
    invalid_order = 20u,
    duplicate_or_unordered_order = 21u,
    invalid_candidate_frontier = 22u,
    missing_stages = 23u,
    invalid_stage = 24u,
    duplicate_or_unordered_stage = 25u,
    invalid_stage_dependency = 26u,
    invalid_complete_cost = 27u,
    invalid_correctness = 28u,
    invalid_correctness_receipt = 29u,
    invalid_performance_freshness = 30u
};

struct execution_export_validation_result_v2 {
    execution_export_validation_code_v2 code =
        execution_export_validation_code_v2::ok;
    std::uint64_t index = 0u;
    std::uint64_t nested_index = 0u;

    constexpr explicit operator bool() const noexcept {
        return code == execution_export_validation_code_v2::ok;
    }
};

execution_export_validation_result_v2 validate_execution_export_v2(
    const execution_export_v2 &value) noexcept;

static_assert(std::is_standard_layout_v<performance_freshness_record_v2>);
static_assert(std::is_trivially_copyable_v<performance_freshness_record_v2>);
static_assert(std::is_standard_layout_v<atom_execution_stage_v2>);
static_assert(std::is_trivially_copyable_v<atom_execution_stage_v2>);
static_assert(std::is_standard_layout_v<execution_export_v2>);
static_assert(std::is_trivially_copyable_v<execution_export_v2>);

}  // namespace cellerator::profiling::joint_compiler
