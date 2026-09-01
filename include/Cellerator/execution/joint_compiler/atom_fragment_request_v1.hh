#pragma once

#include <Cellerator/compute/decomposition/decomposition_v1.hh>
#include <Cellerator/execution/joint_compiler/atom_affordance_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::execution::joint_compiler {

inline constexpr std::uint32_t atom_fragment_request_schema_version_v1 = 1u;

struct atom_binding_request_v1 {
    persistent_identity_v1 atom_identity{};
    persistent_identity_v1 requirement_identity{};
    persistent_identity_v1 affordance_identity{};
};

struct atom_fragment_request_v1 {
    std::uint32_t schema_version = atom_fragment_request_schema_version_v1;
    std::uint32_t record_bytes = sizeof(atom_fragment_request_v1);
    persistent_identity_v1 request_identity{};
    const compute::operation::v2::operation_problem *operation = nullptr;
    const logical_coverage_view_v1 *exact_coverages = nullptr;
    std::uint64_t exact_coverage_count = 0u;
    const hierarchical_index_space_view_v1 *local_index_spaces = nullptr;
    std::uint64_t local_index_space_count = 0u;
    const order_id *external_orders = nullptr;
    std::uint64_t external_order_count = 0u;
    const compute::decomposition::decomposition_portfolio_v1 *decomposition =
        nullptr;
    const atom_binding_request_v1 *atom_bindings = nullptr;
    std::uint64_t atom_binding_count = 0u;
    persistent_identity_v1 global_cost_contract{};
    persistent_identity_v1 target_profile{};
    persistent_identity_v1 desired_output_affordance{};
    persistent_identity_v1 lowering_resumption_stage{};
};

enum class atom_fragment_request_validation_code_v1 : std::uint8_t {
    ok = 0u,
    unsupported_schema = 1u,
    invalid_record_bytes = 2u,
    invalid_request_identity = 3u,
    invalid_operation = 4u,
    missing_coverages = 5u,
    invalid_coverage = 6u,
    duplicate_or_unordered_coverage = 7u,
    missing_index_spaces = 8u,
    invalid_index_space = 9u,
    duplicate_or_unordered_index_space = 10u,
    invalid_index_component = 11u,
    invalid_external_order = 12u,
    duplicate_or_unordered_external_order = 13u,
    invalid_decomposition = 14u,
    missing_atom_bindings = 15u,
    invalid_atom_binding = 16u,
    duplicate_or_unordered_atom_binding = 17u,
    invalid_global_cost_contract = 18u,
    invalid_target_profile = 19u,
    invalid_output_affordance = 20u,
    invalid_resumption_stage = 21u
};

struct atom_fragment_request_validation_result_v1 {
    atom_fragment_request_validation_code_v1 code =
        atom_fragment_request_validation_code_v1::ok;
    std::uint64_t index = 0u;
    std::uint64_t nested_index = 0u;

    constexpr explicit operator bool() const noexcept {
        return code == atom_fragment_request_validation_code_v1::ok;
    }
};

atom_fragment_request_validation_result_v1 validate_atom_fragment_request_v1(
    const atom_fragment_request_v1 &request) noexcept;

static_assert(std::is_standard_layout_v<atom_binding_request_v1>);
static_assert(std::is_trivially_copyable_v<atom_binding_request_v1>);
static_assert(std::is_standard_layout_v<atom_fragment_request_v1>);
static_assert(std::is_trivially_copyable_v<atom_fragment_request_v1>);

}  // namespace cellerator::execution::joint_compiler
