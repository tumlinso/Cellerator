#pragma once

#include <Cellerator/compiler/sema/field/define_execution_field_semantic_ownership_v1.hh>
#include <Cellerator/compiler/sema/field/implement_automatic_lifetime_and_generation_transfer_v1.hh>
#include <Cellerator/compiler/sema/field/implement_named_representative_profile_binding_v1.hh>
#include <Cellerator/compute/operation/operation_core_v2/schema.hh>

#include <cstdint>
#include <string>
#include <vector>

namespace Cellerator::compiler::sema::field {

// The C++ frontend supplies this record only after ordinary name and overload
// resolution. It deliberately carries semantic identities, never a backend or
// physical projection choice.
struct resolved_relation_field_operation_v1 {
    frontend::source::source_span_v1 source{};
    const void* declaration = nullptr;
    std::string qualified_name;
    std::uint64_t operation_identity = 0;
    cellerator::compute::operation::v2::typed_relation relation{};
    cellerator::compute::operation::v2::operation_kind kind =
        cellerator::compute::operation::v2::operation_kind::relation_apply;
    cellerator::compute::operation::v2::relation_orientation orientation =
        cellerator::compute::operation::v2::relation_orientation::forward;
    cellerator::execution::persistent_axis_identity values_axis{};
    cellerator::execution::persistent_axis_identity result_axis{};
    cellerator::compute::operation::v2::numerical_policy numeric{};
    cellerator::compute::operation::v2::output_contract output{};
    cellerator::compute::operation::v2::determinism_contract determinism{};
    std::uint64_t logical_work_items = 0;
    std::uint32_t dense_width = 0;
    std::uint32_t requirement_flags = 0;
};

struct profile_required_semantic_field_request_v1 {
    execution_field_definition_v1 field;
    resolved_relation_field_operation_v1 operation;
    std::vector<representative_profile_state_v1> profiles;
    std::vector<representative_profile_alias_v1> profile_aliases;
    std::string selected_profile;
    automatic_semantic_state_v1 input_state;
    automatic_generation_transfer_v1 generation_transfer;
};

// Pointer-free semantic form of operation_core_v2::operation_problem. Keeping
// the relation inline makes this compiler receipt safe to retain or relocate.
// materialize_operation_problem_v1 supplies the short-lived view required by
// operation_core_v2 without choosing a projection, kernel, or device.
struct semantic_operation_problem_v1 {
    cellerator::compute::operation::v2::stable_id persistent_problem_identity{};
    cellerator::compute::operation::v2::stable_id operation_identity{};
    cellerator::compute::operation::v2::typed_relation relation{};
    cellerator::compute::operation::v2::operation_kind kind =
        cellerator::compute::operation::v2::operation_kind::relation_apply;
    cellerator::compute::operation::v2::relation_orientation orientation =
        cellerator::compute::operation::v2::relation_orientation::forward;
    cellerator::execution::persistent_axis_identity values_axis{};
    cellerator::execution::persistent_axis_identity result_axis{};
    cellerator::execution::value_generation expected_value_generation{};
    cellerator::compute::operation::v2::numerical_policy numeric{};
    cellerator::compute::operation::v2::output_contract output{};
    cellerator::compute::operation::v2::determinism_contract determinism{};
    std::uint64_t logical_work_items = 0;
    std::uint32_t dense_width = 0;
    std::uint32_t requirement_flags = 0;
};

struct profile_required_semantic_field_receipt_v1 {
    execution_field_semantics_v1 field;
    representative_profile_binding_v1 profile_binding;
    resolved_operation_profile_v1 selected_profile;
    automatic_semantic_state_v1 output_state;
    materialized_generation_transition_v1 generation_transition;
    semantic_operation_problem_v1 operation_problem;
    std::string resolved_declaration_name;
    bool physical_execution_selected = false;
};

enum class profile_required_semantic_field_status_v1 : std::uint8_t {
    success = 0,
    invalid_output,
    invalid_field,
    unresolved_cpp_operation,
    operation_outside_field,
    invalid_relation,
    axis_mismatch,
    missing_profile_environment,
    profile_binding_failed,
    generation_transfer_failed,
    operation_problem_invalid,
};

[[nodiscard]] profile_required_semantic_field_status_v1
deliver_the_first_profile_required_semantic_field_slice_v1(
    const profile_required_semantic_field_request_v1& request,
    profile_required_semantic_field_receipt_v1* receipt) noexcept;

[[nodiscard]] cellerator::compute::operation::v2::operation_problem
materialize_operation_problem_v1(
    const semantic_operation_problem_v1& semantics) noexcept;

}  // namespace Cellerator::compiler::sema::field
