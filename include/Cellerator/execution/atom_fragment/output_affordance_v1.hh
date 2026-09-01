#pragma once

#include <Cellerator/execution/atom_fragment/prepared_atom_fragment_v1.hh>
#include <Cellerator/compute/operation/operation_core_v2/schema.hh>

namespace cellerator::execution::atom_fragment {

struct output_affordance_recipe_v1 {
    joint_compiler::persistent_identity_v1 output_atom_identity{};
    joint_compiler::persistent_identity_v1 output_affordance_identity{};
    joint_compiler::persistent_identity_v1 output_plane_identity{};
    joint_compiler::persistent_identity_v1 exact_output_coverage{};
    value_generation output_generation{};
    bool produces_partial = false;
    joint_compiler::persistent_identity_v1 partial_affordance_identity{};
    joint_compiler::persistent_identity_v1 partial_plane_identity{};
    joint_compiler::persistent_identity_v1 partial_algebra{};
};

struct fragment_output_affordance_v1 {
    joint_compiler::persistent_identity_v1 atom_identity{};
    joint_compiler::persistent_identity_v1 affordance_identity{};
    joint_compiler::persistent_identity_v1 plane_identity{};
    joint_compiler::persistent_identity_v1 exact_coverage{};
    order_id order{};
    numeric_type storage = numeric_type::invalid;
    numeric_type logical = numeric_type::invalid;
    value_generation generation{};
    bool partial = false;
    joint_compiler::persistent_identity_v1 partial_algebra{};
};

struct fragment_output_description_v1 {
    fragment_output_affordance_v1 output{};
    fragment_output_affordance_v1 partial{};
    bool has_partial = false;
};

enum class output_affordance_status_code_v1 : std::uint8_t {
    success = 0u,
    null_output,
    invalid_prepared_fragment,
    invalid_operation,
    invalid_recipe,
    invalid_partial_recipe,
    inconsistent_output_order,
};

struct output_affordance_status_v1 {
    output_affordance_status_code_v1 code =
        output_affordance_status_code_v1::success;

    constexpr explicit operator bool() const noexcept {
        return code == output_affordance_status_code_v1::success;
    }
};

output_affordance_status_v1 describe_fragment_output_affordances_v1(
    const prepared_atom_fragment_v1 &prepared,
    const compute::operation::v2::operation_problem &operation,
    const output_affordance_recipe_v1 &recipe,
    fragment_output_description_v1 *description) noexcept;

} // namespace cellerator::execution::atom_fragment
