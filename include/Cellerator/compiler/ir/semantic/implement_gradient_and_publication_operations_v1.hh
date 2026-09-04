#pragma once

#include <Cellerator/compiler/ir/semantic/implement_state_and_value_plane_ir_types_v1.hh>
#include <Cellerator/execution/training_program_v2/interface.hh>

#include <cstdint>
#include <vector>

namespace Cellerator::compiler::ir::semantic {

enum class gradient_publication_operation_ir_v1 : std::uint8_t {
    forward = 1,
    transpose,
    value_gradient,
    publish_generation,
    canonicalize,
    caller_update_boundary,
};

struct gradient_publication_stage_ir_v1 {
    std::uint64_t identity = 0;
    gradient_publication_operation_ir_v1 kind = gradient_publication_operation_ir_v1::forward;
    semantic_identity_v1 input_axis{};
    semantic_identity_v1 output_axis{};
    std::uint64_t consumed_generation = 0;
    std::uint64_t published_generation = 0;
    bool explicit_order_transform = false;
};

struct caller_update_policy_boundary_ir_v1 {
    std::uint64_t caller_policy_identity = 0;
    std::uint64_t prepared_update_candidate_identity = 0;
    bool owned_by_caller = true;
};

struct gradient_publication_program_ir_v1 {
    std::uint64_t program_identity = 0;
    semantic_identity_v1 structure_identity{};
    std::uint64_t structure_epoch = 0;
    std::uint64_t prepared_generation = 0;
    numeric_tuple_ir_v1 numerical{};
    std::vector<gradient_publication_stage_ir_v1> stages;
    caller_update_policy_boundary_ir_v1 update_policy{};
};

enum class gradient_publication_status_ir_v1 : std::uint8_t {
    success = 0,
    invalid_identity,
    invalid_generation,
    invalid_numerical_policy,
    invalid_stage,
    incomplete_gradient_closure,
    invalid_canonicalization,
    update_policy_not_caller_owned,
    training_contract_mismatch,
};

[[nodiscard]] gradient_publication_status_ir_v1
validate_gradient_publication_program_ir_v1(
    const gradient_publication_program_ir_v1& program) noexcept;

[[nodiscard]] gradient_publication_status_ir_v1
compare_gradient_program_with_training_v2(
    const gradient_publication_program_ir_v1& semantic,
    const cellerator::execution::training_v2::training_program_v2& training) noexcept;

[[nodiscard]] cellerator::execution::training_v2::training_stage_kind_v2
lower_gradient_publication_stage_kind_v1(
    gradient_publication_operation_ir_v1 kind) noexcept;

}  // namespace Cellerator::compiler::ir::semantic
