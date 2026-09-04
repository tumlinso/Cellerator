#pragma once

#include <Cellerator/compiler/sema/implement_axis_semantics_v1.hh>
#include <Cellerator/execution/operands.hh>

#include <cstdint>

namespace cellerator::compiler::sema::v1 {

enum class state_mutability : std::uint8_t { read_only = 1, read_write };
enum class generation_class : std::uint8_t {
    immutable = 1,
    launch_bound,
    evolving
};

struct state_type {
    const axis_type *axes = nullptr;
    std::uint8_t rank = 0;
    execution::numeric_type element_type = execution::numeric_type::invalid;
    std::uint32_t feature_width = 1;
    execution::residency_kind residency_intent = execution::residency_kind::host;
    state_mutability mutability = state_mutability::read_only;
    generation_class generation = generation_class::launch_bound;
};

struct state_view {
    void *data = nullptr;
    state_type type{};
};

enum class state_validation : std::uint8_t {
    ok = 0,
    missing_data,
    invalid_rank,
    invalid_type,
    invalid_feature_width,
    residency_mismatch,
    shape_mismatch
};

state_validation validate_state_view(const state_view &view) noexcept;
state_validation validate_against_dense_operand(
    const state_view &state,
    const execution::dense_tensor_view &operand) noexcept;
state_view bind_pointer(void *pointer, state_type type) noexcept;

}  // namespace cellerator::compiler::sema::v1
