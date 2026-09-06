#pragma once
// PROPOSED POST-EPIC DECLARATIONS ONLY; no implementation.
#include <Cellerator/compute/operation/relation_semantics.hh>
#include <cstdint>
namespace cellerator::compute::relation {
enum class gradient_arithmetic : std::uint8_t { full_f32, round_operands_f16_rne };
enum class value_update_kind : std::uint8_t { delta_add, gradient_step };
struct relation_calculus_descriptor {
    operation_descriptor forward;
    operation_descriptor transpose;
    gradient_arithmetic gradient = gradient_arithmetic::full_f32;
    value_update_kind update = value_update_kind::gradient_step;
};
status validate(const relation_calculus_descriptor&) noexcept;
bool equivalent(const relation_calculus_descriptor&,const relation_calculus_descriptor&) noexcept;
} // namespace cellerator::compute::relation
