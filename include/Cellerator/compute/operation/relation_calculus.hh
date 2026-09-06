#pragma once
// Value-owned mathematical contract. Runtime pointers, versions and provenance
// are deliberately absent. This internal API is not an installed SDK contract.
#include <Cellerator/compute/operation/relation_semantics.hh>
#include <array>
#include <cstdint>
#include <type_traits>

namespace cellerator::compute::relation {
enum class gradient_arithmetic : std::uint8_t { full_f32, round_operands_f16_rne };
enum class value_update_kind : std::uint8_t { delta_add, gradient_step };
struct scalar_gradient_policy {
    execution::numeric_type input_storage = execution::numeric_type::f32;
    execution::numeric_type accumulation = execution::numeric_type::f32;
    execution::numeric_type output_storage = execution::numeric_type::f32;
    output_update output = output_update::overwrite;
    std::uint32_t channels_per_edge = 1;
    bool permit_fma = true;
    bool permit_reassociation = true;
    nonfinite_policy nonfinite = nonfinite_policy::propagate;
};
struct relation_calculus_descriptor {
    operation_descriptor forward{};
    operation_descriptor transpose{};
    gradient_arithmetic gradient = gradient_arithmetic::full_f32;
    value_update_kind update = value_update_kind::gradient_step;
    scalar_gradient_policy scalar_gradient{};
};
status validate(const relation_calculus_descriptor&) noexcept;
bool equivalent(const relation_calculus_descriptor&, const relation_calculus_descriptor&) noexcept;

// A bounded straight-line effect witness, not a graph runtime. Dependencies are
// explicit stage-index bits; transitive closure must prove old reads precede a
// mutation and publication precedes use of the new generation.
enum class relation_effect_kind : std::uint8_t {
    forward, transpose, edge_gradient, value_update, publication
};
inline constexpr std::uint32_t max_relation_effects = 16;
struct relation_effect {
    std::uint64_t identity = 0;
    relation_effect_kind kind = relation_effect_kind::forward;
    std::uint32_t dependencies = 0;
    execution::value_generation reads{};
    execution::value_generation writes{};
};
struct relation_effect_sequence {
    execution::value_generation initial_generation{};
    std::uint32_t count = 0;
    std::array<relation_effect, max_relation_effects> stages{};
};
status validate(const relation_effect_sequence&) noexcept;
static_assert(std::is_trivially_copyable<relation_calculus_descriptor>::value,
    "calculus descriptors must remain value-owned");
static_assert(std::is_trivially_copyable<relation_effect_sequence>::value,
    "effect witnesses must not borrow self-referential stage storage");
} // namespace cellerator::compute::relation
