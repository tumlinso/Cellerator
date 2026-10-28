#pragma once

#include <Cellerator/compute/operation/relation_calculus.hh>
#include <vector>

namespace Cellerator::compiler::ir::semantic {
namespace relation_calculus = cellerator::compute::relation;

enum class gradient_publication_operation_ir_v1 : std::uint8_t {
    forward = 1, transpose, value_gradient, delta_add, gradient_step,
    publish_generation, observe_generation,
};

// Source identities are provenance. Axes and edge order carry exact biological
// meaning; dependency bits refer to preceding IR stages, including observations.
struct gradient_publication_stage_ir_v1 {
    std::uint64_t identity = 0;
    gradient_publication_operation_ir_v1 kind = gradient_publication_operation_ir_v1::forward;
    relation_calculus::axis_descriptor input_axis{};
    relation_calculus::axis_descriptor output_axis{};
    cellerator::execution::order_id gradient_order{};
    std::uint64_t consumed_generation = 0;
    std::uint64_t published_generation = 0;
    std::uint32_t dependencies = 0;
    std::size_t source_begin = 0;
    std::size_t source_end = 0;
};

struct gradient_publication_program_ir_v1 {
    std::uint64_t program_identity = 0;
    relation_calculus::relation_calculus_descriptor calculus{};
    std::uint64_t prepared_generation = 0;
    std::vector<gradient_publication_stage_ir_v1> stages;
};

enum class gradient_publication_status_ir_v1 : std::uint8_t {
    success = 0, invalid_identity, invalid_generation, invalid_numerical_policy,
    invalid_stage, incomplete_gradient_closure, invalid_axis, invalid_order,
    invalid_dependency,
};

// Produces the common mathematical witness. Observation remains a distinct
// frontend/runtime lease operation; it cannot masquerade as publication.
[[nodiscard]] gradient_publication_status_ir_v1 lower_gradient_publication_program_ir_v1(
    const gradient_publication_program_ir_v1&,
    relation_calculus::relation_effect_sequence* output) noexcept;
[[nodiscard]] gradient_publication_status_ir_v1 validate_gradient_publication_program_ir_v1(
    const gradient_publication_program_ir_v1&) noexcept;
} // namespace Cellerator::compiler::ir::semantic
