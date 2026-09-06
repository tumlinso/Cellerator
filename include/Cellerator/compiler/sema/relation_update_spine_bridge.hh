#pragma once

#include <Cellerator/compiler/ir/semantic/implement_gradient_and_publication_operations_v1.hh>
#include <Cellerator/compiler/frontend/parser/parse_compiler_semantic_declarations_v1.hh>
#include <string>
#include <string_view>
#include <vector>

namespace Cellerator::compiler::sema {
// Loaded symbol declarations and biological/numeric metadata, never native ops.
// Output names are explicit declarations supplied by the embedding host.
struct relation_update_source_environment {
    std::string relation_name, input_name, cotangent_name, destination_name;
    cellerator::compute::relation::topology_descriptor topology{};
    cellerator::compute::relation::arithmetic_policy arithmetic{};
    std::uint32_t dense_width = 16;
    cellerator::compute::relation::gradient_arithmetic gradient_arithmetic =
        cellerator::compute::relation::gradient_arithmetic::full_f32;
    std::string output_name = "Y", adjoint_name = "dX", gradient_name = "g";
    std::string delta_name = "delta", alpha_name = "alpha";
    std::uint64_t initial_generation = 1, next_generation = 2;
    std::uint64_t program_identity = 1;
};
struct relation_update_source_result {
    cellerator::compute::relation::relation_calculus_descriptor semantic{};
    cellerator::compute::relation::relation_effect_sequence effects{};
    ir::semantic::gradient_publication_program_ir_v1 program{};
    std::vector<frontend::parser::declaration_diagnostic_v1> diagnostics;
    bool lowered = false;
    bool accepted() const noexcept { return lowered && diagnostics.empty(); }
};
// Bounded straight-line embedded statements. Full translation units, imports,
// declarations, arbitrary expressions and implicit output declarations reject.
relation_update_source_result lower_relation_update_source_slice_v1(
    std::string_view source, const relation_update_source_environment&);
} // namespace Cellerator::compiler::sema
