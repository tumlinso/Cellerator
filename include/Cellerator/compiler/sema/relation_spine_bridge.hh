#pragma once

#include <Cellerator/compiler/frontend/parser/parse_compiler_semantic_declarations_v1.hh>
#include <Cellerator/compiler/ir/semantic/implement_relation_apply_and_transpose_operations_v1.hh>
#include <string>
#include <string_view>
#include <vector>

namespace Cellerator::compiler::sema {
namespace spine_ir = ir::semantic;
struct relation_spine_axis_binding {
    std::string name;
    spine_ir::axis_ir_type_v1 axis;
};
struct relation_spine_relation_binding {
    std::string name;
    spine_ir::relation_ir_type_v1 relation;
    cellerator::execution::numeric_type storage = cellerator::execution::numeric_type::invalid;
};
struct relation_spine_state_binding {
    std::string name;
    spine_ir::state_ir_type_v1 state;
};
// Loaded metadata supplies identities and lifetimes, never a prebuilt operation.
struct relation_spine_environment {
    std::vector<relation_spine_axis_binding> axes;
    std::vector<relation_spine_relation_binding> relations;
    std::vector<relation_spine_state_binding> states;
};
struct relation_spine_source_provenance {
    frontend::parser::parser_source_range_v1 expression_range{};
    std::string relation_symbol;
    std::string source_symbol;
    std::string result_symbol;
};
struct relation_spine_source_result {
    spine_ir::lowered_relation_apply_v1 lowered;
    relation_spine_source_provenance provenance;
    std::vector<frontend::parser::declaration_diagnostic_v1> diagnostics;
    [[nodiscard]] bool accepted() const noexcept { return diagnostics.empty(); }
};
// Bounded embedded slice: simple declarations and one assigned relation apply.
// Full .cell translation units, expressions, support filters and execution are
// deliberately not implemented by this binding adapter.
[[nodiscard]] relation_spine_source_result lower_relation_source_slice_v1(
    std::string_view declarations, std::string_view expression,
    const relation_spine_environment& environment,
    spine_ir::semantic_identity_v1 operation_identity);
} // namespace Cellerator::compiler::sema
