#include <Cellerator/compiler/sema/relation_spine_bridge.hh>
#include <Cellerator/compiler/frontend/parser/parse_biological_type_constructors_and_qualifiers_v1.hh>
#include <Cellerator/compiler/frontend/parser/parse_non_relation_operation_families_v1.hh>
#include <Cellerator/compiler/frontend/parser/parse_relation_application_v1.hh>
#include <Cellerator/compiler/sema/implement_numerical_tuple_semantics_v1.hh>
#include <algorithm>
#include <cctype>
#include <map>

namespace Cellerator::compiler::sema {
namespace {
namespace parser = frontend::parser;
using cellerator::execution::numeric_type;
using declaration = parser::semantic_declaration_v1;
using kind = parser::semantic_declaration_kind_v1;

bool identifier(std::string_view text) {
    if (text.empty() || !(std::isalpha(static_cast<unsigned char>(text[0])) || text[0] == '_')) return false;
    return std::all_of(text.begin(), text.end(), [](unsigned char c) { return std::isalnum(c) || c == '_'; });
}
bool same(spine_ir::semantic_identity_v1 a, spine_ir::semantic_identity_v1 b) {
    return a.low == b.low && a.high == b.high;
}
numeric_type numeric(std::string_view name) {
    if (name == "f16") return numeric_type::f16;
    if (name == "f32") return numeric_type::f32;
    if (name == "f64") return numeric_type::f64;
    return numeric_type::invalid;
}
template<class T> const T* lookup(const std::vector<T>& bindings, const std::string& name) {
    const T* found = nullptr;
    for (const auto& binding : bindings) if (binding.name == name) {
        if (found) return nullptr; // Ambiguous runtime symbol binding fails closed.
        found = &binding;
    }
    return found;
}
} // namespace

relation_spine_source_result lower_relation_source_slice_v1(
    std::string_view declarations, std::string_view expression,
    const relation_spine_environment& environment,
    spine_ir::semantic_identity_v1 operation_identity) {
    relation_spine_source_result result;
    const auto error = [&](std::string message, parser::parser_source_range_v1 range) {
        result.diagnostics.push_back({std::move(message), range});
    };
    const auto parsed = parser::parse_semantic_declarations_v1(declarations);
    result.diagnostics = parsed.diagnostics;
    if (!result.accepted()) return result;
    std::map<std::string, declaration> symbols;
    std::map<std::string, parser::biological_type_syntax_v1> types;
    for (const auto& decl : parsed.declarations) {
        const auto suffix_begin = decl.range.begin + decl.type_spelling.size() + decl.name.size();
        auto suffix = declarations.substr(suffix_begin, decl.range.end - suffix_begin - 1);
        if (std::any_of(suffix.begin(), suffix.end(), [](unsigned char c) { return !std::isspace(c); }))
            error("declaration initializers and qualifiers are unsupported in this slice", decl.range);
        if (!symbols.emplace(decl.name, decl).second) error("duplicate declaration: " + decl.name, decl.range);
        if (decl.kind == kind::domain) continue;
        auto type = parser::parse_biological_type_v1(decl.type_spelling);
        if (!type.accepted()) { error(type.diagnostic, decl.range); continue; }
        const auto count = decl.kind == kind::relation ? 3u : decl.kind == kind::state ? 2u : decl.kind == kind::axis ? 1u : 0u;
        if (!count || type.type.arguments.size() != count || !type.type.qualifiers.empty()) {
            error("unsupported declaration in relation slice: " + decl.name, decl.range); continue;
        }
        for (const auto& arg : type.type.arguments)
            if (!identifier(arg.constructor) || !arg.arguments.empty() || !arg.qualifiers.empty())
                error("only concrete scalar and domain arguments are supported", decl.range);
        types.emplace(decl.name, std::move(type.type));
    }
    const auto domain_matches = [&](const std::string& name, const spine_ir::axis_ir_type_v1& axis) {
        const auto it = symbols.find(name);
        return it != symbols.end() && it->second.kind == kind::domain && axis.domain.nominal_tag == name;
    };
    for (const auto& entry : types) {
        const auto& name = entry.first;
        const auto& type = entry.second;
        const auto& decl = symbols.at(name);
        if (decl.kind == kind::axis) {
            const auto* binding = lookup(environment.axes, name);
            if (!binding || !domain_matches(type.arguments[0].constructor, binding->axis))
                error("axis domain or runtime binding mismatch: " + name, decl.range);
        } else if (decl.kind == kind::relation) {
            const auto* binding = lookup(environment.relations, name);
            if (!binding || type.constructor != "relation" || numeric(type.arguments[0].constructor) != binding->storage ||
                binding->storage == numeric_type::invalid ||
                !domain_matches(type.arguments[1].constructor, binding->relation.source_axis) ||
                !domain_matches(type.arguments[2].constructor, binding->relation.destination_axis))
                error("relation numeric, endpoint or runtime binding mismatch: " + name, decl.range);
        } else if (decl.kind == kind::state) {
            const auto* binding = lookup(environment.states, name);
            bool axis_matches = false;
            if (binding && binding->state.axes.size() == 1) for (const auto& axis : environment.axes)
                if (same(axis.axis.identity, binding->state.axes[0]) && domain_matches(type.arguments[1].constructor, axis.axis)) axis_matches = true;
            if (!binding || numeric(type.arguments[0].constructor) == numeric_type::invalid ||
                numeric(type.arguments[0].constructor) != binding->state.numeric.storage || !axis_matches)
                error("state numeric, axis or runtime binding mismatch: " + name, decl.range);
        }
    }
    if (!result.accepted()) return result;
    const auto applications = parser::parse_relation_applications_v1(expression);
    result.diagnostics = applications.diagnostics;
    if (!result.accepted()) return result;
    if (applications.applications.size() != 1) { error("expected one relation application", {0, expression.size()}); return result; }
    const auto& app = applications.applications.front();
    // The existing expression parser accepts a prefix; this bounded adapter must
    // reject any unconsumed statements instead of silently ignoring them.
    auto tail = expression.substr(app.range.end);
    if (!tail.empty() && tail.front() == ';') tail.remove_prefix(1);
    if (std::any_of(tail.begin(), tail.end(), [](unsigned char c) { return !std::isspace(c); }))
        error("unconsumed source after relation application", {app.range.end, expression.size()});
    std::string relation_name = app.selector.relation_expression;
    if (app.selector.orientation == parser::relation_orientation_v1::transpose) {
        const auto transposed = parser::parse_operation_families_v1(relation_name);
        if (!transposed.accepted() || transposed.operations.size() != 1 ||
            transposed.operations[0].family != parser::operation_family_v1::transpose ||
            transposed.operations[0].arguments.size() != 1 ||
            transposed.operations[0].range.end != relation_name.size()) {
            error("invalid transpose selector", app.range); return result;
        }
        relation_name = transposed.operations[0].arguments[0];
    }
    const auto declared = [&](const std::string& name, kind expected) {
        const auto it = symbols.find(name);
        return identifier(name) && it != symbols.end() && it->second.kind == expected;
    };
    if (!declared(relation_name, kind::relation) || !declared(app.source_expression, kind::state) ||
        !declared(app.result_expression, kind::state) || !declared(app.destination_axis_expression, kind::axis) ||
        app.update == parser::relation_update_v1::expression || !app.selector.support_expression.empty()) {
        error("relation slice requires declared relation, input, result and axis; filters are unsupported", app.range); return result;
    }
    if (!result.accepted()) return result;
    const auto* relation = lookup(environment.relations, relation_name);
    const auto* input = lookup(environment.states, app.source_expression);
    const auto* output = lookup(environment.states, app.result_expression);
    const auto* target = lookup(environment.axes, app.destination_axis_expression);
    if (!relation || !input || !output || !target) { error("missing or ambiguous runtime symbol binding", app.range); return result; }
    spine_ir::relation_apply_operation_ir_v1 operation;
    operation.identity = operation_identity;
    operation.relation = relation->relation;
    operation.relation.orientation = app.selector.orientation == parser::relation_orientation_v1::forward
        ? spine_ir::relation_orientation_ir_v1::forward : spine_ir::relation_orientation_ir_v1::transpose;
    operation.source = input->state;
    operation.result = output->state;
    operation.relation_storage = relation->storage;
    const auto& expected_input = operation.relation.orientation == spine_ir::relation_orientation_ir_v1::forward
        ? operation.relation.source_axis : operation.relation.destination_axis;
    const auto& expected_output = operation.relation.orientation == spine_ir::relation_orientation_ir_v1::forward
        ? operation.relation.destination_axis : operation.relation.source_axis;
    if (!same(input->state.order, expected_input.order.identity) ||
        !same(output->state.order, expected_output.order.identity))
        error("state order does not match oriented relation order", app.range);
    if (!same(target->axis.domain.identity, expected_output.domain.identity) ||
        !same(target->axis.order.identity, expected_output.order.identity) ||
        !same(target->axis.geometry.identity, expected_output.geometry.identity) ||
        !same(target->axis.partition.identity, expected_output.partition.identity) ||
        target->axis.extent.kind != expected_output.extent.kind ||
        target->axis.extent.lower_bound != expected_output.extent.lower_bound ||
        target->axis.extent.upper_bound != expected_output.extent.upper_bound)
        error("destination axis metadata does not match relation endpoint", app.range);
    operation.update = app.update == parser::relation_update_v1::accumulate
        ? cellerator::compute::operation::v2::destination_update::accumulate
        : cellerator::compute::operation::v2::destination_update::overwrite;
    if (output->state.axes.size() != 1 || !same(output->state.axes[0], target->axis.identity))
        error("destination axis does not match result state", app.range);
    if (!app.selector.source_axis_expression.empty()) {
        const auto* axis = lookup(environment.axes, app.selector.source_axis_expression);
        if (!declared(app.selector.source_axis_expression, kind::axis) || !axis ||
            input->state.axes.size() != 1 || !same(input->state.axes[0], axis->axis.identity))
            error("explicit source axis does not match input state", app.range);
    }
    cellerator::compiler::sema::v1::numerical_tuple tuple;
    tuple.relation_storage = relation->storage;
    tuple.dense_input = input->state.numeric.storage;
    tuple.compute = input->state.numeric.compute;
    tuple.accumulation = input->state.numeric.accumulation;
    tuple.output = output->state.numeric.output;
    if (!cellerator::compiler::sema::v1::valid_numerical_tuple(tuple)) error("invalid numerical tuple", app.range);
    if (!result.accepted()) return result;
    if (spine_ir::lower_relation_apply_operation_v1(operation, &result.lowered) !=
        spine_ir::relation_apply_ir_validation_code_v1::success)
        error("relation IR rejects identity, axis, width, numeric or effect contract", app.range);
    result.provenance = {app.range, relation_name, app.source_expression, app.result_expression};
    return result;
}
} // namespace Cellerator::compiler::sema
