#include <Cellerator/compiler/sema/relation_update_spine_bridge.hh>
#include <Cellerator/compiler/frontend/source/build_a_lossless_raw_token_stream_v1.hh>
#include <Cellerator/compiler/frontend/parser/parse_relation_application_v1.hh>
#include <algorithm>
#include <cctype>
#include <set>

namespace Cellerator::compiler::sema {
namespace {
namespace raw = frontend::source;
namespace parser = frontend::parser;
namespace irs = ir::semantic;
namespace rel = cellerator::compute::relation;
using kind = irs::gradient_publication_operation_ir_v1;
bool identifier(std::string_view s) {
    return !s.empty() && (std::isalpha(static_cast<unsigned char>(s[0])) || s[0] == '_') &&
        std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isalnum(c) || c == '_'; });
}
}
relation_update_source_result lower_relation_update_source_slice_v1(
    std::string_view source, const relation_update_source_environment& env) {
    relation_update_source_result result{};
    const auto error = [&](std::string message, std::size_t begin, std::size_t end) {
        result.diagnostics.push_back({std::move(message), {begin, end}});
    };
    std::set<std::string> names;
    for (const auto* name : {&env.relation_name, &env.input_name, &env.cotangent_name,
            &env.destination_name, &env.output_name, &env.adjoint_name, &env.gradient_name,
            &env.delta_name, &env.alpha_name}) {
        if (!identifier(*name) || !names.insert(*name).second || *name == "ce")
            error("missing, invalid or shadowed environment symbol: " + *name, 0, 0);
    }
    if (!result.diagnostics.empty()) return result;
    auto& program = result.program;
    program.program_identity = env.program_identity;
    program.prepared_generation = env.initial_generation;
    program.calculus.forward.topology = env.topology;
    program.calculus.forward.arithmetic = env.arithmetic;
    program.calculus.forward.dense_width = env.dense_width;
    program.calculus.transpose = program.calculus.forward;
    program.calculus.transpose.direction = rel::orientation::transpose;
    program.calculus.gradient = env.gradient_arithmetic;
    const auto tokens = raw::build_raw_token_stream_v1(1, source, 0);
    std::size_t cursor = 0;
    bool updated = false, published = false, computed_gradient = false;
    std::uint64_t generation = env.initial_generation;
    while (cursor < tokens.tokens.size()) {
        const auto first = cursor;
        while (cursor < tokens.tokens.size() && tokens.tokens[cursor].spelling != ";") ++cursor;
        const auto begin = tokens.tokens[first].span.begin.byte_offset;
        const auto end = cursor < tokens.tokens.size() ? tokens.tokens[cursor].span.end.byte_offset : source.size();
        if (cursor == tokens.tokens.size()) { error("statement requires a semicolon", begin, end); break; }
        std::vector<std::string> t;
        for (auto j = first; j < cursor; ++j) t.push_back(tokens.tokens[j].spelling);
        ++cursor;
        if (t.empty() || program.stages.size() == rel::max_relation_effects) {
            error("empty statement or closure stage capacity exceeded", begin, end); break;
        }
        irs::gradient_publication_stage_ir_v1 stage{};
        stage.identity = program.stages.size() + 1;
        stage.dependencies = program.stages.empty() ? 0 : 1u << (program.stages.size() - 1);
        stage.source_begin = begin; stage.source_end = end;
        stage.consumed_generation = generation;
        stage.input_axis = env.topology.source; stage.output_axis = env.topology.destination;
        stage.gradient_order = env.topology.logical_edge_order;
        // Existing relation syntax parser supplies apply semantics. Full token
        // consumption below closes its intentionally permissive prefix behavior.
        if (std::find(t.begin(), t.end(), "[") != t.end()) {
            std::string normalized;
            for (const auto& token : t) normalized += token;
            const auto parsed = parser::parse_relation_applications_v1(normalized);
            if (!parsed.accepted() || parsed.applications.size() != 1) {
                error("invalid bounded relation application", begin, end); break;
            }
            const auto& a = parsed.applications[0];
            if (a.range.end != normalized.size() || a.result_expression != env.output_name ||
                a.source_expression != env.input_name || a.destination_axis_expression != env.destination_name ||
                a.selector.relation_expression != env.relation_name || !a.selector.support_expression.empty() ||
                !a.selector.source_axis_expression.empty() || a.update != parser::relation_update_v1::overwrite ||
                (updated && !published)) {
                error("apply requires declared source, relation, output and destination axis in published order", begin, end); break;
            }
            stage.kind = kind::forward;
        } else {
            std::size_t pos = 0;
            std::string output;
            if (t.size() >= 2 && t[1] == "=") { output = t[0]; pos = 2; }
            if (pos + 6 > t.size() || t[pos] != "ce" || t[pos+1] != ":" || t[pos+2] != ":" ||
                !identifier(t[pos+3]) || t[pos+4] != "(" || t.back() != ")") {
                error("unsupported embedded statement", begin, end); break;
            }
            const auto callee = t[pos+3];
            std::vector<std::string> args;
            bool syntax_ok = true;
            for (auto j = pos+5; j < t.size()-1; ++j) {
                if ((j-(pos+5)) % 2 == 0) { if (!identifier(t[j])) syntax_ok = false; args.push_back(t[j]); }
                else if (t[j] != ",") syntax_ok = false;
            }
            if ((t.size()-1-(pos+5)) % 2 == 0) syntax_ok = false;
            if (!syntax_ok || args.empty() || args[0] != env.relation_name) {
                error("call requires declared relation and simple symbol arguments", begin, end); break;
            }
            if (callee == "transpose" && output == env.adjoint_name &&
                args == std::vector<std::string>{env.relation_name, env.cotangent_name} && !updated) {
                stage.kind = kind::transpose;
                stage.input_axis = env.topology.destination; stage.output_axis = env.topology.source;
            } else if (callee == "contract_on" && output == env.gradient_name &&
                args == std::vector<std::string>{env.relation_name, env.input_name, env.cotangent_name} && !updated) {
                stage.kind = kind::value_gradient; computed_gradient = true;
            } else if (callee == "apply_value_delta" && output.empty() && !updated &&
                args == std::vector<std::string>{env.relation_name, env.delta_name}) {
                stage.kind = kind::delta_add; updated = true;
                program.calculus.update = rel::value_update_kind::delta_add;
                stage.published_generation = env.next_generation;
            } else if (callee == "gradient_step" && output.empty() && !updated && computed_gradient &&
                args == std::vector<std::string>{env.relation_name, env.gradient_name, env.alpha_name}) {
                stage.kind = kind::gradient_step; updated = true;
                program.calculus.update = rel::value_update_kind::gradient_step;
                stage.published_generation = env.next_generation;
            } else if (callee == "publish_generation" && output.empty() && updated && !published && args.size() == 1) {
                stage.kind = kind::publish_generation; published = true;
                generation = env.next_generation; stage.consumed_generation = generation;
            } else if (callee == "observe_generation" && output.empty() && published && args.size() == 1) {
                stage.kind = kind::observe_generation;
            } else {
                error("unknown call, undeclared output, wrong operand axes or invalid update/publication order", begin, end); break;
            }
        }
        program.stages.push_back(stage);
    }
    if (!result.diagnostics.empty()) return result;
    const auto status = irs::lower_gradient_publication_program_ir_v1(program, &result.effects);
    if (status != irs::gradient_publication_status_ir_v1::success) {
        error("invalid closure calculus/effects (code " + std::to_string(static_cast<unsigned>(status)) + ")", 0, source.size());
        return result;
    }
    result.semantic = program.calculus;
    result.lowered = true;
    return result;
}
} // namespace Cellerator::compiler::sema
