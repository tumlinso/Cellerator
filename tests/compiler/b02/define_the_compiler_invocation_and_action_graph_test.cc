#include <Cellerator/compiler/driver/define_the_compiler_invocation_and_action_graph_v1.hh>

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

using namespace cellerator::compiler::driver;

bool require(bool condition, const std::string& message) {
    if (!condition) std::cerr << message << '\n';
    return condition;
}

compiler_invocation_v1 invocation(output_mode_v1 mode,
                                  source_kind_v1 source = source_kind_v1::cellerator_cxx) {
    return {mode, source, "input.cc", "output", "sm_70", "clang++", "nvcc"};
}

bool has_exact_actions(const action_graph_result_v1& result,
                       std::initializer_list<action_kind_v1> expected) {
    if (!result || result.graph.job_count != expected.size()) return false;
    std::size_t index = 0;
    for (const auto action : expected) {
        const auto& job = result.graph.jobs[index];
        if (job.kind != action || job.dependency_count != (index == 0 ? 0 : 1) ||
            (index != 0 && job.dependencies[0] != index - 1)) return false;
        ++index;
    }
    return true;
}

}  // namespace

int main() {
    bool valid = true;
    valid &= require(has_exact_actions(define_action_graph_v1(invocation(output_mode_v1::preprocess)),
        {action_kind_v1::preprocess}), "invalid -E graph");
    valid &= require(has_exact_actions(define_action_graph_v1(invocation(output_mode_v1::syntax_only)),
        {action_kind_v1::preprocess, action_kind_v1::analyze}), "invalid syntax-only graph");
    valid &= require(has_exact_actions(define_action_graph_v1(invocation(output_mode_v1::assembly)),
        {action_kind_v1::preprocess, action_kind_v1::analyze, action_kind_v1::emit_ceir,
         action_kind_v1::compile}), "invalid -S graph");
    valid &= require(has_exact_actions(define_action_graph_v1(invocation(output_mode_v1::object)),
        {action_kind_v1::preprocess, action_kind_v1::analyze, action_kind_v1::emit_ceir,
         action_kind_v1::compile, action_kind_v1::assemble}), "invalid -c graph");
    valid &= require(has_exact_actions(define_action_graph_v1(invocation(output_mode_v1::link)),
        {action_kind_v1::preprocess, action_kind_v1::analyze, action_kind_v1::emit_ceir,
         action_kind_v1::compile, action_kind_v1::assemble, action_kind_v1::device_link,
         action_kind_v1::host_link}), "invalid link graph");
    valid &= require(has_exact_actions(define_action_graph_v1(invocation(output_mode_v1::ceir)),
        {action_kind_v1::preprocess, action_kind_v1::analyze, action_kind_v1::emit_ceir}),
        "invalid CEIR-only graph");
    valid &= require(has_exact_actions(define_action_graph_v1(invocation(output_mode_v1::profile_inspection)),
        {action_kind_v1::preprocess, action_kind_v1::analyze, action_kind_v1::emit_ceir,
         action_kind_v1::inspect}), "invalid profile-inspection graph");
    valid &= require(has_exact_actions(define_action_graph_v1(
        invocation(output_mode_v1::link, source_kind_v1::ordinary_cxx)),
        {action_kind_v1::preprocess, action_kind_v1::analyze, action_kind_v1::compile,
         action_kind_v1::assemble, action_kind_v1::host_link}), "ordinary C++ gained CEIR/device jobs");

    auto missing_target = invocation(output_mode_v1::object);
    missing_target.target = {};
    valid &= require(define_action_graph_v1(missing_target).diagnostic ==
        diagnostic_code_v1::unsupported_target, "missing target diagnostic is unstable");
    auto ceir_for_cxx = invocation(output_mode_v1::ceir, source_kind_v1::ordinary_cxx);
    valid &= require(define_action_graph_v1(ceir_for_cxx).diagnostic ==
        diagnostic_code_v1::incompatible_options, "ordinary CEIR request was accepted");

    const auto graph = define_action_graph_v1(invocation(output_mode_v1::link));
    for (std::size_t index = 0; index != graph.graph.job_count; ++index) {
        const auto kind = graph.graph.jobs[index].kind;
        const bool expected_semantic = kind == action_kind_v1::analyze ||
            kind == action_kind_v1::emit_ceir || kind == action_kind_v1::inspect;
        valid &= require(graph.graph.jobs[index].semantic_stage == expected_semantic,
                         "backend policy leaked into semantic action classification");
    }
    if (!valid) return EXIT_FAILURE;
    std::cout << "validated seven driver modes, ordinary C++ fallback, and stable diagnostics\n";
    return EXIT_SUCCESS;
}
