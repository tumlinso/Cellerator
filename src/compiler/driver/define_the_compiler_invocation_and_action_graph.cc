#include <Cellerator/compiler/driver/define_the_compiler_invocation_and_action_graph_v1.hh>

namespace cellerator::compiler::driver {
namespace {

bool append(action_graph_v1& graph, action_kind_v1 kind,
            bool semantic_stage) noexcept {
    if (graph.job_count == graph.capacity) {
        return false;
    }
    auto& job = graph.jobs[graph.job_count];
    job.kind = kind;
    job.semantic_stage = semantic_stage;
    if (graph.job_count != 0) {
        job.dependencies[0] = static_cast<std::uint8_t>(graph.job_count - 1);
        job.dependency_count = 1;
    }
    ++graph.job_count;
    return true;
}

}  // namespace

action_graph_result_v1 define_action_graph_v1(
    const compiler_invocation_v1& invocation) noexcept {
    action_graph_result_v1 result{};
    if (invocation.input.empty()) {
        result.diagnostic = diagnostic_code_v1::invalid_argument;
        return result;
    }
    if (invocation.output_mode != output_mode_v1::syntax_only &&
        invocation.output.empty()) {
        result.diagnostic = diagnostic_code_v1::invalid_argument;
        return result;
    }
    if (invocation.output_mode == output_mode_v1::ceir &&
        invocation.source_kind != source_kind_v1::cellerator_cxx) {
        result.diagnostic = diagnostic_code_v1::incompatible_options;
        return result;
    }

    auto add = [&](action_kind_v1 kind, bool semantic = false) {
        if (!append(result.graph, kind, semantic)) {
            result.diagnostic = diagnostic_code_v1::insufficient_capacity;
            return false;
        }
        return true;
    };

    if (!add(action_kind_v1::preprocess)) return result;
    if (invocation.output_mode == output_mode_v1::preprocess) return result;
    if (!add(action_kind_v1::analyze, true)) return result;
    if (invocation.output_mode == output_mode_v1::syntax_only) return result;

    if (invocation.source_kind == source_kind_v1::cellerator_cxx) {
        if (!add(action_kind_v1::emit_ceir, true)) return result;
        if (invocation.output_mode == output_mode_v1::ceir) return result;
    }
    if (invocation.output_mode == output_mode_v1::profile_inspection) {
        add(action_kind_v1::inspect, true);
        return result;
    }
    if (invocation.target.empty() || invocation.host_toolchain.empty()) {
        result.diagnostic = invocation.target.empty()
            ? diagnostic_code_v1::unsupported_target
            : diagnostic_code_v1::unavailable_toolchain;
        result.graph = {};
        return result;
    }
    if (!add(action_kind_v1::compile)) return result;
    if (invocation.output_mode == output_mode_v1::assembly) return result;
    if (!add(action_kind_v1::assemble)) return result;
    if (invocation.output_mode == output_mode_v1::object) return result;
    if (invocation.source_kind == source_kind_v1::cellerator_cxx) {
        if (invocation.device_toolchain.empty()) {
            result.diagnostic = diagnostic_code_v1::unavailable_toolchain;
            result.graph = {};
            return result;
        }
        if (!add(action_kind_v1::device_link)) return result;
    }
    add(action_kind_v1::host_link);
    return result;
}

std::string_view action_name_v1(action_kind_v1 action) noexcept {
    constexpr std::array names{
        std::string_view{"preprocess"}, std::string_view{"analyze"},
        std::string_view{"emit-ceir"}, std::string_view{"compile"},
        std::string_view{"assemble"}, std::string_view{"device-link"},
        std::string_view{"host-link"}, std::string_view{"inspect"}};
    return names[static_cast<std::size_t>(action)];
}

std::string_view diagnostic_name_v1(diagnostic_code_v1 diagnostic) noexcept {
    constexpr std::array names{
        std::string_view{"success"}, std::string_view{"invalid-argument"},
        std::string_view{"incompatible-options"},
        std::string_view{"unsupported-target"},
        std::string_view{"unavailable-toolchain"},
        std::string_view{"insufficient-capacity"}};
    return names[static_cast<std::size_t>(diagnostic)];
}

}  // namespace cellerator::compiler::driver
