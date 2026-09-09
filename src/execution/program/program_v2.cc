#include "Cellerator/execution/program/program_v2.h"
#include <Cellerator/execution/launch_bindings.hh>

namespace cellerator::execution::program {

program_status validate_prepared_program_v2(
        const prepared_program_v2& program) noexcept {
    if (program.version != 2 ||
        (program.stage_count != 0 && program.stages == nullptr) ||
        (program.dependency_count != 0 && program.dependencies == nullptr)) {
        return program_status::invalid_argument;
    }
    for (std::uint64_t i = 0; i < program.stage_count; ++i) {
        const auto& stage = program.stages[i];
        if (stage.stable_stage_id == 0 || stage.candidate_id == 0 ||
            stage.launch == nullptr ||
            stage.first_dependency > program.dependency_count ||
            stage.dependency_count >
                    program.dependency_count - stage.first_dependency) {
            return program_status::invalid_stage_graph;
        }
        for (std::uint32_t d = 0; d < stage.dependency_count; ++d) {
            if (program.dependencies[stage.first_dependency + d] >= i) {
                return program_status::invalid_stage_graph;
            }
        }
    }
    return program_status::success;
}

program_status preflight_prepared_program_v2(
        const prepared_program_v2& program,
        const launch_binding_v2* bindings,
        std::uint64_t binding_count,
        void* caller_stream) noexcept {
    const auto valid = validate_prepared_program_v2(program);
    if (valid != program_status::success) return valid;
    if (binding_count != 0 && bindings == nullptr) {
        return program_status::invalid_argument;
    }
    for (std::uint64_t i = 0; i < program.stage_count; ++i) {
        const auto& stage = program.stages[i];
        if (stage.binding_index >= binding_count) {
            return program_status::insufficient_bindings;
        }
        const auto& binding = bindings[stage.binding_index];
        if (binding.workspace_bytes < stage.required_workspace_bytes ||
            (stage.required_workspace_bytes != 0 && binding.workspace == nullptr)) {
            return program_status::insufficient_bindings;
        }
        if ((stage.binding_contract == nullptr) != (binding.typed == nullptr)) {
            return program_status::invalid_typed_binding;
        }
        if (binding.typed != nullptr) {
            const auto& typed = *binding.typed;
            if (binding.input != nullptr || binding.output != nullptr ||
                binding.values != nullptr || typed.stream.stream != caller_stream ||
                typed.workspace.data != binding.workspace ||
                typed.workspace.bytes != binding.workspace_bytes ||
                execution::validate_launch_bindings(*stage.binding_contract, typed) !=
                    execution::binding_validation_code::ok) {
                return program_status::invalid_typed_binding;
            }
        }
        if (stage.preflight != nullptr &&
            stage.preflight(stage.prepared_state, binding, caller_stream) !=
                program_status::success) {
            return program_status::invalid_dynamic_binding;
        }
    }
    return program_status::success;
}

program_status execute_prepared_program_v2(
        const prepared_program_v2& program,
        const launch_binding_v2* bindings,
        std::uint64_t binding_count,
        void* caller_stream) noexcept {
    submission_report_v2 report;
    return execute_prepared_program_report_v2(program,bindings,binding_count,caller_stream,report);
}

program_status execute_prepared_program_report_v2(
        const prepared_program_v2& program,const launch_binding_v2* bindings,
        std::uint64_t binding_count,void* caller_stream,submission_report_v2& report) noexcept {
    report={};
    const auto valid = preflight_prepared_program_v2(
        program, bindings, binding_count, caller_stream);
    if (valid != program_status::success) return report.status=valid;
    for (std::uint64_t i = 0; i < program.stage_count; ++i) {
        const auto& stage = program.stages[i];
        const auto& binding = bindings[stage.binding_index];
        ++report.attempted_stages;
        if (stage.launch(stage.prepared_state, binding, caller_stream) !=
            program_status::success) return report.status=program_status::launch_failed;
        ++report.accepted_stages;
    }
    return report.status=program_status::success;
}

}  // namespace cellerator::execution::program
