#include "Cellerator/execution/program/program_v2.h"

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

program_status execute_prepared_program_v2(
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
        if (binding.workspace_bytes < stage.required_workspace_bytes) {
            return program_status::insufficient_bindings;
        }
        if (stage.launch(stage.prepared_state, binding, caller_stream) !=
            program_status::success) return program_status::launch_failed;
    }
    return program_status::success;
}

}  // namespace cellerator::execution::program
