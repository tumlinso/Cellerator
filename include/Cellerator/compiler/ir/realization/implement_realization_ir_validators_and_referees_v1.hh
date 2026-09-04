#pragma once

#include <Cellerator/compiler/ir/realization/commit_selected_exact_cover_and_contribution_ownership_v1.hh>
#include <Cellerator/compiler/ir/realization/implement_launch_and_synchronization_dependencies_v1.hh>
#include <Cellerator/compiler/ir/realization/implement_memory_workspace_and_residency_requirements_v1.hh>
#include <Cellerator/compiler/ir/realization/implement_prepared_stage_graphs_v1.hh>
#include <Cellerator/compiler/ir/realization/implement_realization_ir_text_parser_printer_v1.hh>
#include <Cellerator/compiler/ir/realization/implement_symbolic_runtime_bindings_v1.hh>
#include <Cellerator/compiler/ir/realization/implement_target_and_capability_descriptions_v1.hh>

#include <cstdint>
#include <string>
#include <vector>

namespace cellerator::compiler::ir::realization::v1 {

enum class realization_validation_mode_v1 : std::uint8_t {
    verified = 1u,
    checked,
    unchecked,
};

enum class realization_validation_phase_v1 : std::uint8_t {
    structural = 1u,
    semantic,
    exact_coverage,
    resource_capability,
    host_referee,
};

enum class realization_validation_status_v1 : std::uint8_t {
    valid = 0u,
    structurally_invalid,
    semantically_invalid,
    inexact_coverage,
    resource_incompatible,
    referee_mismatch,
    unsafe_continuation,
};

// A deliberately slow, allocation-owning host reference for a selected
// weighted-row reduction. It is compiler validation evidence, never a runtime
// execution candidate.
struct weighted_row_referee_case_v1 {
    std::vector<std::uint64_t> row_offsets;
    std::vector<std::uint64_t> column_indices;
    std::vector<double> weights;
    std::vector<double> dense_input;
    std::vector<double> expected;
    double absolute_tolerance = 0.0;
    double relative_tolerance = 0.0;
};

struct realization_validation_request_v1 {
    realization_validation_mode_v1 mode = realization_validation_mode_v1::verified;
    bool allow_unsafe_continuation = false;
    std::string serialized_ir;
    const realization_module_v1* module = nullptr;
    const exact_cover_v1* exact_cover = nullptr;
    const target_capability_v1* capability = nullptr;
    const target_requirement_v1* requirement = nullptr;
    const std::vector<memory_requirement_v1>* memory_requirements = nullptr;
    const session_memory_accounting_v1* memory_accounting = nullptr;
    const symbolic_binding_table_v1* bindings = nullptr;
    const prepared_stage_graph_v1* stage_graph = nullptr;
    const launch_dependency_graph_v1* launch_graph = nullptr;
    const weighted_row_referee_case_v1* referee = nullptr;
};

struct realization_validation_receipt_v1 {
    realization_validation_status_v1 status = realization_validation_status_v1::valid;
    realization_validation_phase_v1 failed_phase = realization_validation_phase_v1::structural;
    std::uint32_t phases_run = 0u;
    std::uint32_t phases_skipped = 0u;
    bool unsafe_continuation_used = false;
    std::string detail;
};

[[nodiscard]] realization_validation_status_v1 run_weighted_row_referee_v1(
    const weighted_row_referee_case_v1& test_case,
    std::string* error = nullptr) noexcept;

[[nodiscard]] realization_validation_receipt_v1 validate_realization_ir_v1(
    const realization_validation_request_v1& request) noexcept;

} // namespace cellerator::compiler::ir::realization::v1
