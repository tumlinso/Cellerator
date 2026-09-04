#pragma once

#include <Cellerator/compiler/ir/planning/freeze_planning_ir_module_and_decision_state_model_v1.hh>
#include <Cellerator/compiler/ir/realization/implement_prepared_stage_graphs_v1.hh>
#include <Cellerator/compiler/ir/realization/implement_realization_ir_text_parser_printer_v1.hh>
#include <Cellerator/compiler/ir/realization/implement_symbolic_runtime_bindings_v1.hh>
#include <Cellerator/execution/program/program_v2.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace cellerator::compiler::ir::realization::v1 {

enum class selected_plan_delivery_status_v1 : std::uint8_t {
    success = 0u,
    invalid_planning_module,
    missing_selected_candidate,
    ambiguous_selected_candidate,
    invalid_relation_geometry,
    invalid_realization,
    invalid_runtime_binding,
    execution_failed,
};

struct selected_relation_plan_v1 {
    const cellerator::compiler::ir::planning::v1::planning_ir_module_v1* planning = nullptr;
    stable_identity_v1 realization_module{};
    stable_identity_v1 target{};
    stable_identity_v1 input_order{};
    stable_identity_v1 output_order{};
    std::uint64_t structure_epoch = 0u;
    std::uint64_t value_generation = 0u;
    std::vector<std::uint64_t> row_offsets;
    std::vector<std::uint64_t> column_indices;
};

struct selected_plan_trace_v1 {
    stable_identity_v1 source_operation{};
    stable_identity_v1 selected_candidate{};
    stable_identity_v1 prepared_stage{};
    stable_identity_v1 output{};
};

// Owns only immutable geometry and compiler metadata. Runtime input, value,
// output, workspace, and stream addresses are supplied to each execution.
struct prepared_relation_slice_v1 {
    selected_plan_trace_v1 trace{};
    realization_module_v1 module{};
    prepared_stage_graph_v1 stage_graph{};
    symbolic_binding_table_v1 binding_table{};
    realization_text_document_v1 text_ir{};
    std::string serialized_ir;
    std::vector<std::uint64_t> row_offsets;
    std::vector<std::uint64_t> column_indices;
    std::uint64_t input_element_count = 0u;
};

[[nodiscard]] std::optional<prepared_relation_slice_v1>
lower_selected_relation_plan_v1(
    const selected_relation_plan_v1& plan,
    selected_plan_delivery_status_v1* status = nullptr,
    std::string* error = nullptr);

[[nodiscard]] selected_plan_delivery_status_v1 execute_prepared_relation_slice_v1(
    const prepared_relation_slice_v1& slice,
    const double* input,
    std::uint64_t input_count,
    const double* values,
    std::uint64_t value_count,
    double* output,
    std::uint64_t output_count,
    void* caller_stream = nullptr) noexcept;

} // namespace cellerator::compiler::ir::realization::v1
