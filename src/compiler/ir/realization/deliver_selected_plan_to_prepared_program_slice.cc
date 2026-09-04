#include <Cellerator/compiler/ir/realization/deliver_selected_plan_to_prepared_program_slice_v1.hh>

#include <algorithm>
#include <limits>

namespace cellerator::compiler::ir::realization::v1 {
namespace {

namespace planning = cellerator::compiler::ir::planning::v1;
namespace program = cellerator::execution::program;

stable_identity_v1 convert(planning::planning_identity_v1 identity) noexcept {
    return {identity.high, identity.low};
}

stable_identity_v1 derived(stable_identity_v1 base, std::uint64_t tag) noexcept {
    base.high ^= 0x43454c4c45524154ull;
    base.low ^= tag;
    if (!valid(base)) base.low = tag == 0u ? 1u : tag;
    return base;
}

bool selected(planning::decision_state_v1 state) noexcept {
    return state == planning::decision_state_v1::selected ||
        state == planning::decision_state_v1::forced ||
        state == planning::decision_state_v1::externally_selected ||
        state == planning::decision_state_v1::fallback;
}

void set_status(selected_plan_delivery_status_v1 value,
    selected_plan_delivery_status_v1* status, std::string* error,
    const char* message) {
    if (status != nullptr) *status = value;
    if (error != nullptr) *error = message;
}

program::program_status launch_reference_relation(
    const void* prepared_state,
    const program::launch_binding_v2& binding,
    void*) noexcept {
    if (prepared_state == nullptr || binding.input == nullptr ||
        binding.values == nullptr || binding.output == nullptr) {
        return program::program_status::invalid_argument;
    }
    const auto& slice = *static_cast<const prepared_relation_slice_v1*>(prepared_state);
    const auto* input = static_cast<const double*>(binding.input);
    const auto* values = static_cast<const double*>(binding.values);
    auto* output = static_cast<double*>(binding.output);
    for (std::size_t row = 0; row + 1u < slice.row_offsets.size(); ++row) {
        double sum = 0.0;
        for (std::uint64_t edge = slice.row_offsets[row];
             edge < slice.row_offsets[row + 1u]; ++edge) {
            sum += values[edge] * input[slice.column_indices[edge]];
        }
        output[row] = sum;
    }
    return program::program_status::success;
}

} // namespace

std::optional<prepared_relation_slice_v1> lower_selected_relation_plan_v1(
    const selected_relation_plan_v1& plan,
    selected_plan_delivery_status_v1* status,
    std::string* error) {
    if (plan.planning == nullptr ||
        planning::validate_planning_ir_module_v1(*plan.planning) !=
            planning::planning_ir_status_v1::ok) {
        set_status(selected_plan_delivery_status_v1::invalid_planning_module,
            status, error, "valid Planning IR module required");
        return std::nullopt;
    }
    const planning::decision_record_v1* choice = nullptr;
    for (std::uint32_t index = 0; index < plan.planning->decision_count; ++index) {
        const auto& decision = plan.planning->decisions[index];
        if (!selected(decision.state)) continue;
        if (choice != nullptr) {
            set_status(selected_plan_delivery_status_v1::ambiguous_selected_candidate,
                status, error, "relation plan has more than one selected candidate");
            return std::nullopt;
        }
        choice = &decision;
    }
    if (choice == nullptr) {
        set_status(selected_plan_delivery_status_v1::missing_selected_candidate,
            status, error, "relation plan has no selected candidate");
        return std::nullopt;
    }
    if (choice->candidate.low == 0u || !valid(plan.realization_module) ||
        !valid(plan.target) || !valid(plan.input_order) || !valid(plan.output_order) ||
        plan.structure_epoch == 0u || plan.value_generation == 0u ||
        plan.row_offsets.empty() || plan.row_offsets.front() != 0u ||
        plan.row_offsets.back() != plan.column_indices.size()) {
        set_status(selected_plan_delivery_status_v1::invalid_relation_geometry,
            status, error, "identities, generations, and CSR-like relation geometry required");
        return std::nullopt;
    }
    std::uint64_t input_count = 0u;
    for (std::size_t row = 0; row + 1u < plan.row_offsets.size(); ++row) {
        if (plan.row_offsets[row] > plan.row_offsets[row + 1u]) {
            set_status(selected_plan_delivery_status_v1::invalid_relation_geometry,
                status, error, "relation row offsets are not monotonic");
            return std::nullopt;
        }
    }
    for (const auto column : plan.column_indices) {
        if (column == std::numeric_limits<std::uint64_t>::max()) {
            set_status(selected_plan_delivery_status_v1::invalid_relation_geometry,
                status, error, "relation column cannot be represented as a count");
            return std::nullopt;
        }
        input_count = std::max(input_count, column + 1u);
    }

    prepared_relation_slice_v1 result;
    result.trace.source_operation = convert(choice->source_operation);
    result.trace.selected_candidate = convert(choice->candidate);
    result.trace.prepared_stage = derived(plan.realization_module, 0x5354414745ull);
    result.trace.output = derived(plan.realization_module, 0x4f5554505554ull);
    result.row_offsets = plan.row_offsets;
    result.column_indices = plan.column_indices;
    result.input_element_count = input_count;

    const auto binding_base = derived(plan.realization_module, 0u);
    // program_v2 names bindings by a dense 32-bit index. Retain a full stable
    // identity in the high half while making the compatibility index explicit.
    const stable_identity_v1 binding_identity{binding_base.high, 0u};
    const auto input_slot = derived(binding_identity, 1u);
    const auto output_slot = derived(binding_identity, 2u);
    const auto values_slot = derived(binding_identity, 3u);
    result.module = {realization_ir_contract_version_v1, plan.realization_module,
        "selected-relation-plan", {{plan.target, "host-reference", "planning-v1"}},
        {{result.trace.prepared_stage, plan.target, realization_object_kind_v1::stage,
            "selected-relation-stage",
            {result.trace.source_operation, result.trace.source_operation,
                convert(plan.planning->module)}}}};
    result.binding_table = {binding_identity,
        {{input_slot, binding_slot_kind_v1::input, {}, input_count * sizeof(double),
             alignof(double)},
         {output_slot, binding_slot_kind_v1::output, {},
             (plan.row_offsets.size() - 1u) * sizeof(double), alignof(double)},
         {values_slot, binding_slot_kind_v1::values, {},
             plan.column_indices.size() * sizeof(double), alignof(double)}}};
    result.stage_graph = {derived(plan.realization_module, 0x4752415048ull),
        {{result.trace.prepared_stage, result.trace.selected_candidate, binding_identity,
            prepared_stage_kind_v1::host_stub, {}, {},
            {plan.input_order, order_class_v1::persistent_physical},
            {plan.output_order, order_class_v1::persistent_physical},
            plan.structure_epoch, plan.value_generation, plan.value_generation,
            {0u, plan.row_offsets.size() - 1u, 0u, plan.row_offsets.size() - 1u},
            0u, 0u}}};
    result.text_ir = {plan.realization_module,
        {{"source-operation", result.trace.source_operation, "planning-ir-v1"},
         {"selected-candidate", result.trace.selected_candidate, "exactly-one"},
         {"stage", result.trace.prepared_stage, "host-reference-relation"},
         {"binding", binding_identity, "runtime-rebindable"},
         {"output", result.trace.output, "persistent-physical-order"}}};
    result.serialized_ir = print_realization_text_v1(result.text_ir);

    std::string validation_error;
    if (validate_realization_module_v1(result.module, &validation_error) !=
            realization_module_status_v1::valid ||
        validate_symbolic_binding_table_v1(result.binding_table) !=
            symbolic_binding_status_v1::valid ||
        validate_prepared_stage_graph_v1(result.stage_graph, &validation_error) !=
            stage_graph_status_v1::valid ||
        !parse_realization_text_v1(result.serialized_ir, &validation_error)) {
        set_status(selected_plan_delivery_status_v1::invalid_realization,
            status, error, validation_error.c_str());
        return std::nullopt;
    }
    if (status != nullptr) *status = selected_plan_delivery_status_v1::success;
    if (error != nullptr) error->clear();
    return result;
}

selected_plan_delivery_status_v1 execute_prepared_relation_slice_v1(
    const prepared_relation_slice_v1& slice,
    const double* input,
    std::uint64_t input_count,
    const double* values,
    std::uint64_t value_count,
    double* output,
    std::uint64_t output_count,
    void* caller_stream) noexcept {
    if (input == nullptr || values == nullptr || output == nullptr ||
        input_count < slice.input_element_count ||
        value_count < slice.column_indices.size() ||
        output_count + 1u < slice.row_offsets.size() ||
        slice.binding_table.slots.size() != 3u || slice.stage_graph.stages.size() != 1u) {
        return selected_plan_delivery_status_v1::invalid_runtime_binding;
    }
    const std::vector<live_runtime_binding_v1> live{
        {slice.binding_table.slots[0].identity, const_cast<double*>(input), nullptr, 0u},
        {slice.binding_table.slots[1].identity, output, nullptr, 0u},
        {slice.binding_table.slots[2].identity, const_cast<double*>(values), nullptr, 0u}};
    if (bind_symbolic_runtime_v1(slice.binding_table, live) !=
        symbolic_binding_status_v1::valid) {
        return selected_plan_delivery_status_v1::invalid_runtime_binding;
    }
    const auto& ir_stage = slice.stage_graph.stages.front();
    const program::prepared_stage_v2 stage{ir_stage.identity.low,
        ir_stage.candidate.low, &slice, launch_reference_relation, 0u, 0u, 0u, 0u};
    const program::prepared_program_v2 prepared{2u, 0u, &stage, 1u, nullptr, 0u};
    std::string compare_error;
    if (compare_program_v2_graph_v1(slice.stage_graph, prepared, &compare_error) !=
        stage_graph_status_v1::valid) {
        return selected_plan_delivery_status_v1::invalid_realization;
    }
    const program::launch_binding_v2 launch{input, output, values, nullptr, 0u};
    return program::execute_prepared_program_v2(
        prepared, &launch, 1u, caller_stream) == program::program_status::success
        ? selected_plan_delivery_status_v1::success
        : selected_plan_delivery_status_v1::execution_failed;
}

} // namespace cellerator::compiler::ir::realization::v1
