#include <Cellerator/compute/math/operation_core/operation_core.hh>

#include <cassert>
#include <cstdint>

namespace core = cellerator::compute::math::core;
namespace execution = cellerator::execution;

namespace {

std::uint32_t prepare_count = 0u;
std::uint32_t run_count = 0u;
const void *last_input = nullptr;
void *last_output = nullptr;
void *last_stream = nullptr;
void *last_workspace = nullptr;
std::uint64_t last_alpha = 0u;

execution::axis_identity axis(std::uint32_t base) {
    return {{base, 1u}, {base + 1u, 1u}, {base + 2u, 1u}, {base + 3u, 1u}};
}

execution::device_location device_location() {
    return {execution::residency_kind::device, {0u, 0u, 0u}, 0, 0u};
}

execution::dense_tensor_view dense(
    void *pointer, execution::axis_identity value_axis) {
    execution::dense_tensor_view view{};
    view.data = pointer;
    view.location = device_location();
    view.value_type = execution::numeric_type::f32;
    view.rank = 1u;
    view.axes[0] = value_axis;
    view.shape[0] = 4u;
    view.stride[0] = 1;
    return view;
}

bool supports_fp32(const core::numeric_policy &policy) noexcept {
    return policy.sparse_storage == execution::numeric_type::f32
        && policy.dense_storage == execution::numeric_type::f32
        && policy.output_storage == execution::numeric_type::f32
        && policy.multiply == execution::numeric_type::f32
        && policy.accumulation == execution::numeric_type::f32
        && policy.scalar == execution::numeric_type::f32;
}

core::operation_status fake_run(
    const core::prepared_operation &,
    const execution::launch_bindings &launch) noexcept {
    ++run_count;
    last_input = launch.inputs[0].storage.dense.data;
    last_output = launch.outputs[0].storage.dense.data;
    last_stream = launch.stream.stream;
    last_workspace = launch.workspace.data;
    last_alpha = launch.scalars.values[0].bits;
    return {};
}

core::operation_status fake_prepare(
    const core::operation_candidate &candidate,
    const core::operation_problem &problem,
    const core::structure_set_key &structures,
    const core::projection_key &projection,
    const core::numeric_policy &numeric,
    const core::prepare_policy &,
    core::prepared_operation *prepared) noexcept {
    static execution::operand_axis_contract inputs[1];
    static execution::operand_axis_contract outputs[1];
    static execution::output_axis_contract output_orders[1];
    static execution::output_effect_contract output_effects[1];
    ++prepare_count;
    const execution::axis_identity source_axis = axis(10u);
    const execution::axis_identity destination_axis = axis(20u);
    inputs[0] = {execution::operand_kind::dense_tensor, 1u, {}, {}};
    inputs[0].axes[0] = source_axis;
    outputs[0] = {execution::operand_kind::dense_tensor, 1u, {}, {}};
    outputs[0].axes[0] = destination_axis;
    output_orders[0] = {destination_axis,
        destination_axis,
        execution::order_transition_kind::preserve,
        0u,
        0u,
        0u,
        1u,
        {},
        {}};
    output_effects[0] = {execution::output_update_kind::overwrite,
        false, false, 0u, execution::invalid_scalar_binding_id,
        execution::invalid_scalar_binding_id};
    *prepared = {};
    prepared->problem = problem;
    prepared->structures = structures;
    prepared->projection = projection;
    prepared->numeric = numeric;
    prepared->kernel = candidate.identity;
    prepared->backend = candidate.backend;
    prepared->capability_flags = candidate.capability_flags;
    for (std::uint32_t index = 0u; index < structures.count; ++index)
        prepared->binding_contract.structures[index] = {
            structures.structures[index].runtime,
            structures.structures[index].epoch};
    prepared->binding_contract.inputs = inputs;
    prepared->binding_contract.outputs = outputs;
    prepared->binding_contract.output_orders = output_orders;
    prepared->binding_contract.output_effects = output_effects;
    prepared->binding_contract.input_count = 1u;
    prepared->binding_contract.output_count = 1u;
    prepared->binding_contract.output_order_count = 1u;
    prepared->binding_contract.structure_count = structures.count;
    prepared->binding_contract.output_effect_count = 1u;
    prepared->binding_contract.workspace = {64u, 16u, 0u};
    prepared->run = fake_run;
    return core::validate_prepared_operation(*prepared);
}

} // namespace

int main() {
    core::candidate_registry registry{};
    const core::operation_candidate native{{1u, 1u},
        "native-row-masked",
        core::operation_kind::weighted_relation_reduce,
        core::projection_kind::native_row_masked,
        core::backend_kind::native_direct,
        {},
        core::candidate_deterministic | core::candidate_graph_capture,
        32u,
        64u,
        supports_fp32,
        fake_prepare};
    const core::operation_candidate vendor{{2u, 2u},
        "vendor-csr",
        core::operation_kind::weighted_relation_reduce,
        core::projection_kind::csr,
        core::backend_kind::vendor_library,
        {},
        core::candidate_deterministic,
        48u,
        128u,
        supports_fp32,
        fake_prepare};
    assert(core::register_candidate(&registry, native));
    assert(core::register_candidate(&registry, vendor));
    assert(registry.size == 2u);
    assert(core::find_candidate(registry, native.identity) != nullptr);
    assert(core::find_candidate(registry, vendor.identity) != nullptr);
    assert(core::register_candidate(&registry, native).code
        == core::operation_status_code::duplicate_candidate);

    const core::operation_problem problem{core::operation_core_schema_version,
        core::operation_kind::weighted_relation_reduce,
        0u,
        {7u, 8u},
        1u,
        1u,
        4u};
    core::structure_set_key structures{};
    structures.count = 2u;
    structures.structures[0] = {{11u, 12u}, {31u, 1u}, {4u}};
    structures.structures[1] = {{21u, 22u}, {32u, 1u}, {7u}};
    const core::projection_key projection{{13u, 14u},
        {41u, 1u},
        core::projection_kind::native_row_masked,
        1u,
        3u};
    core::numeric_policy numeric{};
    numeric.sparse_storage = execution::numeric_type::f32;
    numeric.dense_storage = execution::numeric_type::f32;
    numeric.output_storage = execution::numeric_type::f32;
    numeric.multiply = execution::numeric_type::f32;
    numeric.accumulation = execution::numeric_type::f32;
    numeric.scalar = execution::numeric_type::f32;
    numeric.bias = execution::numeric_type::f32;
    const core::prepare_policy policy{true, true, true, true, 8u, 64u, 64u};

    core::prepared_operation prepared{};
    assert(core::prepare_candidate(
        native, problem, structures, projection, numeric, policy, &prepared));
    assert(prepare_count == 1u);

    execution::relation_structure relations[2]{};
    relations[0].identity = structures.structures[0].runtime;
    relations[0].epoch = structures.structures[0].epoch;
    relations[0].source_axis = axis(10u);
    relations[0].destination_axis = axis(15u);
    relations[0].projections = {51u, 1u};
    relations[0].logical_edge_count = 4u;
    relations[1].identity = structures.structures[1].runtime;
    relations[1].epoch = structures.structures[1].epoch;
    relations[1].source_axis = axis(15u);
    relations[1].destination_axis = axis(20u);
    relations[1].projections = {52u, 1u};
    relations[1].logical_edge_count = 4u;

    std::uint32_t input_a = 1u;
    std::uint32_t input_b = 2u;
    std::uint32_t output_a = 0u;
    std::uint32_t output_b = 0u;
    alignas(16) std::uint8_t workspace_a[64]{};
    alignas(16) std::uint8_t workspace_b[64]{};
    execution::biological_operand_view inputs[1]{};
    execution::biological_operand_view outputs[1]{};
    inputs[0].kind = execution::operand_kind::dense_tensor;
    inputs[0].storage.dense = dense(&input_a, axis(10u));
    outputs[0].kind = execution::operand_kind::dense_tensor;
    outputs[0].storage.dense = dense(&output_a, axis(20u));
    execution::launch_bindings launch{};
    launch.structures = relations;
    launch.inputs = inputs;
    launch.outputs = outputs;
    launch.input_count = 1u;
    launch.output_count = 1u;
    launch.structure_count = 2u;
    launch.scalars.count = 1u;
    launch.scalars.values[0] = {1u, execution::numeric_type::f32, {}, 100u};
    launch.stream = {reinterpret_cast<void *>(0x1000), 0, 0u};
    launch.workspace = {workspace_a, sizeof(workspace_a), device_location()};
    assert(core::run_prepared_operation(prepared, launch));
    assert(run_count == 1u && prepare_count == 1u);
    assert(last_input == &input_a && last_output == &output_a);
    assert(last_stream == reinterpret_cast<void *>(0x1000));
    assert(last_workspace == workspace_a && last_alpha == 100u);

    inputs[0].storage.dense.data = &input_b;
    outputs[0].storage.dense.data = &output_b;
    launch.scalars.values[0].bits = 200u;
    launch.stream.stream = reinterpret_cast<void *>(0x2000);
    launch.workspace.data = workspace_b;
    assert(core::run_prepared_operation(prepared, launch));
    assert(run_count == 2u && prepare_count == 1u);
    assert(last_input == &input_b && last_output == &output_b);
    assert(last_stream == reinterpret_cast<void *>(0x2000));
    assert(last_workspace == workspace_b && last_alpha == 200u);

    relations[0].epoch.value = 5u;
    assert(core::run_prepared_operation(prepared, launch).code
        == core::operation_status_code::stale_structure);
    relations[0].epoch = structures.structures[0].epoch;
    relations[1].epoch.value = 8u;
    assert(core::run_prepared_operation(prepared, launch).code
        == core::operation_status_code::stale_structure);
    relations[1].epoch = structures.structures[1].epoch;
    launch.structure_count = 1u;
    assert(core::run_prepared_operation(prepared, launch).binding
        == execution::binding_validation_code::structure_count_mismatch);
    launch.structure_count = 2u;
    const execution::relation_structure second_relation = relations[1];
    relations[1] = relations[0];
    assert(core::run_prepared_operation(prepared, launch).binding
        == execution::binding_validation_code::duplicate_structure);
    relations[1] = second_relation;
    launch.workspace.bytes = 63u;
    assert(core::run_prepared_operation(prepared, launch).binding
        == execution::binding_validation_code::insufficient_workspace);

    core::numeric_policy unsupported = numeric;
    unsupported.dense_storage = execution::numeric_type::f16;
    assert(core::prepare_candidate(
        native, problem, structures, projection, unsupported, policy, &prepared).code
        == core::operation_status_code::unsupported_numeric_policy);
    core::projection_key wrong_projection = projection;
    wrong_projection.kind = core::projection_kind::csr;
    assert(core::prepare_candidate(
        native, problem, structures, wrong_projection, numeric, policy, &prepared).code
        == core::operation_status_code::unsupported_projection);

    return 0;
}
