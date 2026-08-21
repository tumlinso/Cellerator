#include <Cellerator/execution/execution_contract.hh>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace ce = cellerator::execution;

namespace {

void require(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "celleratorExecutionOrderContractTest: "
                  << message << '\n';
        std::exit(1);
    }
}

ce::axis_identity axis(
    ce::u32 domain, ce::u32 order, ce::u32 geometry, ce::u32 partition) {
    return ce::axis_identity{
        {domain, 1u}, {order, 1u}, {geometry, 1u}, {partition, 1u}};
}

ce::device_location location(ce::residency_kind kind, int device = -1) {
    return ce::device_location{kind, {}, device, 0u};
}

ce::relation_structure relation(
    ce::structure_epoch epoch,
    ce::axis_identity source,
    ce::axis_identity destination) {
    return ce::relation_structure{
        {100u, 1u}, epoch, source, destination, {200u, 1u}, 4u};
}

ce::value_plane values(
    ce::structure_epoch epoch,
    ce::value_generation generation,
    float *data) {
    return ce::value_plane{
        {100u, 1u}, epoch, data,
        location(ce::residency_kind::host),
        {ce::numeric_type::f32, ce::numeric_type::f32,
            ce::numeric_type::f32, 0u},
        {ce::quantization_kind::none, ce::numeric_type::invalid,
            ce::numeric_type::invalid, 0u, nullptr, nullptr, 0u},
        ce::value_layout_kind::logical_edge_order, {},
        generation, 4u, 4u * sizeof(float)};
}

ce::biological_operand_view dense_operand(
    float *data, ce::axis_identity value_axis) {
    ce::biological_operand_view result{};
    result.kind = ce::operand_kind::dense_tensor;
    result.storage.dense.data = data;
    result.storage.dense.location = location(ce::residency_kind::host);
    result.storage.dense.value_type = ce::numeric_type::f32;
    result.storage.dense.rank = 1u;
    result.storage.dense.axes[0] = value_axis;
    result.storage.dense.shape[0] = 4u;
    result.storage.dense.stride[0] = 1;
    return result;
}

void test_structure_value_lifetimes() {
    const ce::axis_identity source = axis(1u, 10u, 20u, 30u);
    const ce::axis_identity destination = axis(2u, 11u, 21u, 31u);
    const ce::relation_structure stable = relation({1u}, source, destination);
    std::array<float, 4> shared_storage{};
    const ce::value_plane generation_one = values(
        {1u}, {1u}, shared_storage.data());
    const ce::value_plane generation_two = values(
        {1u}, {2u}, shared_storage.data());

    require(ce::validate_relation_structure(stable)
            == ce::lifetime_validation_code::ok,
        "immutable relation structure failed validation");
    require(ce::validate_value_plane(stable, generation_one)
            == ce::lifetime_validation_code::ok,
        "first value generation failed validation");
    require(ce::validate_value_plane(stable, generation_two)
            == ce::lifetime_validation_code::ok,
        "value update invalidated structural preparation");
    require(generation_one.values == generation_two.values
            && generation_one.generation.value
                != generation_two.generation.value,
        "aliased pointers became value identity");

    ce::value_binding current{&generation_two, {2u}};
    require(ce::validate_value_binding(stable, current)
            == ce::lifetime_validation_code::ok,
        "current value generation failed binding");
    current.expected_generation = {1u};
    require(ce::validate_value_binding(stable, current)
            == ce::lifetime_validation_code::stale_value_generation,
        "stale value generation was accepted");

    const ce::relation_structure changed = relation({2u}, source, destination);
    require(ce::validate_value_plane(changed, generation_two)
            == ce::lifetime_validation_code::stale_structure_epoch,
        "dynamic topology change did not invalidate old values");
}

void test_persistent_execution_order() {
    const ce::axis_identity packed = axis(2u, 12u, 22u, 31u);
    const ce::axis_identity canonical = axis(2u, 11u, 21u, 31u);
    const ce::output_axis_contract preserve{
        packed, packed, ce::order_transition_kind::preserve,
        0u, 0u, 1u, 1u, {}, {0u, 0u}};
    require(ce::validate_output_axis_contract(preserve)
            == ce::order_validation_code::ok,
        "packed-order preservation contract failed");
    require(ce::compatible_without_transform(preserve, packed),
        "two compatible internal operations required canonical order");
    require(!ce::compatible_without_transform(preserve, canonical),
        "canonical consumer silently accepted packed order");

    const ce::output_axis_contract canonicalize{
        packed, canonical, ce::order_transition_kind::canonicalize,
        0u, 0u, 1u, 0u, {}, {300u, 1u}};
    require(ce::validate_output_axis_contract(canonicalize)
            == ce::order_validation_code::ok,
        "explicit canonicalization contract failed");
    ce::output_axis_contract hidden = canonicalize;
    hidden.transform = {0u, 0u};
    require(ce::validate_output_axis_contract(hidden)
            == ce::order_validation_code::missing_transform,
        "implicit canonicalization was accepted");

    const std::array<ce::u32, 4> packed_to_canonical{2u, 0u, 3u, 1u};
    const std::array<ce::u32, 4> canonical_to_packed{1u, 3u, 0u, 2u};
    const ce::order_transform_view transform{
        {300u, 1u}, packed, canonical,
        packed_to_canonical.data(), canonical_to_packed.data(),
        location(ce::residency_kind::host), 4u};
    require(ce::validate_order_transform(transform)
            == ce::order_validation_code::ok,
        "canonical recovery transform failed validation");
    const std::array<float, 4> packed_values{30.0f, 10.0f, 40.0f, 20.0f};
    std::array<float, 4> recovered{};
    for (std::size_t index = 0; index < packed_values.size(); ++index)
        recovered[packed_to_canonical[index]] = packed_values[index];
    require(recovered == std::array<float, 4>{10.0f, 20.0f, 30.0f, 40.0f},
        "explicit canonical recovery changed values");
}

void test_value_maps_and_launch_bindings() {
    const ce::axis_identity packed = axis(2u, 12u, 22u, 31u);
    const ce::relation_structure stable = relation({1u}, packed, packed);
    ce::relation_structure auxiliary = relation({2u}, packed, packed);
    auxiliary.identity = {101u, 1u};
    auxiliary.projections = {201u, 1u};
    const std::array<ce::u32, 4> logical_to_projection{2u, 0u, 3u, 1u};
    const std::array<ce::u32, 4> projection_to_logical{1u, 3u, 0u, 2u};
    ce::value_position_map_view map{
        stable.identity, stable.epoch, ce::value_map_direction::forward, {},
        logical_to_projection.data(), projection_to_logical.data(),
        location(ce::residency_kind::host), 4u};
    require(ce::validate_value_position_map(stable, map)
            == ce::order_validation_code::ok,
        "forward value-position map failed");
    map.direction = ce::value_map_direction::transpose;
    require(ce::validate_value_position_map(stable, map)
            == ce::order_validation_code::ok,
        "transpose value-position map failed");
    map.epoch = {2u};
    require(ce::validate_value_position_map(stable, map)
            == ce::order_validation_code::stale_structure_epoch,
        "stale transpose map was accepted");

    std::array<float, 4> input_values{}, output_values{}, weights{};
    const ce::value_plane plane = values({1u}, {4u}, weights.data());
    const ce::value_binding value_binding{&plane, {4u}};
    ce::biological_operand_view input = dense_operand(
        input_values.data(), packed);
    ce::biological_operand_view output = dense_operand(
        output_values.data(), packed);
    const ce::operand_axis_contract input_contract{
        ce::operand_kind::dense_tensor, 1u, {}, {packed}};
    const ce::operand_axis_contract output_contract{
        ce::operand_kind::dense_tensor, 1u, {}, {packed}};
    const ce::output_axis_contract output_order{
        packed, packed, ce::order_transition_kind::preserve,
        0u, 0u, 1u, 1u, {}, {0u, 0u}};
    const ce::output_effect_contract output_effect{
        ce::output_update_kind::overwrite, false, false, 0u,
        ce::invalid_scalar_binding_id, ce::invalid_scalar_binding_id};
    ce::prepared_binding_contract prepared{};
    prepared.structures[0] = {stable.identity, stable.epoch};
    prepared.structures[1] = {auxiliary.identity, auxiliary.epoch};
    prepared.inputs = &input_contract;
    prepared.outputs = &output_contract;
    prepared.output_orders = &output_order;
    prepared.output_effects = &output_effect;
    prepared.input_count = 1u;
    prepared.output_count = 1u;
    prepared.output_order_count = 1u;
    prepared.structure_count = 2u;
    prepared.output_effect_count = 1u;
    prepared.workspace = {64u, 64u, 0u};
    alignas(64) std::array<std::byte, 64> workspace{};
    const ce::relation_structure structures[2]{stable, auxiliary};
    ce::launch_bindings launch{};
    launch.structures = structures;
    launch.inputs = &input;
    launch.outputs = &output;
    launch.values = &value_binding;
    launch.input_count = 1u;
    launch.output_count = 1u;
    launch.value_count = 1u;
    launch.structure_count = 2u;
    launch.stream = {nullptr, 0, 0u};
    launch.workspace = {workspace.data(), workspace.size(),
        location(ce::residency_kind::device, 0)};
    require(ce::validate_launch_bindings(prepared, launch)
            == ce::binding_validation_code::ok,
        "valid per-launch bindings failed");

    ce::output_effect_contract invalid_overwrite = output_effect;
    invalid_overwrite.requires_initialized_destination = true;
    ce::prepared_binding_contract bad_effect = prepared;
    bad_effect.output_effects = &invalid_overwrite;
    require(ce::validate_launch_bindings(bad_effect, launch)
            == ce::binding_validation_code::invalid_output_effect,
        "overwrite requiring initialized destination was accepted");
    ce::output_effect_contract invalid_accumulate{
        ce::output_update_kind::accumulate, false, false, 0u,
        ce::invalid_scalar_binding_id, ce::invalid_scalar_binding_id};
    bad_effect.output_effects = &invalid_accumulate;
    require(ce::validate_launch_bindings(bad_effect, launch)
            == ce::binding_validation_code::invalid_output_effect,
        "accumulation without initialized destination was accepted");
    ce::output_effect_contract affine{
        ce::output_update_kind::affine_accumulate, true, false, 0u,
        11u, 12u};
    bad_effect.output_effects = &affine;
    require(ce::validate_launch_bindings(bad_effect, launch)
            == ce::binding_validation_code::missing_scalar_binding,
        "affine accumulation without required scalars was accepted");
    launch.scalars.count = 2u;
    launch.scalars.values[0] = {11u, ce::numeric_type::f32, {}, 0u};
    launch.scalars.values[1] = {12u, ce::numeric_type::f32, {}, 0u};
    require(ce::validate_launch_bindings(bad_effect, launch)
            == ce::binding_validation_code::ok,
        "valid affine scalar references were rejected");
    launch.scalars.count = 0u;
    ce::biological_operand_view aliased_output = input;
    launch.outputs = &aliased_output;
    require(ce::validate_launch_bindings(prepared, launch)
            == ce::binding_validation_code::illegal_operand_alias,
        "forbidden input/output alias was accepted");
    launch.outputs = &output;

    ce::value_plane wrong_relation_plane = plane;
    wrong_relation_plane.structure = {99u, 1u};
    ce::value_binding wrong_relation_value{&wrong_relation_plane, {4u}};
    launch.values = &wrong_relation_value;
    require(ce::validate_launch_bindings(prepared, launch)
            == ce::binding_validation_code::unknown_value_structure,
        "value plane referencing an unknown relation was accepted");
    ce::value_plane stale_generation_plane = plane;
    ce::value_binding stale_generation_value{&stale_generation_plane, {5u}};
    launch.values = &stale_generation_value;
    require(ce::validate_launch_bindings(prepared, launch)
            == ce::binding_validation_code::stale_value,
        "stale value generation was accepted");
    launch.values = &value_binding;

    ce::prepared_binding_contract missing_order = prepared;
    missing_order.output_order_count = 0u;
    require(ce::validate_launch_bindings(missing_order, launch)
            == ce::binding_validation_code::invalid_output_order,
        "output axis without explicit order behavior was accepted");

    ce::biological_operand_view stale_output = dense_operand(
        output_values.data(), axis(2u, 11u, 21u, 31u));
    launch.outputs = &stale_output;
    require(ce::validate_launch_bindings(prepared, launch)
            == ce::binding_validation_code::operand_axis_mismatch,
        "stale output order was accepted");
}

} // namespace

int main() {
    test_structure_value_lifetimes();
    test_persistent_execution_order();
    test_value_maps_and_launch_bindings();
    std::cout << "celleratorExecutionOrderContractTest passed"
              << " relation_bytes=" << sizeof(ce::relation_structure)
              << " value_plane_bytes=" << sizeof(ce::value_plane)
              << " launch_bindings_bytes=" << sizeof(ce::launch_bindings)
              << '\n';
    return 0;
}
