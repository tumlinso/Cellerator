#include <Cellerator/compiler/ir/planning/implement_semantic_to_planning_lowering_v1.hh>

#include <array>
#include <cassert>

int main() {
    namespace planning = cellerator::compiler::ir::planning::v1;
    namespace semantic = Cellerator::compiler::ir::semantic;
    namespace profile = cellerator::compiler::profile::v1;
    namespace operation = cellerator::compute::operation::v2;
    using cellerator::execution::numeric_type;

    std::array<semantic::source_linked_semantic_operation_v1, 3> source{};
    source[0].identity = {11u, 1u};
    source[0].kind = operation::operation_kind::relation_apply;
    source[1].identity = {12u, 1u};
    source[1].kind = operation::operation_kind::segment_reduce;
    source[2].identity = {13u, 1u};
    source[2].kind = operation::operation_kind::sparse_axis_update;

    const semantic::numeric_tuple_ir_v1 numeric{
        numeric_type::f32, numeric_type::f32, numeric_type::f64, numeric_type::f32};
    std::array<semantic::semantic_canonical_record_v1, 3> canonical{};
    for (std::size_t index = 0; index != canonical.size(); ++index) {
        canonical[index].operation_identity = {source[index].identity.low, source[index].identity.high};
        canonical[index].operation_spelling = "typed.operation";
        canonical[index].numerical = numeric;
    }
    canonical[0].field_identity = 101u;
    canonical[1].field_identity = 202u;
    canonical[2].field_identity = 101u;

    std::array<semantic::semantic_lifetime_state_v1, 3> lifetimes{};
    for (std::size_t index = 0; index != lifetimes.size(); ++index) {
        lifetimes[index].object = canonical[index].operation_identity;
        lifetimes[index].structure_epoch = 7u;
        lifetimes[index].value_generation = 40u + index;
        lifetimes[index].support_generation = 8u;
        lifetimes[index].order_generation = 9u;
    }

    std::array<semantic::execution_field_region_ir_v1, 2> fields{};
    fields[0].identity = 101u;
    fields[0].constraints = {{"exact_numerics", "required", true},
                             {"layout_hint", "preserve-persistent", false}};
    fields[1].identity = 202u;
    fields[1].constraints = {{"deterministic", "required", true}};

    const profile::profile_state_identity_v1 state_identity{301u, 302u};
    std::array<profile::named_profile_state_v1, 1> states{};
    states[0].state = state_identity;
    states[0].name = {303u, 304u};
    states[0].evidence = {305u, 306u};
    profile::named_profile_environment_v1 environment;
    environment.identity = {401u, 402u};
    environment.default_state = state_identity;
    environment.states = states.data();
    environment.state_count = states.size();
    profile::profile_compile_state_v1 compile_state;
    compile_state.state = state_identity;
    compile_state.structure.structure_epoch = 7u;

    planning::semantic_planning_input_v1 input;
    input.semantic_module = {501u, 502u};
    input.semantic_fingerprint = {503u, 504u};
    input.source_operations = source.data();
    input.canonical_operations = canonical.data();
    input.lifetime_states = lifetimes.data();
    input.operation_count = source.size();
    input.fields = fields.data();
    input.field_count = fields.size();

    planning::semantic_to_planning_status_v1 status{};
    auto result = planning::lower_semantic_to_planning_v1(
        input, {&environment, &compile_state},
        {planning::planning_target_class_v1::nvidia_gpu,
         planning::planning_objective_latency_v1 | planning::planning_objective_memory_v1},
        &status);
    assert(result.has_value());
    assert(status == planning::semantic_to_planning_status_v1::success);
    assert(result->problems.size() == 2u);
    assert(result->operations.size() == 3u);
    assert(result->constraints.size() == 3u);

    // Field grouping is stable and produces valid contiguous Planning IR slices.
    assert(result->problems[0].field.low == 101u);
    assert(result->problems[0].operation_count == 2u);
    assert(result->problems[0].operations[0].operation.low == 11u);
    assert(result->problems[0].operations[1].operation.low == 13u);
    assert((result->problems[0].constraints &
            planning::planning_constraint_exact_numerics_v1) != 0u);
    assert(result->problems[1].field.low == 202u);
    assert((result->problems[1].constraints &
            planning::planning_constraint_deterministic_v1) != 0u);

    // Planner-facing requests preserve operation kind, numeric policy, and generations.
    const auto& emitted = result->operations[0];
    assert(emitted.kind == source[0].kind);
    assert(emitted.planner_request.kind == source[0].kind);
    assert(emitted.planner_request.numeric.accumulation == numeric_type::f64);
    assert(emitted.planner_request.expected_value_generation.value == 40u);
    assert(emitted.structure_epoch == 7u && emitted.support_generation == 8u &&
           emitted.order_generation == 9u);
    assert(result->constraints[1].name == "layout_hint");
    assert(!result->constraints[1].hard);

    // Copying a public writable result must rebind its pointer views.
    const auto copy = *result;
    assert(copy.problems[0].operations == copy.operation_scopes.data());

    canonical[0].field_identity = 999u;
    result = planning::lower_semantic_to_planning_v1(
        input, {&environment, &compile_state}, {}, &status);
    assert(!result.has_value());
    assert(status == planning::semantic_to_planning_status_v1::invalid_field);
}
