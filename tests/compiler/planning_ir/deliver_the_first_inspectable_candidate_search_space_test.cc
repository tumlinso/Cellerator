#include <Cellerator/compiler/ir/planning/planning_ir_v1.hh>

#include <array>
#include <cassert>

namespace planning = cellerator::compiler::ir::planning::v1;

planning::complete_cost_vector_v1 cost(double preparation, double execution,
                                       std::uint64_t persistent_bytes) {
    planning::complete_cost_vector_v1 result;
    result.nanoseconds[static_cast<std::size_t>(planning::cost_dimension_v1::preparation)] =
        preparation;
    result.nanoseconds[static_cast<std::size_t>(planning::cost_dimension_v1::execution)] =
        execution;
    result.persistent_bytes = persistent_bytes;
    result.transient_bytes = 4096u;
    result.reuse_count = 8u;
    result.total_nanoseconds = preparation + execution;
    return result;
}

int main() {
    static_assert(planning::planning_ir_contract_version_v1 == 1u);
    std::array<planning::semantic_operation_scope_v1, 1> operations{{
        {{11u, 12u}, {21u, 0u}, 0u, 0u}}};
    planning::planning_problem_v1 problem;
    problem.problem = {31u, 32u};
    problem.semantic_module = {33u, 34u};
    problem.semantic_fingerprint = {35u, 36u};
    problem.field = {21u, 0u};
    problem.profile_family = {41u, 42u};
    problem.operations = operations.data();
    problem.operation_count = operations.size();
    problem.target = planning::planning_target_class_v1::nvidia_gpu;

    planning::first_search_space_input_v1 input;
    input.problem = &problem;
    input.conventional = {{51u, 52u}, {61u, 62u}, cost(0.0, 120.0, 0u)};
    input.structure_dependent = {{53u, 54u}, {63u, 64u}, cost(15.0, 60.0, 8192u)};
    input.profile_evidence = {71u, 72u};
    input.profiled_support_count = 1024u;
    input.profile_confidence = 0.95;

    planning::first_search_space_status_v1 status{};
    auto search = planning::build_first_inspectable_candidate_search_space_v1(input, &status);
    assert(search.has_value());
    assert(status == planning::first_search_space_status_v1::success);
    assert(search->candidates.size() == 2u);
    assert(search->decisions[0].state == planning::decision_state_v1::dominated);
    assert(search->decisions[1].state == planning::decision_state_v1::selected);
    assert(search->explanations[0].reason == planning::removal_reason_v1::cost);
    assert(search->explanations[0].candidate.low == 51u);

    const auto dump = planning::compile_selected_plan_dump_v1(*search, &status);
    assert(dump.has_value());
    assert(dump->find("kind=structure-dependent") != std::string::npos);
    assert(dump->find("total_ns=75") != std::string::npos);
    assert(dump->find("profile=000000000000002a:0000000000000029") !=
           std::string::npos);

    // A profile with no observed support retains the structured alternative and
    // its full cost, rejects it explicitly, and selects the conventional fallback.
    input.profiled_support_count = 0u;
    search = planning::build_first_inspectable_candidate_search_space_v1(input, &status);
    assert(search.has_value());
    assert(search->decisions[0].state == planning::decision_state_v1::fallback);
    assert(search->decisions[1].state == planning::decision_state_v1::rejected);
    assert(search->explanations[0].reason == planning::removal_reason_v1::profile);
    const auto fallback_dump = planning::compile_selected_plan_dump_v1(*search, &status);
    assert(fallback_dump.has_value());
    assert(fallback_dump->find("kind=conventional-fallback") != std::string::npos);

    // Public Planning IR remains self-contained after a copy.
    const auto copy = *search;
    assert(copy.module.decisions == copy.decisions.data());
    assert(planning::validate_planning_ir_module_v1(copy.module) ==
           planning::planning_ir_status_v1::ok);
}
