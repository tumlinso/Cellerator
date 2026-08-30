#include <Cellerator/geometry/admissibility.hh>
#include <Cellerator/geometry/compiler/strategy_registry.hh>

namespace cellerator::geometry::compiler {
namespace {

bool valid_operation_kind(geometry_operation_kind kind) noexcept {
    return kind == geometry_operation_kind::relation_apply
        || kind == geometry_operation_kind::relation_apply_transpose
        || kind == geometry_operation_kind::contract_on_support
        || kind == geometry_operation_kind::segment_operation
        || kind == geometry_operation_kind::relation_bundle_apply;
}

bool valid_support_evidence(
    const portable_support_evidence_v1 &evidence) noexcept {
    const bool absent = evidence.evidence_identity == 0u
        && evidence.schema_version == 0u && evidence.evidence_kind == 0u
        && evidence.data == nullptr && evidence.data_bytes == 0u;
    const bool present = evidence.evidence_identity != 0u
        && evidence.schema_version != 0u && evidence.evidence_kind != 0u
        && evidence.data != nullptr && evidence.data_bytes != 0u;
    return absent || present;
}

bool valid_problem(const geometry_problem_v1 &problem) noexcept {
    if (problem.schema_version != geometry_problem_schema_version
        || problem.reserved != 0u
        || execution::validate_sparse_relation(problem.primary_relation)
            != execution::biological_validation_code::ok
        || problem.primary_relation.epoch.value == 0u
        || !validate_work_window(problem.work_window))
        return false;
    if (problem.work_window.kind == work_window_kind::relation_rows
        && !execution::same_axis_identity(problem.work_window.axis,
            problem.primary_relation.destination_axis))
        return false;
    if (problem.workload.schema_version
            != geometry_workload_profile_schema_version
        || problem.workload.reserved != 0u
        || problem.workload.reserved2 != 0u
        || !valid_operation_kind(problem.workload.operation)
        || problem.workload.relation_value_type
            == execution::numeric_type::invalid
        || problem.workload.dense_input_type
            == execution::numeric_type::invalid
        || problem.workload.accumulation_type
            == execution::numeric_type::invalid
        || problem.workload.output_type == execution::numeric_type::invalid
        || problem.workload.dense_width == 0u
        || problem.workload.expected_reuse == 0u)
        return false;
    if (problem.admissibility != nullptr
        && !validate_admissibility(
            problem.work_window, *problem.admissibility))
        return false;
    return valid_support_evidence(problem.support_evidence);
}

bool cover_matches_problem(
    const geometry_problem_v1 &problem,
    const relation_cover_view_v1 &cover) noexcept {
    return execution::same_handle(
               cover.structure, problem.primary_relation.structure)
        && cover.structure_epoch.value == problem.primary_relation.epoch.value
        && execution::same_axis_identity(
            cover.source_axis, problem.primary_relation.source_axis)
        && execution::same_axis_identity(
            cover.destination_axis,
            problem.primary_relation.destination_axis)
        && cover.logical_edge_count
            == problem.primary_relation.logical_edge_count;
}

} // namespace

geometry_strategy_status compile_geometry(
    geometry_strategy_registry_v1 registry,
    const geometry_problem_v1 &problem,
    const geometry_search_policy_v1 &policy,
    geometry_strategy_workspace_v1 strategy_workspace,
    geometry_solution_buffers_v1 buffers,
    relation_cover_validation_workspace validation_workspace,
    geometry_solution_v1 *solution) noexcept {
    if (solution == nullptr || !valid_problem(problem)
        || policy.schema_version != geometry_search_policy_schema_version
        || policy.reserved[0] != 0u || policy.reserved[1] != 0u
        || policy.reserved[2] != 0u
        || geometry_search_tier_bit(policy.tier) == 0u
        || policy.strategy_id == 0u)
        return geometry_strategy_status::invalid_argument;

    geometry_solution_v1 candidate{};
    const geometry_strategy_status strategy_status = run_geometry_strategy(
        registry, problem, policy, strategy_workspace, buffers, &candidate);
    if (strategy_status != geometry_strategy_status::ok)
        return strategy_status;
    if (candidate.strategy_id != policy.strategy_id
        || candidate.tier != policy.tier
        || candidate.reserved[0] != 0u || candidate.reserved[1] != 0u
        || candidate.reserved[2] != 0u || candidate.reserved[3] != 0u
        || candidate.reserved[4] != 0u || candidate.reserved[5] != 0u
        || candidate.reserved[6] != 0u
        || !validate_work_layout(problem.work_window, candidate.work_layout)
        || !cover_matches_problem(problem, candidate.relation_cover)
        || !validate_relation_cover(
            candidate.relation_cover, validation_workspace))
        return geometry_strategy_status::strategy_failed;

    *solution = candidate;
    return geometry_strategy_status::ok;
}

} // namespace cellerator::geometry::compiler
