#include <Cellerator/compiler/planning/deliver_source_to_selected_plan_vertical_slice_v1.hh>

#include <algorithm>
#include <cstdio>
#include <utility>

namespace Cellerator::compiler::planning {
namespace {

namespace planning_ir = cellerator::compiler::ir::planning::v1;
namespace profile = cellerator::compiler::profile::v1;

void set_status(source_to_selected_plan_status_v1* destination,
                source_to_selected_plan_status_v1 value) noexcept {
    if (destination != nullptr) *destination = value;
}

bool valid(planning_ir::planning_identity_v1 identity) noexcept {
    return identity.low != 0u || identity.high != 0u;
}

planning_ir::planning_identity_v1 identity(std::uint64_t low,
                                           std::uint64_t high) noexcept {
    planning_ir::planning_identity_v1 result{low, high};
    if (!valid(result)) result.low = 1u;
    return result;
}

planning_ir::planning_identity_v1 identity(
    cellerator::compute::operation::v2::stable_id value) noexcept {
    return identity(value.low, value.high);
}

planning_ir::planning_identity_v1 identity(profile::profile_state_identity_v1 value) noexcept {
    return identity(value.low, value.high);
}

planning_ir::planning_identity_v1 derive(
    planning_ir::planning_identity_v1 seed, std::uint64_t tag) noexcept {
    return identity(seed.low ^ tag, seed.high ^ (tag << 1u));
}

bool valid_profile(const profile::profile_compile_state_v1& value) noexcept {
    return value.contract_version == profile::profile_environment_contract_version_v1 &&
        (value.state.low != 0u || value.state.high != 0u) &&
        value.structure.structure_epoch != 0u &&
        value.structure.confidence >= 0.0 && value.structure.confidence <= 1.0;
}

}  // namespace

void source_to_selected_plan_result_v1::refresh_views() noexcept {
    problem.operations = operation_scopes.empty() ? nullptr : operation_scopes.data();
    problem.operation_count = static_cast<std::uint32_t>(operation_scopes.size());
    planning_module.decisions = decisions.empty() ? nullptr : decisions.data();
    planning_module.decision_count = static_cast<std::uint32_t>(decisions.size());
}

source_to_selected_plan_result_v1::source_to_selected_plan_result_v1(
    const source_to_selected_plan_result_v1& other)
    : semantic(other.semantic), source_receipt(other.source_receipt),
      operation_scopes(other.operation_scopes), problem(other.problem),
      decomposition(other.decomposition), candidates(other.candidates),
      decisions(other.decisions), planning_module(other.planning_module),
      selected_candidate(other.selected_candidate), portable_ruleset(other.portable_ruleset) {
    refresh_views();
}

source_to_selected_plan_result_v1& source_to_selected_plan_result_v1::operator=(
    const source_to_selected_plan_result_v1& other) {
    if (this != &other) {
        semantic = other.semantic;
        source_receipt = other.source_receipt;
        operation_scopes = other.operation_scopes;
        problem = other.problem;
        decomposition = other.decomposition;
        candidates = other.candidates;
        decisions = other.decisions;
        planning_module = other.planning_module;
        selected_candidate = other.selected_candidate;
        portable_ruleset = other.portable_ruleset;
        refresh_views();
    }
    return *this;
}

source_to_selected_plan_result_v1::source_to_selected_plan_result_v1(
    source_to_selected_plan_result_v1&& other) noexcept
    : semantic(std::move(other.semantic)), source_receipt(other.source_receipt),
      operation_scopes(std::move(other.operation_scopes)), problem(other.problem),
      decomposition(other.decomposition), candidates(std::move(other.candidates)),
      decisions(std::move(other.decisions)), planning_module(other.planning_module),
      selected_candidate(other.selected_candidate), portable_ruleset(std::move(other.portable_ruleset)) {
    refresh_views();
}

source_to_selected_plan_result_v1& source_to_selected_plan_result_v1::operator=(
    source_to_selected_plan_result_v1&& other) noexcept {
    if (this != &other) {
        semantic = std::move(other.semantic);
        source_receipt = other.source_receipt;
        operation_scopes = std::move(other.operation_scopes);
        problem = other.problem;
        decomposition = other.decomposition;
        candidates = std::move(other.candidates);
        decisions = std::move(other.decisions);
        planning_module = other.planning_module;
        selected_candidate = other.selected_candidate;
        portable_ruleset = std::move(other.portable_ruleset);
        refresh_views();
    }
    return *this;
}

std::optional<source_to_selected_plan_result_v1>
deliver_source_to_selected_plan_vertical_slice_v1(
    const source_to_selected_plan_request_v1& request,
    source_to_selected_plan_status_v1* status) noexcept {
    set_status(status, source_to_selected_plan_status_v1::invalid_source);
    const auto semantic =
        Cellerator::compiler::ir::semantic::lower_cell_source_to_semantic_ir_v1(request.source);
    if (!semantic) return std::nullopt;
    if (semantic->operations.size() != 2u) {
        set_status(status, source_to_selected_plan_status_v1::wrong_operation_count);
        return std::nullopt;
    }
    if (request.profile == nullptr || !valid_profile(*request.profile)) {
        set_status(status, source_to_selected_plan_status_v1::invalid_profile);
        return std::nullopt;
    }

    const auto conventional = normalize_complete_cost_v1(
        request.conventional_cost, request.required_cost_phases);
    const auto data_dependent = normalize_complete_cost_v1(
        request.data_dependent_cost, request.required_cost_phases);
    if (!conventional || !data_dependent) {
        set_status(status, source_to_selected_plan_status_v1::invalid_cost);
        return std::nullopt;
    }
    const auto portfolio = built_in_decomposition_portfolio_v1();
    if (validate_decomposition_portfolio_v1(portfolio) !=
            decomposition_portfolio_validation_code_v1::ok ||
        find_decomposition_provider_v1(portfolio, decomposition_provider_kind_v1::greedy) ==
            nullptr) {
        set_status(status, source_to_selected_plan_status_v1::unavailable_decomposition);
        return std::nullopt;
    }

    source_to_selected_plan_result_v1 result;
    result.semantic = *semantic;
    const auto receipt = Cellerator::compiler::ir::semantic::make_source_linked_receipt_v1(
        request.source, result.semantic);
    if (!receipt) return std::nullopt;
    result.source_receipt = *receipt;

    const auto module_identity = identity(receipt->semantic_hash, receipt->source_hash);
    const auto field_identity = derive(module_identity, 0x4649454c44ULL);
    for (std::uint32_t index = 0u; index != result.semantic.operations.size(); ++index) {
        result.operation_scopes.push_back(
            {identity(result.semantic.operations[index].identity), field_identity, index, 0u});
    }
    result.problem.problem = derive(module_identity, 0x504c414eULL);
    result.problem.semantic_module = module_identity;
    result.problem.semantic_fingerprint = identity(receipt->semantic_hash, receipt->semantic_hash);
    result.problem.field = field_identity;
    result.problem.profile_family = identity(request.profile->state);
    result.problem.scope = planning_ir::planning_scope_kind_v1::field;
    result.problem.target = planning_ir::planning_target_class_v1::portable_host;
    result.problem.constraints = planning_ir::planning_constraint_exact_numerics_v1;
    result.problem.objectives = planning_ir::planning_objective_latency_v1;

    const auto fallback_candidate = derive(result.problem.problem, 0x46414c4c4241434bULL);
    const auto structured_candidate = derive(result.problem.problem, 0x44415441504c414eULL);
    const bool data_admissible = request.profile->structure.support_count != 0u &&
        request.profile->structure.confidence > 0.0;
    result.candidates = {
        {fallback_candidate, identity(1u, 0u),
         vertical_slice_candidate_kind_v1::conventional_fallback,
         conventional, true, true},
        {structured_candidate, identity(2u, 0u),
         vertical_slice_candidate_kind_v1::data_dependent,
         data_dependent, true, data_admissible},
    };
    const bool select_data = data_admissible &&
        data_dependent.mean_nanoseconds < conventional.mean_nanoseconds;
    result.selected_candidate = select_data ? structured_candidate : fallback_candidate;
    result.decisions.resize(2u);
    for (std::size_t index = 0u; index != result.decisions.size(); ++index) {
        auto& decision = result.decisions[index];
        decision.decision = derive(result.candidates[index].candidate, 0xdec1deULL);
        decision.candidate = result.candidates[index].candidate;
        decision.source_operation = result.operation_scopes.front().operation;
        decision.flags = planning_ir::decision_flag_correct_v1;
    }
    result.decisions[0].state = select_data
        ? planning_ir::decision_state_v1::dominated
        : planning_ir::decision_state_v1::fallback;
    result.decisions[1].state = !data_admissible
        ? planning_ir::decision_state_v1::rejected
        : (select_data ? planning_ir::decision_state_v1::selected
                       : planning_ir::decision_state_v1::dominated);
    if (!data_admissible) result.decisions[1].flags = planning_ir::decision_flag_none_v1;
    result.planning_module.module = derive(result.problem.problem, 0x52554c45534554ULL);
    result.refresh_views();

    char ruleset[512]{};
    const auto& selected = select_data ? result.candidates[1] : result.candidates[0];
    const int written = std::snprintf(
        ruleset, sizeof(ruleset),
        "portable-ruleset-v1\nfield=%s\nprofile=%s\noperations=%u\n"
        "decomposition=greedy\nselected=%s\ncomplete_cost_ns=%.17g\n"
        "fallback_present=true\ndata_dependent_present=true\n",
        result.semantic.field.c_str(), result.semantic.profile.c_str(),
        static_cast<unsigned>(result.operation_scopes.size()),
        select_data ? "data-dependent" : "conventional-fallback",
        selected.complete_cost.mean_nanoseconds);
    if (written < 0 || static_cast<std::size_t>(written) >= sizeof(ruleset)) {
        set_status(status, source_to_selected_plan_status_v1::invalid_planning_ir);
        return std::nullopt;
    }
    result.portable_ruleset.assign(ruleset, static_cast<std::size_t>(written));
    set_status(status, source_to_selected_plan_status_v1::success);
    return result;
}

}  // namespace Cellerator::compiler::planning
