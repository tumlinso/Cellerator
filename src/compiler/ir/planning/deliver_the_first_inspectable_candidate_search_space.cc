#include <Cellerator/compiler/ir/planning/deliver_the_first_inspectable_candidate_search_space_v1.hh>

#include <cstdio>
#include <utility>

namespace cellerator::compiler::ir::planning::v1 {
namespace {

bool valid(planning_identity_v1 identity) noexcept {
    return identity.low != 0u || identity.high != 0u;
}

bool same(planning_identity_v1 left, planning_identity_v1 right) noexcept {
    return left.low == right.low && left.high == right.high;
}

planning_identity_v1 derive_identity(planning_identity_v1 seed,
                                     std::uint64_t tag) noexcept {
    planning_identity_v1 result{seed.low ^ tag, seed.high ^ (tag << 1u)};
    if (!valid(result)) result.low = tag == 0u ? 1u : tag;
    return result;
}

void set_status(first_search_space_status_v1* destination,
                first_search_space_status_v1 value) noexcept {
    if (destination != nullptr) *destination = value;
}

}  // namespace

void inspectable_candidate_search_space_v1::refresh_views() noexcept {
    module.decisions = decisions.empty() ? nullptr : decisions.data();
    module.decision_count = static_cast<std::uint32_t>(decisions.size());
}

inspectable_candidate_search_space_v1::inspectable_candidate_search_space_v1(
    const inspectable_candidate_search_space_v1& other)
    : problem(other.problem), profile_family(other.profile_family),
      profile_evidence(other.profile_evidence), candidates(other.candidates),
      decisions(other.decisions), explanations(other.explanations), module(other.module) {
    refresh_views();
}

inspectable_candidate_search_space_v1&
inspectable_candidate_search_space_v1::operator=(
    const inspectable_candidate_search_space_v1& other) {
    if (this != &other) {
        problem = other.problem;
        profile_family = other.profile_family;
        profile_evidence = other.profile_evidence;
        candidates = other.candidates;
        decisions = other.decisions;
        explanations = other.explanations;
        module = other.module;
        refresh_views();
    }
    return *this;
}

inspectable_candidate_search_space_v1::inspectable_candidate_search_space_v1(
    inspectable_candidate_search_space_v1&& other) noexcept
    : problem(other.problem), profile_family(other.profile_family),
      profile_evidence(other.profile_evidence), candidates(std::move(other.candidates)),
      decisions(std::move(other.decisions)), explanations(std::move(other.explanations)),
      module(other.module) {
    refresh_views();
}

inspectable_candidate_search_space_v1&
inspectable_candidate_search_space_v1::operator=(
    inspectable_candidate_search_space_v1&& other) noexcept {
    if (this != &other) {
        problem = other.problem;
        profile_family = other.profile_family;
        profile_evidence = other.profile_evidence;
        candidates = std::move(other.candidates);
        decisions = std::move(other.decisions);
        explanations = std::move(other.explanations);
        module = other.module;
        refresh_views();
    }
    return *this;
}

std::optional<inspectable_candidate_search_space_v1>
build_first_inspectable_candidate_search_space_v1(
    const first_search_space_input_v1& input,
    first_search_space_status_v1* status) noexcept {
    set_status(status, first_search_space_status_v1::invalid_argument);
    if (input.problem == nullptr) return std::nullopt;
    if (validate_planning_problem_v1(*input.problem) != planning_problem_status_v1::ok) {
        set_status(status, first_search_space_status_v1::invalid_problem);
        return std::nullopt;
    }
    if (!valid(input.problem->profile_family) || !valid(input.profile_evidence) ||
        !valid(input.conventional.candidate) || !valid(input.conventional.provider) ||
        !valid(input.structure_dependent.candidate) ||
        !valid(input.structure_dependent.provider)) {
        set_status(status, first_search_space_status_v1::invalid_identity);
        return std::nullopt;
    }
    if (same(input.conventional.candidate, input.structure_dependent.candidate)) {
        set_status(status, first_search_space_status_v1::duplicate_candidate);
        return std::nullopt;
    }
    if (input.profile_confidence < 0.0 || input.profile_confidence > 1.0) {
        set_status(status, first_search_space_status_v1::invalid_profile);
        return std::nullopt;
    }
    if (validate_complete_cost_vector_v1(input.conventional.cost) !=
            complete_cost_status_v1::ok ||
        validate_complete_cost_vector_v1(input.structure_dependent.cost) !=
            complete_cost_status_v1::ok) {
        set_status(status, first_search_space_status_v1::invalid_cost);
        return std::nullopt;
    }

    inspectable_candidate_search_space_v1 result;
    result.problem = input.problem->problem;
    result.profile_family = input.problem->profile_family;
    result.profile_evidence = input.profile_evidence;
    const bool structure_admissible =
        input.profiled_support_count != 0u && input.profile_confidence > 0.0;
    result.candidates = {
        {input.conventional.candidate, input.conventional.provider,
         first_search_candidate_kind_v1::conventional_fallback,
         input.conventional.cost, true},
        {input.structure_dependent.candidate, input.structure_dependent.provider,
         first_search_candidate_kind_v1::structure_dependent,
         input.structure_dependent.cost, structure_admissible},
    };

    const bool select_structure = structure_admissible &&
        input.structure_dependent.cost.total_nanoseconds <
            input.conventional.cost.total_nanoseconds;
    const auto source_operation = input.problem->operations[0].operation;
    decision_record_v1 conventional;
    conventional.decision = derive_identity(input.conventional.candidate, 0x10u);
    conventional.candidate = input.conventional.candidate;
    conventional.source_operation = source_operation;
    conventional.state = select_structure ? decision_state_v1::dominated
                                          : decision_state_v1::fallback;
    conventional.flags = decision_flag_correct_v1;
    decision_record_v1 structured;
    structured.decision = derive_identity(input.structure_dependent.candidate, 0x20u);
    structured.candidate = input.structure_dependent.candidate;
    structured.source_operation = source_operation;
    structured.state = !structure_admissible
        ? decision_state_v1::rejected
        : (select_structure ? decision_state_v1::selected : decision_state_v1::dominated);
    structured.flags = structure_admissible ? decision_flag_correct_v1 : decision_flag_none_v1;
    result.decisions = {conventional, structured};

    removal_explanation_v1 explanation;
    explanation.evidence = input.profile_evidence;
    if (!structure_admissible) {
        explanation.candidate = input.structure_dependent.candidate;
        explanation.related_candidate = input.conventional.candidate;
        explanation.reason = removal_reason_v1::profile;
        explanation.observed = static_cast<double>(input.profiled_support_count);
        explanation.limit = 1.0;
    } else if (select_structure) {
        explanation.candidate = input.conventional.candidate;
        explanation.related_candidate = input.structure_dependent.candidate;
        explanation.reason = removal_reason_v1::cost;
        explanation.observed = input.conventional.cost.total_nanoseconds;
        explanation.limit = input.structure_dependent.cost.total_nanoseconds;
    } else {
        explanation.candidate = input.structure_dependent.candidate;
        explanation.related_candidate = input.conventional.candidate;
        explanation.reason = removal_reason_v1::cost;
        explanation.observed = input.structure_dependent.cost.total_nanoseconds;
        explanation.limit = input.conventional.cost.total_nanoseconds;
    }
    result.explanations.push_back(explanation);
    result.module.module = derive_identity(input.problem->problem, 0x53454152434831ULL);
    result.refresh_views();
    if (validate_planning_ir_module_v1(result.module) != planning_ir_status_v1::ok) {
        set_status(status, first_search_space_status_v1::invalid_module);
        return std::nullopt;
    }
    set_status(status, first_search_space_status_v1::success);
    return result;
}

std::optional<std::string> compile_selected_plan_dump_v1(
    const inspectable_candidate_search_space_v1& search_space,
    first_search_space_status_v1* status) noexcept {
    if (validate_planning_ir_module_v1(search_space.module) != planning_ir_status_v1::ok ||
        search_space.candidates.size() != search_space.decisions.size()) {
        set_status(status, first_search_space_status_v1::invalid_module);
        return std::nullopt;
    }
    std::size_t selected = search_space.decisions.size();
    for (std::size_t index = 0u; index != search_space.decisions.size(); ++index) {
        const auto state = search_space.decisions[index].state;
        if (state == decision_state_v1::selected || state == decision_state_v1::forced ||
            state == decision_state_v1::externally_selected ||
            state == decision_state_v1::fallback) {
            if (selected != search_space.decisions.size()) {
                set_status(status, first_search_space_status_v1::invalid_module);
                return std::nullopt;
            }
            selected = index;
        }
    }
    if (selected == search_space.decisions.size()) {
        set_status(status, first_search_space_status_v1::no_selected_candidate);
        return std::nullopt;
    }
    const auto& candidate = search_space.candidates[selected];
    char dump[512]{};
    const int written = std::snprintf(
        dump, sizeof(dump),
        "selected-plan-v1\nproblem=%016llx:%016llx\nprofile=%016llx:%016llx\n"
        "candidate=%016llx:%016llx\nprovider=%016llx:%016llx\nkind=%s\n"
        "total_ns=%.17g\npersistent_bytes=%llu\ntransient_bytes=%llu\n",
        static_cast<unsigned long long>(search_space.problem.high),
        static_cast<unsigned long long>(search_space.problem.low),
        static_cast<unsigned long long>(search_space.profile_family.high),
        static_cast<unsigned long long>(search_space.profile_family.low),
        static_cast<unsigned long long>(candidate.candidate.high),
        static_cast<unsigned long long>(candidate.candidate.low),
        static_cast<unsigned long long>(candidate.provider.high),
        static_cast<unsigned long long>(candidate.provider.low),
        candidate.kind == first_search_candidate_kind_v1::conventional_fallback
            ? "conventional-fallback" : "structure-dependent",
        candidate.cost.total_nanoseconds,
        static_cast<unsigned long long>(candidate.cost.persistent_bytes),
        static_cast<unsigned long long>(candidate.cost.transient_bytes));
    if (written < 0 || static_cast<std::size_t>(written) >= sizeof(dump)) {
        set_status(status, first_search_space_status_v1::invalid_argument);
        return std::nullopt;
    }
    set_status(status, first_search_space_status_v1::success);
    return std::string(dump, static_cast<std::size_t>(written));
}

}  // namespace cellerator::compiler::ir::planning::v1
