#include <Cellerator/compiler/ir/planning/implement_semantic_to_planning_lowering_v1.hh>

#include <algorithm>
#include <utility>

namespace cellerator::compiler::ir::planning::v1 {
namespace {

using Cellerator::compiler::ir::semantic::execution_field_constraint_ir_v1;
using Cellerator::compiler::ir::semantic::execution_field_region_ir_v1;
using Cellerator::compiler::ir::semantic::semantic_identity_v1;
using cellerator::compiler::profile::v1::profile_state_identity_v1;

bool valid(planning_identity_v1 identity) noexcept {
    return identity.low != 0u || identity.high != 0u;
}

planning_identity_v1 planning_identity(profile_state_identity_v1 identity) noexcept {
    return {identity.low, identity.high};
}

planning_identity_v1 operation_identity(
    cellerator::compute::operation::v2::stable_id identity) noexcept {
    return {identity.low, identity.high};
}

bool same_identity(semantic_identity_v1 semantic,
                   cellerator::compute::operation::v2::stable_id source) noexcept {
    return semantic.low == source.low && semantic.high == source.high;
}

const execution_field_region_ir_v1* find_field(
    const semantic_planning_input_v1& input, std::uint64_t identity) noexcept {
    for (std::uint32_t index = 0u; index != input.field_count; ++index) {
        if (input.fields[index].identity == identity) return input.fields + index;
    }
    return nullptr;
}

bool known_operation_kind(
    cellerator::compute::operation::v2::operation_kind kind) noexcept {
    return cellerator::compute::operation::v2::valid_operation_kind(kind);
}

bool valid_profile_environment(
    const cellerator::compiler::profile::v1::named_profile_environment_v1& environment) noexcept {
    if (environment.schema_version !=
            cellerator::compiler::profile::v1::named_profile_environment_schema_version_v1 ||
        environment.reserved != 0u || !valid(planning_identity(environment.identity)) ||
        environment.state_count == 0u || environment.states == nullptr) {
        return false;
    }
    return true;
}

bool same(profile_state_identity_v1 left, profile_state_identity_v1 right) noexcept {
    return left.low == right.low && left.high == right.high;
}

const cellerator::compiler::profile::v1::named_profile_state_v1* find_profile_state(
    const cellerator::compiler::profile::v1::named_profile_environment_v1& environment,
    profile_state_identity_v1 state) noexcept {
    for (std::uint32_t index = 0u; index != environment.state_count; ++index) {
        if (same(environment.states[index].state, state)) return environment.states + index;
    }
    return nullptr;
}

std::uint32_t constraint_flag(const execution_field_constraint_ir_v1& constraint) noexcept {
    if (!constraint.hard) return planning_constraint_none_v1;
    if (constraint.name == "exact_numerics") return planning_constraint_exact_numerics_v1;
    if (constraint.name == "deterministic") return planning_constraint_deterministic_v1;
    if (constraint.name == "memory_bounded") return planning_constraint_memory_bounded_v1;
    if (constraint.name == "graph_capture") return planning_constraint_graph_capture_v1;
    return planning_constraint_none_v1;
}

planning_identity_v1 make_problem_identity(planning_identity_v1 module,
                                           planning_identity_v1 field) noexcept {
    planning_identity_v1 result{
        module.low ^ field.high ^ 0x706c616e6e696e67ULL,
        module.high ^ field.low ^ 0x70726f626c656d31ULL};
    if (!valid(result)) result.low = 1u;
    return result;
}

void set_status(semantic_to_planning_status_v1* destination,
                semantic_to_planning_status_v1 value) noexcept {
    if (destination != nullptr) *destination = value;
}

}  // namespace

void semantic_to_planning_result_v1::refresh_views() noexcept {
    auto* scopes = operation_scopes.empty() ? nullptr : operation_scopes.data();
    std::uint32_t offset = 0u;
    for (auto& problem : problems) {
        problem.operations = scopes == nullptr ? nullptr : scopes + offset;
        problem.first_operation = 0u;
        offset += problem.operation_count;
    }
}

semantic_to_planning_result_v1::semantic_to_planning_result_v1(
    const semantic_to_planning_result_v1& other)
    : profile_environment(other.profile_environment), profile_state(other.profile_state),
      operation_scopes(other.operation_scopes), operations(other.operations),
      constraints(other.constraints), problems(other.problems) {
    refresh_views();
}

semantic_to_planning_result_v1& semantic_to_planning_result_v1::operator=(
    const semantic_to_planning_result_v1& other) {
    if (this != &other) {
        profile_environment = other.profile_environment;
        profile_state = other.profile_state;
        operation_scopes = other.operation_scopes;
        operations = other.operations;
        constraints = other.constraints;
        problems = other.problems;
        refresh_views();
    }
    return *this;
}

semantic_to_planning_result_v1::semantic_to_planning_result_v1(
    semantic_to_planning_result_v1&& other) noexcept
    : profile_environment(other.profile_environment), profile_state(other.profile_state),
      operation_scopes(std::move(other.operation_scopes)),
      operations(std::move(other.operations)), constraints(std::move(other.constraints)),
      problems(std::move(other.problems)) {
    refresh_views();
}

semantic_to_planning_result_v1& semantic_to_planning_result_v1::operator=(
    semantic_to_planning_result_v1&& other) noexcept {
    if (this != &other) {
        profile_environment = other.profile_environment;
        profile_state = other.profile_state;
        operation_scopes = std::move(other.operation_scopes);
        operations = std::move(other.operations);
        constraints = std::move(other.constraints);
        problems = std::move(other.problems);
        refresh_views();
    }
    return *this;
}

std::optional<semantic_to_planning_result_v1> lower_semantic_to_planning_v1(
    const semantic_planning_input_v1& semantic,
    const semantic_planning_profile_v1& profile,
    semantic_to_planning_options_v1 options,
    semantic_to_planning_status_v1* status) noexcept {
    set_status(status, semantic_to_planning_status_v1::invalid_argument);
    if (semantic.operation_count == 0u || semantic.source_operations == nullptr ||
        semantic.canonical_operations == nullptr || semantic.lifetime_states == nullptr ||
        semantic.field_count == 0u || semantic.fields == nullptr ||
        profile.environment == nullptr || profile.state == nullptr) {
        return std::nullopt;
    }
    if (!valid(semantic.semantic_module) || !valid(semantic.semantic_fingerprint)) {
        set_status(status, semantic_to_planning_status_v1::invalid_module);
        return std::nullopt;
    }
    if (!valid_profile_environment(*profile.environment)) {
        set_status(status, semantic_to_planning_status_v1::invalid_profile_environment);
        return std::nullopt;
    }
    const auto* selected = find_profile_state(*profile.environment, profile.state->state);
    if (selected == nullptr) {
        set_status(status, semantic_to_planning_status_v1::profile_state_not_found);
        return std::nullopt;
    }
    if (profile.state->contract_version !=
            cellerator::compiler::profile::v1::profile_environment_contract_version_v1 ||
        profile.state->structure.structure_epoch == 0u ||
        !same(selected->state, profile.state->state)) {
        set_status(status, semantic_to_planning_status_v1::profile_state_mismatch);
        return std::nullopt;
    }

    semantic_to_planning_result_v1 result;
    result.profile_environment = planning_identity(profile.environment->identity);
    result.profile_state = planning_identity(profile.state->state);

    // Fields define planning boundaries. Keep first-appearance order stable and
    // make each problem's operation slice contiguous.
    std::vector<std::uint64_t> field_order;
    for (std::uint32_t index = 0u; index != semantic.operation_count; ++index) {
        const auto& source = semantic.source_operations[index];
        const auto& canonical = semantic.canonical_operations[index];
        const auto& lifetime = semantic.lifetime_states[index];
        if (!known_operation_kind(source.kind) ||
            !cellerator::compute::operation::v2::valid_stable_id(source.identity)) {
            set_status(status, semantic_to_planning_status_v1::invalid_operation);
            return std::nullopt;
        }
        if (!canonical.operation_identity.valid() ||
            !same_identity(canonical.operation_identity, source.identity) ||
            canonical.operation_spelling.empty() || canonical.field_identity == 0u ||
            Cellerator::compiler::ir::semantic::validate_numeric_tuple_ir_v1(
                canonical.numerical) !=
                Cellerator::compiler::ir::semantic::state_value_ir_validation_code_v1::success) {
            set_status(status, semantic_to_planning_status_v1::invalid_canonical_operation);
            return std::nullopt;
        }
        if (find_field(semantic, canonical.field_identity) == nullptr) {
            set_status(status, semantic_to_planning_status_v1::invalid_field);
            return std::nullopt;
        }
        if (lifetime.value_generation == 0u || lifetime.structure_epoch == 0u ||
            !lifetime.values_valid || !lifetime.structure_valid) {
            set_status(status, semantic_to_planning_status_v1::invalid_generation);
            return std::nullopt;
        }
        if (std::find(field_order.begin(), field_order.end(), canonical.field_identity) ==
            field_order.end()) {
            field_order.push_back(canonical.field_identity);
        }
    }

    for (const auto field_identity : field_order) {
        const auto* field = find_field(semantic, field_identity);
        const auto planning_field = planning_identity_v1{field_identity, 0u};
        const auto first_operation = static_cast<std::uint32_t>(result.operations.size());
        std::uint32_t flags = planning_constraint_none_v1;
        const auto first_constraint = static_cast<std::uint32_t>(result.constraints.size());
        for (const auto& constraint : field->constraints) {
            if (constraint.name.empty() || constraint.value.empty()) {
                set_status(status, semantic_to_planning_status_v1::invalid_constraint);
                return std::nullopt;
            }
            flags |= constraint_flag(constraint);
            result.constraints.push_back(
                {planning_field, constraint.name, constraint.value, constraint.hard});
        }
        const auto constraint_count =
            static_cast<std::uint32_t>(result.constraints.size()) - first_constraint;

        for (std::uint32_t index = 0u; index != semantic.operation_count; ++index) {
            const auto& canonical = semantic.canonical_operations[index];
            if (canonical.field_identity != field_identity) continue;
            const auto& source = semantic.source_operations[index];
            const auto& lifetime = semantic.lifetime_states[index];
            lowered_semantic_operation_v1 lowered;
            lowered.scope.operation = operation_identity(source.identity);
            lowered.scope.field = planning_field;
            lowered.scope.ordinal = static_cast<std::uint32_t>(result.operations.size()) -
                first_operation;
            lowered.kind = source.kind;
            lowered.numeric = canonical.numerical;
            lowered.structure_epoch = lifetime.structure_epoch;
            lowered.value_generation = lifetime.value_generation;
            lowered.support_generation = lifetime.support_generation;
            lowered.order_generation = lifetime.order_generation;
            lowered.first_constraint = first_constraint;
            lowered.constraint_count = constraint_count;
            lowered.planner_request.kind = source.kind;
            lowered.planner_request.persistent_problem_identity = source.identity;
            lowered.planner_request.operation_identity = source.identity;
            lowered.planner_request.expected_value_generation = {lifetime.value_generation};
            lowered.planner_request.numeric =
                Cellerator::compiler::ir::semantic::to_operation_numeric_policy_v1(
                    canonical.numerical);
            lowered.planner_request.determinism.deterministic_required =
                (flags & planning_constraint_deterministic_v1) != 0u;
            lowered.planner_request.determinism.stable_work_order =
                lowered.planner_request.determinism.deterministic_required;
            result.operation_scopes.push_back(lowered.scope);
            result.operations.push_back(std::move(lowered));
        }

        planning_problem_v1 problem;
        problem.problem = make_problem_identity(semantic.semantic_module, planning_field);
        problem.semantic_module = semantic.semantic_module;
        problem.semantic_fingerprint = semantic.semantic_fingerprint;
        problem.field = planning_field;
        problem.profile_family = result.profile_environment;
        problem.operation_count = static_cast<std::uint32_t>(result.operations.size()) -
            first_operation;
        problem.first_operation = 0u;
        problem.scope = planning_scope_kind_v1::field;
        problem.target = options.target;
        problem.constraints = flags;
        problem.objectives = options.objectives;
        result.problems.push_back(problem);
    }
    result.refresh_views();
    for (const auto& problem : result.problems) {
        if (validate_planning_problem_v1(problem) != planning_problem_status_v1::ok) {
            set_status(status, semantic_to_planning_status_v1::invalid_planning_problem);
            return std::nullopt;
        }
    }
    set_status(status, semantic_to_planning_status_v1::success);
    return result;
}

}  // namespace cellerator::compiler::ir::planning::v1
