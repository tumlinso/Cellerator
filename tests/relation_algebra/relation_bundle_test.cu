#include <Cellerator/compute/operation/relation_bundle.hh>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace operation = cellerator::compute::operation;
namespace execution = cellerator::execution;
namespace core = cellerator::compute::math::core;

namespace {

template<typename Condition>
void require(Condition condition, const char *message) {
    if (!static_cast<bool>(condition)) {
        std::fprintf(stderr, "relation_bundle_test: %s\n", message);
        std::exit(1);
    }
}

execution::persistent_axis_identity axis(std::uint64_t seed) {
    return {{execution::biological_abi_version,
                execution::serialized_record_kind::persistent_axis_identity,
                sizeof(execution::persistent_axis_identity)},
        {seed + 1u, seed + 2u}, {seed + 3u, seed + 4u},
        {seed + 5u, seed + 6u}, {seed + 7u, seed + 8u}};
}

operation::relation_numeric_semantics_v1 numeric() {
    return {execution::numeric_type::f32, execution::numeric_type::f32,
        execution::numeric_type::f32, execution::numeric_type::f32,
        execution::numeric_type::f32, execution::numeric_type::f32,
        core::rounding_policy::nearest_even, core::saturation_policy::none,
        operation::nan_policy_v1::propagate, {}};
}

operation::typed_relation_v1 relation(std::uint64_t identity,
    execution::persistent_axis_identity source,
    execution::persistent_axis_identity destination) {
    return {{identity, identity + 1u}, {3u}, source, destination, 3u};
}

operation::relation_bundle_plan_v1 plan(
    operation::relation_bundle_kind_v1 kind,
    const operation::typed_relation_v1 *relations,
    const core::stable_id *operations,
    std::uint32_t count,
    execution::persistent_axis_identity destination) {
    operation::relation_bundle_plan_v1 result{};
    result.kind = kind;
    result.bundle = {relations, count, 0u, destination};
    result.operation_identities = operations;
    result.numeric = numeric();
    result.dense_width = 1u;
    return result;
}

struct execution_trace {
    std::array<float, 3> destination{};
    std::array<operation::relation_algebra_kind_v1, 4> kinds{};
    std::array<execution::output_update_kind, 4> updates{};
    std::uint32_t calls = 0u;
    std::uint32_t fail_at = 0xffffffffu;
};

bool execute_bundle_step(const operation::relation_algebra_problem_v1 &problem,
    execution::output_update_kind update, void *opaque) noexcept {
    auto &trace = *static_cast<execution_trace *>(opaque);
    const std::uint32_t call = trace.calls++;
    trace.kinds[call] = problem.kind;
    trace.updates[call] = update;
    if (call == trace.fail_at) return false;
    if (operation::validate_relation_algebra_problem_v1(problem)
        != operation::relation_algebra_status_v1::ok)
        return false;
    const std::array<float, 3> contribution =
        problem.relation.structure.low == 101u
        ? std::array<float, 3>{1, 2, 0}
        : std::array<float, 3>{3, 0, 4};
    if (update == execution::output_update_kind::overwrite)
        trace.destination = contribution;
    else
        for (std::size_t index = 0u; index < trace.destination.size(); ++index)
            trace.destination[index] += contribution[index];
    return true;
}

struct incidence_trace {
    std::array<float, 3> source{5, 7, 11};
    std::array<float, 2> pooled{};
    std::array<float, 2> pool_input{2, 3};
    std::array<float, 3> broadcast{};
    operation::relation_algebra_kind_v1 seen_kind{};
    execution::output_update_kind seen_update{};
};

bool execute_incidence_step(
    const operation::relation_algebra_problem_v1 &problem,
    execution::output_update_kind update, void *opaque) noexcept {
    auto &trace = *static_cast<incidence_trace *>(opaque);
    trace.seen_kind = problem.kind;
    trace.seen_update = update;
    if (problem.kind == operation::relation_algebra_kind_v1::relation_apply) {
        trace.pooled = {trace.source[0] + trace.source[1], trace.source[2]};
        return true;
    }
    if (problem.kind
        == operation::relation_algebra_kind_v1::relation_apply_transpose) {
        trace.broadcast = {
            trace.pool_input[0], trace.pool_input[0], trace.pool_input[1]};
        return true;
    }
    return false;
}

void destination_accumulation_is_explicit_and_ordered() {
    const auto destination = axis(30u);
    const operation::typed_relation_v1 relations[2] = {
        relation(101u, axis(10u), destination),
        relation(201u, axis(20u), destination)};
    const core::stable_id operations[2] = {{1u, 2u}, {3u, 4u}};
    const auto bundle = plan(operation::relation_bundle_kind_v1::
        destination_accumulate, relations, operations, 2u, destination);
    require(operation::validate_relation_bundle_plan_v1(bundle),
        "validate destination bundle");
    execution_trace trace{};
    require(operation::run_relation_bundle_v1(
        bundle, execute_bundle_step, &trace), "run destination bundle");
    require(trace.calls == 2u
        && trace.kinds[0] == operation::relation_algebra_kind_v1::relation_apply
        && trace.kinds[1] == operation::relation_algebra_kind_v1::relation_apply,
        "destination bundle did not compose forward relation applies");
    require(trace.updates[0] == execution::output_update_kind::overwrite
        && trace.updates[1] == execution::output_update_kind::accumulate,
        "destination accumulation effects are ambiguous");
    require(trace.destination == std::array<float, 3>{4, 2, 4},
        "destination bundle accumulation is wrong");

    trace = {};
    trace.fail_at = 1u;
    const auto failed = operation::run_relation_bundle_v1(
        bundle, execute_bundle_step, &trace);
    require(failed.code == operation::relation_bundle_status_v1::execution_failed
        && failed.member_index == 1u && trace.calls == 2u,
        "destination bundle did not stop at the failed member");
}

void incidence_uses_forward_pool_and_transpose_broadcast() {
    const auto member_axis = axis(40u);
    const auto pool_axis = axis(50u);
    const operation::typed_relation_v1 incidence =
        relation(301u, member_axis, pool_axis);
    const core::stable_id operation_id{5u, 6u};
    incidence_trace trace{};

    const auto pool = plan(operation::relation_bundle_kind_v1::incidence_pool,
        &incidence, &operation_id, 1u, pool_axis);
    require(operation::run_relation_bundle_v1(
        pool, execute_incidence_step, &trace), "run incidence pool");
    require(trace.seen_kind
            == operation::relation_algebra_kind_v1::relation_apply
        && trace.seen_update == execution::output_update_kind::overwrite
        && trace.pooled == std::array<float, 2>{12, 11},
        "incidence pool was not a forward relation apply");

    const auto broadcast = plan(
        operation::relation_bundle_kind_v1::incidence_broadcast,
        &incidence, &operation_id, 1u, pool_axis);
    require(operation::run_relation_bundle_v1(
        broadcast, execute_incidence_step, &trace),
        "run incidence broadcast");
    require(trace.seen_kind
            == operation::relation_algebra_kind_v1::relation_apply_transpose
        && trace.seen_update == execution::output_update_kind::overwrite
        && trace.broadcast == std::array<float, 3>{2, 2, 3},
        "incidence broadcast was not a transpose relation apply");
}

void reject_untyped_or_incompatible_bundles() {
    const auto destination = axis(60u);
    operation::typed_relation_v1 relations[2] = {
        relation(401u, axis(61u), destination),
        relation(501u, axis(62u), axis(70u))};
    const core::stable_id operations[2] = {{7u, 8u}, {9u, 10u}};
    auto bundle = plan(operation::relation_bundle_kind_v1::
        destination_accumulate, relations, operations, 2u, destination);
    const auto incompatible = operation::validate_relation_bundle_plan_v1(bundle);
    require(incompatible.code
            == operation::relation_bundle_status_v1::incompatible_destination
        && incompatible.member_index == 1u,
        "bundle accepted a different destination identity");
    bundle.kind = operation::relation_bundle_kind_v1::incidence_pool;
    require(operation::validate_relation_bundle_plan_v1(bundle).code
            == operation::relation_bundle_status_v1::invalid_shape,
        "incidence pool accepted multiple relations");
}

} // namespace

int main() {
    destination_accumulation_is_explicit_and_ordered();
    incidence_uses_forward_pool_and_transpose_broadcast();
    reject_untyped_or_incompatible_bundles();
    std::puts("relation bundle composition passed");
    return 0;
}
