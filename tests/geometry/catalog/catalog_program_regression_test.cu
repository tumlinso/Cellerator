#include <Cellerator/execution/program.hh>

#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace core = cellerator::compute::math::core;
namespace execution = cellerator::execution;
namespace planner = cellerator::planner;
namespace runtime = cellerator::runtime;

namespace cellerator::runtime {

session_status insert_session_cache(execution_session *session,
    session_cache_kind kind, session_cache_key key, void *state,
    std::uint64_t structure_epoch, std::uint64_t generation) noexcept {
    if (session == nullptr || kind != session_cache_kind::plan
        || state == nullptr || session->plans.size >= execution_session_cache_capacity)
        return session_status::invalid_argument;
    session_cache_entry &entry = session->plans.entries[session->plans.size++];
    entry.key = key;
    entry.state = state;
    entry.structure_epoch = structure_epoch;
    entry.generation = generation;
    entry.occupied = true;
    return session_status::success;
}

} // namespace cellerator::runtime

namespace {

void require(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "catalog_program_regression_test: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

constexpr core::stable_id candidate_a{0xa1u, 0xa2u};
constexpr core::stable_id candidate_b{0xb1u, 0xb2u};
constexpr core::stable_id provider_a{0x1a1u, 0x1a2u};
constexpr core::stable_id provider_b{0x1b1u, 0x1b2u};
constexpr core::stable_id view_a{0x2a1u, 0x2a2u};
constexpr core::stable_id view_b{0x2b1u, 0x2b2u};
constexpr core::stable_id capability_b{0x3b1u, 0x3b2u};

struct fake_view {
    std::uint64_t owner = 0u;
};

bool supports_numeric(const core::numeric_policy &) noexcept {
    return true;
}

core::operation_status unused_compact_prepare(
    const core::operation_candidate &,
    const core::operation_problem &,
    const core::structure_set_key &,
    const core::projection_key &,
    const core::numeric_policy &,
    const core::prepare_policy &,
    core::prepared_operation *) noexcept {
    return {core::operation_status_code::preparation_failed,
        execution::binding_validation_code::ok,
        "program v2 bypassed the candidate-owned erased adapter"};
}

std::uint32_t calls_a = 0u;
std::uint32_t calls_b = 0u;

core::operation_status prepare_provider(
    const core::candidate_preparation_adapter_v2 &adapter,
    const core::candidate_preparation_request_v2 &request,
    const execution::activated_projection_reference_v2 &projection,
    core::prepared_operation *prepared) noexcept {
    const core::stable_id identity = adapter.candidate->candidate.identity;
    const bool is_a = core::same_stable_id(identity, candidate_a);
    const bool is_b = core::same_stable_id(identity, candidate_b);
    if ((!is_a && !is_b) || projection.view_bytes != sizeof(fake_view)
        || static_cast<const fake_view *>(projection.view)->owner
            != identity.low)
        return {core::operation_status_code::unsupported_projection,
            execution::binding_validation_code::ok,
            "provider typed-view ownership mismatch"};
    if (is_a) ++calls_a;
    if (is_b) ++calls_b;
    prepared->problem = request.problem;
    prepared->structures = request.structures;
    prepared->projection = projection.key;
    prepared->numeric = request.numeric;
    prepared->kernel = identity;
    prepared->persistent = {request.state.data, request.state.bytes};
    return {};
}

core::candidate_descriptor_v2 descriptor(
    core::stable_id identity,
    core::stable_id provider,
    core::stable_id view,
    core::stable_id capability,
    std::uint16_t schema,
    std::uint16_t variant) {
    core::candidate_descriptor_v2 result{};
    result.candidate.identity = identity;
    result.candidate.name = core::same_stable_id(identity, candidate_a)
        ? "provider-a" : "provider-b-extension";
    result.candidate.operation = core::operation_kind::weighted_relation_reduce;
    result.candidate.projection = core::projection_kind::architecture_specific;
    result.candidate.capability_flags = core::candidate_deterministic;
    result.candidate.supports_numeric = supports_numeric;
    result.candidate.prepare = unused_compact_prepare;
    result.provider_identity = provider;
    result.capability_identity = capability;
    result.projection_contract = {view, 1u, 0u, schema, variant};
    result.flags = core::valid_catalog_identity_v2(capability)
        ? core::candidate_descriptor_requires_capability : 0u;
    result.minimum_dense_width = 1u;
    result.maximum_dense_width = 4u;
    result.state_bytes = sizeof(std::uint64_t);
    result.state_alignment = alignof(std::uint64_t);
    return result;
}

execution::activated_projection_reference_v2 projection(
    const core::candidate_descriptor_v2 &candidate,
    fake_view *view,
    std::uint64_t projection_seed) {
    execution::activated_projection_reference_v2 result{};
    result.key = {{projection_seed, projection_seed + 1u},
        {static_cast<std::uint32_t>(projection_seed), 1u},
        candidate.candidate.projection,
        candidate.projection_contract.schema_version,
        candidate.projection_contract.variant};
    result.provider_identity = candidate.provider_identity;
    result.capability_identity = candidate.capability_identity;
    result.contract = candidate.projection_contract;
    result.location = {execution::residency_kind::device, {}, 0, 0u};
    result.view = view;
    result.view_bytes = sizeof(*view);
    return result;
}

execution::program_axis axis(
    std::uint32_t live, std::uint64_t domain, std::uint64_t order) {
    return {{{live, 1u}, {live + 1u, 1u},
                {live + 2u, 1u}, {live + 3u, 1u}},
        {{execution::biological_abi_version,
             execution::serialized_record_kind::persistent_axis_identity,
             sizeof(execution::persistent_axis_identity)},
            {domain, 1u}, {order, 1u}, {0x700u, 1u}, {0x800u, 1u}}};
}

core::numeric_policy numeric() {
    core::numeric_policy result{};
    result.sparse_storage = execution::numeric_type::f32;
    result.dense_storage = execution::numeric_type::f32;
    result.output_storage = execution::numeric_type::f32;
    result.multiply = execution::numeric_type::f32;
    result.accumulation = execution::numeric_type::f32;
    result.scalar = execution::numeric_type::u32;
    return result;
}

execution::executable_program_request_v2 request(
    runtime::execution_session *session,
    const core::candidate_preparation_adapter_v2 *catalog,
    std::uint32_t catalog_count,
    const execution::activated_projection_reference_v2 *projections,
    std::uint32_t projection_count,
    const execution::program_candidate_cost *costs,
    std::uint32_t cost_count,
    std::uint64_t *state) {
    execution::executable_program_request_v2 result{};
    result.problem.kind = core::operation_kind::weighted_relation_reduce;
    result.problem.operation = {0x401u, 0x402u};
    result.problem.input_count = 1u;
    result.problem.output_count = 1u;
    result.problem.logical_work_items = 32u;
    result.structures.count = 1u;
    result.structures.structures[0] = {
        {0x501u, 0x502u}, {51u, 1u}, {7u}};
    result.numeric = numeric();
    result.preparation = {true, false, true, true, 8u, 0u, 0u};
    result.source_axis = axis(10u, 0x100u, 0x200u);
    result.destination_axis = axis(20u, 0x300u, 0x400u);
    result.dense_column_axis = axis(30u, 0x500u, 0x600u);
    result.planning.problem.identity = result.problem.operation;
    require(planner::make_persistent_structure_set_key(
                result.structures, &result.planning.structures),
        "persistent structure key");
    result.planning.geometry = {
        result.source_axis.persistent.domain,
        result.destination_axis.persistent.domain,
        result.source_axis.persistent.geometry,
        result.source_axis.persistent.order,
        result.destination_axis.persistent.order,
        result.source_axis.persistent.partition};
    result.planning.device = {1u, 7u, 0u, 700u};
    result.planning.build = {1u, 2u, 3u, 4u};
    result.planning.policy = {8u, 8u, 8u, 1u, 1u, 1u, 0u};
    result.planner_policy.deterministic = true;
    result.current_evidence_revision = 1u;
    result.catalog = {catalog, catalog_count, 0u};
    result.projections = projections;
    result.projection_count = projection_count;
    result.costs = costs;
    result.cost_count = cost_count;
    result.session = session;
    result.dense_width = 2u;
    result.preparation_state = {state, sizeof(*state)};
    return result;
}

execution::program_candidate_cost cost(
    core::stable_id candidate,
    execution::projection_id projection,
    double kernel_ns) {
    execution::program_candidate_cost result{};
    result.candidate = candidate;
    result.projection = projection;
    result.phases.kernel_ns = kernel_ns;
    result.planner_flags = planner::planner_candidate_correct
        | planner::planner_candidate_deterministic;
    return result;
}

void test_enumeration_extension_and_atomic_rejection() {
    const core::candidate_descriptor_v2 descriptors[]{
        descriptor(candidate_a, provider_a, view_a, {}, 9u, 1u),
        descriptor(candidate_b, provider_b, view_b, capability_b, 10u, 2u)};
    const core::candidate_preparation_adapter_v2 adapters[]{
        {core::candidate_preparation_adapter_schema_version_v2,
            sizeof(core::candidate_preparation_adapter_v2), &descriptors[0],
            prepare_provider, {}},
        {core::candidate_preparation_adapter_schema_version_v2,
            sizeof(core::candidate_preparation_adapter_v2), &descriptors[1],
            prepare_provider, {}}};
    fake_view views[]{ {candidate_a.low}, {candidate_b.low} };
    execution::activated_projection_reference_v2 projections[]{
        projection(descriptors[0], &views[0], 0x601u),
        projection(descriptors[1], &views[1], 0x701u)};
    execution::program_candidate_cost costs[]{
        cost(candidate_a, projections[0].key.persistent, 50.0),
        cost(candidate_b, projections[1].key.persistent, 5.0)};
    runtime::execution_session session{};
    session.initialized = true;
    session.device = 0;
    std::uint64_t state = 0u;

    auto compile_request = request(
        &session, adapters, 1u, projections, 1u, costs, 1u, &state);
    execution::executable_program program{};
    require(static_cast<bool>(execution::compile_executable_program_v2(
                compile_request, &program))
            && program.candidate_count == 1u
            && core::same_stable_id(program.selected_candidate, candidate_a)
            && calls_a == 1u && calls_b == 0u,
        "base catalog enumeration or preparation");

    compile_request = request(
        &session, adapters, 2u, projections, 2u, costs, 2u, &state);
    require(static_cast<bool>(execution::compile_executable_program_v2(
                compile_request, &program))
            && program.candidate_count == 2u
            && program.legal_count == 2u
            && core::same_stable_id(program.selected_candidate, candidate_b)
            && program.selection == planner::selection_source::analytical
            && calls_a == 1u && calls_b == 1u,
        "source-linked provider extension required central program edits");

    const core::candidate_preparation_adapter_v2 duplicate[]{
        adapters[0], adapters[0]};
    auto invalid = request(
        &session, duplicate, 2u, projections, 2u, costs, 2u, &state);
    require(execution::compile_executable_program_v2(invalid, &program).code
            == execution::executable_program_status_code::invalid_argument
            && program.candidate_count == 0u
            && program.preparation_count == 0u,
        "duplicate catalog rejection was not atomic");

    auto wrong_capability = projections[1];
    wrong_capability.capability_identity.low += 1u;
    invalid = request(&session, &adapters[1], 1u, &wrong_capability, 1u,
        &costs[1], 1u, &state);
    require(execution::compile_executable_program_v2(invalid, &program).code
            == execution::executable_program_status_code::no_compatible_candidate
            && program.candidate_count == 0u && calls_b == 1u,
        "capability mismatch reached candidate preparation");
}

} // namespace

int main() {
    test_enumeration_extension_and_atomic_rejection();
    std::cout << "catalog_program_regression_test: ok\n";
    return EXIT_SUCCESS;
}
