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
        std::cerr << "program_v2_test: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

constexpr core::stable_id candidate_id{0x101u, 0x102u};
constexpr core::stable_id provider_id{0x201u, 0x202u};
constexpr core::stable_id view_type{0x301u, 0x302u};

bool supports_numeric(const core::numeric_policy &) noexcept {
    return true;
}

core::operation_status unused_legacy_prepare(
    const core::operation_candidate &,
    const core::operation_problem &,
    const core::structure_set_key &,
    const core::projection_key &,
    const core::numeric_policy &,
    const core::prepare_policy &,
    core::prepared_operation *) noexcept {
    return {core::operation_status_code::preparation_failed,
        execution::binding_validation_code::ok,
        "program v2 called the compact candidate prepare function"};
}

struct fake_projection_view {
    std::uint64_t semantic_value = 0xabcdefu;
};

std::uint32_t adapter_calls = 0u;

core::operation_status prepare_fake_provider(
    const core::candidate_preparation_adapter_v2 &adapter,
    const core::candidate_preparation_request_v2 &request,
    const execution::activated_projection_reference_v2 &projection,
    core::prepared_operation *prepared) noexcept {
    ++adapter_calls;
    if (adapter.candidate == nullptr
        || projection.view_bytes != sizeof(fake_projection_view)
        || static_cast<const fake_projection_view *>(projection.view)
                ->semantic_value != 0xabcdefu)
        return {core::operation_status_code::unsupported_projection,
            execution::binding_validation_code::ok,
            "fake provider rejected erased view"};
    prepared->problem = request.problem;
    prepared->structures = request.structures;
    prepared->projection = projection.key;
    prepared->numeric = request.numeric;
    prepared->kernel = adapter.candidate->candidate.identity;
    prepared->persistent = {request.state.data, request.state.bytes};
    return {};
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

core::candidate_descriptor_v2 descriptor() {
    core::candidate_descriptor_v2 result{};
    result.candidate.identity = candidate_id;
    result.candidate.name = "external-fake-provider";
    result.candidate.operation = core::operation_kind::weighted_relation_reduce;
    result.candidate.projection = core::projection_kind::architecture_specific;
    result.candidate.capability_flags = core::candidate_deterministic;
    result.candidate.supports_numeric = supports_numeric;
    result.candidate.prepare = unused_legacy_prepare;
    result.provider_identity = provider_id;
    result.projection_contract = {view_type, 1u, 0u, 9u, 2u};
    result.minimum_dense_width = 1u;
    result.maximum_dense_width = 4u;
    result.state_bytes = sizeof(std::uint64_t);
    result.state_alignment = alignof(std::uint64_t);
    return result;
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
    const execution::activated_projection_reference_v2 *projection,
    const execution::program_candidate_cost *cost,
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
    result.projections = projection;
    result.projection_count = 1u;
    result.costs = cost;
    result.cost_count = 1u;
    result.session = session;
    result.dense_width = 2u;
    result.preparation_state = {state, sizeof(*state)};
    return result;
}

void test_external_provider_without_program_switch() {
    const core::candidate_descriptor_v2 candidate = descriptor();
    const core::candidate_preparation_adapter_v2 adapter{
        core::candidate_preparation_adapter_schema_version_v2,
        sizeof(core::candidate_preparation_adapter_v2), &candidate,
        prepare_fake_provider, {}};
    fake_projection_view view{};
    execution::activated_projection_reference_v2 projection{};
    projection.key = {{0x601u, 0x602u}, {61u, 1u},
        core::projection_kind::architecture_specific, 9u, 2u};
    projection.provider_identity = provider_id;
    projection.contract = candidate.projection_contract;
    projection.location = {
        execution::residency_kind::device, {}, 0, 0u};
    projection.view = &view;
    projection.view_bytes = sizeof(view);
    execution::program_candidate_cost cost{};
    cost.candidate = candidate_id;
    cost.projection = projection.key.persistent;
    cost.phases.kernel_ns = 13.0;
    cost.planner_flags = planner::planner_candidate_correct
        | planner::planner_candidate_deterministic;
    runtime::execution_session session{};
    session.initialized = true;
    session.device = 0;
    std::uint64_t state = 0u;
    const auto compile_request = request(
        &session, &adapter, 1u, &projection, &cost, &state);
    execution::executable_program program{};
    require(static_cast<bool>(execution::compile_executable_program_v2(
                compile_request, &program)),
        "external provider program compilation");
    require(adapter_calls == 1u && program.candidate_count == 1u
            && core::same_stable_id(
                program.selected_candidate, candidate_id)
            && program.selected_projection.runtime.slot == 61u
            && program.preparation_count == 1u
            && session.plans.size == 1u,
        "v2 enumeration, selection, or candidate-owned preparation");

    core::candidate_preparation_adapter_v2 duplicates[2]{adapter, adapter};
    auto invalid = request(
        &session, duplicates, 2u, &projection, &cost, &state);
    require(execution::compile_executable_program_v2(invalid, &program).code
            == execution::executable_program_status_code::invalid_argument,
        "duplicate candidate identities were accepted");
    invalid = compile_request;
    auto mismatched = projection;
    mismatched.provider_identity.low += 1u;
    invalid.projections = &mismatched;
    require(execution::compile_executable_program_v2(invalid, &program).code
            == execution::executable_program_status_code::no_compatible_candidate
            && adapter_calls == 1u,
        "provider mismatch reached preparation callback");
}

} // namespace

int main() {
    test_external_provider_without_program_switch();
    std::cout << "program_v2_test: ok\n";
    return EXIT_SUCCESS;
}
