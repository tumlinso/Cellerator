#include <Cellerator/execution/preparation/preparation_factory.hh>

#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace core = cellerator::execution::operation_core;
namespace execution = cellerator::execution;
namespace runtime = cellerator::runtime;

namespace {

void require(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "erased_prepare_test: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

constexpr core::stable_id candidate_id{0x101u, 0x102u};
constexpr core::stable_id provider_id{0x201u, 0x202u};
constexpr core::stable_id view_type{0x301u, 0x302u};
constexpr std::uint64_t expected_magic = 0xfeedfacecafebeefull;

struct fake_projection_view {
    std::uint64_t magic = expected_magic;
    std::uint64_t payload = 17u;
};

bool supports_numeric(const core::numeric_policy &) noexcept {
    return true;
}

core::operation_status legacy_prepare_should_not_run(
    const core::operation_candidate &,
    const core::operation_problem &,
    const core::structure_set_key &,
    const core::projection_key &,
    const core::numeric_policy &,
    const core::prepare_policy &,
    core::prepared_operation *) noexcept {
    return {core::operation_status_code::preparation_failed,
        execution::binding_validation_code::ok,
        "legacy prepare callback unexpectedly ran"};
}

std::uint32_t erased_prepare_calls = 0u;

core::operation_status provider_owned_prepare(
    const core::candidate_preparation_adapter_v2 &adapter,
    const core::candidate_preparation_request_v2 &request,
    const execution::activated_projection_reference_v2 &projection,
    core::prepared_operation *prepared) noexcept {
    ++erased_prepare_calls;
    if (adapter.candidate == nullptr
        || !core::same_stable_id(
            adapter.candidate->candidate.identity, candidate_id)
        || projection.view_bytes != sizeof(fake_projection_view)
        || static_cast<const fake_projection_view *>(projection.view)->magic
            != expected_magic)
        return {core::operation_status_code::unsupported_projection,
            execution::binding_validation_code::ok,
            "provider-owned typed view validation failed"};
    prepared->problem = request.problem;
    prepared->structures = request.structures;
    prepared->projection = projection.key;
    prepared->numeric = request.numeric;
    prepared->kernel = candidate_id;
    prepared->persistent = {request.state.data, request.state.bytes};
    return {};
}

core::candidate_descriptor_v2 descriptor() {
    core::candidate_descriptor_v2 result{};
    result.candidate.identity = candidate_id;
    result.candidate.name = "fake-provider-erased-prepare";
    result.candidate.operation = core::operation_kind::weighted_relation_reduce;
    result.candidate.projection = core::projection_kind::architecture_specific;
    result.candidate.supports_numeric = supports_numeric;
    result.candidate.prepare = legacy_prepare_should_not_run;
    result.provider_identity = provider_id;
    result.projection_contract = {view_type, 1u, 0u, 7u, 3u};
    result.minimum_dense_width = 1u;
    result.maximum_dense_width = 8u;
    result.state_bytes = sizeof(std::uint64_t);
    result.state_alignment = alignof(std::uint64_t);
    return result;
}

execution::activated_projection_reference_v2 reference(
    const fake_projection_view *view) {
    execution::projection_reference_binding_v2 binding{};
    binding.key = {{0x401u, 0x402u}, {41u, 1u},
        core::projection_kind::architecture_specific, 7u, 3u};
    binding.provider_identity = provider_id;
    binding.contract = {view_type, 1u, 0u, 7u, 3u};
    binding.location = {execution::residency_kind::device, {}, 0, 0u};
    binding.view = view;
    binding.view_bytes = sizeof(*view);
    execution::activated_projection_reference_v2 result{};
    require(execution::make_activated_projection_reference_v2(binding, &result)
            == execution::projection_reference_status_v2::success,
        "activated fake projection construction");
    return result;
}

core::candidate_preparation_request_v2 request(
    runtime::execution_session *session,
    std::uint64_t *state) {
    core::candidate_preparation_request_v2 result{};
    result.problem.kind = core::operation_kind::weighted_relation_reduce;
    result.problem.operation = {0x501u, 0x502u};
    result.problem.input_count = 1u;
    result.problem.output_count = 1u;
    result.problem.logical_work_items = 2u;
    result.structures.count = 1u;
    result.structures.structures[0] = {
        {0x601u, 0x602u}, {61u, 1u}, {9u}};
    result.session = session;
    result.dense_width = 4u;
    result.state = {state, sizeof(*state)};
    return result;
}

void test_provider_owned_erased_dispatch() {
    const core::candidate_descriptor_v2 candidate = descriptor();
    const core::candidate_preparation_adapter_v2 adapter{
        core::candidate_preparation_adapter_schema_version_v2,
        sizeof(core::candidate_preparation_adapter_v2), &candidate,
        provider_owned_prepare, {}};
    require(static_cast<bool>(
            core::validate_candidate_preparation_adapter_v2(adapter)),
        "valid provider adapter rejected");
    const core::candidate_preparation_catalog_v2 catalog{&adapter, 1u, 0u};
    require(core::find_candidate_preparation_adapter_v2(catalog, candidate_id)
            == &adapter,
        "candidate-owned adapter lookup failed");

    runtime::execution_session session{};
    session.initialized = true;
    session.device = 0;
    alignas(std::uint64_t) std::uint64_t state = 0u;
    fake_projection_view typed_view{};
    const auto activated = reference(&typed_view);
    const auto preparation = request(&session, &state);
    core::prepared_operation prepared{};
    require(static_cast<bool>(core::prepare_catalog_candidate_v2(adapter,
                preparation, activated, &prepared)),
        "erased preparation failed");
    require(erased_prepare_calls == 1u
            && core::same_stable_id(prepared.kernel, candidate_id)
            && prepared.projection.runtime.slot == 41u
            && prepared.persistent.data == &state
            && prepared.structures.structures[0].epoch.value == 9u,
        "erased callback result or caller-owned state mismatch");

    // Distinct preparations used to exhaust the fixed 64-entry session table.
    std::uint64_t states[65]{};
    auto next = preparation;
    for (auto &candidate_state : states) {
        ++next.structures.structures[0].epoch.value;
        next.state = {&candidate_state, sizeof(candidate_state)};
        require(static_cast<bool>(core::prepare_catalog_candidate_v2(adapter,
                    next, activated, &prepared)),
            "independent preparation in the same session");
        require(prepared.persistent.data == &candidate_state
                && prepared.structures.structures[0].epoch.value
                    == next.structures.structures[0].epoch.value,
            "independent preparation did not retain its own state and epoch");
    }
}

void test_mismatch_rejected_before_provider_callback() {
    const core::candidate_descriptor_v2 candidate = descriptor();
    const core::candidate_preparation_adapter_v2 adapter{
        core::candidate_preparation_adapter_schema_version_v2,
        sizeof(core::candidate_preparation_adapter_v2), &candidate,
        provider_owned_prepare, {}};
    runtime::execution_session session{};
    session.initialized = true;
    session.device = 0;
    std::uint64_t state = 0u;
    fake_projection_view typed_view{};
    auto activated = reference(&typed_view);
    activated.contract.variant = 4u;
    auto preparation = request(&session, &state);
    core::prepared_operation prepared{};
    const std::uint32_t before = erased_prepare_calls;
    require(core::prepare_catalog_candidate_v2(adapter, preparation,
                activated, &prepared).code
            == core::operation_status_code::unsupported_projection
            && erased_prepare_calls == before,
        "contract mismatch reached provider callback");

    activated = reference(&typed_view);
    preparation.state.bytes = sizeof(state) - 1u;
    require(core::prepare_catalog_candidate_v2(adapter, preparation,
                activated, &prepared).code
            == core::operation_status_code::preparation_failed
            && erased_prepare_calls == before,
        "undersized state reached provider callback");

    auto malformed = adapter;
    malformed.reserved[0] = 1u;
    require(!core::validate_candidate_preparation_adapter_v2(malformed),
        "nonzero adapter reserve accepted");
}

} // namespace

int main() {
    test_provider_owned_erased_dispatch();
    test_mismatch_rejected_before_provider_callback();
    std::cout << "erased_prepare_test: ok\n";
    return EXIT_SUCCESS;
}
