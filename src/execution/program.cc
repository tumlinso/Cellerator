#include <Cellerator/execution/program.hh>

#include <cstdint>
#include <new>

namespace cellerator::execution {
namespace {

executable_program_status fail(
    executable_program_status_code code,
    const char *message) noexcept {
    executable_program_status result{};
    result.code = code;
    result.message = message;
    return result;
}

void reset_program(executable_program *program) noexcept {
    program->~executable_program();
    ::new (static_cast<void *>(program)) executable_program;
}

bool same_projection_id(projection_id lhs, projection_id rhs) noexcept {
    return same_identity(lhs, rhs);
}

bool valid_program_axis(const program_axis &axis) noexcept {
    return valid_axis_identity(axis.live)
        && validate_persistent_axis_identity(axis.persistent)
            == biological_validation_code::ok;
}

template<typename Request>
bool matches_planning_identity(const Request &request) noexcept {
    if (!operation_core::same_stable_id(
            request.problem.operation, request.planning.problem.identity))
        return false;
    planner::persistent_structure_set_key structures{};
    if (!planner::make_persistent_structure_set_key(
            request.structures, &structures))
        return false;
    if (structures.count != request.planning.structures.count) return false;
    for (std::uint32_t index = 0u; index < structures.count; ++index)
        if (!same_identity(structures.structures[index].identity,
                request.planning.structures.structures[index].identity)
            || structures.structures[index].epoch.value
                != request.planning.structures.structures[index].epoch.value)
            return false;
    const auto &geometry = request.planning.geometry;
    return same_identity(request.source_axis.persistent.domain,
               geometry.source_domain)
        && same_identity(request.destination_axis.persistent.domain,
            geometry.destination_domain)
        && same_identity(request.source_axis.persistent.order,
            geometry.source_order)
        && same_identity(request.destination_axis.persistent.order,
            geometry.destination_order)
        && same_identity(request.source_axis.persistent.geometry,
            geometry.geometry)
        && same_identity(request.destination_axis.persistent.geometry,
            geometry.geometry)
        && same_identity(request.source_axis.persistent.partition,
            geometry.partition)
        && same_identity(request.destination_axis.persistent.partition,
            geometry.partition);
}

bool projection_type_matches(
    activated_projection_type type,
    operation_core::projection_kind kind) noexcept {
    switch (type) {
    case activated_projection_type::row_masked:
        return kind == operation_core::projection_kind::native_row_masked;
    case activated_projection_type::csr:
        return kind == operation_core::projection_kind::csr;
    case activated_projection_type::feature_major:
        return kind == operation_core::projection_kind::native_feature_major;
    case activated_projection_type::transpose:
        return kind == operation_core::projection_kind::transpose_or_backward;
    }
    return false;
}

const program_candidate_cost *find_cost(
    const program_candidate_cost *costs,
    std::uint32_t cost_count,
    operation_core::stable_id candidate,
    projection_id projection) noexcept {
    for (std::uint32_t index = 0u; index < cost_count; ++index)
        if (operation_core::same_stable_id(
                costs[index].candidate, candidate)
            && same_projection_id(
                costs[index].projection, projection))
            return &costs[index];
    return nullptr;
}

bool catalog_is_canonical(
    operation_core::built_in_candidate_catalog_view catalog) noexcept {
    const auto canonical = operation_core::built_in_candidate_catalog();
    return catalog.entries == canonical.entries
        && catalog.size == canonical.size
        && static_cast<bool>(
            operation_core::validate_built_in_candidate_catalog());
}

std::uint64_t compatibility_view_bytes(
    activated_projection_type type) noexcept {
    switch (type) {
    case activated_projection_type::row_masked:
        return sizeof(cellpack::persistent_packing_payload_view);
    case activated_projection_type::csr:
        return sizeof(compute::math::execution_csr_view);
    case activated_projection_type::feature_major:
        return sizeof(compute::math::feature_major_projection_view);
    case activated_projection_type::transpose:
        return sizeof(compute::math::transpose_projection_view);
    }
    return 0u;
}

bool same_contract_v2(
    const operation_core::candidate_projection_contract_v2 &lhs,
    const operation_core::candidate_projection_contract_v2 &rhs) noexcept {
    return operation_core::same_stable_id(lhs.view_type, rhs.view_type)
        && lhs.abi_major == rhs.abi_major && lhs.abi_minor == rhs.abi_minor
        && lhs.schema_version == rhs.schema_version
        && lhs.variant == rhs.variant;
}

bool projection_matches_candidate_v2(
    const activated_projection_reference_v2 &projection,
    const operation_core::candidate_descriptor_v2 &candidate) noexcept {
    if (projection.schema_version
            != activated_projection_reference_schema_version_v2
        || projection.record_bytes != sizeof(activated_projection_reference_v2)
        || !valid_identity(projection.key.persistent)
        || !valid_handle(projection.key.runtime)
        || !valid_location(projection.location)
        || projection.location.residency == residency_kind::host
        || projection.view == nullptr || projection.view_bytes == 0u
        || !operation_core::same_stable_id(
            projection.provider_identity, candidate.provider_identity)
        || projection.key.kind != candidate.candidate.projection
        || projection.key.schema_version
            != candidate.projection_contract.schema_version
        || projection.key.variant != candidate.projection_contract.variant
        || !same_contract_v2(
            projection.contract, candidate.projection_contract))
        return false;
    for (std::uint32_t value : projection.reserved)
        if (value != 0u) return false;
    const bool names_capability =
        operation_core::valid_catalog_identity_v2(
            candidate.capability_identity);
    const bool requires_capability =
        (candidate.flags
            & operation_core::candidate_descriptor_requires_capability) != 0u;
    return (!names_capability && !requires_capability)
        || (operation_core::valid_catalog_identity_v2(
                projection.capability_identity)
            && operation_core::same_stable_id(
                projection.capability_identity,
                candidate.capability_identity));
}

bool valid_catalog_v2(
    operation_core::candidate_preparation_catalog_v2 catalog) noexcept {
    if (catalog.entries == nullptr || catalog.entry_count == 0u
        || catalog.reserved != 0u)
        return false;
    for (std::uint32_t index = 0u; index < catalog.entry_count; ++index) {
        if (!operation_core::validate_candidate_preparation_adapter_v2(
                catalog.entries[index]))
            return false;
        for (std::uint32_t previous = 0u; previous < index; ++previous)
            if (operation_core::same_stable_id(
                    catalog.entries[index].candidate->candidate.identity,
                    catalog.entries[previous].candidate->candidate.identity))
                return false;
    }
    return true;
}

std::uint32_t selected_index(
    const planner::planner_result &result,
    const planner::planner_candidate *candidates,
    std::uint32_t candidate_count) noexcept {
    for (std::uint32_t index = 0u; index < candidate_count; ++index)
        if (result.selected == &candidates[index]) return index;
    return maximum_program_candidates;
}

} // namespace

activated_projection_reference program_projection(
    operation_core::projection_key key,
    const cellpack::persistent_packing_payload_view &view) noexcept {
    return {key, activated_projection_type::row_masked, &view};
}

activated_projection_reference program_projection(
    operation_core::projection_key key,
    const compute::math::execution_csr_view &view) noexcept {
    return {key, activated_projection_type::csr, &view};
}

activated_projection_reference program_projection(
    operation_core::projection_key key,
    const compute::math::feature_major_projection_view &view) noexcept {
    return {key, activated_projection_type::feature_major, &view};
}

activated_projection_reference program_projection(
    operation_core::projection_key key,
    const compute::math::transpose_projection_view &view) noexcept {
    return {key, activated_projection_type::transpose, &view};
}

executable_program_status compile_executable_program_v2(
    const executable_program_request_v2 &request,
    executable_program *program) noexcept {
    if (program == nullptr) return fail(
        executable_program_status_code::invalid_argument,
        "executable program output is null");
    reset_program(program);
    if (request.schema_version != executable_program_schema_version_v2
        || request.reserved != 0u || request.reserved2 != 0u
        || request.reserved3 != 0u || request.reserved4 != 0u
        || request.session == nullptr || !request.session->initialized
        || request.session->sealed || request.dense_width == 0u
        || request.projections == nullptr || request.projection_count == 0u
        || request.costs == nullptr || request.cost_count == 0u
        || request.current_evidence_revision == 0u
        || !valid_catalog_v2(request.catalog))
        return fail(executable_program_status_code::invalid_argument,
            "executable program v2 request is incomplete");
    const auto problem = operation_core::validate_operation_problem(
        request.problem, request.structures);
    if (!problem) {
        auto result = fail(executable_program_status_code::stale_structure,
            "operation or structure identity is invalid");
        result.operation = problem;
        return result;
    }
    const auto numeric = operation_core::validate_numeric_policy(
        request.numeric);
    if (!numeric) {
        auto result = fail(executable_program_status_code::invalid_argument,
            "numeric policy is invalid");
        result.operation = numeric;
        return result;
    }
    if (!valid_program_axis(request.source_axis)
        || !valid_program_axis(request.destination_axis)
        || !valid_program_axis(request.dense_column_axis)
        || !matches_planning_identity(request))
        return fail(executable_program_status_code::identity_mismatch,
            "live axes and persistent planning identity disagree");

    planner::planner_candidate candidates[maximum_program_candidates]{};
    std::uint32_t adapter_indices[maximum_program_candidates]{};
    std::uint32_t projection_indices[maximum_program_candidates]{};
    std::uint32_t count = 0u;
    for (std::uint32_t catalog_index = 0u;
         catalog_index < request.catalog.entry_count; ++catalog_index) {
        const auto &entry = request.catalog.entries[catalog_index];
        const auto &descriptor = *entry.candidate;
        const auto &operation = descriptor.candidate;
        if (operation.operation != request.problem.kind
            || request.dense_width < descriptor.minimum_dense_width
            || (descriptor.maximum_dense_width != 0u
                && request.dense_width > descriptor.maximum_dense_width))
            continue;
        if (operation.supports_numeric == nullptr
            || !operation.supports_numeric(request.numeric))
            continue;
        for (std::uint32_t projection_index = 0u;
             projection_index < request.projection_count; ++projection_index) {
            const auto &projection = request.projections[projection_index];
            if (!projection_matches_candidate_v2(projection, descriptor))
                continue;
            const auto *cost = find_cost(
                request.costs, request.cost_count, operation.identity,
                projection.key.persistent);
            if (cost == nullptr) continue;
            if (count == maximum_program_candidates)
                return fail(executable_program_status_code::invalid_argument,
                    "compatible candidates exceed program capacity");
            candidates[count].identity = operation.identity;
            candidates[count].name = operation.name;
            candidates[count].operation = &operation;
            candidates[count].projection = projection.key;
            candidates[count].analytical = cost->phases;
            if (candidates[count].analytical.persistent_bytes
                    < operation.persistent_bytes)
                candidates[count].analytical.persistent_bytes =
                    operation.persistent_bytes;
            if (candidates[count].analytical.transient_bytes
                    < operation.transient_bytes)
                candidates[count].analytical.transient_bytes =
                    operation.transient_bytes;
            candidates[count].flags = cost->planner_flags;
            adapter_indices[count] = catalog_index;
            projection_indices[count] = projection_index;
            ++count;
        }
    }
    if (count == 0u)
        return fail(executable_program_status_code::no_compatible_candidate,
            "no catalog candidate matches operation, numeric policy, width, and projection");

    planner::planner_request planning{};
    planning.problem = request.problem;
    planning.keys = request.planning;
    planning.candidates = candidates;
    planning.candidate_count = count;
    planning.policy = request.planner_policy;
    planning.measurement = request.measurement;
    planning.cache = request.cache;
    planning.current_evidence_revision = request.current_evidence_revision;
    planner::planner_result selection{};
    const auto planned = planner::plan_end_to_end(planning, &selection);
    if (!planned) {
        auto result = fail(executable_program_status_code::planner_failed,
            "existing planner did not select a legal candidate");
        result.planning = planned;
        return result;
    }
    const std::uint32_t winner = selected_index(selection, candidates, count);
    if (winner == maximum_program_candidates)
        return fail(executable_program_status_code::planner_failed,
            "planner winner is not part of the enumerated candidate set");

    operation_core::prepared_operation prepared{};
    operation_core::candidate_preparation_request_v2 preparation{};
    preparation.problem = request.problem;
    preparation.structures = request.structures;
    preparation.numeric = request.numeric;
    preparation.policy = request.preparation;
    preparation.session = request.session;
    preparation.dense_width = request.dense_width;
    preparation.feature_axis = request.source_axis.live;
    preparation.row_axis = request.destination_axis.live;
    preparation.dense_column_axis = request.dense_column_axis.live;
    preparation.state = request.preparation_state;
    const auto prepared_status = operation_core::prepare_catalog_candidate_v2(
        request.catalog.entries[adapter_indices[winner]], preparation,
        request.projections[projection_indices[winner]], &prepared);
    if (!prepared_status) {
        auto result = fail(executable_program_status_code::preparation_failed,
            "selected candidate preparation failed");
        result.operation = prepared_status;
        return result;
    }

    program->prepared = prepared;
    program->selected_candidate = candidates[winner].identity;
    program->selected_projection = candidates[winner].projection;
    program->selection = selection.source;
    program->cache = selection.cache;
    program->conventional_winner = selection.conventional_winner;
    program->candidate_count = count;
    program->legal_count = selection.legal_count;
    program->shortlist_count = selection.shortlist_count;
    program->measurement_count = selection.measurement_count;
    program->selection_reason = selection.reason;
    program->session = request.session;
    program->preparation_count = 1u;
    for (std::uint32_t index = 0u; index < count; ++index) {
        const auto &diagnostic = selection.diagnostics[index];
        program->candidates[index] = {candidates[index].identity,
            candidates[index].name, candidates[index].projection,
            diagnostic.analytical, diagnostic.rejection,
            diagnostic.shortlisted, diagnostic.measured,
            diagnostic.conventional, 0u};
    }
    program->expected_cost = selection.source == planner::selection_source::empirical
            && selection.diagnostics[winner].measured
        ? selection.diagnostics[winner].empirical
        : selection.diagnostics[winner].analytical;
    return {};
}

executable_program_status compile_executable_program(
    const executable_program_request &request,
    executable_program *program) noexcept {
    if (program == nullptr)
        return fail(executable_program_status_code::invalid_argument,
            "executable program output is null");
    reset_program(program);
    if (request.schema_version != executable_program_schema_version
        || request.projection_count > maximum_program_candidates
        || request.projections == nullptr || request.projection_count == 0u
        || !catalog_is_canonical(request.catalog))
        return fail(executable_program_status_code::invalid_argument,
            "v1 compatibility program request is incomplete");

    const operation_core::candidate_preparation_catalog_v2 catalog =
        operation_core::built_in_candidate_preparation_catalog_v2();
    activated_projection_reference_v2
        projections[maximum_program_candidates]{};
    std::uint32_t projection_count = 0u;
    for (std::uint32_t projection_index = 0u;
         projection_index < request.projection_count; ++projection_index) {
        const activated_projection_reference &legacy =
            request.projections[projection_index];
        if (legacy.view == nullptr
            || !projection_type_matches(legacy.type, legacy.key.kind))
            continue;
        const operation_core::candidate_descriptor_v2 *descriptor = nullptr;
        for (std::uint32_t catalog_index = 0u;
             catalog_index < catalog.entry_count; ++catalog_index) {
            const auto *candidate = catalog.entries[catalog_index].candidate;
            if (candidate != nullptr
                && candidate->candidate.projection == legacy.key.kind
                && candidate->projection_contract.schema_version
                    == legacy.key.schema_version
                && candidate->projection_contract.variant
                    == legacy.key.variant) {
                descriptor = candidate;
                break;
            }
        }
        if (descriptor == nullptr) continue;
        activated_projection_reference_v2 converted{};
        converted.key = legacy.key;
        converted.provider_identity = descriptor->provider_identity;
        converted.capability_identity = descriptor->capability_identity;
        converted.contract = descriptor->projection_contract;
        converted.location = {
            residency_kind::device, {}, request.session == nullptr
                ? -1 : request.session->device, 0u};
        converted.view = legacy.view;
        converted.view_bytes = compatibility_view_bytes(legacy.type);
        projections[projection_count++] = converted;
    }
    if (projection_count == 0u)
        return fail(executable_program_status_code::no_compatible_candidate,
            "v1 projections cannot be represented by catalog v2");

    executable_program_request_v2 v2{};
    v2.problem = request.problem;
    v2.structures = request.structures;
    v2.numeric = request.numeric;
    v2.preparation = request.preparation;
    v2.planning = request.planning;
    v2.planner_policy = request.planner_policy;
    v2.measurement = request.measurement;
    v2.cache = request.cache;
    v2.current_evidence_revision = request.current_evidence_revision;
    v2.catalog = catalog;
    v2.projections = projections;
    v2.projection_count = projection_count;
    v2.costs = request.costs;
    v2.cost_count = request.cost_count;
    v2.session = request.session;
    v2.dense_width = request.dense_width;
    v2.source_axis = request.source_axis;
    v2.destination_axis = request.destination_axis;
    v2.dense_column_axis = request.dense_column_axis;
    v2.preparation_state = request.preparation_state;
    return compile_executable_program_v2(v2, program);
}

executable_program_status run_executable_program(
    executable_program *program,
    const executable_program_launch &launch,
    executable_program_result *result) noexcept {
    if (result != nullptr) *result = executable_program_result{};
    if (program == nullptr || result == nullptr
        || program->schema_version != executable_program_schema_version_v2
        || program->session == nullptr || !program->session->initialized)
        return fail(executable_program_status_code::invalid_argument,
            "executable program launch is incomplete");
    if (program->prepared.structures.count == 0u
        || launch.expected_structure_epoch.value == 0u
        || launch.expected_structure_epoch.value
            != program->prepared.structures.structures[0].epoch.value)
        return fail(executable_program_status_code::stale_structure,
            "launch structure epoch does not match prepared topology");
    if (launch.bindings.stream.device_ordinal != program->session->device)
        return fail(executable_program_status_code::invalid_launch,
            "caller stream device does not match execution session");
    if (launch.bindings.value_count != 0u) {
        if (launch.value_readiness == nullptr
            || launch.expected_value_generation.value == 0u)
            return fail(executable_program_status_code::stale_or_unready_value,
                "value-bearing launch lacks explicit readiness and generation");
        for (std::uint32_t index = 0u;
             index < launch.bindings.value_count; ++index)
            if (launch.bindings.values == nullptr
                || launch.bindings.values[index].expected_generation.value
                    != launch.expected_value_generation.value)
                return fail(
                    executable_program_status_code::stale_or_unready_value,
                    "launch value bindings disagree on generation");
        const auto readiness = runtime::wait_for_value_generation(
            *launch.value_readiness,
            launch.expected_structure_epoch.value,
            launch.expected_value_generation.value,
            static_cast<cudaStream_t>(launch.bindings.stream.stream),
            launch.bindings.stream.device_ordinal);
        if (readiness != runtime::value_readiness_status::success) {
            auto status = fail(
                executable_program_status_code::stale_or_unready_value,
                "value generation is stale or not ready for caller stream");
            status.readiness = readiness;
            return status;
        }
    }
    const auto executed = operation_core::run_prepared_operation(
        program->prepared, launch.bindings);
    if (!executed) {
        auto status = fail(
            executed.code == operation_core::operation_status_code::stale_structure
                ? executable_program_status_code::stale_structure
                : executable_program_status_code::execution_failed,
            "prepared operation rejected or failed the launch");
        status.operation = executed;
        return status;
    }
    ++program->run_count;
    result->candidate = program->selected_candidate;
    result->projection = program->selected_projection;
    result->selection = program->selection;
    result->expected_cost = program->expected_cost;
    result->output_orders = program->prepared.binding_contract.output_orders;
    result->output_order_count =
        program->prepared.binding_contract.output_order_count;
    result->structure_epoch_value = launch.expected_structure_epoch;
    result->consumed_generation = launch.expected_value_generation;
    result->completion_stream = launch.bindings.stream;
    result->enqueued = true;
    return {};
}

} // namespace cellerator::execution
