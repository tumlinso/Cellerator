#include <Cellerator/execution/program.hh>

#include <cstdint>

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

bool same_projection_id(projection_id lhs, projection_id rhs) noexcept {
    return same_identity(lhs, rhs);
}

bool valid_program_axis(const program_axis &axis) noexcept {
    return valid_axis_identity(axis.live)
        && validate_persistent_axis_identity(axis.persistent)
            == biological_validation_code::ok;
}

bool matches_planning_identity(
    const executable_program_request &request) noexcept {
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
    const executable_program_request &request,
    operation_core::stable_id candidate,
    projection_id projection) noexcept {
    for (std::uint32_t index = 0u; index < request.cost_count; ++index)
        if (operation_core::same_stable_id(
                request.costs[index].candidate, candidate)
            && same_projection_id(
                request.costs[index].projection, projection))
            return &request.costs[index];
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

operation_core::operation_status prepare_selected(
    const executable_program_request &request,
    const operation_core::built_in_candidate_descriptor &entry,
    const activated_projection_reference &projection,
    operation_core::prepared_operation *prepared) noexcept {
    operation_core::preparation_factory_request factory{};
    factory.catalog_entry = &entry;
    factory.problem = request.problem;
    factory.structures = request.structures;
    factory.projection = projection.key;
    factory.numeric = request.numeric;
    factory.policy = request.preparation;
    factory.session = request.session;
    factory.dense_width = request.dense_width;
    factory.feature_axis = request.source_axis.live;
    factory.row_axis = request.destination_axis.live;
    factory.dense_column_axis = request.dense_column_axis.live;
    factory.state = request.preparation_state;
    switch (projection.type) {
    case activated_projection_type::row_masked:
        return operation_core::prepare_catalog_row_masked(factory,
            *static_cast<const cellpack::persistent_packing_payload_view *>(
                projection.view), prepared);
    case activated_projection_type::csr:
        return operation_core::prepare_catalog_csr(factory,
            *static_cast<const compute::math::execution_csr_view *>(
                projection.view), prepared);
    case activated_projection_type::feature_major:
        return operation_core::prepare_catalog_feature_major(factory,
            *static_cast<const compute::math::feature_major_projection_view *>(
                projection.view), prepared);
    case activated_projection_type::transpose:
        return operation_core::prepare_catalog_transpose(factory,
            *static_cast<const compute::math::transpose_projection_view *>(
                projection.view), prepared);
    }
    return {operation_core::operation_status_code::unsupported_projection,
        binding_validation_code::ok, "unknown activated projection type"};
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

executable_program_status compile_executable_program(
    const executable_program_request &request,
    executable_program *program) noexcept {
    if (program == nullptr) return fail(
        executable_program_status_code::invalid_argument,
        "executable program output is null");
    *program = executable_program{};
    if (request.schema_version != executable_program_schema_version
        || request.session == nullptr || !request.session->initialized
        || request.session->sealed || request.dense_width == 0u
        || request.projections == nullptr || request.projection_count == 0u
        || request.costs == nullptr || request.cost_count == 0u
        || request.current_evidence_revision == 0u
        || request.preparation_state.data == nullptr
        || !catalog_is_canonical(request.catalog))
        return fail(executable_program_status_code::invalid_argument,
            "executable program request is incomplete");
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

    operation_core::operation_candidate operations[maximum_program_candidates]{};
    planner::planner_candidate candidates[maximum_program_candidates]{};
    const operation_core::built_in_candidate_descriptor
        *entries[maximum_program_candidates]{};
    const activated_projection_reference
        *projections[maximum_program_candidates]{};
    std::uint32_t count = 0u;
    for (std::uint32_t catalog_index = 0u;
         catalog_index < request.catalog.size; ++catalog_index) {
        const auto &entry = request.catalog.entries[catalog_index];
        if (entry.operation != request.problem.kind
            || request.dense_width < entry.minimum_dense_width
            || request.dense_width > entry.maximum_dense_width)
            continue;
        const auto operation = entry.factory();
        if (operation.supports_numeric == nullptr
            || !operation.supports_numeric(request.numeric))
            continue;
        for (std::uint32_t projection_index = 0u;
             projection_index < request.projection_count; ++projection_index) {
            const auto &projection = request.projections[projection_index];
            if (projection.view == nullptr
                || projection.key.kind != entry.projection
                || projection.key.schema_version
                    != entry.projection_schema_version
                || projection.key.variant != entry.projection_variant
                || !projection_type_matches(
                    projection.type, projection.key.kind))
                continue;
            const auto *cost = find_cost(
                request, entry.identity, projection.key.persistent);
            if (cost == nullptr) continue;
            if (count == maximum_program_candidates)
                return fail(executable_program_status_code::invalid_argument,
                    "compatible candidates exceed program capacity");
            operations[count] = operation;
            candidates[count].identity = entry.identity;
            candidates[count].name = entry.name;
            candidates[count].operation = &operations[count];
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
            entries[count] = &entry;
            projections[count] = &projection;
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
    const auto prepared_status = prepare_selected(
        request, *entries[winner], *projections[winner], &prepared);
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

executable_program_status run_executable_program(
    executable_program *program,
    const executable_program_launch &launch,
    executable_program_result *result) noexcept {
    if (result != nullptr) *result = executable_program_result{};
    if (program == nullptr || result == nullptr
        || program->schema_version != executable_program_schema_version
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
