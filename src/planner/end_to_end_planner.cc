#include <Cellerator/planner/end_to_end_planner.hh>

#include <algorithm>
#include <cmath>
#include <limits>

namespace cellerator::planner {
namespace {

bool finite_nonnegative(double value) noexcept {
    return std::isfinite(value) && value >= 0.0;
}

bool valid_phases(const phase_costs &phases) noexcept {
    return finite_nonnegative(phases.host_preparation_ns)
        && finite_nonnegative(phases.semantic_packing_ns)
        && finite_nonnegative(phases.projection_construction_ns)
        && finite_nonnegative(phases.backend_prepare_ns)
        && finite_nonnegative(phases.static_value_pack_ns)
        && finite_nonnegative(phases.h2d_ns)
        && finite_nonnegative(phases.dynamic_input_pack_ns)
        && finite_nonnegative(phases.kernel_ns)
        && finite_nonnegative(phases.epilogue_ns)
        && finite_nonnegative(phases.order_transform_ns)
        && finite_nonnegative(phases.synchronization_ns)
        && finite_nonnegative(phases.communication_ns)
        && finite_nonnegative(phases.d2h_ns);
}

bool less_persistent_structure(
    const persistent_structure_dependency &lhs,
    const persistent_structure_dependency &rhs) noexcept {
    return lhs.identity.high < rhs.identity.high
        || (lhs.identity.high == rhs.identity.high
            && lhs.identity.low < rhs.identity.low);
}

bool valid_structure_set(
    const persistent_structure_set_key &structures) noexcept {
    if (structures.count == 0u
        || structures.count > execution::maximum_operation_structures)
        return false;
    for (std::uint32_t index = 0u; index < structures.count; ++index) {
        const persistent_structure_dependency &current =
            structures.structures[index];
        if (!execution::valid_identity(current.identity)
            || current.epoch.value == 0u)
            return false;
        if (index != 0u
            && !less_persistent_structure(
                structures.structures[index - 1u], current))
            return false;
    }
    return true;
}

bool same_structure_set(
    const persistent_structure_set_key &lhs,
    const persistent_structure_set_key &rhs) noexcept {
    if (lhs.count != rhs.count) return false;
    for (std::uint32_t index = 0u; index < lhs.count; ++index)
        if (!execution::same_identity(
                lhs.structures[index].identity,
                rhs.structures[index].identity)
            || lhs.structures[index].epoch.value
                != rhs.structures[index].epoch.value)
            return false;
    return true;
}

bool same_geometry_key(
    const semantic_geometry_key &lhs,
    const semantic_geometry_key &rhs) noexcept {
    return execution::same_identity(lhs.source_domain, rhs.source_domain)
        && execution::same_identity(
            lhs.destination_domain, rhs.destination_domain)
        && execution::same_identity(lhs.geometry, rhs.geometry)
        && execution::same_identity(lhs.source_order, rhs.source_order)
        && execution::same_identity(lhs.destination_order, rhs.destination_order)
        && execution::same_identity(lhs.partition, rhs.partition);
}

candidate_rejection reject_candidate(
    const planner_candidate &candidate,
    const planner_policy &policy,
    const operation_core::operation_problem &problem) noexcept {
    if (!operation_core::same_stable_id(candidate.identity,
            candidate.operation == nullptr
                ? operation_core::stable_id{}
                : candidate.operation->identity)
        || candidate.name == nullptr
        || candidate.operation->operation != problem.kind
        || !execution::valid_identity(candidate.projection.persistent)
        || !execution::valid_handle(candidate.projection.runtime)
        || candidate.projection.schema_version == 0u
        || candidate.projection.kind != candidate.operation->projection
        || ((candidate.flags & planner_candidate_deterministic) != 0u
            && (candidate.operation->capability_flags
                & operation_core::candidate_deterministic) == 0u)
        || ((candidate.flags & planner_candidate_graph_capture) != 0u
            && (candidate.operation->capability_flags
                & operation_core::candidate_graph_capture) == 0u)
        || candidate.analytical.persistent_bytes
            < candidate.operation->persistent_bytes
        || candidate.analytical.transient_bytes
            < candidate.operation->transient_bytes
        || !valid_phases(candidate.analytical))
        return candidate_rejection::malformed;
    if ((candidate.flags & planner_candidate_correct) == 0u)
        return candidate_rejection::incorrect;
    if (policy.deterministic
        && (candidate.flags & planner_candidate_deterministic) == 0u)
        return candidate_rejection::nondeterministic;
    if (policy.graph_capture_required
        && (candidate.flags & planner_candidate_graph_capture) == 0u)
        return candidate_rejection::graph_incompatible;
    if (policy.maximum_persistent_bytes != 0u
        && candidate.analytical.persistent_bytes
            > policy.maximum_persistent_bytes)
        return candidate_rejection::persistent_memory;
    if (policy.maximum_transient_bytes != 0u
        && candidate.analytical.transient_bytes
            > policy.maximum_transient_bytes)
        return candidate_rejection::transient_memory;
    return candidate_rejection::none;
}

bool less_identity(
    operation_core::stable_id lhs,
    operation_core::stable_id rhs) noexcept {
    return lhs.high < rhs.high || (lhs.high == rhs.high && lhs.low < rhs.low);
}

bool within_tolerance(double lhs, double rhs, double percent) noexcept {
    const double scale = std::max(std::fabs(lhs), std::fabs(rhs));
    return std::fabs(lhs - rhs) <= scale * percent / 100.0;
}

std::uint32_t find_candidate_index(
    const planner_request &request,
    operation_core::stable_id identity,
    const persistent_projection_key &projection,
    const planner_result &result) noexcept {
    for (std::uint32_t index = 0u; index < request.candidate_count; ++index)
        if (result.diagnostics[index].rejection == candidate_rejection::none
            && operation_core::same_stable_id(
                request.candidates[index].identity, identity)
            && same_persistent_projection_key(
                projection, request.candidates[index].projection))
            return index;
    return maximum_planner_candidates;
}

bool valid_cache_evidence(
    const plan_cache_entry &entry,
    const planner_request &request) noexcept {
    return entry.occupied
        && entry.evidence.evidence_revision == request.current_evidence_revision
        && entry.evidence.sample_count != 0u
        && finite_nonnegative(entry.evidence.median_total_ns)
        && finite_nonnegative(entry.evidence.spread_percent)
        && entry.evidence.spread_percent <= request.policy.maximum_spread_percent
        && entry.evidence.practical_tolerance_percent
            == request.policy.practical_tolerance_percent
        && std::isfinite(entry.evidence.confidence)
        && entry.evidence.confidence >= request.policy.minimum_cache_confidence;
}

double measurement_confidence(
    const candidate_diagnostic &selected,
    const candidate_diagnostic *runner_up,
    const planner_policy &policy) noexcept {
    const double sample_factor = std::min(
        1.0, static_cast<double>(selected.sample_count) / 5.0);
    const double spread_scale = std::max(
        1.0, policy.maximum_spread_percent);
    const double spread_factor = std::max(
        0.0, 1.0 - selected.spread_percent / spread_scale);
    double separation_factor = 0.5;
    if (runner_up != nullptr) {
        const double selected_cost = selected.empirical.amortized_total_ns;
        const double runner_cost = runner_up->empirical.amortized_total_ns;
        const double scale = std::max(std::fabs(selected_cost),
            std::fabs(runner_cost));
        const double separation_percent = scale == 0.0 ? 0.0
            : std::fabs(runner_cost - selected_cost) * 100.0 / scale;
        separation_factor = std::min(1.0, separation_percent
            / std::max(1.0, policy.practical_tolerance_percent));
    }
    return 0.5 * sample_factor + 0.3 * spread_factor
        + 0.2 * separation_factor;
}

} // namespace

planner_status compute_total_cost(
    const phase_costs &phases,
    std::uint64_t structure_reuse,
    std::uint64_t projection_reuse,
    std::uint64_t value_reuse,
    total_cost *out) noexcept {
    if (out == nullptr || structure_reuse == 0u || projection_reuse == 0u
        || value_reuse == 0u)
        return {planner_status_code::invalid_argument,
            "cost accounting requires output and nonzero reuse"};
    *out = total_cost{};
    if (!valid_phases(phases))
        return {planner_status_code::invalid_cost,
            "phase costs must be finite and nonnegative"};
    const double structure = static_cast<double>(structure_reuse);
    const double projection = static_cast<double>(projection_reuse);
    const double values = static_cast<double>(value_reuse);
    const double total = phases.host_preparation_ns
        + phases.semantic_packing_ns / structure
        + phases.projection_construction_ns / projection
        + phases.backend_prepare_ns / projection
        + phases.static_value_pack_ns / values
        + phases.h2d_ns + phases.dynamic_input_pack_ns + phases.kernel_ns
        + phases.epilogue_ns + phases.order_transform_ns
        + phases.synchronization_ns + phases.communication_ns + phases.d2h_ns;
    if (!finite_nonnegative(total))
        return {planner_status_code::invalid_cost,
            "amortized total cost overflowed"};
    out->phases = phases;
    out->structure_reuse = structure_reuse;
    out->projection_reuse = projection_reuse;
    out->value_reuse = value_reuse;
    out->amortized_total_ns = total;
    return {};
}

bool same_planning_keys(
    const planning_keys &lhs,
    const planning_keys &rhs) noexcept {
    return operation_core::same_stable_id(
            lhs.problem.identity, rhs.problem.identity)
        && same_structure_set(lhs.structures, rhs.structures)
        && same_geometry_key(lhs.geometry, rhs.geometry)
        && lhs.device.vendor == rhs.device.vendor
        && lhs.device.architecture_major == rhs.device.architecture_major
        && lhs.device.architecture_minor == rhs.device.architecture_minor
        && lhs.device.performance_class == rhs.device.performance_class
        && lhs.build.runtime == rhs.build.runtime
        && lhs.build.kernel_build == rhs.build.kernel_build
        && lhs.build.driver == rhs.build.driver
        && lhs.build.library == rhs.build.library
        && lhs.policy.structure_reuse == rhs.policy.structure_reuse
        && lhs.policy.projection_reuse == rhs.policy.projection_reuse
        && lhs.policy.value_reuse == rhs.policy.value_reuse
        && lhs.policy.numeric_policy == rhs.policy.numeric_policy
        && lhs.policy.determinism_policy == rhs.policy.determinism_policy
        && lhs.policy.output_order_policy == rhs.policy.output_order_policy
        && lhs.policy.graph_policy == rhs.policy.graph_policy;
}

bool make_persistent_structure_set_key(
    const operation_core::structure_set_key &live,
    persistent_structure_set_key *persistent) noexcept {
    if (persistent == nullptr || live.count == 0u
        || live.count > execution::maximum_operation_structures)
        return false;
    *persistent = {};
    persistent->count = live.count;
    for (std::uint32_t index = 0u; index < live.count; ++index) {
        if (!execution::valid_identity(live.structures[index].persistent)
            || live.structures[index].epoch.value == 0u)
            return false;
        persistent->structures[index] = {
            live.structures[index].persistent, live.structures[index].epoch};
    }
    std::sort(persistent->structures,
        persistent->structures + persistent->count,
        less_persistent_structure);
    return valid_structure_set(*persistent);
}

bool same_persistent_projection_key(
    const persistent_projection_key &persistent,
    const operation_core::projection_key &live) noexcept {
    return execution::same_identity(persistent.identity, live.persistent)
        && persistent.kind == live.kind
        && persistent.schema_version == live.schema_version
        && persistent.variant == live.variant;
}

planner_status plan_end_to_end(
    const planner_request &request,
    planner_result *out) noexcept {
    if (out == nullptr) return {planner_status_code::invalid_argument,
        "planner output is null"};
    *out = planner_result{};
    out->reason = "planning did not complete";
    out->practical_tolerance_percent = request.policy.practical_tolerance_percent;
    if (request.schema_version != planner_schema_version
        || request.candidates == nullptr || request.candidate_count == 0u
        || request.candidate_count > maximum_planner_candidates
        || request.problem.input_count == 0u || request.problem.output_count == 0u
        || request.problem.logical_work_items == 0u
        || !operation_core::same_stable_id(
            request.problem.operation, request.keys.problem.identity)
        || !valid_structure_set(request.keys.structures)
        || !execution::valid_identity(request.keys.geometry.source_domain)
        || !execution::valid_identity(request.keys.geometry.destination_domain)
        || !execution::valid_identity(request.keys.geometry.geometry)
        || !execution::valid_identity(request.keys.geometry.source_order)
        || !execution::valid_identity(request.keys.geometry.destination_order)
        || !execution::valid_identity(request.keys.geometry.partition)
        || request.keys.device.vendor == 0u
        || request.keys.device.performance_class == 0u
        || request.keys.build.runtime == 0u
        || request.keys.build.kernel_build == 0u
        || request.keys.build.driver == 0u
        || request.keys.build.library == 0u
        || request.current_evidence_revision == 0u
        || request.keys.policy.structure_reuse == 0u
        || request.keys.policy.projection_reuse == 0u
        || request.keys.policy.value_reuse == 0u
        || request.policy.shortlist_size == 0u
        || request.policy.shortlist_size > maximum_planner_candidates
        || !finite_nonnegative(request.policy.practical_tolerance_percent)
        || !finite_nonnegative(request.policy.maximum_spread_percent)
        || !std::isfinite(request.policy.minimum_cache_confidence)) {
        out->status = {planner_status_code::invalid_argument,
            "planner request is invalid"};
        return out->status;
    }

    std::uint32_t order[maximum_planner_candidates]{};
    for (std::uint32_t index = 0u; index < request.candidate_count; ++index) {
        candidate_diagnostic &diagnostic = out->diagnostics[index];
        diagnostic.identity = request.candidates[index].identity;
        diagnostic.conventional = (request.candidates[index].flags
            & planner_candidate_conventional) != 0u;
        diagnostic.rejection = reject_candidate(
            request.candidates[index], request.policy, request.problem);
        if (diagnostic.rejection != candidate_rejection::none) continue;
        const planner_status cost = compute_total_cost(
            request.candidates[index].analytical,
            request.keys.policy.structure_reuse,
            request.keys.policy.projection_reuse,
            request.keys.policy.value_reuse,
            &diagnostic.analytical);
        if (!cost) {
            diagnostic.rejection = candidate_rejection::malformed;
            continue;
        }
        order[out->legal_count++] = index;
    }
    if (out->legal_count == 0u) {
        out->status = {planner_status_code::no_legal_candidate,
            "no candidate satisfies correctness and policy"};
        out->reason = "all candidates were rejected before ranking";
        return out->status;
    }
    std::sort(order, order + out->legal_count,
        [&](std::uint32_t lhs, std::uint32_t rhs) noexcept {
            const double lhs_cost =
                out->diagnostics[lhs].analytical.amortized_total_ns;
            const double rhs_cost =
                out->diagnostics[rhs].analytical.amortized_total_ns;
            return lhs_cost < rhs_cost
                || (lhs_cost == rhs_cost && less_identity(
                    request.candidates[lhs].identity,
                    request.candidates[rhs].identity));
        });
    out->shortlist_count = std::min(
        out->legal_count, request.policy.shortlist_size);
    for (std::uint32_t rank = 0u; rank < out->shortlist_count; ++rank)
        out->diagnostics[order[rank]].shortlisted = true;

    if (request.cache.lookup != nullptr) {
        plan_cache_entry cached{};
        if (!request.cache.lookup(request.cache.context, request.keys, &cached)) {
            out->cache = cache_state::miss;
        } else if (!same_planning_keys(cached.keys, request.keys)
            || !valid_cache_evidence(cached, request)) {
            out->cache = cache_state::stale;
        } else {
            const std::uint32_t cached_index = find_candidate_index(
                request, cached.winner, cached.winner_projection, *out);
            if (cached_index == maximum_planner_candidates) {
                out->cache = cache_state::winner_unavailable;
            } else {
                out->cache = cache_state::hit;
                out->winner = cached.winner;
                out->selected = &request.candidates[cached_index];
                out->source = selection_source::cache;
                out->confidence = cached.evidence.confidence;
                out->practical_tolerance_percent =
                    cached.evidence.practical_tolerance_percent;
                out->conventional_winner =
                    out->diagnostics[cached_index].conventional;
                out->reason = "fresh measured cache winner remains legal";
                out->status = {};
                return {};
            }
        }
    }

    const bool one_shot = request.keys.policy.structure_reuse == 1u
        && request.keys.policy.projection_reuse == 1u
        && request.keys.policy.value_reuse == 1u;
    const bool tune = request.measurement.measure != nullptr
        && request.policy.maximum_measurements != 0u
        && request.problem.logical_work_items
            >= request.policy.minimum_tuning_work_items
        && (!one_shot || request.policy.tune_one_shot);
    const bool analytical_requires_measurement =
        (request.candidates[order[0]].flags
            & planner_candidate_empirical_required) != 0u;
    if (!tune) {
        if (analytical_requires_measurement) {
            out->status = {planner_status_code::no_correct_measurement,
                "uncertain objective v2 estimate requires empirical measurement"};
            out->reason = "calibrated analytical ranking was not used as final authority";
            return out->status;
        }
        const std::uint32_t selected_index = order[0];
        out->winner = request.candidates[selected_index].identity;
        out->selected = &request.candidates[selected_index];
        out->source = selection_source::analytical;
        out->tuning_skipped = true;
        out->conventional_winner =
            out->diagnostics[selected_index].conventional;
        out->reason = one_shot
            ? "bounded tuning skipped for one-shot reuse"
            : "bounded tuning skipped for tiny workload or missing hook";
        out->status = {};
        return {};
    }

    const std::uint32_t budget = std::min(
        out->shortlist_count, request.policy.maximum_measurements);
    std::uint32_t measured_order[maximum_planner_candidates]{};
    std::uint32_t measured_count = 0u;
    for (std::uint32_t rank = 0u; rank < budget; ++rank) {
        const std::uint32_t index = order[rank];
        measured_candidate measured{};
        ++out->measurement_count;
        out->diagnostics[index].measured = true;
        if (!request.measurement.measure(
                request.measurement.context,
                request.candidates[index], &measured)) {
            out->diagnostics[index].rejection =
                candidate_rejection::measurement_failed;
            continue;
        }
        if (!measured.correct) {
            out->diagnostics[index].rejection = candidate_rejection::incorrect;
            continue;
        }
        if (measured.contaminated
            || !finite_nonnegative(measured.spread_percent)
            || measured.spread_percent > request.policy.maximum_spread_percent
            || measured.sample_count == 0u) {
            out->diagnostics[index].rejection = candidate_rejection::contaminated;
            continue;
        }
        out->diagnostics[index].sample_count = measured.sample_count;
        out->diagnostics[index].spread_percent = measured.spread_percent;
        const planner_status cost = compute_total_cost(
            measured.phases,
            request.keys.policy.structure_reuse,
            request.keys.policy.projection_reuse,
            request.keys.policy.value_reuse,
            &out->diagnostics[index].empirical);
        if (!cost) {
            out->diagnostics[index].rejection =
                candidate_rejection::measurement_failed;
            continue;
        }
        measured_order[measured_count++] = index;
    }
    if (measured_count == 0u) {
        if (request.policy.allow_analytical_fallback_after_measurement_failure
            && !analytical_requires_measurement) {
            const std::uint32_t selected_index = order[0];
            out->winner = request.candidates[selected_index].identity;
            out->selected = &request.candidates[selected_index];
            out->source = selection_source::analytical;
            out->conventional_winner =
                out->diagnostics[selected_index].conventional;
            out->reason = "all empirical measurements failed; selected best legal analytical candidate without persistence";
            out->status = {};
            return {};
        }
        out->status = {planner_status_code::no_correct_measurement,
            "bounded tuning produced no clean correct measurement"};
        out->reason = "no empirical winner was safe to persist";
        return out->status;
    }
    std::sort(measured_order, measured_order + measured_count,
        [&](std::uint32_t lhs, std::uint32_t rhs) noexcept {
            const double lhs_cost =
                out->diagnostics[lhs].empirical.amortized_total_ns;
            const double rhs_cost =
                out->diagnostics[rhs].empirical.amortized_total_ns;
            return lhs_cost < rhs_cost
                || (lhs_cost == rhs_cost && less_identity(
                    request.candidates[lhs].identity,
                    request.candidates[rhs].identity));
        });
    std::uint32_t selected_index = measured_order[0];
    const double best_cost =
        out->diagnostics[selected_index].empirical.amortized_total_ns;
    for (std::uint32_t rank = 1u; rank < measured_count; ++rank) {
        const std::uint32_t alternative = measured_order[rank];
        const double alternative_cost =
            out->diagnostics[alternative].empirical.amortized_total_ns;
        if (!within_tolerance(best_cost, alternative_cost,
                request.policy.practical_tolerance_percent)) break;
        if (less_identity(request.candidates[alternative].identity,
                request.candidates[selected_index].identity))
            selected_index = alternative;
    }
    out->winner = request.candidates[selected_index].identity;
    out->selected = &request.candidates[selected_index];
    out->source = selection_source::empirical;
    const candidate_diagnostic *runner_up = measured_count > 1u
        ? &out->diagnostics[measured_order[
            measured_order[0] == selected_index ? 1u : 0u]]
        : nullptr;
    out->confidence = measurement_confidence(
        out->diagnostics[selected_index], runner_up, request.policy);
    out->conventional_winner = out->diagnostics[selected_index].conventional;
    out->reason = out->conventional_winner
        ? "measured conventional fallback won end to end"
        : "measured candidate won end to end";
    out->status = {};

    if (request.cache.store != nullptr) {
        const operation_core::projection_key &live_projection =
            request.candidates[selected_index].projection;
        const plan_cache_entry entry{request.keys,
            out->winner,
            {live_projection.persistent, live_projection.kind,
                live_projection.schema_version, live_projection.variant},
            {request.current_evidence_revision,
                out->diagnostics[selected_index].sample_count,
                0u,
                out->diagnostics[selected_index].empirical.amortized_total_ns,
                out->diagnostics[selected_index].spread_percent,
                out->confidence,
                request.policy.practical_tolerance_percent},
            true};
        if (!request.cache.store(request.cache.context, entry))
            out->cache_store_failed = true;
    }
    return {};
}

bool same_connected_planning_keys(
    const connected_planning_keys &lhs,
    const connected_planning_keys &rhs) noexcept {
    if (!operation_core::same_stable_id(
            lhs.graph_identity, rhs.graph_identity)
        || lhs.stage_count != rhs.stage_count
        || lhs.stage_count == 0u
        || lhs.stage_count > maximum_connected_operations)
        return false;
    for (std::uint32_t stage = 0u; stage < lhs.stage_count; ++stage)
        if (!same_planning_keys(lhs.stages[stage], rhs.stages[stage]))
            return false;
    return true;
}

planner_status plan_connected_operations(
    const connected_planner_request &request,
    connected_planner_result *out) noexcept {
    if (out == nullptr)
        return {planner_status_code::invalid_argument,
            "connected planner output is null"};
    *out = connected_planner_result{};
    out->reason = "connected planning did not complete";
    const auto stable_present = [](operation_core::stable_id id) noexcept {
        return id.low != 0u || id.high != 0u;
    };
    if (request.schema_version != connected_planner_schema_version
        || !stable_present(request.graph_identity)
        || request.stages == nullptr || request.stage_count < 2u
        || request.stage_count > maximum_connected_operations
        || request.transitions == nullptr || request.transition_count == 0u
        || request.transition_count > maximum_connected_transitions
        || request.shortlist_size == 0u
        || request.shortlist_size > maximum_connected_stage_candidates
        || request.maximum_measurements > request.shortlist_size
        || request.current_evidence_revision == 0u
        || !finite_nonnegative(request.practical_tolerance_percent)
        || !finite_nonnegative(request.maximum_spread_percent)
        || !std::isfinite(request.minimum_cache_confidence)
        || request.minimum_cache_confidence < 0.0
        || request.minimum_cache_confidence > 1.0) {
        out->status = {planner_status_code::invalid_argument,
            "connected planner request is invalid"};
        return out->status;
    }

    connected_planning_keys durable_keys{};
    durable_keys.graph_identity = request.graph_identity;
    durable_keys.stage_count = request.stage_count;
    for (std::uint32_t stage = 0u; stage < request.stage_count; ++stage) {
        const connected_operation_stage &node = request.stages[stage];
        if (node.candidates == nullptr || node.candidate_count == 0u
            || node.candidate_count > maximum_connected_stage_candidates
            || node.problem.input_count == 0u || node.problem.output_count == 0u
            || node.problem.logical_work_items == 0u
            || !operation_core::same_stable_id(
                node.problem.operation, node.keys.problem.identity)
            || !valid_structure_set(node.keys.structures)
            || !execution::valid_identity(node.keys.geometry.source_domain)
            || !execution::valid_identity(node.keys.geometry.destination_domain)
            || !execution::valid_identity(node.keys.geometry.geometry)
            || !execution::valid_identity(node.keys.geometry.source_order)
            || !execution::valid_identity(node.keys.geometry.destination_order)
            || !execution::valid_identity(node.keys.geometry.partition)
            || node.keys.device.vendor == 0u
            || node.keys.device.performance_class == 0u
            || node.keys.build.runtime == 0u
            || node.keys.build.kernel_build == 0u
            || node.keys.build.driver == 0u
            || node.keys.build.library == 0u
            || node.keys.policy.structure_reuse == 0u
            || node.keys.policy.projection_reuse == 0u
            || node.keys.policy.value_reuse == 0u) {
            out->status = {planner_status_code::invalid_argument,
                "connected operation stage is invalid"};
            return out->status;
        }
        durable_keys.stages[stage] = node.keys;
    }
    for (std::uint32_t index = 0u; index < request.transition_count; ++index) {
        const connected_transition_cost &transition = request.transitions[index];
        const bool conversion_present = stable_present(transition.conversion);
        if (transition.boundary >= request.stage_count - 1u
            || !stable_present(transition.producer)
            || !stable_present(transition.consumer)
            || (transition.order != execution::order_transition_kind::preserve
                && transition.order
                    != execution::order_transition_kind::transform
                && transition.order
                    != execution::order_transition_kind::canonicalize)
            || ((transition.format_conversion
                    || transition.order
                        != execution::order_transition_kind::preserve)
                != conversion_present)
            || !valid_phases(transition.phases)) {
            out->status = {planner_status_code::invalid_argument,
                "connected transition contract is invalid"};
            return out->status;
        }
        for (std::uint32_t prior = 0u; prior < index; ++prior)
            if (request.transitions[prior].boundary == transition.boundary
                && operation_core::same_stable_id(
                    request.transitions[prior].producer,
                    transition.producer)
                && operation_core::same_stable_id(
                    request.transitions[prior].consumer,
                    transition.consumer)) {
                out->status = {planner_status_code::invalid_argument,
                    "connected transition pair is duplicated"};
                return out->status;
            }
    }

    struct path_state {
        bool reachable = false;
        bool empirical_required = false;
        std::uint16_t reserved = 0u;
        std::uint64_t path_count = 0u;
        double total_ns = 0.0;
        std::uint32_t candidate_indices[maximum_connected_operations]{};
        total_cost stage_costs[maximum_connected_operations]{};
    };
    path_state previous[maximum_connected_stage_candidates]{};
    path_state current[maximum_connected_stage_candidates]{};

    const auto candidate_cost = [&](std::uint32_t stage,
                                    std::uint32_t index,
                                    total_cost *cost) noexcept {
        const connected_operation_stage &node = request.stages[stage];
        const planner_candidate &candidate = node.candidates[index];
        if (reject_candidate(candidate, node.policy, node.problem)
                != candidate_rejection::none)
            return false;
        return static_cast<bool>(compute_total_cost(candidate.analytical,
            node.keys.policy.structure_reuse,
            node.keys.policy.projection_reuse,
            node.keys.policy.value_reuse, cost));
    };
    const auto find_transition = [&](std::uint32_t boundary,
                                     operation_core::stable_id producer,
                                     operation_core::stable_id consumer,
                                     total_cost *cost) noexcept {
        for (std::uint32_t index = 0u; index < request.transition_count; ++index) {
            const connected_transition_cost &transition =
                request.transitions[index];
            if (transition.boundary != boundary
                || !operation_core::same_stable_id(
                    transition.producer, producer)
                || !operation_core::same_stable_id(
                    transition.consumer, consumer)) continue;
            if (!transition.legal) return false;
            const planning_keys &keys = request.stages[boundary + 1u].keys;
            return static_cast<bool>(compute_total_cost(transition.phases,
                keys.policy.structure_reuse, keys.policy.projection_reuse,
                keys.policy.value_reuse, cost));
        }
        return false;
    };
    const auto path_less = [&](const path_state &lhs,
                               const path_state &rhs,
                               std::uint32_t stage_count) noexcept {
        if (lhs.total_ns != rhs.total_ns) return lhs.total_ns < rhs.total_ns;
        for (std::uint32_t stage = 0u; stage < stage_count; ++stage) {
            const auto left = request.stages[stage].candidates[
                lhs.candidate_indices[stage]].identity;
            const auto right = request.stages[stage].candidates[
                rhs.candidate_indices[stage]].identity;
            if (!operation_core::same_stable_id(left, right))
                return less_identity(left, right);
        }
        return false;
    };

    for (std::uint32_t candidate = 0u;
         candidate < request.stages[0].candidate_count; ++candidate) {
        total_cost cost{};
        if (!candidate_cost(0u, candidate, &cost)) continue;
        previous[candidate].reachable = true;
        previous[candidate].path_count = 1u;
        previous[candidate].total_ns = cost.amortized_total_ns;
        previous[candidate].candidate_indices[0] = candidate;
        previous[candidate].stage_costs[0] = cost;
        previous[candidate].empirical_required =
            (request.stages[0].candidates[candidate].flags
                & planner_candidate_empirical_required) != 0u;
    }
    for (std::uint32_t stage = 1u; stage < request.stage_count; ++stage) {
        for (auto &state : current) state = {};
        for (std::uint32_t candidate = 0u;
             candidate < request.stages[stage].candidate_count; ++candidate) {
            total_cost isolated{};
            if (!candidate_cost(stage, candidate, &isolated)) continue;
            for (std::uint32_t producer = 0u;
                 producer < request.stages[stage - 1u].candidate_count;
                 ++producer) {
                if (!previous[producer].reachable) continue;
                total_cost transition{};
                if (!find_transition(stage - 1u,
                        request.stages[stage - 1u].candidates[producer].identity,
                        request.stages[stage].candidates[candidate].identity,
                        &transition)) continue;
                current[candidate].path_count += previous[producer].path_count;
                path_state proposal = previous[producer];
                proposal.reachable = true;
                proposal.path_count = current[candidate].path_count;
                proposal.candidate_indices[stage] = candidate;
                proposal.stage_costs[stage] = isolated;
                proposal.total_ns += isolated.amortized_total_ns
                    + transition.amortized_total_ns;
                proposal.empirical_required = proposal.empirical_required
                    || (request.stages[stage].candidates[candidate].flags
                        & planner_candidate_empirical_required) != 0u;
                if (!current[candidate].reachable
                    || path_less(proposal, current[candidate], stage + 1u))
                    current[candidate] = proposal;
            }
        }
        for (std::uint32_t index = 0u;
             index < maximum_connected_stage_candidates; ++index)
            previous[index] = current[index];
    }

    std::uint32_t final_order[maximum_connected_stage_candidates]{};
    std::uint32_t final_count = 0u;
    std::uint64_t legal_path_count = 0u;
    for (std::uint32_t candidate = 0u;
         candidate < request.stages[request.stage_count - 1u].candidate_count;
         ++candidate)
        if (previous[candidate].reachable) {
            final_order[final_count++] = candidate;
            legal_path_count += previous[candidate].path_count;
        }
    out->legal_path_count = legal_path_count
            > std::numeric_limits<std::uint32_t>::max()
        ? std::numeric_limits<std::uint32_t>::max()
        : static_cast<std::uint32_t>(legal_path_count);
    if (final_count == 0u) {
        out->status = {planner_status_code::no_legal_candidate,
            "no connected candidate path satisfies contracts"};
        out->reason = "all connected paths were rejected";
        return out->status;
    }
    std::sort(final_order, final_order + final_count,
        [&](std::uint32_t lhs, std::uint32_t rhs) noexcept {
            return path_less(previous[lhs], previous[rhs], request.stage_count);
        });
    out->shortlist_count = std::min(
        final_count, request.shortlist_size);

    const auto fill_path = [&](const path_state &state,
                               connected_plan_path *path) noexcept {
        *path = connected_plan_path{};
        path->stage_count = request.stage_count;
        for (std::uint32_t stage = 0u; stage < request.stage_count; ++stage) {
            const planner_candidate &candidate = request.stages[stage].candidates[
                state.candidate_indices[stage]];
            path->candidates[stage] = candidate.identity;
            path->projections[stage] = {candidate.projection.persistent,
                candidate.projection.kind, candidate.projection.schema_version,
                candidate.projection.variant};
        }
    };
    const auto same_path = [&](const connected_plan_path &lhs,
                               const connected_plan_path &rhs) noexcept {
        if (lhs.stage_count != rhs.stage_count) return false;
        for (std::uint32_t stage = 0u; stage < lhs.stage_count; ++stage)
            if (!operation_core::same_stable_id(
                    lhs.candidates[stage], rhs.candidates[stage])
                || !execution::same_identity(lhs.projections[stage].identity,
                    rhs.projections[stage].identity)
                || lhs.projections[stage].kind != rhs.projections[stage].kind
                || lhs.projections[stage].schema_version
                    != rhs.projections[stage].schema_version
                || lhs.projections[stage].variant
                    != rhs.projections[stage].variant)
                return false;
        return true;
    };
    const auto select_state = [&](const path_state &state,
                                  selection_source source,
                                  double empirical_ns) noexcept {
        fill_path(state, &out->winner);
        for (std::uint32_t stage = 0u; stage < request.stage_count; ++stage) {
            out->stages[stage].candidate = &request.stages[stage].candidates[
                state.candidate_indices[stage]];
            out->stages[stage].analytical = state.stage_costs[stage];
        }
        out->source = source;
        out->analytical_total_ns = state.total_ns;
        out->empirical_total_ns = empirical_ns;
        out->empirical_required = request.force_empirical
            || state.empirical_required;
    };

    if (request.cache.lookup != nullptr) {
        connected_plan_cache_entry cached{};
        if (!request.cache.lookup(request.cache.context,
                durable_keys, &cached)) {
            out->cache = cache_state::miss;
        } else if (!cached.occupied
            || !same_connected_planning_keys(cached.keys, durable_keys)
            || cached.evidence.evidence_revision
                != request.current_evidence_revision
            || cached.evidence.sample_count == 0u
            || !finite_nonnegative(cached.evidence.median_total_ns)
            || !finite_nonnegative(cached.evidence.spread_percent)
            || cached.evidence.spread_percent
                > request.maximum_spread_percent
            || cached.evidence.practical_tolerance_percent
                != request.practical_tolerance_percent
            || !std::isfinite(cached.evidence.confidence)
            || cached.evidence.confidence
                < request.minimum_cache_confidence) {
            out->cache = cache_state::stale;
        } else {
            for (std::uint32_t rank = 0u;
                 rank < final_count; ++rank) {
                connected_plan_path path{};
                fill_path(previous[final_order[rank]], &path);
                if (!same_path(path, cached.winner)) continue;
                out->cache = cache_state::hit;
                select_state(previous[final_order[rank]],
                    selection_source::cache,
                    cached.evidence.median_total_ns);
                out->confidence = cached.evidence.confidence;
                out->reason = "fresh measured connected path remains legal";
                out->status = {};
                return {};
            }
            out->cache = cache_state::winner_unavailable;
        }
    }

    const path_state &analytical = previous[final_order[0]];
    const bool measurement_required = request.force_empirical
        || analytical.empirical_required;
    if (request.measurement.measure == nullptr
        || request.maximum_measurements == 0u) {
        if (measurement_required) {
            out->status = {planner_status_code::no_correct_measurement,
                "connected path requires empirical measurement"};
            out->reason = "uncertain connected objective was not final authority";
            return out->status;
        }
        select_state(analytical, selection_source::analytical, 0.0);
        out->reason = "bounded connected analytical path selected";
        out->status = {};
        return {};
    }

    struct measured_path {
        std::uint32_t final_candidate = 0u;
        std::uint32_t sample_count = 0u;
        double total_ns = 0.0;
        double spread_percent = 0.0;
    } measured[maximum_connected_stage_candidates]{};
    std::uint32_t measured_count = 0u;
    const std::uint32_t budget = std::min(
        out->shortlist_count, request.maximum_measurements);
    for (std::uint32_t rank = 0u; rank < budget; ++rank) {
        connected_plan_path path{};
        fill_path(previous[final_order[rank]], &path);
        measured_connected_plan evidence{};
        ++out->measurement_count;
        if (!request.measurement.measure(request.measurement.context,
                path, &evidence)
            || !evidence.correct || evidence.contaminated
            || evidence.sample_count == 0u
            || !finite_nonnegative(evidence.amortized_total_ns)
            || !finite_nonnegative(evidence.spread_percent)
            || evidence.spread_percent > request.maximum_spread_percent)
            continue;
        measured[measured_count++] = {final_order[rank],
            evidence.sample_count, evidence.amortized_total_ns,
            evidence.spread_percent};
    }
    if (measured_count == 0u) {
        if (request.allow_analytical_fallback_after_measurement_failure
            && !measurement_required) {
            select_state(analytical, selection_source::analytical, 0.0);
            out->reason = "connected measurements failed; analytical path was not persisted";
            out->status = {};
            return {};
        }
        out->status = {planner_status_code::no_correct_measurement,
            "bounded connected tuning produced no clean measurement"};
        out->reason = "no measured connected path was safe to persist";
        return out->status;
    }
    std::sort(measured, measured + measured_count,
        [&](const measured_path &lhs, const measured_path &rhs) noexcept {
            if (lhs.total_ns != rhs.total_ns) return lhs.total_ns < rhs.total_ns;
            return path_less(previous[lhs.final_candidate],
                previous[rhs.final_candidate], request.stage_count);
        });
    std::uint32_t selected = 0u;
    for (std::uint32_t rank = 1u; rank < measured_count; ++rank) {
        if (!within_tolerance(measured[0].total_ns, measured[rank].total_ns,
                request.practical_tolerance_percent)) break;
        if (path_less(previous[measured[rank].final_candidate],
                previous[measured[selected].final_candidate],
                request.stage_count)) selected = rank;
    }
    const measured_path &winner = measured[selected];
    select_state(previous[winner.final_candidate],
        selection_source::empirical, winner.total_ns);
    const double sample_factor = std::min(
        1.0, static_cast<double>(winner.sample_count) / 5.0);
    const double spread_factor = std::max(0.0,
        1.0 - winner.spread_percent
            / std::max(1.0, request.maximum_spread_percent));
    double separation_factor = 0.5;
    if (measured_count > 1u) {
        const measured_path &runner = measured[selected == 0u ? 1u : 0u];
        const double scale = std::max(
            std::fabs(winner.total_ns), std::fabs(runner.total_ns));
        const double separation = scale == 0.0 ? 0.0
            : std::fabs(runner.total_ns - winner.total_ns) * 100.0 / scale;
        separation_factor = std::min(1.0, separation
            / std::max(1.0, request.practical_tolerance_percent));
    }
    out->confidence = 0.5 * sample_factor + 0.3 * spread_factor
        + 0.2 * separation_factor;
    out->reason = "measured connected path won total end-to-end cost";
    out->status = {};
    if (request.cache.store != nullptr) {
        const connected_plan_cache_entry entry{durable_keys, out->winner,
            {request.current_evidence_revision, winner.sample_count, 0u,
                winner.total_ns, winner.spread_percent, out->confidence,
                request.practical_tolerance_percent}, true};
        if (!request.cache.store(request.cache.context, entry))
            out->cache_store_failed = true;
    }
    return {};
}

} // namespace cellerator::planner
