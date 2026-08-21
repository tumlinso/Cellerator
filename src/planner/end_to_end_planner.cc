#include <Cellerator/planner/end_to_end_planner.hh>

#include <algorithm>
#include <cmath>

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
        && finite_nonnegative(phases.h2d_ns)
        && finite_nonnegative(phases.dynamic_input_pack_ns)
        && finite_nonnegative(phases.kernel_ns)
        && finite_nonnegative(phases.epilogue_ns)
        && finite_nonnegative(phases.order_transform_ns)
        && finite_nonnegative(phases.synchronization_ns)
        && finite_nonnegative(phases.communication_ns)
        && finite_nonnegative(phases.d2h_ns);
}

bool same_persistent_identity(
    const execution::structure_id &lhs,
    const execution::structure_id &rhs) noexcept {
    return execution::same_identity(lhs, rhs);
}

bool same_projection_key(
    const operation_core::projection_key &lhs,
    const operation_core::projection_key &rhs) noexcept {
    return execution::same_identity(lhs.persistent, rhs.persistent)
        && execution::same_handle(lhs.runtime, rhs.runtime)
        && lhs.kind == rhs.kind && lhs.schema_version == rhs.schema_version
        && lhs.variant == rhs.variant;
}

bool same_structure_key(
    const operation_core::structure_key &lhs,
    const operation_core::structure_key &rhs) noexcept {
    return same_persistent_identity(lhs.persistent, rhs.persistent)
        && execution::same_handle(lhs.runtime, rhs.runtime)
        && lhs.epoch.value == rhs.epoch.value;
}

bool same_geometry_key(
    const semantic_geometry_key &lhs,
    const semantic_geometry_key &rhs) noexcept {
    return execution::same_identity(lhs.geometry, rhs.geometry)
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
    const planner_result &result) noexcept {
    for (std::uint32_t index = 0u; index < request.candidate_count; ++index)
        if (result.diagnostics[index].rejection == candidate_rejection::none
            && operation_core::same_stable_id(
                request.candidates[index].identity, identity))
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

} // namespace

planner_status compute_total_cost(
    const phase_costs &phases,
    std::uint64_t structure_reuse,
    std::uint64_t projection_reuse,
    total_cost *out) noexcept {
    if (out == nullptr || structure_reuse == 0u || projection_reuse == 0u)
        return {planner_status_code::invalid_argument,
            "cost accounting requires output and nonzero reuse"};
    *out = total_cost{};
    if (!valid_phases(phases))
        return {planner_status_code::invalid_cost,
            "phase costs must be finite and nonnegative"};
    const double structure = static_cast<double>(structure_reuse);
    const double projection = static_cast<double>(projection_reuse);
    const double total = phases.host_preparation_ns
        + phases.semantic_packing_ns / structure
        + phases.projection_construction_ns / projection
        + phases.backend_prepare_ns / projection
        + phases.h2d_ns + phases.dynamic_input_pack_ns + phases.kernel_ns
        + phases.epilogue_ns + phases.order_transform_ns
        + phases.synchronization_ns + phases.communication_ns + phases.d2h_ns;
    if (!finite_nonnegative(total))
        return {planner_status_code::invalid_cost,
            "amortized total cost overflowed"};
    out->phases = phases;
    out->structure_reuse = structure_reuse;
    out->projection_reuse = projection_reuse;
    out->amortized_total_ns = total;
    return {};
}

bool same_planning_keys(
    const planning_keys &lhs,
    const planning_keys &rhs) noexcept {
    return operation_core::same_stable_id(
            lhs.problem.identity, rhs.problem.identity)
        && same_structure_key(lhs.structure, rhs.structure)
        && same_geometry_key(lhs.geometry, rhs.geometry)
        && lhs.device.vendor == rhs.device.vendor
        && lhs.device.architecture_major == rhs.device.architecture_major
        && lhs.device.architecture_minor == rhs.device.architecture_minor
        && lhs.device.performance_class == rhs.device.performance_class
        && lhs.build.runtime == rhs.build.runtime
        && lhs.build.kernel_build == rhs.build.kernel_build
        && lhs.build.driver == rhs.build.driver
        && lhs.policy.structure_reuse == rhs.policy.structure_reuse
        && lhs.policy.projection_reuse == rhs.policy.projection_reuse
        && lhs.policy.numeric_policy == rhs.policy.numeric_policy
        && lhs.policy.determinism_policy == rhs.policy.determinism_policy
        && lhs.policy.output_order_policy == rhs.policy.output_order_policy
        && lhs.policy.graph_policy == rhs.policy.graph_policy;
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
        || !execution::valid_identity(request.keys.structure.persistent)
        || !execution::valid_handle(request.keys.structure.runtime)
        || request.keys.structure.epoch.value == 0u
        || !execution::valid_identity(request.keys.geometry.geometry)
        || !execution::valid_identity(request.keys.geometry.source_order)
        || !execution::valid_identity(request.keys.geometry.destination_order)
        || !execution::valid_identity(request.keys.geometry.partition)
        || request.keys.device.vendor == 0u
        || request.keys.device.performance_class == 0u
        || request.keys.build.runtime == 0u
        || request.keys.build.kernel_build == 0u
        || request.current_evidence_revision == 0u
        || request.keys.policy.structure_reuse == 0u
        || request.keys.policy.projection_reuse == 0u
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
                request, cached.winner, *out);
            if (cached_index == maximum_planner_candidates
                || !same_projection_key(cached.winner_projection,
                    request.candidates[cached_index].projection)) {
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
        && request.keys.policy.projection_reuse == 1u;
    const bool tune = request.measurement.measure != nullptr
        && request.policy.maximum_measurements != 0u
        && request.problem.logical_work_items
            >= request.policy.minimum_tuning_work_items
        && (!one_shot || request.policy.tune_one_shot);
    if (!tune) {
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
            &out->diagnostics[index].empirical);
        if (!cost) {
            out->diagnostics[index].rejection =
                candidate_rejection::measurement_failed;
            continue;
        }
        measured_order[measured_count++] = index;
    }
    if (measured_count == 0u) {
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
    out->confidence = std::max(0.0,
        1.0 - out->diagnostics[selected_index].spread_percent / 100.0);
    out->conventional_winner = out->diagnostics[selected_index].conventional;
    out->reason = out->conventional_winner
        ? "measured conventional fallback won end to end"
        : "measured candidate won end to end";
    out->status = {};

    if (request.cache.store != nullptr) {
        const plan_cache_entry entry{request.keys,
            out->winner,
            request.candidates[selected_index].projection,
            {request.current_evidence_revision,
                out->diagnostics[selected_index].sample_count,
                0u,
                out->diagnostics[selected_index].empirical.amortized_total_ns,
                out->diagnostics[selected_index].spread_percent,
                out->confidence,
                request.policy.practical_tolerance_percent},
            true};
        request.cache.store(request.cache.context, entry);
    }
    return {};
}

} // namespace cellerator::planner
