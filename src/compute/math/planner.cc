#include <Cellerator/compute/math/planner.hh>

#include <algorithm>
#include <limits>

namespace cellerator::compute::math {
namespace {

constexpr std::size_t capability_count = 10u;

void reject(
    planner_diagnostics *diagnostics,
    backend_capability capability) noexcept {
    const std::size_t code = static_cast<std::size_t>(capability.code);
    if (code < capability_count) ++diagnostics->rejection_count[code];
    if (!diagnostics->has_rejection) {
        diagnostics->first_rejection = capability;
        diagnostics->has_rejection = true;
    }
}

backend_capability malformed(const char *message) noexcept {
    return {capability_code::backend_unavailable,
        request_validation_code::ok,
        message};
}

spmm_request without_epilogue(const spmm_request &request) noexcept {
    spmm_request base = request;
    base.epilogue = epilogue_descriptor{};
    return base;
}

bool valid_epilogue_strategy(
    const spmm_request &request,
    epilogue_strategy_kind strategy) noexcept {
    if (request.epilogue.kind == epilogue_kind::none)
        return strategy == epilogue_strategy_kind::none;
    return strategy == epilogue_strategy_kind::backend_fused
        || strategy == epilogue_strategy_kind::generic_unfused;
}

backend_capability query_candidate(
    SpMMBackend *backend,
    const spmm_request &request,
    const DeviceCapabilities &device,
    candidate_origin *origin) noexcept {
    backend_capability capability = backend->query(request, device);
    if (capability
        && capability.epilogue_strategy
            == epilogue_strategy_kind::generic_unfused) {
        const backend_capability generic =
            query_generic_unfused_epilogue_capability(request);
        return generic ? capability : generic;
    }
    if (capability
        && valid_epilogue_strategy(request, capability.epilogue_strategy))
        return capability;

    const bool may_compose = request.epilogue.kind != epilogue_kind::none
        && (capability.code == capability_code::unsupported_epilogue
            || (capability
                && capability.epilogue_strategy == epilogue_strategy_kind::none));
    if (!may_compose) {
        if (capability && !valid_epilogue_strategy(
                request, capability.epilogue_strategy)) {
            return malformed("backend returned an invalid epilogue strategy");
        }
        return capability;
    }

    backend_capability base = backend->query(without_epilogue(request), device);
    if (!base) return base;
    if (base.epilogue_strategy != epilogue_strategy_kind::none)
        return malformed("backend base capability has an epilogue strategy");
    const backend_capability generic =
        query_generic_unfused_epilogue_capability(request);
    if (!generic) return generic;
    base.workspace_bytes = std::max(
        base.workspace_bytes, generic.workspace_bytes);
    base.epilogue_strategy = epilogue_strategy_kind::generic_unfused;
    *origin = candidate_origin::generic_epilogue_composed;
    return base;
}

bool workspace_legal(
    const spmm_request &request,
    const DeviceCapabilities &device,
    const planner_policy &policy,
    backend_capability *capability,
    bool *structural) noexcept {
    const u64 bytes = capability->workspace_bytes;
    const workspace_policy &workspace = request.workspace;
    if ((workspace.kind == workspace_policy_kind::no_additional_workspace
            && bytes != 0u)
        || (workspace.kind == workspace_policy_kind::caller_limit
            && bytes > workspace.byte_limit)) {
        capability->code = capability_code::workspace_policy_rejected;
        capability->message = "backend workspace exceeds request policy";
        return false;
    }
    u64 physical_limit = device.total_global_memory_bytes;
    if (policy.workspace_soft_limit_bytes != 0u)
        physical_limit = physical_limit == 0u
            ? policy.workspace_soft_limit_bytes
            : std::min(physical_limit, policy.workspace_soft_limit_bytes);
    if (physical_limit != 0u && bytes > physical_limit) {
        capability->code = capability_code::workspace_policy_rejected;
        capability->message = "backend workspace exceeds structural planner limit";
        *structural = true;
        return false;
    }
    return true;
}

bool implemented_candidate(
    SpMMBackend *backend,
    const backend_capability &capability) noexcept {
    return backend != nullptr && backend->identity() != 0u
        && capability.algorithm_identity != 0u
        && capability.kernel_variant_identity != 0u;
}

bool candidate_less(
    const planner_candidate &lhs,
    const planner_candidate &rhs) noexcept {
    const auto epilogue_rank = [](epilogue_strategy_kind strategy) noexcept {
        return strategy == epilogue_strategy_kind::generic_unfused ? 1u : 0u;
    };
    const auto preprocess_rank = [](preprocessing_kind kind) noexcept {
        return kind == preprocessing_kind::none ? 0u : 1u;
    };
    const u32 lhs_epilogue = epilogue_rank(lhs.capability.epilogue_strategy);
    const u32 rhs_epilogue = epilogue_rank(rhs.capability.epilogue_strategy);
    if (lhs_epilogue != rhs_epilogue) return lhs_epilogue < rhs_epilogue;
    const u32 lhs_preprocess = preprocess_rank(lhs.capability.preprocessing);
    const u32 rhs_preprocess = preprocess_rank(rhs.capability.preprocessing);
    if (lhs_preprocess != rhs_preprocess) return lhs_preprocess < rhs_preprocess;
    if (lhs.capability.workspace_bytes != rhs.capability.workspace_bytes)
        return lhs.capability.workspace_bytes < rhs.capability.workspace_bytes;
    if (lhs.backend->identity() != rhs.backend->identity())
        return lhs.backend->identity() < rhs.backend->identity();
    if (lhs.capability.algorithm_identity != rhs.capability.algorithm_identity)
        return lhs.capability.algorithm_identity
            < rhs.capability.algorithm_identity;
    return lhs.capability.kernel_variant_identity
        < rhs.capability.kernel_variant_identity;
}

execution_plan make_plan(
    const spmm_request &request,
    const DeviceFingerprint &fingerprint,
    const planner_candidate &candidate) noexcept {
    execution_plan plan;
    plan.operation = make_operation_signature(request);
    plan.physical_view_schema_version =
        candidate.capability.physical_view_schema_version;
    plan.backend_identity = candidate.backend->identity();
    plan.algorithm_identity = candidate.capability.algorithm_identity;
    plan.kernel_variant_identity = candidate.capability.kernel_variant_identity;
    plan.workspace_bytes = candidate.capability.workspace_bytes;
    plan.preprocessing = candidate.capability.preprocessing;
    plan.epilogue_strategy = candidate.capability.epilogue_strategy;
    plan.device_fingerprint = detail::device_fingerprint_identity(fingerprint);
    plan.toolchain_fingerprint = detail::toolchain_fingerprint_identity(fingerprint);
    plan.tuning_identity = candidate.capability.tuning_identity;
    return plan;
}

bool same_signature(
    const operation_signature &lhs,
    const operation_signature &rhs) noexcept {
    return lhs.schema_version == rhs.schema_version
        && lhs.operation == rhs.operation
        && lhs.low == rhs.low && lhs.high == rhs.high;
}

bool same_candidate_plan(
    const execution_plan &cached,
    const execution_plan &legal) noexcept {
    return cached.schema_version == execution_plan_schema_version
        && same_signature(cached.operation, legal.operation)
        && cached.physical_view_schema_version
            == legal.physical_view_schema_version
        && cached.backend_identity == legal.backend_identity
        && cached.algorithm_identity == legal.algorithm_identity
        && cached.kernel_variant_identity == legal.kernel_variant_identity
        && cached.workspace_bytes == legal.workspace_bytes
        && cached.preprocessing == legal.preprocessing
        && cached.epilogue_strategy == legal.epilogue_strategy
        && cached.device_fingerprint == legal.device_fingerprint
        && cached.toolchain_fingerprint == legal.toolchain_fingerprint
        && cached.tuning_identity == legal.tuning_identity;
}

planner_status no_candidate_status(
    const planner_diagnostics &diagnostics) noexcept {
    if (diagnostics.has_rejection) {
        return {planner_status_code::no_legal_candidate,
            diagnostics.first_rejection.validation,
            diagnostics.first_rejection.code,
            diagnostics.first_rejection.message};
    }
    return {planner_status_code::no_legal_candidate,
        request_validation_code::ok,
        capability_code::backend_unavailable,
        "no SpMM backend is registered"};
}

} // namespace

planner_status plan_spmm(
    const planner_input &input,
    planner_result *out) noexcept {
    if (out == nullptr) {
        return {planner_status_code::invalid_argument,
            request_validation_code::ok,
            capability_code::supported,
            "planner requires an output"};
    }
    *out = planner_result{};
    if (input.request == nullptr || input.device == nullptr
        || input.fingerprint == nullptr || input.registry == nullptr) {
        return {planner_status_code::invalid_argument,
            request_validation_code::ok,
            capability_code::supported,
            "planner requires request, device, fingerprint, and registry"};
    }
    const request_validation_result validation =
        validate_math_request(*input.request);
    if (!validation) {
        return {planner_status_code::invalid_request,
            validation.code,
            capability_code::invalid_request,
            validation.message};
    }
    if (input.device->schema_version != device_math_runtime_schema_version
        || input.fingerprint->schema_version != device_math_runtime_schema_version
        || input.device->device_ordinal != input.fingerprint->device_ordinal) {
        return {planner_status_code::invalid_device,
            request_validation_code::ok,
            capability_code::unsupported_device,
            "planner device identity is invalid"};
    }
    if (input.policy.candidate_limit > max_planner_candidate_count) {
        return {planner_status_code::invalid_argument,
            request_validation_code::ok,
            capability_code::supported,
            "planner candidate limit exceeds fixed capacity"};
    }

    const spmm_request &request = input.request->operation;
    out->trivial = classify_trivial_operation(request);
    out->plan.operation = make_operation_signature(request);
    out->plan.device_fingerprint =
        detail::device_fingerprint_identity(*input.fingerprint);
    out->plan.toolchain_fingerprint =
        detail::toolchain_fingerprint_identity(*input.fingerprint);
    if (out->trivial != trivial_operation_kind::none) return {};

    planner_diagnostics &diagnostics = out->diagnostics;
    diagnostics.registered_backend_count = input.registry->size();
    const std::size_t candidate_limit = input.policy.candidate_limit == 0u
        ? max_planner_candidate_count
        : input.policy.candidate_limit;
    for (std::size_t index = 0u; index < input.registry->size(); ++index) {
        SpMMBackend *const backend = input.registry->at(index);
        ++diagnostics.queried_backend_count;
        if (backend == nullptr || backend->identity() == 0u) {
            ++diagnostics.malformed_candidate_count;
            reject(&diagnostics, malformed("registry contains an invalid backend"));
            continue;
        }
        candidate_origin origin = candidate_origin::backend_native;
        backend_capability capability = query_candidate(
            backend, request, *input.device, &origin);
        if (!capability) {
            reject(&diagnostics, capability);
            continue;
        }
        bool structural = false;
        if (!workspace_legal(request, *input.device, input.policy,
                &capability, &structural)) {
            if (structural) ++diagnostics.structurally_pruned_count;
            reject(&diagnostics, capability);
            continue;
        }
        if (!implemented_candidate(backend, capability)) {
            ++diagnostics.malformed_candidate_count;
            reject(&diagnostics,
                malformed("backend capability lacks executable identities"));
            continue;
        }
        out->candidates[out->candidate_count++] = {backend, capability, origin};
        if (origin == candidate_origin::generic_epilogue_composed)
            ++diagnostics.composed_epilogue_count;
    }
    if (out->candidate_count == 0u) return no_candidate_status(diagnostics);

    std::sort(out->candidates, out->candidates + out->candidate_count,
        candidate_less);
    if (out->candidate_count > candidate_limit) {
        diagnostics.structurally_pruned_count +=
            out->candidate_count - candidate_limit;
        out->candidate_count = candidate_limit;
    }
    diagnostics.legal_candidate_count = out->candidate_count;
    execution_plan legal_plans[max_planner_candidate_count]{};
    for (std::size_t index = 0u; index < out->candidate_count; ++index)
        legal_plans[index] = make_plan(
            request, *input.fingerprint, out->candidates[index]);

    std::size_t selected = 0u;
    if (input.cache.lookup != nullptr) {
        execution_plan cached;
        const operation_signature signature = make_operation_signature(request);
        const u64 device_identity =
            detail::device_fingerprint_identity(*input.fingerprint);
        const u64 toolchain_identity =
            detail::toolchain_fingerprint_identity(*input.fingerprint);
        if (!input.cache.lookup(input.cache.context, signature,
                device_identity, toolchain_identity, &cached)) {
            out->cache_state = cache_lookup_state::miss;
        } else {
            out->cache_state = cache_lookup_state::stale_or_illegal;
            for (std::size_t index = 0u; index < out->candidate_count; ++index) {
                if (!same_candidate_plan(cached, legal_plans[index])) continue;
                selected = index;
                out->cache_state = cache_lookup_state::hit;
                break;
            }
        }
    }
    out->selected_index = selected;
    out->selected_backend = out->candidates[selected].backend;
    out->plan = legal_plans[selected];
    return {};
}

} // namespace cellerator::compute::math
