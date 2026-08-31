#include <Cellerator/execution/geometry_acquisition_v2/assembly.hh>

namespace cellerator::execution::acquisition_v2 {
namespace {

const compiled_provider *find_provider(
    const provider_registry &registry, std::uint32_t kind) noexcept {
    for (std::uint64_t index = 0; index < registry.provider_count; ++index) {
        if (registry.providers[index].provider_kind == kind) {
            return &registry.providers[index];
        }
    }
    return nullptr;
}

const catalog_candidate *find_candidate(
    const candidate_catalog &catalog, stable_identity identity) noexcept {
    for (std::uint64_t index = 0; index < catalog.candidate_count; ++index) {
        if (catalog.candidates[index].identity.low == identity.low
            && catalog.candidates[index].identity.high == identity.high) {
            return &catalog.candidates[index];
        }
    }
    return nullptr;
}

status validate_request_against_assembly(
    const default_assembly &assembly, const acquisition_request &request) noexcept {
    for (std::uint64_t index = 0; index < request.projection_requirement_count; ++index) {
        const projection_requirement &required = request.projection_requirements[index];
        const catalog_candidate *candidate = find_candidate(assembly.catalog, required.candidate);
        if (candidate == nullptr || candidate->provider_kind != required.provider_kind
            || candidate->projection_kind != required.projection_kind
            || (candidate->experimental && !assembly.include_experimental)) {
            return {status_code::invalid_argument, index};
        }
    }
    return {};
}

}  // namespace

status validate_default_assembly(const default_assembly &assembly) noexcept {
    if (!valid_stable_identity(assembly.registry.identity)
        || !valid_stable_identity(assembly.catalog.identity)
        || !valid_stable_identity(assembly.planner.identity)
        || assembly.registry.providers == nullptr || assembly.registry.provider_count == 0
        || assembly.catalog.candidates == nullptr || assembly.catalog.candidate_count == 0
        || assembly.planner.facade.query == nullptr
        || assembly.planner.facade.acquire == nullptr) {
        return {status_code::invalid_argument, 0};
    }
    std::uint64_t primary_count = 0;
    for (std::uint64_t index = 0; index < assembly.registry.provider_count; ++index) {
        const compiled_provider &provider = assembly.registry.providers[index];
        if (!valid_stable_identity(provider.identity) || provider.provider_kind == 0) {
            return {status_code::invalid_identity, index};
        }
        primary_count += provider.primary ? 1u : 0u;
        for (std::uint64_t prior = 0; prior < index; ++prior) {
            if (assembly.registry.providers[prior].provider_kind == provider.provider_kind) {
                return {status_code::invalid_argument, index};
            }
        }
    }
    if (primary_count != 1) {
        return {status_code::invalid_argument, primary_count};
    }
    for (std::uint64_t index = 0; index < assembly.catalog.candidate_count; ++index) {
        const catalog_candidate &candidate = assembly.catalog.candidates[index];
        if (!valid_stable_identity(candidate.identity)
            || find_provider(assembly.registry, candidate.provider_kind) == nullptr) {
            return {status_code::invalid_identity, index};
        }
        for (std::uint64_t prior = 0; prior < index; ++prior) {
            const stable_identity prior_id = assembly.catalog.candidates[prior].identity;
            if (prior_id.low == candidate.identity.low
                && prior_id.high == candidate.identity.high) {
                return {status_code::invalid_identity, index};
            }
        }
    }
    return {};
}

status query_default_assembly(const default_assembly &assembly,
    const acquisition_request &request,
    acquisition_requirements *requirements) noexcept {
    const status assembly_status = validate_default_assembly(assembly);
    if (!assembly_status) {
        return assembly_status;
    }
    const status request_status = validate_request_against_assembly(assembly, request);
    if (!request_status) {
        return request_status;
    }
    return query_requirements(assembly.planner.facade, request, requirements);
}

status acquire_default_assembly(const default_assembly &assembly,
    const acquisition_request &request,
    const acquisition_requirements &requirements,
    const acquisition_buffers &buffers,
    acquired_geometry *result) noexcept {
    const status assembly_status = validate_default_assembly(assembly);
    if (!assembly_status) {
        return assembly_status;
    }
    const status request_status = validate_request_against_assembly(assembly, request);
    if (!request_status) {
        return request_status;
    }
    return acquire(assembly.planner.facade, request, requirements, buffers, result);
}

}  // namespace cellerator::execution::acquisition_v2
