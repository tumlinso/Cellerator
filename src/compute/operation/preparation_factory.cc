#include <Cellerator/compute/operation/preparation_factory.hh>

#include <cstddef>
#include <cstdint>

namespace cellerator::compute::math::core {
namespace {

operation_status fail(operation_status_code code, const char *message) noexcept {
    return {code, execution::binding_validation_code::ok, message};
}

std::uint64_t mix(std::uint64_t value) noexcept {
    value += 0x9e3779b97f4a7c15ull;
    value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27u)) * 0x94d049bb133111ebull;
    return value ^ (value >> 31u);
}

bool same_projection_contract_v2(
    const candidate_projection_contract_v2 &lhs,
    const candidate_projection_contract_v2 &rhs) noexcept {
    return same_stable_id(lhs.view_type, rhs.view_type)
        && lhs.abi_major == rhs.abi_major && lhs.abi_minor == rhs.abi_minor
        && lhs.schema_version == rhs.schema_version
        && lhs.variant == rhs.variant;
}

bool projection_matches_candidate_v2(
    const execution::activated_projection_reference_v2 &projection,
    const candidate_descriptor_v2 &candidate) noexcept {
    if (projection.schema_version
            != execution::activated_projection_reference_schema_version_v2
        || projection.record_bytes
            != sizeof(execution::activated_projection_reference_v2)
        || !execution::valid_identity(projection.key.persistent)
        || !execution::valid_handle(projection.key.runtime)
        || !execution::valid_location(projection.location)
        || projection.location.residency == execution::residency_kind::host
        || projection.view == nullptr || projection.view_bytes == 0u
        || !same_stable_id(
            projection.provider_identity, candidate.provider_identity)
        || projection.key.kind != candidate.candidate.projection
        || projection.key.schema_version
            != candidate.projection_contract.schema_version
        || projection.key.variant != candidate.projection_contract.variant
        || !same_projection_contract_v2(
            projection.contract, candidate.projection_contract))
        return false;
    for (std::uint32_t value : projection.reserved)
        if (value != 0u) return false;
    const bool names_capability =
        valid_catalog_identity_v2(candidate.capability_identity);
    const bool requires_capability =
        (candidate.flags & candidate_descriptor_requires_capability) != 0u;
    return (!names_capability && !requires_capability)
        || (valid_catalog_identity_v2(projection.capability_identity)
            && same_stable_id(projection.capability_identity,
                candidate.capability_identity));
}

runtime::session_cache_key plan_cache_key(
    const preparation_factory_request &request) noexcept {
    const auto &entry = *request.catalog_entry;
    const auto &structure = request.structures.structures[0];
    std::uint64_t low = mix(entry.identity.low ^ structure.persistent.low);
    low = mix(low ^ request.projection.persistent.low);
    low = mix(low ^ request.dense_width);
    std::uint64_t high = mix(entry.identity.high ^ structure.persistent.high);
    high = mix(high ^ request.projection.persistent.high);
    high = mix(high ^ structure.epoch.value);
    return {high, low};
}

runtime::session_cache_key plan_cache_key_v2(
    const candidate_preparation_adapter_v2 &adapter,
    const candidate_preparation_request_v2 &request,
    const execution::activated_projection_reference_v2 &projection) noexcept {
    const auto &structure = request.structures.structures[0];
    std::uint64_t low = mix(
        adapter.candidate->candidate.identity.low ^ structure.persistent.low);
    low = mix(low ^ projection.key.persistent.low);
    low = mix(low ^ request.dense_width);
    std::uint64_t high = mix(
        adapter.candidate->candidate.identity.high ^ structure.persistent.high);
    high = mix(high ^ projection.key.persistent.high);
    high = mix(high ^ structure.epoch.value);
    return {high, low};
}

operation_status validate_request(
    const preparation_factory_request &request,
    preparation_family expected) noexcept {
    if (request.schema_version != preparation_factory_schema_version
        || request.catalog_entry == nullptr || request.session == nullptr
        || request.state.data == nullptr)
        return fail(operation_status_code::invalid_argument,
            "preparation factory request is incomplete");
    const operation_status catalog = validate_built_in_candidate_catalog();
    if (!catalog) return catalog;
    const built_in_candidate_descriptor *canonical =
        find_built_in_candidate(request.catalog_entry->identity);
    if (canonical == nullptr || canonical != request.catalog_entry)
        return fail(operation_status_code::invalid_argument,
            "preparation requires a canonical built-in catalog entry");
    const auto &entry = *canonical;
    if (entry.preparation != expected)
        return fail(operation_status_code::unsupported_projection,
            "catalog preparation family does not match typed projection");
    if (request.projection.kind != entry.projection
        || request.projection.schema_version
            != entry.projection_schema_version
        || request.projection.variant != entry.projection_variant)
        return fail(operation_status_code::unsupported_projection,
            "projection key does not match catalog schema and variant");
    if (request.dense_width < entry.minimum_dense_width
        || request.dense_width > entry.maximum_dense_width)
        return fail(operation_status_code::unsupported_problem,
            "dense width is outside the catalog candidate regime");
    if (request.state.bytes < entry.state_bytes
        || reinterpret_cast<std::uintptr_t>(request.state.data)
            % entry.state_alignment != 0u)
        return fail(operation_status_code::preparation_failed,
            "caller-owned candidate state has invalid capacity or alignment");
    if (!request.session->initialized || request.session->sealed
        || request.session->device < 0)
        return fail(operation_status_code::preparation_failed,
            "preparation requires an initialized unsealed execution session");
    if (request.structures.count != 1u
        || request.structures.structures[0].epoch.value == 0u)
        return fail(operation_status_code::stale_structure,
            "preparation factory structure set is invalid or stale");
    return {};
}

operation_status cache_prepared_state(
    const preparation_factory_request &request) noexcept {
    const auto status = runtime::insert_session_cache(request.session,
        runtime::session_cache_kind::plan, plan_cache_key(request),
        request.state.data, request.structures.structures[0].epoch.value, 0u);
    if (status != runtime::session_status::success)
        return fail(operation_status_code::preparation_failed,
            "execution session plan cache rejected prepared candidate state");
    return {};
}

operation_status cache_prepared_state_v2(
    const candidate_preparation_adapter_v2 &adapter,
    const candidate_preparation_request_v2 &request,
    const execution::activated_projection_reference_v2 &projection) noexcept {
    const auto status = runtime::insert_session_cache(request.session,
        runtime::session_cache_kind::plan,
        plan_cache_key_v2(adapter, request, projection), request.state.data,
        request.structures.structures[0].epoch.value, 0u);
    if (status != runtime::session_status::success)
        return fail(operation_status_code::preparation_failed,
            "execution session plan cache rejected erased candidate state");
    return {};
}

operation_status prepare_row_masked_v2(
    const candidate_preparation_adapter_v2 &,
    const candidate_preparation_request_v2 &request,
    const execution::activated_projection_reference_v2 &projection,
    prepared_operation *prepared) noexcept {
    if (projection.view_bytes
            != sizeof(cellpack::persistent_packing_payload_view))
        return fail(operation_status_code::unsupported_projection,
            "row-masked activated view size is incompatible");
    return prepare_row_masked_n1_operation(request.problem,
        request.structures, projection.key, request.numeric, request.policy,
        *static_cast<const cellpack::persistent_packing_payload_view *>(
            projection.view),
        request.feature_axis, request.row_axis,
        static_cast<row_masked_n1_prepared_state *>(request.state.data),
        prepared);
}

operation_status prepare_csr_v2(
    const candidate_preparation_adapter_v2 &,
    const candidate_preparation_request_v2 &request,
    const execution::activated_projection_reference_v2 &projection,
    prepared_operation *prepared) noexcept {
    if (projection.view_bytes != sizeof(execution_csr_view))
        return fail(operation_status_code::unsupported_projection,
            "CSR activated view size is incompatible");
    return prepare_csr_fallback_operation(request.problem,
        request.structures, projection.key, request.numeric, request.policy,
        *static_cast<const execution_csr_view *>(projection.view),
        request.session->device, request.feature_axis, request.row_axis,
        static_cast<csr_fallback_prepared_state *>(request.state.data),
        prepared);
}

operation_status prepare_feature_major_small_n_v2(
    const candidate_preparation_adapter_v2 &,
    const candidate_preparation_request_v2 &request,
    const execution::activated_projection_reference_v2 &projection,
    prepared_operation *prepared) noexcept {
    if (projection.view_bytes != sizeof(feature_major_projection_view))
        return fail(operation_status_code::unsupported_projection,
            "feature-major activated view size is incompatible");
    return prepare_feature_major_small_n_operation(request.problem,
        request.structures, projection.key, request.numeric, request.policy,
        *static_cast<const feature_major_projection_view *>(projection.view),
        request.session->device, request.dense_width, request.feature_axis,
        request.row_axis, request.dense_column_axis,
        static_cast<feature_major_small_n_prepared_state *>(
            request.state.data),
        prepared);
}

operation_status prepare_feature_major_cta_v2(
    const candidate_preparation_adapter_v2 &,
    const candidate_preparation_request_v2 &request,
    const execution::activated_projection_reference_v2 &projection,
    prepared_operation *prepared) noexcept {
    if (projection.view_bytes != sizeof(feature_major_projection_view))
        return fail(operation_status_code::unsupported_projection,
            "feature-major activated view size is incompatible");
    return prepare_feature_major_cta_medium_n_operation(request.problem,
        request.structures, projection.key, request.numeric, request.policy,
        *static_cast<const feature_major_projection_view *>(projection.view),
        request.session->device, request.dense_width, request.feature_axis,
        request.row_axis, request.dense_column_axis,
        static_cast<feature_major_small_n_prepared_state *>(
            request.state.data),
        prepared);
}

operation_status prepare_transpose_v2(
    const candidate_preparation_adapter_v2 &,
    const candidate_preparation_request_v2 &request,
    const execution::activated_projection_reference_v2 &projection,
    prepared_operation *prepared) noexcept {
    if (projection.view_bytes != sizeof(transpose_projection_view))
        return fail(operation_status_code::unsupported_projection,
            "transpose activated view size is incompatible");
    return prepare_transpose_backward_n1_operation(request.problem,
        request.structures, projection.key, request.numeric, request.policy,
        *static_cast<const transpose_projection_view *>(projection.view),
        request.session->device, request.feature_axis, request.row_axis,
        request.dense_column_axis,
        static_cast<transpose_backward_prepared_state *>(request.state.data),
        prepared);
}

template<typename Prepare>
operation_status prepare_and_cache(
    const preparation_factory_request &request,
    preparation_family family,
    prepared_operation *prepared,
    Prepare &&prepare) noexcept {
    if (prepared == nullptr)
        return fail(operation_status_code::invalid_argument,
            "preparation factory output is null");
    const operation_status valid = validate_request(request, family);
    if (!valid) return valid;
    const operation_status status = prepare();
    if (!status) return status;
    const operation_status cached = cache_prepared_state(request);
    if (!cached) {
        *prepared = {};
        return cached;
    }
    return {};
}

} // namespace

operation_status validate_candidate_preparation_adapter_v2(
    const candidate_preparation_adapter_v2 &adapter) noexcept {
    if (adapter.schema_version
            != candidate_preparation_adapter_schema_version_v2
        || adapter.record_bytes != sizeof(candidate_preparation_adapter_v2)
        || adapter.candidate == nullptr || adapter.prepare == nullptr
        || adapter.reserved[0] != 0u || adapter.reserved[1] != 0u
        || validate_candidate_descriptor_v2(*adapter.candidate)
            != candidate_catalog_status_v2::success)
        return fail(operation_status_code::invalid_argument,
            "candidate preparation adapter is invalid");
    return {};
}

const candidate_preparation_adapter_v2 *find_candidate_preparation_adapter_v2(
    candidate_preparation_catalog_v2 catalog,
    stable_id candidate_identity) noexcept {
    if (catalog.entries == nullptr || catalog.entry_count == 0u
        || catalog.reserved != 0u)
        return nullptr;
    for (std::uint32_t index = 0u; index < catalog.entry_count; ++index) {
        const candidate_preparation_adapter_v2 &entry = catalog.entries[index];
        if (entry.candidate != nullptr
            && same_stable_id(entry.candidate->candidate.identity,
                candidate_identity))
            return &entry;
    }
    return nullptr;
}

candidate_preparation_catalog_v2
built_in_candidate_preparation_catalog_v2() noexcept {
    const candidate_descriptor_v2 *entries =
        built_in_candidate_catalog_fragment_v2().entries;
    static const candidate_preparation_adapter_v2 adapters[]{
        {candidate_preparation_adapter_schema_version_v2,
            sizeof(candidate_preparation_adapter_v2), &entries[0],
            prepare_row_masked_v2, {}},
        {candidate_preparation_adapter_schema_version_v2,
            sizeof(candidate_preparation_adapter_v2), &entries[1],
            prepare_csr_v2, {}},
        {candidate_preparation_adapter_schema_version_v2,
            sizeof(candidate_preparation_adapter_v2), &entries[2],
            prepare_feature_major_small_n_v2, {}},
        {candidate_preparation_adapter_schema_version_v2,
            sizeof(candidate_preparation_adapter_v2), &entries[3],
            prepare_feature_major_cta_v2, {}},
        {candidate_preparation_adapter_schema_version_v2,
            sizeof(candidate_preparation_adapter_v2), &entries[4],
            prepare_transpose_v2, {}}};
    return {adapters, builtin_candidate_count, 0u};
}

operation_status prepare_catalog_candidate_v2(
    const candidate_preparation_adapter_v2 &adapter,
    const candidate_preparation_request_v2 &request,
    const execution::activated_projection_reference_v2 &projection,
    prepared_operation *prepared) noexcept {
    if (prepared == nullptr)
        return fail(operation_status_code::invalid_argument,
            "erased preparation output is null");
    *prepared = {};
    const operation_status valid_adapter =
        validate_candidate_preparation_adapter_v2(adapter);
    if (!valid_adapter) return valid_adapter;
    if (request.schema_version
            != candidate_preparation_request_schema_version_v2
        || request.reserved != 0u || request.reserved2 != 0u
        || request.session == nullptr || request.dense_width == 0u)
        return fail(operation_status_code::invalid_argument,
            "erased preparation request is incomplete");
    if (!request.session->initialized || request.session->sealed
        || request.session->device < 0)
        return fail(operation_status_code::preparation_failed,
            "erased preparation requires an initialized unsealed session");
    if (request.structures.count != 1u
        || request.structures.structures[0].epoch.value == 0u)
        return fail(operation_status_code::stale_structure,
            "erased preparation structure set is invalid or stale");
    const candidate_descriptor_v2 &candidate = *adapter.candidate;
    if (request.dense_width < candidate.minimum_dense_width
        || (candidate.maximum_dense_width != 0u
            && request.dense_width > candidate.maximum_dense_width))
        return fail(operation_status_code::unsupported_problem,
            "dense width is outside the candidate-owned adapter regime");
    if (candidate.state_bytes != 0u
        && (request.state.data == nullptr
            || request.state.bytes < candidate.state_bytes
            || (candidate.state_alignment != 0u
                && reinterpret_cast<std::uintptr_t>(request.state.data)
                    % candidate.state_alignment != 0u)))
        return fail(operation_status_code::preparation_failed,
            "caller-owned erased candidate state is invalid");
    if (!projection_matches_candidate_v2(projection, candidate))
        return fail(operation_status_code::unsupported_projection,
            "activated projection does not match selected catalog entry");

    const operation_status status =
        adapter.prepare(adapter, request, projection, prepared);
    if (!status) {
        *prepared = {};
        return status;
    }
    if (candidate.state_bytes == 0u) return {};
    const operation_status cached =
        cache_prepared_state_v2(adapter, request, projection);
    if (!cached) *prepared = {};
    return cached;
}

operation_status prepare_catalog_row_masked(
    const preparation_factory_request &request,
    const cellpack::persistent_packing_payload_view &projection,
    prepared_operation *prepared) noexcept {
    return prepare_and_cache(request, preparation_family::row_masked_n1,
        prepared, [&]() noexcept {
            return prepare_row_masked_n1_operation(request.problem,
                request.structures, request.projection, request.numeric,
                request.policy, projection, request.feature_axis,
                request.row_axis,
                static_cast<row_masked_n1_prepared_state *>(request.state.data),
                prepared);
        });
}

operation_status prepare_catalog_csr(
    const preparation_factory_request &request,
    const execution_csr_view &projection,
    prepared_operation *prepared) noexcept {
    return prepare_and_cache(request, preparation_family::csr_n1,
        prepared, [&]() noexcept {
            return prepare_csr_fallback_operation(request.problem,
                request.structures, request.projection, request.numeric,
                request.policy, projection, request.session->device,
                request.feature_axis, request.row_axis,
                static_cast<csr_fallback_prepared_state *>(request.state.data),
                prepared);
        });
}

operation_status prepare_catalog_feature_major(
    const preparation_factory_request &request,
    const feature_major_projection_view &projection,
    prepared_operation *prepared) noexcept {
    if (request.catalog_entry == nullptr)
        return fail(operation_status_code::invalid_argument,
            "feature-major preparation lacks a catalog entry");
    const preparation_family family = request.catalog_entry->preparation;
    if (family != preparation_family::feature_major_small_n
        && family != preparation_family::feature_major_cta_medium_n)
        return fail(operation_status_code::unsupported_projection,
            "catalog entry is not a feature-major preparation family");
    return prepare_and_cache(request, family, prepared, [&]() noexcept {
        auto *state = static_cast<feature_major_small_n_prepared_state *>(
            request.state.data);
        if (family == preparation_family::feature_major_small_n)
            return prepare_feature_major_small_n_operation(request.problem,
                request.structures, request.projection, request.numeric,
                request.policy, projection, request.session->device,
                request.dense_width, request.feature_axis, request.row_axis,
                request.dense_column_axis, state, prepared);
        return prepare_feature_major_cta_medium_n_operation(request.problem,
            request.structures, request.projection, request.numeric,
            request.policy, projection, request.session->device,
            request.dense_width, request.feature_axis, request.row_axis,
            request.dense_column_axis, state, prepared);
    });
}

operation_status prepare_catalog_transpose(
    const preparation_factory_request &request,
    const transpose_projection_view &projection,
    prepared_operation *prepared) noexcept {
    return prepare_and_cache(request, preparation_family::transpose_backward_n1,
        prepared, [&]() noexcept {
            return prepare_transpose_backward_n1_operation(request.problem,
                request.structures, request.projection, request.numeric,
                request.policy, projection, request.session->device,
                request.feature_axis, request.row_axis,
                request.dense_column_axis,
                static_cast<transpose_backward_prepared_state *>(
                    request.state.data),
                prepared);
        });
}

} // namespace cellerator::compute::math::core
