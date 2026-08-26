#include <Cellerator/compute/math/operation_core/preparation_factory.hh>

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
