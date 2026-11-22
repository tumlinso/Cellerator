#include <Cellerator/compute/operation/preparation_factory.hh>

#include <cstddef>
#include <cstdint>

namespace cellerator::compute::math::core {
namespace {

operation_status fail(operation_status_code code, const char *message) noexcept {
    return {code, execution::binding_validation_code::ok, message};
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
    if (projection.view_bytes != sizeof(::cellerator::layout::projection::CsrViewV1))
        return fail(operation_status_code::unsupported_projection,
            "CSR activated view size is incompatible");
    return prepare_csr_fallback_operation(request.problem,
        request.structures, projection.key, request.numeric, request.policy,
        *static_cast<const ::cellerator::layout::projection::CsrViewV1 *>(projection.view),
        request.session->device, request.feature_axis, request.row_axis,
        static_cast<csr_fallback_prepared_state *>(request.state.data),
        prepared);
}

operation_status prepare_feature_major_small_n_v2(
    const candidate_preparation_adapter_v2 &,
    const candidate_preparation_request_v2 &request,
    const execution::activated_projection_reference_v2 &projection,
    prepared_operation *prepared) noexcept {
    if (projection.view_bytes != sizeof(::cellerator::layout::projection::FeatureMajorViewV1))
        return fail(operation_status_code::unsupported_projection,
            "feature-major activated view size is incompatible");
    const execution::FeatureMajorBinding binding{
        *static_cast<const ::cellerator::layout::projection::FeatureMajorViewV1 *>(
            projection.view),
        request.structures.structures[0].runtime, projection.key.runtime};
    return prepare_feature_major_small_n_operation(request.problem,
        request.structures, projection.key, request.numeric, request.policy,
        binding,
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
    if (projection.view_bytes != sizeof(::cellerator::layout::projection::FeatureMajorViewV1))
        return fail(operation_status_code::unsupported_projection,
            "feature-major activated view size is incompatible");
    const execution::FeatureMajorBinding binding{
        *static_cast<const ::cellerator::layout::projection::FeatureMajorViewV1 *>(
            projection.view),
        request.structures.structures[0].runtime, projection.key.runtime};
    return prepare_feature_major_cta_medium_n_operation(request.problem,
        request.structures, projection.key, request.numeric, request.policy,
        binding,
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
    if (projection.view_bytes != sizeof(::cellerator::layout::projection::TransposeViewV1))
        return fail(operation_status_code::unsupported_projection,
            "transpose activated view size is incompatible");
    const execution::TransposeBinding binding{
        *static_cast<const ::cellerator::layout::projection::TransposeViewV1 *>(
            projection.view),
        request.structures.structures[0].runtime, projection.key.runtime,
        projection.runtime_forward_projection};
    return prepare_transpose_backward_n1_operation(request.problem,
        request.structures, projection.key, request.numeric, request.policy,
        binding,
        request.session->device, request.feature_axis, request.row_axis,
        request.dense_column_axis,
        static_cast<transpose_backward_prepared_state *>(request.state.data),
        prepared);
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
    if (execution::match_activated_projection_reference_v2(projection, candidate)
        != execution::projection_reference_status_v2::success)
        return fail(operation_status_code::unsupported_projection,
            "activated projection does not match selected catalog entry");

    const operation_status status =
        adapter.prepare(adapter, request, projection, prepared);
    if (!status) *prepared = {};
    return status;
}

} // namespace cellerator::compute::math::core
