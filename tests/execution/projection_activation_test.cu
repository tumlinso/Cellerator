#include <Cellerator/execution/projection_activation.hh>

#include <cassert>
#include <cstdint>

#include <cuda_runtime.h>

namespace {

using namespace cellerator;

__global__ void verify_row_masked_device_view(
    cellpack::persistent_packing_payload_view view,
    const unsigned char *expected_base,
    int *result) {
    if (blockIdx.x == 0u && threadIdx.x == 0u) {
        const auto *permutation = reinterpret_cast<const unsigned char *>(
            view.inverse_feature_permutation);
        *result = view.image_base == expected_base
            && permutation == expected_base + 64u ? 1 : 0;
    }
}

void require_cuda(cudaError_t status) {
    assert(status == cudaSuccess);
}

execution::projection_activation_context context(
    execution::projection_id projection = {31u, 37u}) {
    execution::projection_activation_context result;
    result.structure = {11u, 13u};
    result.runtime_structure = {2u, 1u};
    result.epoch = {17u};
    result.projection = projection;
    result.runtime_projection = {3u, 1u};
    result.location = {execution::residency_kind::device, {}, 0, 1u};
    return result;
}

cellpack::persistence::prebound_projection_view_v1 prebound(
    cellpack::persistence::execution_projection_kind kind,
    std::uint32_t schema,
    execution::relation_orientation orientation,
    std::uint32_t capability,
    execution::projection_id identity = {31u, 37u}) {
    cellpack::persistence::prebound_projection_view_v1 result;
    result.descriptor.identity_low = identity.low;
    result.descriptor.identity_high = identity.high;
    result.descriptor.kind = kind;
    result.descriptor.schema_version = schema;
    result.descriptor.orientation = static_cast<std::uint16_t>(orientation);
    result.descriptor.flags =
        cellpack::persistence::directory_device_readable | capability;
    return result;
}

void row_masked_activation_is_typed_and_non_owning() {
    alignas(64) unsigned char host_payload[128]{};
    unsigned char *device_payload = nullptr;
    int *device_result = nullptr;
    require_cuda(cudaMalloc(&device_payload, sizeof(host_payload)));
    require_cuda(cudaMalloc(&device_result, sizeof(int)));
    cellpack::persistent_packing_payload_view host;
    host.payload_schema_version =
        cellpack::persistent_packing_payload_schema_version;
    host.payload_kind = cellpack::persistent_packing_payload_kind;
    host.payload_identity = 41u;
    host.image_base = host_payload;
    host.image_bytes = sizeof(host_payload);
    host.inverse_feature_permutation =
        reinterpret_cast<const std::uint32_t *>(host_payload + 64u);

    auto bound = prebound(
        cellpack::persistence::execution_projection_kind::native_row_masked,
        cellpack::persistent_packing_payload_schema_version,
        execution::relation_orientation::forward,
        cellpack::persistence::projection_forward_capable);
    bound.payload = device_payload;
    bound.payload_bytes = sizeof(host_payload);

    cellpack::persistent_packing_payload_view activated;
    auto status = execution::activate_row_masked_projection(
        bound, context(), host, &activated);
    assert(status);
    assert(activated.image_base == device_payload);
    assert(activated.inverse_feature_permutation
        == reinterpret_cast<const std::uint32_t *>(device_payload + 64u));
    verify_row_masked_device_view<<<1, 1>>>(
        activated, device_payload, device_result);
    require_cuda(cudaGetLastError());
    int result = 0;
    require_cuda(cudaMemcpy(
        &result, device_result, sizeof(result), cudaMemcpyDeviceToHost));
    assert(result == 1);

    bound.payload_bytes -= 1u;
    status = execution::activate_row_masked_projection(
        bound, context(), host, &activated);
    assert(!status);
    assert(status.code == execution::projection_activation_code::size_mismatch);
    require_cuda(cudaFree(device_result));
    require_cuda(cudaFree(device_payload));
}

void feature_major_activation_rejects_identity_and_stale_epoch() {
    alignas(64) unsigned char host_payload[256]{};
    alignas(64) unsigned char device_payload[256]{};
    compute::math::feature_major_projection_view host;
    host.header.payload_bytes = sizeof(host_payload);
    host.header.structure_identity = {11u, 13u};
    host.header.projection_identity = {31u, 37u};
    host.header.structure_epoch = 17u;
    host.payload_base = host_payload;
    host.runtime_structure = {2u, 1u};
    host.runtime_projection = {3u, 1u};

    auto bound = prebound(
        cellpack::persistence::execution_projection_kind::native_feature_major,
        compute::math::feature_major_projection_schema_version,
        execution::relation_orientation::forward,
        cellpack::persistence::projection_forward_capable);
    bound.payload = device_payload;
    bound.payload_bytes = sizeof(device_payload);

    compute::math::feature_major_projection_view activated;
    auto status = execution::activate_feature_major_projection(
        bound, context(), host, &activated);
    assert(status);
    assert(activated.payload_base == device_payload);
    assert(execution::same_handle(
        activated.runtime_projection, context().runtime_projection));

    host.header.structure_epoch = 18u;
    status = execution::activate_feature_major_projection(
        bound, context(), host, &activated);
    assert(!status);
    assert(status.code == execution::projection_activation_code::stale_structure);

    host.header.structure_epoch = 17u;
    bound.descriptor.identity_low = 99u;
    status = execution::activate_feature_major_projection(
        bound, context(), host, &activated);
    assert(!status);
    assert(status.code == execution::projection_activation_code::identity_mismatch);
}

void transpose_activation_requires_explicit_map_and_orientation() {
    alignas(64) unsigned char host_payload[256]{};
    alignas(64) unsigned char device_payload[256]{};
    std::uint32_t transpose_maps[6] = {2u, 0u, 1u, 1u, 2u, 0u};
    compute::math::transpose_projection_view host;
    host.header.payload_bytes = sizeof(host_payload);
    host.header.structure_identity = {11u, 13u};
    host.header.projection_identity = {31u, 37u};
    host.header.forward_projection_identity = {43u, 47u};
    host.header.structure_epoch = 17u;
    host.header.nnz_count = 3u;
    host.payload_base = host_payload;
    host.runtime_structure = {2u, 1u};
    host.runtime_projection = {3u, 1u};
    host.runtime_forward_projection = {4u, 1u};

    auto bound = prebound(
        cellpack::persistence::execution_projection_kind::transpose_backward,
        compute::math::transpose_projection_schema_version,
        execution::relation_orientation::transpose,
        cellpack::persistence::projection_transpose_capable);
    bound.payload = device_payload;
    bound.payload_bytes = sizeof(device_payload);
    bound.transpose_map = transpose_maps;
    bound.transpose_map_bytes = sizeof(transpose_maps);

    execution::transpose_projection_activation_context activation;
    activation.projection = context();
    activation.forward_projection = {43u, 47u};
    activation.runtime_forward_projection = {4u, 1u};
    compute::math::transpose_projection_view activated;
    auto status = execution::activate_transpose_projection(
        bound, activation, host, &activated);
    assert(status);
    assert(activated.payload_base == device_payload);
    assert(activated.logical_to_transpose == transpose_maps);
    assert(activated.transpose_to_logical == transpose_maps + 3u);

    bound.transpose_map = nullptr;
    bound.transpose_map_bytes = 0u;
    status = execution::activate_transpose_projection(
        bound, activation, host, &activated);
    assert(!status);
    assert(status.code == execution::projection_activation_code::map_mismatch);

    bound.transpose_map = transpose_maps;
    bound.transpose_map_bytes = sizeof(transpose_maps);
    bound.descriptor.orientation = static_cast<std::uint16_t>(
        execution::relation_orientation::forward);
    status = execution::activate_transpose_projection(
        bound, activation, host, &activated);
    assert(!status);
    assert(status.code
        == execution::projection_activation_code::orientation_mismatch);
}

void csr_activation_cannot_hide_materialization() {
    std::uint32_t row_offsets[2] = {0u, 1u};
    std::uint32_t features[1] = {0u};
    float values[1] = {1.0f};
    compute::math::execution_csr_view prepared;
    prepared.row_count = 1u;
    prepared.feature_count = 1u;
    prepared.nnz_count = 1u;
    prepared.value_size_bytes = sizeof(float);
    prepared.structure = {
        compute::math::sparse_structure_identity_schema_version, 1u, 53u};
    prepared.row_offsets = row_offsets;
    prepared.execution_feature_ids = features;
    prepared.values = values;

    auto bound = prebound(
        cellpack::persistence::execution_projection_kind::csr,
        compute::math::execution_csr_schema_version,
        execution::relation_orientation::forward,
        cellpack::persistence::projection_forward_capable);
    compute::math::execution_csr_view activated;
    auto status = execution::activate_csr_projection(
        bound, context(), prepared, &activated);
    assert(status);
    assert(activated.values == values);

    bound.payload = values;
    bound.payload_bytes = sizeof(values);
    status = execution::activate_csr_projection(
        bound, context(), prepared, &activated);
    assert(!status);
    assert(status.code == execution::projection_activation_code::map_mismatch);
}

void generic_validation_rejects_location_schema_kind_and_capacity() {
    compute::math::execution_csr_view prepared;
    std::uint32_t row_offsets[1] = {0u};
    prepared.row_offsets = row_offsets;
    prepared.value_size_bytes = sizeof(float);
    prepared.structure = {
        compute::math::sparse_structure_identity_schema_version, 1u, 1u};
    auto bound = prebound(
        cellpack::persistence::execution_projection_kind::csr,
        compute::math::execution_csr_schema_version,
        execution::relation_orientation::forward,
        cellpack::persistence::projection_forward_capable);
    compute::math::execution_csr_view activated;

    auto activation = context();
    activation.location = {execution::residency_kind::host, {}, -1, 0u};
    auto status = execution::activate_csr_projection(
        bound, activation, prepared, &activated);
    assert(!status);
    assert(status.code == execution::projection_activation_code::location_mismatch);

    activation = context();
    bound.descriptor.schema_version += 1u;
    status = execution::activate_csr_projection(
        bound, activation, prepared, &activated);
    assert(!status);
    assert(status.code == execution::projection_activation_code::schema_mismatch);

    bound.descriptor.schema_version = compute::math::execution_csr_schema_version;
    bound.descriptor.kind =
        cellpack::persistence::execution_projection_kind::native_row_masked;
    status = execution::activate_csr_projection(
        bound, activation, prepared, &activated);
    assert(!status);
    assert(status.code == execution::projection_activation_code::kind_mismatch);
}

} // namespace

int main() {
    row_masked_activation_is_typed_and_non_owning();
    feature_major_activation_rejects_identity_and_stale_epoch();
    transpose_activation_requires_explicit_map_and_orientation();
    csr_activation_cannot_hide_materialization();
    generic_validation_rejects_location_schema_kind_and_capacity();
    return 0;
}
