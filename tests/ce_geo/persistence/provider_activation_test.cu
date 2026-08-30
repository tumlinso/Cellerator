#include <Cellerator/execution/projection_activation.hh>

#include <Cellerator/geometry/persistence/execution_capability_manifest_v1.hh>

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

namespace architecture = cellerator::compute::architecture;
namespace execution = cellerator::execution;
namespace persistence = cellpack::persistence;

constexpr architecture::architecture_identity_v1 provider_identity{
    0x50524f5641435431ull, 0x434547454f343300ull};
constexpr architecture::architecture_identity_v1 capability_identity{
    0x4341504143543031ull, 0x434547454f343300ull};

const architecture::matrix_engine_capability_v1 capability{
    architecture::matrix_engine_capability_schema_version_v1,
    sizeof(architecture::matrix_engine_capability_v1),
    capability_identity,
    provider_identity,
    {},
    architecture::architecture_vendor_v1::generic,
    70u,
    7u, 0u, 7u, 9u,
    architecture::matrix_instruction_family_v1::generic_multiply_accumulate,
    architecture::collective_scope_v1::thread,
    0u,
    1u,
    4u, 4u, 4u,
    execution::numeric_type::f32,
    execution::numeric_type::f32,
    execution::numeric_type::f32,
    execution::numeric_type::f32,
    architecture::matrix_layout_v1::row_major,
    architecture::matrix_layout_v1::row_major,
    architecture::matrix_layout_v1::not_applicable,
    architecture::matrix_layout_v1::row_major,
    architecture::instruction_sparsity_v1::dense,
    architecture::structured_operand_v1::none,
    architecture::structured_group_semantics_v1::none,
    0u,
    architecture::capability_source_linked_implementation,
    architecture::matrix_engine_multiply_accumulate,
    {}};

const architecture::architecture_provider_v1 architecture_provider{
    architecture::architecture_provider_schema_version_v1,
    sizeof(architecture::architecture_provider_v1),
    provider_identity,
    "ce_geo_projection_provider",
    &capability,
    1u,
    nullptr,
    0u,
    0u,
    {}};

struct provider_payload {
    std::uint32_t magic;
    std::uint32_t count;
    float values[4];
};

struct validated_payload {
    std::uint32_t count;
};

struct activated_payload {
    const float *values;
    std::uint32_t count;
};

int validation_calls = 0;
int activation_calls = 0;

execution::projection_activation_status validate_payload(
    const persistence::prebound_projection_view_v2 &prebound,
    const execution::projection_activation_context &,
    void *storage,
    std::size_t storage_bytes) noexcept {
    ++validation_calls;
    if (storage_bytes != sizeof(validated_payload)
        || prebound.projection_v1.payload == nullptr
        || prebound.projection_v1.payload_bytes != sizeof(provider_payload))
        return {execution::projection_activation_code::size_mismatch,
            "test provider payload size mismatch"};
    const auto &payload = *static_cast<const provider_payload *>(
        prebound.projection_v1.payload);
    if (payload.magic != 0x50524f4au || payload.count != 4u)
        return {execution::projection_activation_code::invalid_projection,
            "test provider payload is invalid"};
    *static_cast<validated_payload *>(storage) = {payload.count};
    return {};
}

execution::projection_activation_status activate_payload(
    const persistence::prebound_projection_view_v2 &prebound,
    const execution::projection_activation_context &,
    const void *validated_storage,
    std::size_t validated_bytes,
    void *activated_storage,
    std::size_t activated_bytes) noexcept {
    ++activation_calls;
    if (validated_bytes != sizeof(validated_payload)
        || activated_bytes != sizeof(activated_payload)
        || prebound.projection_v1.payload == nullptr
        || prebound.projection_v1.payload_bytes != sizeof(provider_payload))
        return {execution::projection_activation_code::size_mismatch,
            "test provider activation storage mismatch"};
    const auto &validated = *static_cast<const validated_payload *>(
        validated_storage);
    auto *activated = static_cast<activated_payload *>(activated_storage);
    activated->values = reinterpret_cast<const provider_payload *>(
        prebound.projection_v1.payload)->values;
    activated->count = validated.count;
    return {};
}

__global__ void sum_projection(activated_payload projection, float *out) {
    if (threadIdx.x == 0u && blockIdx.x == 0u) {
        float sum = 0.0f;
        for (std::uint32_t index = 0u; index < projection.count; ++index)
            sum += projection.values[index];
        *out = sum;
    }
}

void require(bool condition, const char *message) {
    if (!condition) {
        std::fprintf(stderr, "provider_activation_test: %s\n", message);
        std::exit(1);
    }
}

void require_cuda(cudaError_t status, const char *message) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "provider_activation_test: %s: %s\n", message,
            cudaGetErrorString(status));
        std::exit(1);
    }
}

persistence::execution_capability_manifest_v1 manifest() {
    persistence::execution_capability_manifest_v1 result{};
    result.schema_version =
        persistence::execution_capability_manifest_v1_schema_version;
    result.record_bytes = sizeof(result);
    result.endian = persistence::execution_capability_manifest_v1_endian_marker;
    result.flags = persistence::capability_source_linked_implementation;
    result.provider_identity_low = provider_identity.low;
    result.provider_identity_high = provider_identity.high;
    result.provider_abi_identity_low = 1u;
    result.capability_identity_low = capability_identity.low;
    result.capability_identity_high = capability_identity.high;
    result.hardware_compatibility_identity_low = 1u;
    result.runtime_build_identity_low = 1u;
    result.kernel_build_identity_low = 1u;
    result.vendor = persistence::execution_capability_vendor_v1::generic;
    result.architecture_class = 70u;
    result.minimum_compute_capability_major = 7u;
    result.maximum_compute_capability_major = 7u;
    result.maximum_compute_capability_minor = 9u;
    result.instruction_family =
        persistence::execution_instruction_family_v1::generic_scalar;
    result.collective_scope = persistence::execution_collective_scope_v1::thread;
    result.collective_threads = 1u;
    result.instruction_m = 4u;
    result.instruction_n = 4u;
    result.instruction_k = 4u;
    result.relation_storage_type =
        persistence::execution_capability_numeric_type_v1::f32;
    result.dense_input_type =
        persistence::execution_capability_numeric_type_v1::f32;
    result.accumulation_type =
        persistence::execution_capability_numeric_type_v1::f32;
    result.output_type = persistence::execution_capability_numeric_type_v1::f32;
    result.operand_a_layout = persistence::execution_matrix_layout_v1::row_major;
    result.operand_b_layout = persistence::execution_matrix_layout_v1::row_major;
    result.accumulation_layout =
        persistence::execution_matrix_layout_v1::not_applicable;
    result.output_layout = persistence::execution_matrix_layout_v1::row_major;
    result.instruction_sparsity =
        persistence::execution_instruction_sparsity_v1::dense;
    result.structured_operand = persistence::execution_structured_operand_v1::none;
    result.structured_group_semantics =
        persistence::execution_structured_group_semantics_v1::none;
    result.required_engine_capability = 1u;
    return result;
}

} // namespace

int main() {
    provider_payload host_payload{0x50524f4au, 4u, {1.0f, 2.0f, 3.0f, 4.0f}};
    provider_payload *device_payload = nullptr;
    float *device_result = nullptr;
    require_cuda(cudaMalloc(&device_payload, sizeof(host_payload)),
        "allocate device payload");
    require_cuda(cudaMalloc(&device_result, sizeof(float)),
        "allocate device result");
    require_cuda(cudaMemcpy(device_payload, &host_payload, sizeof(host_payload),
        cudaMemcpyHostToDevice), "copy provider payload");

    auto capability_manifest = manifest();
    persistence::execution_projection_entry_v1 entry{};
    entry.identity_low = 31u;
    entry.identity_high = 32u;
    entry.kind = persistence::execution_projection_kind::architecture_specific;
    entry.schema_version = 7u;
    entry.flags = persistence::directory_device_readable
        | persistence::projection_forward_capable;
    entry.orientation =
        static_cast<std::uint16_t>(execution::relation_orientation::forward);
    entry.architecture_class = 70u;

    persistence::prebound_projection_view_v2 host_prebound{};
    host_prebound.projection_v1.descriptor = entry;
    host_prebound.projection_v1.payload = &host_payload;
    host_prebound.projection_v1.payload_bytes = sizeof(host_payload);
    host_prebound.capability = &capability_manifest;
    host_prebound.capability_bytes = sizeof(capability_manifest);
    persistence::prebound_projection_view_v2 device_prebound = host_prebound;
    device_prebound.projection_v1.payload = device_payload;
    device_prebound.capability = device_payload; // Opaque device-image address.

    execution::projection_provider_descriptor_v1 provider{};
    provider.architecture = &architecture_provider;
    provider.capability_identity = capability_identity;
    provider.projection_kind = entry.kind;
    provider.projection_schema_version = entry.schema_version;
    provider.orientation = execution::relation_orientation::forward;
    provider.required_directory_capability =
        persistence::projection_forward_capable;
    provider.validated_host_view_bytes = sizeof(validated_payload);
    provider.activated_device_view_bytes = sizeof(activated_payload);
    provider.validate_host = &validate_payload;
    provider.activate_device = &activate_payload;

    execution::projection_activation_context context{};
    context.structure = {11u, 12u};
    context.runtime_structure = {13u, 1u};
    context.epoch = {14u};
    context.projection = {entry.identity_low, entry.identity_high};
    context.runtime_projection = {15u, 1u};
    context.location = {execution::residency_kind::device, {0u, 0u, 0u}, 0, 0u};

    validated_payload validated{};
    activated_payload activated{};
    auto status = execution::validate_and_activate_projection_via_provider_v1(
        provider, host_prebound, device_prebound, context, &validated,
        sizeof(validated), &activated, sizeof(activated));
    require(static_cast<bool>(status), status.message);
    require(validation_calls == 1 && activation_calls == 1,
        "provider callbacks were not routed exactly once");

    sum_projection<<<1, 1>>>(activated, device_result);
    require_cuda(cudaGetLastError(), "launch projection consumer");
    float result = 0.0f;
    require_cuda(cudaMemcpy(&result, device_result, sizeof(result),
        cudaMemcpyDeviceToHost), "copy projection result");
    require(result == 10.0f, "activated device view produced wrong result");

    capability_manifest.provider_identity_low ^= 1u;
    status = execution::validate_and_activate_projection_via_provider_v1(
        provider, host_prebound, device_prebound, context, &validated,
        sizeof(validated), &activated, sizeof(activated));
    require(status.code == execution::projection_activation_code::provider_mismatch,
        "mismatched persistent provider identity was accepted");
    require(validation_calls == 1 && activation_calls == 1,
        "identity mismatch reached provider callbacks");

    auto malformed = provider;
    malformed.capability_identity.low ^= 1u;
    status = execution::validate_projection_provider_descriptor_v1(malformed);
    require(status.code == execution::projection_activation_code::provider_mismatch,
        "unlinked capability identity was accepted");

    require_cuda(cudaFree(device_result), "free device result");
    require_cuda(cudaFree(device_payload), "free device payload");
    std::puts("provider activation route passed");
    return 0;
}
