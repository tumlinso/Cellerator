#include <Cellerator/compute/architecture/providers/nvidia/sm70_provider.hh>

#include <Cellerator/runtime/device_descriptor.hh>

namespace cellerator::compute::architecture::providers::nvidia {
namespace {

constexpr matrix_memory_operand_contract_v1 global_f16_matrix_read{
    32u,
    8u,
    16u,
    memory_address_generic | memory_address_global,
    memory_operand_read,
    {}};

const matrix_memory_interface_v1 memory_interface{
    matrix_memory_interface_schema_version_v1,
    sizeof(matrix_memory_interface_v1),
    sm70_wmma_f16_memory_interface_identity_v1,
    memory_interface_operand_a | memory_interface_operand_b,
    0u,
    global_f16_matrix_read,
    global_f16_matrix_read,
    {},
    {},
    {}};

const matrix_engine_capability_v1 capability{
    matrix_engine_capability_schema_version_v1,
    sizeof(matrix_engine_capability_v1),
    sm70_wmma_f16_f32_identity_v1,
    sm70_provider_identity_v1,
    sm70_wmma_f16_memory_interface_identity_v1,
    architecture_vendor_v1::nvidia,
    static_cast<std::uint16_t>(
        runtime::device_architecture_class_v1::nvidia_volta),
    7u,
    0u,
    7u,
    0u,
    matrix_instruction_family_v1::nvidia_wmma,
    collective_scope_v1::warp,
    0u,
    32u,
    16u,
    16u,
    16u,
    execution::numeric_type::f16,
    execution::numeric_type::f16,
    execution::numeric_type::f32,
    execution::numeric_type::f32,
    matrix_layout_v1::row_major,
    matrix_layout_v1::row_major,
    matrix_layout_v1::opaque,
    matrix_layout_v1::row_major,
    instruction_sparsity_v1::dense,
    structured_operand_v1::none,
    structured_group_semantics_v1::none,
    0u,
    capability_source_linked_implementation
        | capability_fragment_layout_opaque
        | capability_requires_converged_collective
        | capability_memory_interface_present,
    matrix_engine_multiply_accumulate,
    {}};

const architecture_provider_v1 provider{
    architecture_provider_schema_version_v1,
    sizeof(architecture_provider_v1),
    sm70_provider_identity_v1,
    "nvidia_sm70",
    &capability,
    1u,
    &memory_interface,
    1u,
    0u,
    {}};

} // namespace

const matrix_memory_interface_v1 &
sm70_wmma_f16_memory_interface_v1() noexcept {
    return memory_interface;
}

const matrix_engine_capability_v1 &
sm70_wmma_f16_f32_capability_v1() noexcept {
    return capability;
}

const architecture_provider_v1 &sm70_provider_v1() noexcept {
    return provider;
}

provider_status_v1 register_sm70_provider_v1(
    architecture_provider_registry_v1 *registry) noexcept {
    return register_architecture_provider_v1(registry, provider);
}

} // namespace cellerator::compute::architecture::providers::nvidia
