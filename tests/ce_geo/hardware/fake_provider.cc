#include <Cellerator/compute/architecture/provider.hh>

namespace cellerator::tests::ce_geo::fake_provider {

namespace architecture = compute::architecture;

constexpr architecture::architecture_identity_v1 provider_identity{
    0x46414b4550524f56ull, 0x4944455256310001ull};
constexpr architecture::architecture_identity_v1 capability_identity{
    0x46414b4543415041ull, 0x42494c4954590001ull};
constexpr architecture::architecture_identity_v1 memory_identity{
    0x46414b454d454d4full, 0x5259563100000001ull};

namespace {

constexpr architecture::matrix_memory_operand_contract_v1 matrix_read{
    16u,
    4u,
    4u,
    architecture::memory_address_generic
        | architecture::memory_address_global,
    architecture::memory_operand_read,
    {}};

const architecture::matrix_memory_interface_v1 memory_interface{
    architecture::matrix_memory_interface_schema_version_v1,
    sizeof(architecture::matrix_memory_interface_v1),
    memory_identity,
    architecture::memory_interface_operand_a
        | architecture::memory_interface_operand_b,
    0u,
    matrix_read,
    matrix_read,
    {},
    {},
    {}};

const architecture::matrix_engine_capability_v1 capability{
    architecture::matrix_engine_capability_schema_version_v1,
    sizeof(architecture::matrix_engine_capability_v1),
    capability_identity,
    provider_identity,
    memory_identity,
    architecture::architecture_vendor_v1::generic,
    static_cast<std::uint16_t>(
        runtime::device_architecture_class_v1::nvidia_volta),
    7u,
    0u,
    7u,
    9u,
    architecture::matrix_instruction_family_v1::generic_multiply_accumulate,
    architecture::collective_scope_v1::thread,
    0u,
    1u,
    4u,
    4u,
    4u,
    execution::numeric_type::f32,
    execution::numeric_type::f32,
    execution::numeric_type::f32,
    execution::numeric_type::f32,
    architecture::matrix_layout_v1::row_major,
    architecture::matrix_layout_v1::column_major,
    architecture::matrix_layout_v1::not_applicable,
    architecture::matrix_layout_v1::row_major,
    architecture::instruction_sparsity_v1::dense,
    architecture::structured_operand_v1::none,
    architecture::structured_group_semantics_v1::none,
    0u,
    architecture::capability_source_linked_implementation
        | architecture::capability_memory_interface_present,
    architecture::matrix_engine_multiply_accumulate,
    {}};

const architecture::architecture_provider_v1 provider{
    architecture::architecture_provider_schema_version_v1,
    sizeof(architecture::architecture_provider_v1),
    provider_identity,
    "ce_geo_fake_provider",
    &capability,
    1u,
    &memory_interface,
    1u,
    0u,
    {}};

} // namespace

const architecture::architecture_provider_v1 &description() noexcept {
    return provider;
}

architecture::provider_status_v1 register_provider(
    architecture::architecture_provider_registry_v1 *registry) noexcept {
    return architecture::register_architecture_provider_v1(registry, provider);
}

float execute_candidate(float lhs, float rhs, float accumulator) noexcept {
    return lhs * rhs + accumulator;
}

} // namespace cellerator::tests::ce_geo::fake_provider
