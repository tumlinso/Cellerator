#include <Cellerator/compute/architecture/provider.hh>

#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>

namespace architecture = cellerator::compute::architecture;
namespace runtime = cellerator::runtime;

namespace {

constexpr architecture::architecture_identity_v1 provider_id{1u, 2u};

architecture::matrix_engine_capability_v1 capability() {
    architecture::matrix_engine_capability_v1 result{};
    result.identity = {3u, 4u};
    result.provider_identity = provider_id;
    result.vendor = architecture::architecture_vendor_v1::nvidia;
    result.architecture_class = static_cast<std::uint16_t>(
        runtime::device_architecture_class_v1::nvidia_volta);
    result.minimum_compute_major = 7u;
    result.maximum_compute_major = 7u;
    result.instruction_family =
        architecture::matrix_instruction_family_v1::nvidia_wmma;
    result.collective_scope = architecture::collective_scope_v1::warp;
    result.collective_threads = 32u;
    result.instruction_m = 16u;
    result.instruction_n = 16u;
    result.instruction_k = 16u;
    result.operand_a_type = cellerator::execution::numeric_type::f16;
    result.operand_b_type = cellerator::execution::numeric_type::f16;
    result.accumulation_type = cellerator::execution::numeric_type::f32;
    result.output_type = cellerator::execution::numeric_type::f32;
    result.operand_a_layout = architecture::matrix_layout_v1::row_major;
    result.operand_b_layout = architecture::matrix_layout_v1::column_major;
    result.accumulation_layout = architecture::matrix_layout_v1::opaque;
    result.output_layout = architecture::matrix_layout_v1::row_major;
    result.instruction_sparsity = architecture::instruction_sparsity_v1::dense;
    result.flags = architecture::capability_source_linked_implementation
        | architecture::capability_fragment_layout_opaque
        | architecture::capability_requires_converged_collective;
    result.engine_requirements = architecture::matrix_engine_multiply_accumulate;
    return result;
}

architecture::architecture_provider_v1 provider() {
    static const architecture::matrix_engine_capability_v1 capabilities[]{
        capability()};
    architecture::architecture_provider_v1 result{};
    result.identity = provider_id;
    result.name = "fake-volta-provider";
    result.capabilities = capabilities;
    result.capability_count = 1u;
    return result;
}

architecture::provider_status_v1 register_fake(
    architecture::architecture_provider_registry_v1 *registry) noexcept {
    return architecture::register_architecture_provider_v1(
        registry, provider());
}

runtime::device_descriptor_v1 volta_device() {
    runtime::device_descriptor_v1 result{};
    result.vendor = runtime::nvidia_pci_vendor_id;
    result.ordinal = 0;
    result.compute_major = 7u;
    result.architecture =
        runtime::device_architecture_class_v1::nvidia_volta;
    result.multiprocessor_count = 80u;
    result.warp_size = 32u;
    result.maximum_threads_per_block = 1024u;
    result.maximum_threads_per_multiprocessor = 2048u;
    result.maximum_blocks_per_multiprocessor = 32u;
    result.maximum_thread_dimensions[0] = 1024u;
    result.maximum_thread_dimensions[1] = 1024u;
    result.maximum_thread_dimensions[2] = 64u;
    result.maximum_grid_dimensions[0] = 2147483647u;
    result.maximum_grid_dimensions[1] = 65535u;
    result.maximum_grid_dimensions[2] = 65535u;
    result.registers_per_block = 65536u;
    result.registers_per_multiprocessor = 65536u;
    result.shared_memory_per_block_bytes = 49152u;
    result.optin_shared_memory_per_block_bytes = 98304u;
    result.shared_memory_per_multiprocessor_bytes = 98304u;
    result.global_memory_bytes = 16ull << 30u;
    result.l2_cache_bytes = 6ull << 20u;
    result.hardware_compatibility_identity = 1u;
    result.performance_class_identity = 2u;
    return result;
}

} // namespace

int main() {
    architecture::architecture_provider_registry_v1 registry{};
    const architecture::provider_registration_function_v1 registrations[]{
        register_fake};
    assert(architecture::register_compiled_providers_v1(
            &registry, {registrations, 1u})
        == architecture::provider_status_v1::success);
    assert(registry.size == 1u);

    const architecture::architecture_provider_registry_v1 baseline = registry;
    assert(architecture::register_architecture_provider_v1(
            &registry, provider())
        == architecture::provider_status_v1::duplicate_provider);
    assert(std::memcmp(&registry, &baseline, sizeof(registry)) == 0);
    assert(architecture::seal_architecture_provider_registry_v1(&registry)
        == architecture::provider_status_v1::success);

    const architecture::architecture_provider_v1 *active[1]{};
    std::uint32_t active_count = 0u;
    const runtime::device_descriptor_v1 volta = volta_device();
    assert(architecture::active_architecture_providers_v1(
            registry, volta, active, 1u, &active_count)
        == architecture::provider_status_v1::success);
    assert(active_count == 1u && active[0] == &registry.providers[0]);

    runtime::device_descriptor_v1 ampere = volta;
    ampere.compute_major = 8u;
    ampere.architecture =
        runtime::device_architecture_class_v1::nvidia_ampere;
    assert(architecture::active_architecture_providers_v1(
            registry, ampere, active, 1u, &active_count)
        == architecture::provider_status_v1::success);
    assert(active_count == 0u);

    std::cout << "provider_registry_test passed providers=" << registry.size
              << " active_volta=1 active_ampere=0\n";
    return 0;
}
