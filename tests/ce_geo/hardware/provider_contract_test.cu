#include <Cellerator/compute/architecture/provider.hh>
#include <Cellerator/runtime/device_descriptor.hh>

#include <cmath>
#include <cstdint>
#include <iostream>

namespace architecture = cellerator::compute::architecture;
namespace runtime = cellerator::runtime;

namespace cellerator::tests::ce_geo::fake_provider {

const architecture::architecture_provider_v1 &description() noexcept;
architecture::provider_status_v1 register_provider(
    architecture::architecture_provider_registry_v1 *) noexcept;
float execute_candidate(float lhs, float rhs, float accumulator) noexcept;

} // namespace cellerator::tests::ce_geo::fake_provider

namespace {

namespace fake = cellerator::tests::ce_geo::fake_provider;

bool require(bool condition, const char *message) {
    if (!condition)
        std::cerr << "provider_contract_test: " << message << '\n';
    return condition;
}

runtime::device_descriptor_v1 compatible_device() {
    runtime::device_descriptor_v1 device{};
    device.vendor = 0x1234u;
    device.ordinal = 0;
    device.compute_major = 7u;
    device.compute_minor = 5u;
    device.architecture =
        runtime::device_architecture_class_v1::nvidia_volta;
    device.multiprocessor_count = 1u;
    device.warp_size = 32u;
    device.maximum_threads_per_block = 1024u;
    device.maximum_threads_per_multiprocessor = 2048u;
    device.maximum_blocks_per_multiprocessor = 32u;
    device.maximum_thread_dimensions[0] = 1024u;
    device.maximum_thread_dimensions[1] = 1024u;
    device.maximum_thread_dimensions[2] = 64u;
    device.maximum_grid_dimensions[0] = 2147483647u;
    device.maximum_grid_dimensions[1] = 65535u;
    device.maximum_grid_dimensions[2] = 65535u;
    device.registers_per_block = 65536u;
    device.registers_per_multiprocessor = 65536u;
    device.shared_memory_per_block_bytes = 49152u;
    device.optin_shared_memory_per_block_bytes = 98304u;
    device.shared_memory_per_multiprocessor_bytes = 98304u;
    device.global_memory_bytes = 1u << 20u;
    device.l2_cache_bytes = 1u << 16u;
    device.hardware_compatibility_identity = 1u;
    device.performance_class_identity = 2u;
    return device;
}

} // namespace

int main() {
    const architecture::architecture_provider_v1 &provider =
        fake::description();
    if (!require(architecture::validate_architecture_provider_v1(provider)
            == architecture::provider_status_v1::success,
            "valid fake provider rejected")
        || !require(provider.capability_count == 1u
                && provider.memory_interface_count == 1u,
            "provider did not expose one capability and memory interface")
        || !require(architecture::validate_matrix_engine_capability_v1(
                provider.capabilities[0])
                == architecture::capability_status_v1::success,
            "fake candidate capability rejected")
        || !require(architecture::validate_matrix_memory_interface_v1(
                provider.memory_interfaces[0])
                == architecture::capability_status_v1::success,
            "fake memory interface rejected"))
        return 1;

    architecture::architecture_provider_registry_v1 registry{};
    const architecture::provider_registration_function_v1 manifest_entries[]{
        fake::register_provider};
    if (!require(architecture::register_compiled_providers_v1(
                &registry, {manifest_entries, 1u})
                == architecture::provider_status_v1::success,
            "compiled manifest registration failed")
        || !require(registry.size == 1u,
            "manifest did not add exactly one provider")
        || !require(architecture::seal_architecture_provider_registry_v1(
                &registry) == architecture::provider_status_v1::success,
            "registry seal failed")
        || !require(fake::register_provider(&registry)
                == architecture::provider_status_v1::invalid_argument,
            "sealed registry accepted late provider registration"))
        return 1;

    const architecture::architecture_provider_v1 *active[1]{};
    std::uint32_t active_count = 0u;
    runtime::device_descriptor_v1 device = compatible_device();
    if (!require(runtime::valid_device_descriptor_v1(device),
            "compatible device fixture is invalid")
        || !require(architecture::active_architecture_providers_v1(
                registry, device, active, 1u, &active_count)
                == architecture::provider_status_v1::success,
            "active-provider selection failed")
        || !require(active_count == 1u
                && active[0] == &registry.providers[0],
            "fake provider was not selected")
        || !require(std::fabs(fake::execute_candidate(2.0f, 3.0f, 4.0f)
                - 10.0f) == 0.0f,
            "source-linked fake candidate did not execute"))
        return 1;

    std::uint64_t query_count = 0u;
    runtime::device_descriptor_v1 forbidden_query{};
    if (!require(runtime::query_device_descriptor_v1(
                0, true, &forbidden_query, &query_count)
                == runtime::device_descriptor_status_v1::invalid_state,
            "sealed session did not reject hardware discovery")
        || !require(query_count == 0u,
            "sealed session reached CUDA hardware discovery"))
        return 1;

    device.compute_major = 8u;
    device.architecture =
        runtime::device_architecture_class_v1::nvidia_ampere;
    if (!require(architecture::active_architecture_providers_v1(
                registry, device, active, 1u, &active_count)
                == architecture::provider_status_v1::success
            && active_count == 0u,
            "incompatible device activated fake provider"))
        return 1;

    std::cout << "provider_contract_test: ok providers=" << registry.size
              << " sealed_query_count=" << query_count << '\n';
    return 0;
}
