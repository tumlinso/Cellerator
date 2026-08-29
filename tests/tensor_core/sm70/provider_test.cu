#include <Cellerator/compute/architecture/providers/nvidia/sm70_provider.hh>

#include <Cellerator/runtime/device_descriptor.hh>

#include <cstdlib>
#include <iostream>

namespace architecture = cellerator::compute::architecture;
namespace provider =
    cellerator::compute::architecture::providers::nvidia;
namespace runtime = cellerator::runtime;

namespace {

void require(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "sm70_provider_test: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

} // namespace

int main() {
    const architecture::matrix_engine_capability_v1 &capability =
        provider::sm70_wmma_f16_f32_capability_v1();
    require(architecture::validate_matrix_engine_capability_v1(capability)
            == architecture::capability_status_v1::success,
        "capability validation failed");
    require(capability.minimum_compute_major == 7u
            && capability.maximum_compute_major == 7u
            && capability.minimum_compute_minor == 0u
            && capability.maximum_compute_minor == 0u,
        "provider advertised beyond sm_70");
    require(capability.instruction_m == 16u
            && capability.instruction_n == 16u
            && capability.instruction_k == 16u,
        "WMMA instruction shape changed");
    require(capability.operand_a_type == cellerator::execution::numeric_type::f16
            && capability.operand_b_type
                == cellerator::execution::numeric_type::f16
            && capability.accumulation_type
                == cellerator::execution::numeric_type::f32
            && capability.output_type
                == cellerator::execution::numeric_type::f32,
        "provider numeric truth is too broad");
    require(capability.instruction_sparsity
            == architecture::instruction_sparsity_v1::dense,
        "Volta provider falsely advertised sparse MMA");

    const architecture::matrix_memory_interface_v1 &memory =
        provider::sm70_wmma_f16_memory_interface_v1();
    require(architecture::validate_matrix_memory_interface_v1(memory)
            == architecture::capability_status_v1::success,
        "memory-interface validation failed");
    require(memory.flags == (architecture::memory_interface_operand_a
            | architecture::memory_interface_operand_b),
        "provider advertised an unimplemented matrix memory operation");
    require(memory.operand_a.base_alignment_bytes == 32u
            && memory.operand_b.base_alignment_bytes == 32u
            && memory.operand_a.leading_dimension_multiple_elements == 8u
            && memory.operand_b.leading_dimension_multiple_elements == 8u,
        "WMMA load alignment or stride contract changed");

    architecture::architecture_provider_registry_v1 registry{};
    require(provider::register_sm70_provider_v1(&registry)
            == architecture::provider_status_v1::success,
        "explicit provider registration failed");
    require(architecture::seal_architecture_provider_registry_v1(&registry)
            == architecture::provider_status_v1::success,
        "provider registry sealing failed");

    runtime::device_descriptor_v1 device{};
    std::uint64_t query_count = 0u;
    require(runtime::query_device_descriptor_v1(
            -1, false, &device, &query_count)
            == runtime::device_descriptor_status_v1::success,
        "active-device descriptor query failed");
    require(device.compute_major == 7u && device.compute_minor == 0u,
        "test requires an sm_70 device");

    const architecture::architecture_provider_v1 *active[1]{};
    std::uint32_t active_count = 0u;
    require(architecture::active_architecture_providers_v1(
            registry, device, active, 1u, &active_count)
            == architecture::provider_status_v1::success,
        "active provider filtering failed");
    require(active_count == 1u
            && active[0] == &registry.providers[0]
            && query_count != 0u,
        "sm_70 provider did not activate from canonical device truth");

    std::cout << "sm70_provider_test passed capability=16x16x16-f16-f32"
              << " device=" << device.compute_major << '.'
              << device.compute_minor << " queries=" << query_count << '\n';
    return EXIT_SUCCESS;
}
