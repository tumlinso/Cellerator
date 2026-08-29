#include <Cellerator/compute/architecture/capability.hh>

#include <cstdlib>
#include <iostream>
#include <type_traits>

namespace architecture = cellerator::compute::architecture;
namespace execution = cellerator::execution;

namespace {

void require(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "capability_test: " << message << '\n';
        std::exit(1);
    }
}

architecture::matrix_memory_operand_contract_v1 readable_operand() {
    architecture::matrix_memory_operand_contract_v1 result{};
    result.base_alignment_bytes = 32u;
    result.leading_dimension_multiple_elements = 8u;
    result.contiguous_extent_multiple_elements = 1u;
    result.address_space_flags = architecture::memory_address_global
        | architecture::memory_address_shared;
    result.access_flags = architecture::memory_operand_read;
    return result;
}

architecture::matrix_memory_interface_v1 volta_memory_interface() {
    architecture::matrix_memory_interface_v1 result{};
    result.identity = {0x701u, 0x702u};
    result.flags = architecture::memory_interface_operand_a
        | architecture::memory_interface_operand_b
        | architecture::memory_interface_output;
    result.operand_a = readable_operand();
    result.operand_b = readable_operand();
    result.output = readable_operand();
    result.output.access_flags = architecture::memory_operand_write;
    return result;
}

architecture::matrix_engine_capability_v1 volta_matrix_engine() {
    architecture::matrix_engine_capability_v1 result{};
    result.identity = {0x301u, 0x302u};
    result.provider_identity = {0x101u, 0x102u};
    result.vendor = architecture::architecture_vendor_v1::nvidia;
    result.architecture_class = 70u;
    result.minimum_compute_major = 7u;
    result.minimum_compute_minor = 0u;
    result.maximum_compute_major = 7u;
    result.maximum_compute_minor = 5u;
    result.instruction_family =
        architecture::matrix_instruction_family_v1::nvidia_wmma;
    result.collective_scope = architecture::collective_scope_v1::warp;
    result.collective_threads = 32u;
    result.instruction_m = 16u;
    result.instruction_n = 16u;
    result.instruction_k = 16u;
    result.operand_a_type = execution::numeric_type::f16;
    result.operand_b_type = execution::numeric_type::f16;
    result.accumulation_type = execution::numeric_type::f32;
    result.output_type = execution::numeric_type::f32;
    result.operand_a_layout = architecture::matrix_layout_v1::row_major;
    result.operand_b_layout = architecture::matrix_layout_v1::column_major;
    result.accumulation_layout = architecture::matrix_layout_v1::opaque;
    result.output_layout = architecture::matrix_layout_v1::row_major;
    result.instruction_sparsity = architecture::instruction_sparsity_v1::dense;
    result.flags = architecture::capability_source_linked_implementation
        | architecture::capability_fragment_layout_opaque
        | architecture::capability_requires_converged_collective;
    result.engine_requirements =
        architecture::matrix_engine_multiply_accumulate;
    return result;
}

} // namespace

int main() {
    const auto interface = volta_memory_interface();
    require(architecture::validate_matrix_memory_interface_v1(interface)
            == architecture::capability_status_v1::success,
        "valid memory interface rejected");

    auto capability = volta_matrix_engine();
    require(architecture::validate_matrix_engine_capability_v1(capability)
            == architecture::capability_status_v1::success,
        "valid register-level capability rejected");
    require(architecture::compute_capability_in_range_v1(capability, 7u, 0u)
            && architecture::compute_capability_in_range_v1(capability, 7u, 5u)
            && !architecture::compute_capability_in_range_v1(capability, 6u, 1u)
            && !architecture::compute_capability_in_range_v1(capability, 8u, 0u),
        "compute capability range changed");

    capability.flags |= architecture::capability_memory_interface_present;
    capability.memory_interface_identity = interface.identity;
    require(architecture::validate_matrix_engine_capability_v1(capability)
            == architecture::capability_status_v1::success,
        "capability did not accept separately identified memory interface");
    require(architecture::same_architecture_identity_v1(
                capability.memory_interface_identity, interface.identity),
        "memory interface identity mismatch");

    auto invalid_capability = capability;
    invalid_capability.flags &= ~architecture::capability_memory_interface_present;
    require(architecture::validate_matrix_engine_capability_v1(invalid_capability)
            == architecture::capability_status_v1::invalid_memory_interface,
        "orphan memory interface identity accepted");
    invalid_capability = capability;
    invalid_capability.minimum_compute_major = 8u;
    require(architecture::validate_matrix_engine_capability_v1(invalid_capability)
            == architecture::capability_status_v1::invalid_compute_range,
        "descending compute capability range accepted");
    invalid_capability = capability;
    invalid_capability.instruction_sparsity =
        architecture::instruction_sparsity_v1::structured;
    require(architecture::validate_matrix_engine_capability_v1(invalid_capability)
            == architecture::capability_status_v1::invalid_sparsity_contract,
        "partial structured sparsity contract accepted");
    invalid_capability.structured_operand =
        architecture::structured_operand_v1::operand_a;
    invalid_capability.structured_group_semantics =
        architecture::structured_group_semantics_v1::two_of_four;
    require(architecture::validate_matrix_engine_capability_v1(invalid_capability)
            == architecture::capability_status_v1::success,
        "complete structured sparsity contract rejected");
    invalid_capability = capability;
    invalid_capability.flags |= 1u << 31u;
    require(architecture::validate_matrix_engine_capability_v1(invalid_capability)
            == architecture::capability_status_v1::invalid_flags,
        "unknown capability flag accepted");
    invalid_capability = capability;
    invalid_capability.reserved[0] = 1u;
    require(architecture::validate_matrix_engine_capability_v1(invalid_capability)
            == architecture::capability_status_v1::nonzero_reserved,
        "nonzero capability reserved field accepted");

    auto invalid_interface = interface;
    invalid_interface.operand_a.base_alignment_bytes = 24u;
    require(architecture::validate_matrix_memory_interface_v1(invalid_interface)
            == architecture::capability_status_v1::invalid_memory_interface,
        "non-power-of-two alignment accepted");
    invalid_interface = interface;
    invalid_interface.flags &= ~architecture::memory_interface_output;
    require(architecture::validate_matrix_memory_interface_v1(invalid_interface)
            == architecture::capability_status_v1::invalid_memory_interface,
        "unadvertised output restrictions accepted");
    invalid_interface = interface;
    invalid_interface.operand_b.address_space_flags = 1u << 31u;
    require(architecture::validate_matrix_memory_interface_v1(invalid_interface)
            == architecture::capability_status_v1::invalid_memory_interface,
        "unknown address space accepted");
    invalid_interface = interface;
    invalid_interface.reserved[0] = 1u;
    require(architecture::validate_matrix_memory_interface_v1(invalid_interface)
            == architecture::capability_status_v1::nonzero_reserved,
        "nonzero memory-interface reserved field accepted");

    require(std::is_trivially_copyable<
                architecture::matrix_engine_capability_v1>::value
            && std::is_standard_layout<
                architecture::matrix_engine_capability_v1>::value
            && std::is_trivially_copyable<
                architecture::matrix_memory_interface_v1>::value,
        "cold capability contracts lost POD properties");

    std::cout << "capability_test passed capability_bytes="
              << sizeof(capability) << " memory_interface_bytes="
              << sizeof(interface) << '\n';
    return 0;
}
