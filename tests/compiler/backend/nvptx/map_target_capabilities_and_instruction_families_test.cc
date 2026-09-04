#include <Cellerator/compiler/backend/nvptx/map_target_capabilities_and_instruction_families_v1.hh>

#include <cassert>
#include <iostream>

using namespace Cellerator::compiler::backend::nvptx;
using namespace cellpack::persistence;

namespace {

execution_capability_manifest_v1 manifest() {
    execution_capability_manifest_v1 value{};
    value.schema_version = execution_capability_manifest_v1_schema_version;
    value.record_bytes = sizeof(value);
    value.endian = execution_capability_manifest_v1_endian_marker;
    value.flags = capability_source_linked_implementation |
        capability_requires_converged_collective | capability_memory_interface_present;
    value.provider_identity_low = 1u;
    value.provider_abi_identity_low = 2u;
    value.capability_identity_low = 3u;
    value.hardware_compatibility_identity_low = 4u;
    value.runtime_build_identity_low = 5u;
    value.kernel_build_identity_low = 6u;
    value.memory_interface_identity_low = 7u;
    value.vendor = execution_capability_vendor_v1::nvidia;
    value.architecture_class = 70u;
    value.minimum_compute_capability_major = 7u;
    value.maximum_compute_capability_major = 7u;
    value.instruction_family = execution_instruction_family_v1::nvidia_wmma;
    value.collective_scope = execution_collective_scope_v1::warp;
    value.collective_threads = 32u;
    value.instruction_m = 16u;
    value.instruction_n = 16u;
    value.instruction_k = 16u;
    value.relation_storage_type = execution_capability_numeric_type_v1::f16;
    value.dense_input_type = execution_capability_numeric_type_v1::f16;
    value.accumulation_type = execution_capability_numeric_type_v1::f32;
    value.output_type = execution_capability_numeric_type_v1::f32;
    value.operand_a_layout = execution_matrix_layout_v1::row_major;
    value.operand_b_layout = execution_matrix_layout_v1::column_major;
    value.accumulation_layout = execution_matrix_layout_v1::not_applicable;
    value.output_layout = execution_matrix_layout_v1::row_major;
    value.instruction_sparsity = execution_instruction_sparsity_v1::dense;
    value.memory_interface_flags = 0x0fu;
    value.required_engine_capability = 1u;
    return value;
}

target_instruction_requirement_v1 requirement() {
    const auto source = manifest();
    target_instruction_requirement_v1 value;
    value.compute_major = 7u;
    value.instruction_family = source.instruction_family;
    value.collective_scope = source.collective_scope;
    value.collective_threads = source.collective_threads;
    value.instruction_m = source.instruction_m;
    value.instruction_n = source.instruction_n;
    value.instruction_k = source.instruction_k;
    value.relation_storage_type = source.relation_storage_type;
    value.dense_input_type = source.dense_input_type;
    value.accumulation_type = source.accumulation_type;
    value.output_type = source.output_type;
    value.operand_a_layout = source.operand_a_layout;
    value.operand_b_layout = source.operand_b_layout;
    value.accumulation_layout = source.accumulation_layout;
    value.output_layout = source.output_layout;
    value.instruction_sparsity = source.instruction_sparsity;
    value.structured_operand = source.structured_operand;
    value.structured_group_semantics = source.structured_group_semantics;
    value.required_memory_interface_flags = 0x03u;
    return value;
}

}  // namespace

int main() {
    const auto capability = manifest();
    auto request = requirement();
    const auto supported = map_target_capability_v1(
        request, capability, target_capability_validation_mode_v1::checked);
    assert(supported && supported.status == target_capability_mapping_status_v1::supported);

    request.instruction_family = execution_instruction_family_v1::nvidia_mma_sync;
    request.instruction_m = 8u;
    const auto rejected = map_target_capability_v1(
        request, capability, target_capability_validation_mode_v1::checked);
    assert(!rejected && rejected.status == target_capability_mapping_status_v1::rejected &&
           rejected.mismatches.size() == 2u);
    const auto warned = map_target_capability_v1(
        request, capability, target_capability_validation_mode_v1::trusted);
    assert(warned && !warned.unsafe_override &&
           warned.status == target_capability_mapping_status_v1::supported_with_warning);
    const auto unsafe = map_target_capability_v1(
        request, capability, target_capability_validation_mode_v1::unsafe);
    assert(unsafe && unsafe.unsafe_override);

    auto invalid = capability;
    invalid.memory_interface_identity_low = 0u;
    assert(map_target_capability_v1(requirement(), invalid,
               target_capability_validation_mode_v1::checked).status ==
           target_capability_mapping_status_v1::invalid_manifest);
    std::cout << "capability tuple supported; mismatches rejected/warned by validation mode\n";
}
