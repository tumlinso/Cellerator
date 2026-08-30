#include <Cellerator/compute/architecture/provider.hh>
#include <Cellerator/execution/projection_activation_v2.hh>
#include <Cellerator/geometry/persistence/execution_capability_manifest_v1.hh>
#include <Cellerator/geometry/persistence/execution_image_v2.hh>

#include <cstddef>
#include <cstdint>

namespace cellerator::compute::projection {
bool make_physical_mma_hybrid_cpe2_source_v1(
    const void *, std::size_t, std::uint64_t, std::uint64_t, std::uint64_t,
    std::uint64_t, std::uint32_t,
    cellpack::persistence::execution_section_source *,
    cellpack::persistence::execution_projection_source *) noexcept;
}

namespace cellerator::compute::architecture::providers::nvidia {
namespace {

namespace core = compute::math::core;
namespace persistence = cellpack::persistence;

persistence::execution_capability_numeric_type_v1 numeric_type(
    execution::numeric_type type) noexcept {
    using out = persistence::execution_capability_numeric_type_v1;
    switch (type) {
    case execution::numeric_type::bit: return out::bit;
    case execution::numeric_type::u8: return out::u8;
    case execution::numeric_type::u16: return out::u16;
    case execution::numeric_type::u32: return out::u32;
    case execution::numeric_type::i32: return out::i32;
    case execution::numeric_type::f16: return out::f16;
    case execution::numeric_type::bf16: return out::bf16;
    case execution::numeric_type::f32: return out::f32;
    case execution::numeric_type::f64: return out::f64;
    default: return out::invalid;
    }
}

persistence::execution_matrix_layout_v1 matrix_layout(
    matrix_layout_v1 layout) noexcept {
    using out = persistence::execution_matrix_layout_v1;
    switch (layout) {
    case matrix_layout_v1::not_applicable: return out::not_applicable;
    case matrix_layout_v1::row_major: return out::row_major;
    case matrix_layout_v1::column_major: return out::column_major;
    case matrix_layout_v1::opaque: return out::opaque;
    default: return out::invalid;
    }
}

persistence::execution_capability_vendor_v1 vendor(
    architecture_vendor_v1 value) noexcept {
    using out = persistence::execution_capability_vendor_v1;
    switch (value) {
    case architecture_vendor_v1::generic: return out::generic;
    case architecture_vendor_v1::nvidia: return out::nvidia;
    default: return out::invalid;
    }
}

persistence::execution_instruction_family_v1 instruction_family(
    matrix_instruction_family_v1 value) noexcept {
    using out = persistence::execution_instruction_family_v1;
    switch (value) {
    case matrix_instruction_family_v1::generic_multiply_accumulate:
        return out::generic_scalar;
    case matrix_instruction_family_v1::nvidia_wmma: return out::nvidia_wmma;
    case matrix_instruction_family_v1::nvidia_mma_sync:
        return out::nvidia_mma_sync;
    default: return out::invalid;
    }
}

persistence::execution_collective_scope_v1 collective_scope(
    collective_scope_v1 value) noexcept {
    using out = persistence::execution_collective_scope_v1;
    switch (value) {
    case collective_scope_v1::thread: return out::thread;
    case collective_scope_v1::warp: return out::warp;
    case collective_scope_v1::warp_group: return out::warp_group;
    case collective_scope_v1::cooperative_thread_array:
        return out::cooperative_thread_array;
    default: return out::invalid;
    }
}

persistence::execution_instruction_sparsity_v1 instruction_sparsity(
    instruction_sparsity_v1 value) noexcept {
    using out = persistence::execution_instruction_sparsity_v1;
    switch (value) {
    case instruction_sparsity_v1::dense: return out::dense;
    case instruction_sparsity_v1::structured: return out::structured;
    default: return out::invalid;
    }
}

persistence::execution_structured_operand_v1 structured_operand(
    structured_operand_v1 value) noexcept {
    using out = persistence::execution_structured_operand_v1;
    switch (value) {
    case structured_operand_v1::none: return out::none;
    case structured_operand_v1::operand_a: return out::operand_a;
    case structured_operand_v1::operand_b: return out::operand_b;
    default: return out::none;
    }
}

persistence::execution_structured_group_semantics_v1 structured_group(
    structured_group_semantics_v1 value) noexcept {
    using out = persistence::execution_structured_group_semantics_v1;
    switch (value) {
    case structured_group_semantics_v1::none: return out::none;
    case structured_group_semantics_v1::implementation_defined:
        return out::implementation_defined;
    case structured_group_semantics_v1::two_of_four: return out::two_of_four;
    default: return out::none;
    }
}

const matrix_memory_interface_v1 *find_memory_interface(
    const architecture_provider_v1 &provider,
    architecture_identity_v1 identity) noexcept {
    for (std::uint32_t i = 0u; i < provider.memory_interface_count; ++i)
        if (same_architecture_identity_v1(
                provider.memory_interfaces[i].identity, identity))
            return &provider.memory_interfaces[i];
    return nullptr;
}

} // namespace

bool realize_mma_provider_projection_v1(
    const architecture_provider_v1 &provider,
    const matrix_engine_capability_v1 &capability,
    const void *host_payload,
    std::size_t payload_bytes,
    core::stable_id provider_abi_identity,
    core::stable_id hardware_compatibility_identity,
    core::stable_id runtime_build_identity,
    core::stable_id kernel_build_identity,
    core::stable_id section_identity,
    core::stable_id projection_identity,
    std::uint32_t payload_section_index,
    std::uint32_t capability_section_index,
    core::projection_key key,
    core::candidate_projection_contract_v2 contract,
    execution::device_location device_location,
    const void *device_view,
    persistence::execution_section_source *sections,
    persistence::execution_projection_source *projection_source,
    persistence::execution_capability_manifest_v1 *manifest,
    execution::activated_projection_reference_v2 *activated) noexcept {
    if (sections == nullptr || projection_source == nullptr || manifest == nullptr
        || activated == nullptr || device_view == nullptr
        || !same_architecture_identity_v1(
            capability.provider_identity, provider.identity)
        || !valid_architecture_identity_v1(provider.identity)
        || !valid_architecture_identity_v1(capability.identity)
        || !core::valid_catalog_identity_v2(provider_abi_identity)
        || !core::valid_catalog_identity_v2(hardware_compatibility_identity)
        || !core::valid_catalog_identity_v2(runtime_build_identity)
        || !core::valid_catalog_identity_v2(kernel_build_identity)
        || !core::valid_catalog_identity_v2(section_identity)
        || !core::valid_catalog_identity_v2(projection_identity)
        || !execution::valid_location(device_location)
        || device_location.residency == execution::residency_kind::host)
        return false;
    const matrix_memory_interface_v1 *memory = find_memory_interface(
        provider, capability.memory_interface_identity);
    const bool requires_memory = (capability.flags
        & architecture::capability_memory_interface_present) != 0u;
    if (requires_memory != (memory != nullptr)) return false;

    persistence::execution_section_source payload{};
    persistence::execution_projection_source projection{};
    if (!compute::projection::make_physical_mma_hybrid_cpe2_source_v1(
            host_payload, payload_bytes, section_identity.low,
            section_identity.high, projection_identity.low,
            projection_identity.high, payload_section_index, &payload,
            &projection))
        return false;

    persistence::execution_capability_manifest_v1 typed{};
    typed.schema_version =
        persistence::execution_capability_manifest_v1_schema_version;
    typed.record_bytes = sizeof(typed);
    typed.endian = persistence::execution_capability_manifest_v1_endian_marker;
    typed.flags = persistence::capability_source_linked_implementation;
    if ((capability.flags & architecture::capability_fragment_layout_opaque) != 0u)
        typed.flags |= persistence::capability_fragment_layout_opaque;
    if ((capability.flags
            & architecture::capability_requires_converged_collective) != 0u)
        typed.flags |= persistence::capability_requires_converged_collective;
    if (requires_memory)
        typed.flags |= persistence::capability_memory_interface_present;
    typed.provider_identity_low = provider.identity.low;
    typed.provider_identity_high = provider.identity.high;
    typed.provider_abi_identity_low = provider_abi_identity.low;
    typed.provider_abi_identity_high = provider_abi_identity.high;
    typed.capability_identity_low = capability.identity.low;
    typed.capability_identity_high = capability.identity.high;
    typed.hardware_compatibility_identity_low =
        hardware_compatibility_identity.low;
    typed.hardware_compatibility_identity_high =
        hardware_compatibility_identity.high;
    typed.runtime_build_identity_low = runtime_build_identity.low;
    typed.runtime_build_identity_high = runtime_build_identity.high;
    typed.kernel_build_identity_low = kernel_build_identity.low;
    typed.kernel_build_identity_high = kernel_build_identity.high;
    typed.memory_interface_identity_low = capability.memory_interface_identity.low;
    typed.memory_interface_identity_high = capability.memory_interface_identity.high;
    typed.vendor = vendor(capability.vendor);
    typed.architecture_class = capability.architecture_class;
    typed.minimum_compute_capability_major = capability.minimum_compute_major;
    typed.minimum_compute_capability_minor = capability.minimum_compute_minor;
    typed.maximum_compute_capability_major = capability.maximum_compute_major;
    typed.maximum_compute_capability_minor = capability.maximum_compute_minor;
    typed.instruction_family = instruction_family(capability.instruction_family);
    typed.collective_scope = collective_scope(capability.collective_scope);
    typed.collective_threads = capability.collective_threads;
    typed.instruction_m = capability.instruction_m;
    typed.instruction_n = capability.instruction_n;
    typed.instruction_k = capability.instruction_k;
    typed.relation_storage_type = numeric_type(capability.operand_a_type);
    typed.dense_input_type = numeric_type(capability.operand_b_type);
    typed.accumulation_type = numeric_type(capability.accumulation_type);
    typed.output_type = numeric_type(capability.output_type);
    typed.operand_a_layout = matrix_layout(capability.operand_a_layout);
    typed.operand_b_layout = matrix_layout(capability.operand_b_layout);
    typed.accumulation_layout = matrix_layout(capability.accumulation_layout);
    typed.output_layout = matrix_layout(capability.output_layout);
    typed.instruction_sparsity = instruction_sparsity(
        capability.instruction_sparsity);
    typed.structured_operand = structured_operand(capability.structured_operand);
    typed.structured_group_semantics = structured_group(
        capability.structured_group_semantics);
    typed.memory_interface_flags = memory == nullptr ? 0u : memory->flags;
    typed.required_engine_capability = capability.engine_requirements;
    if (!persistence::validate_execution_capability_manifest_v1(typed))
        return false;

    persistence::execution_section_source capability_section{};
    capability_section.kind =
        persistence::execution_capability_manifest_v1_section_kind;
    capability_section.schema_version =
        persistence::execution_capability_manifest_v1_schema_version;
    capability_section.flags = persistence::directory_optional
        | persistence::directory_device_readable;
    capability_section.alignment = persistence::execution_image_v2_alignment;
    capability_section.identity_low = capability.identity.low;
    capability_section.identity_high = capability.identity.high;
    capability_section.data = manifest;
    capability_section.bytes = sizeof(typed);
    capability_section.element_count = 1u;
    capability_section.element_bytes = sizeof(typed);
    projection.entry.capability_section = capability_section_index;

    execution::projection_reference_binding_v2 binding{};
    binding.key = key;
    binding.provider_identity = {provider.identity.low, provider.identity.high};
    binding.capability_identity = {capability.identity.low, capability.identity.high};
    binding.contract = contract;
    binding.location = device_location;
    binding.view = device_view;
    binding.view_bytes = payload_bytes;
    execution::activated_projection_reference_v2 reference{};
    if (execution::make_activated_projection_reference_v2(binding, &reference)
        != execution::projection_reference_status_v2::success)
        return false;

    *manifest = typed;
    sections[0] = payload;
    sections[1] = capability_section;
    *projection_source = projection;
    *activated = reference;
    return true;
}

} // namespace cellerator::compute::architecture::providers::nvidia
