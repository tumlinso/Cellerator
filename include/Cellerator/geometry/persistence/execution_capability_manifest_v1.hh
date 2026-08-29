#pragma once

#include "Cellerator/geometry/persistence/execution_image_v2.hh"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellpack::persistence {

inline constexpr u32 execution_capability_manifest_v1_schema_version = 1u;
inline constexpr u32 execution_capability_manifest_v1_endian_marker =
    execution_image_v2_endian_marker;
inline constexpr execution_section_kind
    execution_capability_manifest_v1_section_kind =
        static_cast<execution_section_kind>(0x80000001u);

enum execution_capability_manifest_flags_v1 : u32 {
    capability_source_linked_implementation = 1u << 0u,
    capability_fragment_layout_opaque = 1u << 1u,
    capability_requires_converged_collective = 1u << 2u,
    capability_memory_interface_present = 1u << 3u
};

inline constexpr u32 execution_capability_manifest_known_flags_v1 =
    capability_source_linked_implementation
    | capability_fragment_layout_opaque
    | capability_requires_converged_collective
    | capability_memory_interface_present;

enum class execution_capability_vendor_v1 : u32 {
    invalid = 0u,
    generic = 1u,
    nvidia = 2u
};

enum class execution_instruction_family_v1 : u32 {
    invalid = 0u,
    generic_scalar = 1u,
    nvidia_wmma = 2u,
    nvidia_mma_sync = 3u
};

enum class execution_collective_scope_v1 : u32 {
    invalid = 0u,
    thread = 1u,
    warp = 2u,
    warp_group = 3u,
    cooperative_thread_array = 4u
};

enum class execution_capability_numeric_type_v1 : u32 {
    invalid = 0u,
    bit = 1u,
    u8 = 2u,
    u16 = 3u,
    u32 = 4u,
    i32 = 5u,
    f16 = 6u,
    bf16 = 7u,
    f32 = 8u,
    f64 = 9u
};

enum class execution_matrix_layout_v1 : u32 {
    invalid = 0u,
    not_applicable = 1u,
    row_major = 2u,
    column_major = 3u,
    opaque = 4u
};

enum class execution_instruction_sparsity_v1 : u32 {
    invalid = 0u,
    dense = 1u,
    structured = 2u
};

enum class execution_structured_operand_v1 : u32 {
    none = 0u,
    operand_a = 1u,
    operand_b = 2u
};

enum class execution_structured_group_semantics_v1 : u32 {
    none = 0u,
    implementation_defined = 1u,
    two_of_four = 2u
};

// This record is cold, pointer-free persistent metadata. It describes the
// source-linked capability required by one device-specific projection; it does
// not claim that the active device provides the capability. Runtime activation
// compares these identities and bounds with the canonical device descriptor.
struct execution_capability_manifest_v1 {
    u32 schema_version;
    u32 record_bytes;
    u32 endian;
    u32 flags;

    u64 provider_identity_low;
    u64 provider_identity_high;
    u64 provider_abi_identity_low;
    u64 provider_abi_identity_high;
    u64 capability_identity_low;
    u64 capability_identity_high;
    u64 hardware_compatibility_identity_low;
    u64 hardware_compatibility_identity_high;
    u64 runtime_build_identity_low;
    u64 runtime_build_identity_high;
    u64 kernel_build_identity_low;
    u64 kernel_build_identity_high;
    u64 memory_interface_identity_low;
    u64 memory_interface_identity_high;

    execution_capability_vendor_v1 vendor;
    u32 architecture_class;
    u32 minimum_compute_capability_major;
    u32 minimum_compute_capability_minor;
    u32 maximum_compute_capability_major;
    u32 maximum_compute_capability_minor;

    execution_instruction_family_v1 instruction_family;
    execution_collective_scope_v1 collective_scope;
    u32 collective_threads;
    u32 instruction_m;
    u32 instruction_n;
    u32 instruction_k;

    execution_capability_numeric_type_v1 relation_storage_type;
    execution_capability_numeric_type_v1 dense_input_type;
    execution_capability_numeric_type_v1 accumulation_type;
    execution_capability_numeric_type_v1 output_type;
    execution_matrix_layout_v1 operand_a_layout;
    execution_matrix_layout_v1 operand_b_layout;
    execution_matrix_layout_v1 accumulation_layout;
    execution_matrix_layout_v1 output_layout;

    execution_instruction_sparsity_v1 instruction_sparsity;
    execution_structured_operand_v1 structured_operand;
    execution_structured_group_semantics_v1 structured_group_semantics;
    u32 memory_interface_flags;
    u32 required_engine_capability;
    u32 reserved[7];
};

inline bool execution_capability_identity_present_v1(u64 low, u64 high) noexcept {
    return low != 0u || high != 0u;
}

inline bool execution_capability_compute_range_valid_v1(
    const execution_capability_manifest_v1 &manifest) noexcept {
    if (manifest.minimum_compute_capability_major == 0u
        || manifest.maximum_compute_capability_major == 0u
        || manifest.minimum_compute_capability_minor > 99u
        || manifest.maximum_compute_capability_minor > 99u)
        return false;
    return manifest.minimum_compute_capability_major
            < manifest.maximum_compute_capability_major
        || (manifest.minimum_compute_capability_major
                == manifest.maximum_compute_capability_major
            && manifest.minimum_compute_capability_minor
                <= manifest.maximum_compute_capability_minor);
}

inline validation_result validate_execution_capability_manifest_v1(
    const execution_capability_manifest_v1 &manifest) noexcept {
    if (manifest.schema_version != execution_capability_manifest_v1_schema_version
        || manifest.record_bytes != sizeof(execution_capability_manifest_v1)
        || manifest.endian != execution_capability_manifest_v1_endian_marker)
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "execution capability manifest header is invalid");
    if ((manifest.flags & ~execution_capability_manifest_known_flags_v1) != 0u
        || (manifest.flags & capability_source_linked_implementation) == 0u)
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "execution capability manifest flags are invalid");
    if (!execution_capability_identity_present_v1(manifest.provider_identity_low,
            manifest.provider_identity_high)
        || !execution_capability_identity_present_v1(
            manifest.provider_abi_identity_low, manifest.provider_abi_identity_high)
        || !execution_capability_identity_present_v1(
            manifest.capability_identity_low, manifest.capability_identity_high)
        || !execution_capability_identity_present_v1(
            manifest.hardware_compatibility_identity_low,
            manifest.hardware_compatibility_identity_high)
        || !execution_capability_identity_present_v1(
            manifest.runtime_build_identity_low, manifest.runtime_build_identity_high)
        || !execution_capability_identity_present_v1(
            manifest.kernel_build_identity_low, manifest.kernel_build_identity_high))
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "execution capability manifest identity is invalid");

    const bool memory_interface_present =
        (manifest.flags & capability_memory_interface_present) != 0u;
    const bool memory_interface_identity_present =
        execution_capability_identity_present_v1(
            manifest.memory_interface_identity_low,
            manifest.memory_interface_identity_high);
    if (memory_interface_present != memory_interface_identity_present
        || (!memory_interface_present && manifest.memory_interface_flags != 0u))
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "execution capability memory interface is inconsistent");

    if (manifest.vendor == execution_capability_vendor_v1::invalid
        || manifest.architecture_class == 0u
        || !execution_capability_compute_range_valid_v1(manifest)
        || manifest.instruction_family == execution_instruction_family_v1::invalid
        || manifest.collective_scope == execution_collective_scope_v1::invalid
        || manifest.collective_threads == 0u || manifest.instruction_m == 0u
        || manifest.instruction_n == 0u || manifest.instruction_k == 0u
        || manifest.relation_storage_type
            == execution_capability_numeric_type_v1::invalid
        || manifest.dense_input_type == execution_capability_numeric_type_v1::invalid
        || manifest.accumulation_type == execution_capability_numeric_type_v1::invalid
        || manifest.output_type == execution_capability_numeric_type_v1::invalid
        || manifest.operand_a_layout == execution_matrix_layout_v1::invalid
        || manifest.operand_b_layout == execution_matrix_layout_v1::invalid
        || manifest.accumulation_layout == execution_matrix_layout_v1::invalid
        || manifest.output_layout == execution_matrix_layout_v1::invalid
        || manifest.instruction_sparsity
            == execution_instruction_sparsity_v1::invalid
        || manifest.required_engine_capability == 0u)
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "execution capability instruction contract is invalid");

    const bool structured = manifest.instruction_sparsity
        == execution_instruction_sparsity_v1::structured;
    if (structured != (manifest.structured_operand
                != execution_structured_operand_v1::none)
        || structured != (manifest.structured_group_semantics
                != execution_structured_group_semantics_v1::none))
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "execution capability structured sparsity contract is inconsistent");
    for (u32 value : manifest.reserved)
        if (value != 0u)
            return validation_error(validation_code::invalid_matrix_view, invalid_id,
                "execution capability reserved fields are nonzero");
    return validation_ok();
}

// Resolve the existing CPE2 capability-section hook after the image has passed
// ordinary CPE2 validation. Older readers continue to ignore this optional
// extension and prebound_projection_view_v1 remains unchanged.
inline validation_result bind_execution_capability_manifest_v1_host(
    const execution_image_v2_view &validated_host_view,
    u32 projection_index,
    const execution_capability_manifest_v1 **out) noexcept {
    if (validated_host_view.image_base == nullptr
        || validated_host_view.sections == nullptr
        || validated_host_view.projections == nullptr || out == nullptr)
        return validation_error(validation_code::null_pointer, invalid_id,
            "execution capability bind input is null");
    if (projection_index >= validated_host_view.header.projection_count)
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "execution capability projection index is out of range");
    const execution_projection_entry_v1 &projection =
        validated_host_view.projections[projection_index];
    if (projection.capability_section == invalid_directory_index
        || projection.capability_section >= validated_host_view.header.section_count)
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "execution projection has no capability manifest");
    const execution_section_entry_v1 &section =
        validated_host_view.sections[projection.capability_section];
    if (section.kind != execution_capability_manifest_v1_section_kind
        || section.schema_version != execution_capability_manifest_v1_schema_version
        || (section.flags & (directory_optional | directory_device_readable))
            != (directory_optional | directory_device_readable)
        || section.bytes != sizeof(execution_capability_manifest_v1)
        || section.element_count != 1u
        || section.element_bytes != sizeof(execution_capability_manifest_v1))
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "execution capability section shape is invalid");
    const auto *manifest = reinterpret_cast<const execution_capability_manifest_v1 *>(
        static_cast<const unsigned char *>(validated_host_view.image_base)
        + section.offset);
    const validation_result status =
        validate_execution_capability_manifest_v1(*manifest);
    if (!static_cast<bool>(status)) return status;
    if (projection.architecture_class != manifest->architecture_class)
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "projection architecture and capability manifest differ");
    *out = manifest;
    return validation_ok();
}

static_assert(sizeof(execution_capability_manifest_v1) == 256u,
    "execution capability manifest v1 must remain one 256-byte wire record");
static_assert(std::is_trivially_copyable<execution_capability_manifest_v1>::value,
    "execution capability manifest must remain pointer-free");
static_assert(std::is_standard_layout<execution_capability_manifest_v1>::value,
    "execution capability manifest must remain field-addressable");
static_assert(sizeof(execution_image_v2_header) == 256u,
    "capability extension must not change the CPE2 header");
static_assert(sizeof(execution_section_entry_v1) == 64u,
    "capability extension must not change CPE2 section entries");
static_assert(sizeof(execution_projection_entry_v1) == 64u,
    "capability extension must not change CPE2 projection entries");

} // namespace cellpack::persistence
