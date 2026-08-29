#pragma once

#include <Cellerator/execution/operands.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellerator::compute::architecture {

inline constexpr std::uint32_t matrix_engine_capability_schema_version_v1 = 1u;
inline constexpr std::uint32_t matrix_memory_interface_schema_version_v1 = 1u;

struct architecture_identity_v1 {
    std::uint64_t low = 0u;
    std::uint64_t high = 0u;
};

enum class capability_status_v1 : std::uint8_t {
    success = 0u,
    invalid_header = 1u,
    invalid_identity = 2u,
    invalid_flags = 3u,
    invalid_compute_range = 4u,
    invalid_instruction = 5u,
    invalid_numeric_contract = 6u,
    invalid_sparsity_contract = 7u,
    invalid_memory_interface = 8u,
    nonzero_reserved = 9u
};

enum class architecture_vendor_v1 : std::uint16_t {
    invalid = 0u,
    generic = 1u,
    nvidia = 2u
};

enum class matrix_instruction_family_v1 : std::uint16_t {
    invalid = 0u,
    generic_multiply_accumulate = 1u,
    nvidia_wmma = 2u,
    nvidia_mma_sync = 3u
};

enum class collective_scope_v1 : std::uint8_t {
    invalid = 0u,
    thread = 1u,
    warp = 2u,
    warp_group = 3u,
    cooperative_thread_array = 4u
};

enum class matrix_layout_v1 : std::uint8_t {
    invalid = 0u,
    not_applicable = 1u,
    row_major = 2u,
    column_major = 3u,
    opaque = 4u
};

enum class instruction_sparsity_v1 : std::uint8_t {
    invalid = 0u,
    dense = 1u,
    structured = 2u
};

enum class structured_operand_v1 : std::uint8_t {
    none = 0u,
    operand_a = 1u,
    operand_b = 2u
};

enum class structured_group_semantics_v1 : std::uint8_t {
    none = 0u,
    implementation_defined = 1u,
    two_of_four = 2u
};

enum matrix_engine_capability_flag_v1 : std::uint32_t {
    capability_source_linked_implementation = 1u << 0u,
    capability_fragment_layout_opaque = 1u << 1u,
    capability_requires_converged_collective = 1u << 2u,
    capability_memory_interface_present = 1u << 3u
};

inline constexpr std::uint32_t matrix_engine_capability_known_flags_v1 =
    capability_source_linked_implementation
    | capability_fragment_layout_opaque
    | capability_requires_converged_collective
    | capability_memory_interface_present;

enum matrix_engine_requirement_v1 : std::uint32_t {
    matrix_engine_multiply_accumulate = 1u << 0u
};

enum memory_address_space_flag_v1 : std::uint32_t {
    memory_address_generic = 1u << 0u,
    memory_address_global = 1u << 1u,
    memory_address_shared = 1u << 2u
};

inline constexpr std::uint32_t memory_address_space_known_flags_v1 =
    memory_address_generic | memory_address_global | memory_address_shared;

enum matrix_memory_operand_flag_v1 : std::uint32_t {
    memory_operand_read = 1u << 0u,
    memory_operand_write = 1u << 1u
};

inline constexpr std::uint32_t matrix_memory_operand_known_flags_v1 =
    memory_operand_read | memory_operand_write;

enum matrix_memory_interface_flag_v1 : std::uint32_t {
    memory_interface_operand_a = 1u << 0u,
    memory_interface_operand_b = 1u << 1u,
    memory_interface_accumulator = 1u << 2u,
    memory_interface_output = 1u << 3u
};

inline constexpr std::uint32_t matrix_memory_interface_known_flags_v1 =
    memory_interface_operand_a | memory_interface_operand_b
    | memory_interface_accumulator | memory_interface_output;

// One field describes one matrix operand at a versioned memory API boundary.
// Alignment is in bytes; stride and contiguous extent are in logical elements.
// A zero stride/extent multiple means that the interface imposes no restriction.
struct matrix_memory_operand_contract_v1 {
    std::uint32_t base_alignment_bytes = 0u;
    std::uint32_t leading_dimension_multiple_elements = 0u;
    std::uint32_t contiguous_extent_multiple_elements = 0u;
    std::uint32_t address_space_flags = 0u;
    std::uint32_t access_flags = 0u;
    std::uint32_t reserved[3]{};
};

// This record is separate from the register-level matrix-engine contract.
// Providers may therefore advertise an instruction without claiming that a
// particular load/store helper is implemented, or publish several memory
// interfaces for the same instruction capability.
struct matrix_memory_interface_v1 {
    std::uint32_t schema_version = matrix_memory_interface_schema_version_v1;
    std::uint32_t record_bytes = sizeof(matrix_memory_interface_v1);
    architecture_identity_v1 identity{};
    std::uint32_t flags = 0u;
    std::uint32_t reserved0 = 0u;
    matrix_memory_operand_contract_v1 operand_a{};
    matrix_memory_operand_contract_v1 operand_b{};
    matrix_memory_operand_contract_v1 accumulator{};
    matrix_memory_operand_contract_v1 output{};
    std::uint32_t reserved[4]{};
};

// Cold source-linked implementation truth. This record describes code that is
// compiled into a provider, not every instruction documented for compatible
// hardware. Architecture-specific fragment types never cross this boundary.
struct matrix_engine_capability_v1 {
    std::uint32_t schema_version = matrix_engine_capability_schema_version_v1;
    std::uint32_t record_bytes = sizeof(matrix_engine_capability_v1);
    architecture_identity_v1 identity{};
    architecture_identity_v1 provider_identity{};
    architecture_identity_v1 memory_interface_identity{};

    architecture_vendor_v1 vendor = architecture_vendor_v1::invalid;
    std::uint16_t architecture_class = 0u;
    std::uint16_t minimum_compute_major = 0u;
    std::uint16_t minimum_compute_minor = 0u;
    std::uint16_t maximum_compute_major = 0u;
    std::uint16_t maximum_compute_minor = 0u;

    matrix_instruction_family_v1 instruction_family =
        matrix_instruction_family_v1::invalid;
    collective_scope_v1 collective_scope = collective_scope_v1::invalid;
    std::uint8_t reserved0 = 0u;
    std::uint16_t collective_threads = 0u;
    std::uint16_t instruction_m = 0u;
    std::uint16_t instruction_n = 0u;
    std::uint16_t instruction_k = 0u;

    execution::numeric_type operand_a_type = execution::numeric_type::invalid;
    execution::numeric_type operand_b_type = execution::numeric_type::invalid;
    execution::numeric_type accumulation_type = execution::numeric_type::invalid;
    execution::numeric_type output_type = execution::numeric_type::invalid;
    matrix_layout_v1 operand_a_layout = matrix_layout_v1::invalid;
    matrix_layout_v1 operand_b_layout = matrix_layout_v1::invalid;
    matrix_layout_v1 accumulation_layout = matrix_layout_v1::invalid;
    matrix_layout_v1 output_layout = matrix_layout_v1::invalid;

    instruction_sparsity_v1 instruction_sparsity =
        instruction_sparsity_v1::invalid;
    structured_operand_v1 structured_operand = structured_operand_v1::none;
    structured_group_semantics_v1 structured_group_semantics =
        structured_group_semantics_v1::none;
    std::uint8_t reserved1 = 0u;
    std::uint32_t flags = 0u;
    std::uint32_t engine_requirements = 0u;
    std::uint32_t reserved[6]{};
};

constexpr bool valid_architecture_identity_v1(
    architecture_identity_v1 identity) noexcept {
    return identity.low != 0u || identity.high != 0u;
}

constexpr bool same_architecture_identity_v1(
    architecture_identity_v1 lhs,
    architecture_identity_v1 rhs) noexcept {
    return lhs.low == rhs.low && lhs.high == rhs.high;
}

constexpr bool compute_capability_in_range_v1(
    const matrix_engine_capability_v1 &capability,
    std::uint16_t major,
    std::uint16_t minor) noexcept {
    const std::uint32_t value = static_cast<std::uint32_t>(major) * 100u + minor;
    const std::uint32_t minimum =
        static_cast<std::uint32_t>(capability.minimum_compute_major) * 100u
        + capability.minimum_compute_minor;
    const std::uint32_t maximum =
        static_cast<std::uint32_t>(capability.maximum_compute_major) * 100u
        + capability.maximum_compute_minor;
    return value >= minimum && value <= maximum;
}

constexpr capability_status_v1 validate_matrix_memory_operand_contract_v1(
    const matrix_memory_operand_contract_v1 &contract) noexcept {
    if ((contract.address_space_flags
            & ~memory_address_space_known_flags_v1) != 0u
        || contract.address_space_flags == 0u
        || (contract.access_flags & ~matrix_memory_operand_known_flags_v1) != 0u
        || contract.access_flags == 0u
        || contract.base_alignment_bytes == 0u
        || (contract.base_alignment_bytes
            & (contract.base_alignment_bytes - 1u)) != 0u)
        return capability_status_v1::invalid_memory_interface;
    for (std::uint32_t value : contract.reserved)
        if (value != 0u) return capability_status_v1::nonzero_reserved;
    return capability_status_v1::success;
}

constexpr capability_status_v1 validate_matrix_memory_interface_v1(
    const matrix_memory_interface_v1 &interface) noexcept {
    if (interface.schema_version != matrix_memory_interface_schema_version_v1
        || interface.record_bytes != sizeof(matrix_memory_interface_v1))
        return capability_status_v1::invalid_header;
    if (!valid_architecture_identity_v1(interface.identity))
        return capability_status_v1::invalid_identity;
    if (interface.flags == 0u
        || (interface.flags & ~matrix_memory_interface_known_flags_v1) != 0u)
        return capability_status_v1::invalid_flags;
    if (interface.reserved0 != 0u)
        return capability_status_v1::nonzero_reserved;

    const matrix_memory_operand_contract_v1 contracts[4] = {
        interface.operand_a, interface.operand_b,
        interface.accumulator, interface.output};
    const std::uint32_t present[4] = {
        memory_interface_operand_a, memory_interface_operand_b,
        memory_interface_accumulator, memory_interface_output};
    for (std::size_t index = 0u; index < 4u; ++index) {
        if ((interface.flags & present[index]) != 0u) {
            const capability_status_v1 status =
                validate_matrix_memory_operand_contract_v1(contracts[index]);
            if (status != capability_status_v1::success) return status;
        } else if (contracts[index].base_alignment_bytes != 0u
            || contracts[index].leading_dimension_multiple_elements != 0u
            || contracts[index].contiguous_extent_multiple_elements != 0u
            || contracts[index].address_space_flags != 0u
            || contracts[index].access_flags != 0u) {
            return capability_status_v1::invalid_memory_interface;
        }
    }
    for (std::uint32_t value : interface.reserved)
        if (value != 0u) return capability_status_v1::nonzero_reserved;
    return capability_status_v1::success;
}

constexpr capability_status_v1 validate_matrix_engine_capability_v1(
    const matrix_engine_capability_v1 &capability) noexcept {
    if (capability.schema_version != matrix_engine_capability_schema_version_v1
        || capability.record_bytes != sizeof(matrix_engine_capability_v1))
        return capability_status_v1::invalid_header;
    if (!valid_architecture_identity_v1(capability.identity)
        || !valid_architecture_identity_v1(capability.provider_identity))
        return capability_status_v1::invalid_identity;
    if ((capability.flags & ~matrix_engine_capability_known_flags_v1) != 0u
        || (capability.flags & capability_source_linked_implementation) == 0u)
        return capability_status_v1::invalid_flags;
    if (capability.minimum_compute_major == 0u
        || capability.maximum_compute_major == 0u
        || capability.minimum_compute_minor > 99u
        || capability.maximum_compute_minor > 99u
        || !compute_capability_in_range_v1(capability,
            capability.minimum_compute_major,
            capability.minimum_compute_minor)
        || !compute_capability_in_range_v1(capability,
            capability.maximum_compute_major,
            capability.maximum_compute_minor))
        return capability_status_v1::invalid_compute_range;
    if (capability.vendor == architecture_vendor_v1::invalid
        || capability.architecture_class == 0u
        || capability.instruction_family == matrix_instruction_family_v1::invalid
        || capability.collective_scope == collective_scope_v1::invalid
        || capability.collective_threads == 0u
        || capability.instruction_m == 0u || capability.instruction_n == 0u
        || capability.instruction_k == 0u
        || capability.engine_requirements == 0u)
        return capability_status_v1::invalid_instruction;
    if (capability.operand_a_type == execution::numeric_type::invalid
        || capability.operand_b_type == execution::numeric_type::invalid
        || capability.accumulation_type == execution::numeric_type::invalid
        || capability.output_type == execution::numeric_type::invalid
        || capability.operand_a_layout == matrix_layout_v1::invalid
        || capability.operand_b_layout == matrix_layout_v1::invalid
        || capability.accumulation_layout == matrix_layout_v1::invalid
        || capability.output_layout == matrix_layout_v1::invalid)
        return capability_status_v1::invalid_numeric_contract;

    const bool structured = capability.instruction_sparsity
        == instruction_sparsity_v1::structured;
    if (capability.instruction_sparsity == instruction_sparsity_v1::invalid
        || structured != (capability.structured_operand
            != structured_operand_v1::none)
        || structured != (capability.structured_group_semantics
            != structured_group_semantics_v1::none))
        return capability_status_v1::invalid_sparsity_contract;

    const bool memory_interface_present =
        (capability.flags & capability_memory_interface_present) != 0u;
    if (memory_interface_present
        != valid_architecture_identity_v1(capability.memory_interface_identity))
        return capability_status_v1::invalid_memory_interface;
    if (capability.reserved0 != 0u || capability.reserved1 != 0u)
        return capability_status_v1::nonzero_reserved;
    for (std::uint32_t value : capability.reserved)
        if (value != 0u) return capability_status_v1::nonzero_reserved;
    return capability_status_v1::success;
}

static_assert(std::is_trivially_copyable<architecture_identity_v1>::value,
    "architecture identities must remain trivially copyable");
static_assert(std::is_trivially_copyable<matrix_engine_capability_v1>::value,
    "matrix-engine capabilities must remain trivially copyable");
static_assert(std::is_standard_layout<matrix_engine_capability_v1>::value,
    "matrix-engine capabilities must remain field-addressable");
static_assert(std::is_trivially_copyable<matrix_memory_interface_v1>::value,
    "matrix memory-interface contracts must remain trivially copyable");
static_assert(std::is_standard_layout<matrix_memory_interface_v1>::value,
    "matrix memory-interface contracts must remain field-addressable");

} // namespace cellerator::compute::architecture
