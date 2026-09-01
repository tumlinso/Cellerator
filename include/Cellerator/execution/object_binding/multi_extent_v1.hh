#pragma once

#include <Cellerator/execution/geometry_acquisition_v2/schema.hh>

#include <cstdint>
#include <cstring>
#include <type_traits>

namespace cellerator::execution::object_binding {

using stable_identity_v1 = acquisition_v2::stable_identity;

enum class binding_status_code_v1 : std::uint8_t {
    success = 0,
    invalid_argument,
    invalid_identity,
    duplicate_port,
    duplicate_atom,
    invalid_extent,
    insufficient_capacity,
    incompatible_requirement,
};

struct binding_status_v1 {
    binding_status_code_v1 code = binding_status_code_v1::success;
    std::uint64_t index = 0u;
    std::uint64_t required_capacity = 0u;

    constexpr explicit operator bool() const noexcept {
        return code == binding_status_code_v1::success;
    }
};

enum class port_access_v1 : std::uint8_t {
    read_only = 1u,
    write_only = 2u,
    read_write = 3u,
};

struct atom_port_binding_v1 {
    stable_identity_v1 atom_identity{};
    std::uint64_t logical_begin = 0u;
    std::uint64_t logical_extent = 0u;
};

struct multi_atom_port_binding_v1 {
    stable_identity_v1 port_identity{};
    stable_identity_v1 domain_identity{};
    stable_identity_v1 order_identity{};
    const atom_port_binding_v1 *atoms = nullptr;
    std::uint64_t atom_count = 0u;
    port_access_v1 access = port_access_v1::read_only;
    std::uint8_t reserved[7]{};
};

struct multi_atom_port_binding_list_v1 {
    const multi_atom_port_binding_v1 *ports = nullptr;
    std::uint64_t port_count = 0u;
};

enum class extent_residency_v1 : std::uint8_t {
    host = 1u,
    device = 2u,
    managed = 3u,
};

struct physical_extent_binding_v1 {
    stable_identity_v1 atom_identity{};
    const void *data = nullptr;
    std::uint64_t byte_count = 0u;
    std::uint64_t element_count = 0u;
    std::uint64_t element_stride_bytes = 0u;
    std::uint64_t alignment_bytes = 1u;
    std::uint64_t value_generation = 0u;
    extent_residency_v1 residency = extent_residency_v1::host;
    std::uint8_t reserved[7]{};
};

struct multi_extent_physical_binding_v1 {
    stable_identity_v1 port_identity{};
    const physical_extent_binding_v1 *extents = nullptr;
    std::uint64_t extent_count = 0u;
};

struct multi_extent_physical_binding_list_v1 {
    const multi_extent_physical_binding_v1 *ports = nullptr;
    std::uint64_t port_count = 0u;
};

enum class contiguity_requirement_v1 : std::uint8_t {
    multi_extent_allowed = 1u,
    contiguous_required = 2u,
};

struct port_extent_requirement_v1 {
    stable_identity_v1 port_identity{};
    std::uint64_t minimum_alignment_bytes = 1u;
    std::uint64_t maximum_extent_count = UINT64_MAX;
    contiguity_requirement_v1 contiguity =
        contiguity_requirement_v1::multi_extent_allowed;
    bool uniform_value_generation_required = false;
    std::uint8_t reserved[6]{};
};

struct port_extent_query_result_v1 {
    stable_identity_v1 port_identity{};
    bool directly_compatible = false;
    bool assembly_required = false;
    std::uint8_t reserved[6]{};
    std::uint64_t extent_count = 0u;
    std::uint64_t logical_element_count = 0u;
    std::uint64_t contiguous_bytes = 0u;
    std::uint64_t required_alignment_bytes = 1u;
};

struct contiguous_assembly_segment_v1 {
    const void *source = nullptr;
    std::uint64_t source_bytes = 0u;
    std::uint64_t destination_offset_bytes = 0u;
};

struct contiguous_assembly_plan_v1 {
    stable_identity_v1 port_identity{};
    const contiguous_assembly_segment_v1 *segments = nullptr;
    std::uint64_t segment_count = 0u;
    std::uint64_t destination_bytes = 0u;
    std::uint64_t destination_alignment_bytes = 1u;
};

struct index_permutation_v1 {
    const std::uint64_t *source_index_for_destination = nullptr;
    std::uint64_t element_count = 0u;
};

enum extent_residency_flag_v1 : std::uint8_t {
    host_extent_v1 = 1u << 0u,
    device_extent_v1 = 1u << 1u,
    managed_extent_v1 = 1u << 2u,
};

struct direct_multi_extent_candidate_requirements_v1 {
    std::uint64_t maximum_extent_count = 0u;
    std::uint64_t minimum_alignment_bytes = 1u;
    std::uint64_t element_stride_bytes = 0u;
    std::uint8_t accepted_residencies = device_extent_v1;
    bool accepts_mixed_value_generations = false;
    bool preserves_logical_order = true;
    std::uint8_t reserved[5]{};
};

using direct_multi_extent_launch_v1 = binding_status_v1 (*)(
    const void *prepared_state,
    const multi_extent_physical_binding_v1 &input,
    void *output, std::uint64_t output_bytes,
    void *caller_stream) noexcept;

struct direct_multi_extent_candidate_v1 {
    stable_identity_v1 candidate_identity{};
    direct_multi_extent_candidate_requirements_v1 requirements{};
    const void *prepared_state = nullptr;
    direct_multi_extent_launch_v1 launch = nullptr;
};

constexpr bool valid_identity_v1(stable_identity_v1 identity) noexcept {
    return identity.low != 0u || identity.high != 0u;
}

inline binding_status_v1 validate_multi_atom_port_bindings_v1(
    const multi_atom_port_binding_list_v1 &list) noexcept {
    if (list.port_count != 0u && list.ports == nullptr) {
        return {binding_status_code_v1::invalid_argument};
    }
    for (std::uint64_t port_index = 0u; port_index < list.port_count;
         ++port_index) {
        const auto &port = list.ports[port_index];
        if (!valid_identity_v1(port.port_identity) ||
            !valid_identity_v1(port.domain_identity) ||
            !valid_identity_v1(port.order_identity)) {
            return {binding_status_code_v1::invalid_identity, port_index};
        }
        if (port.atom_count == 0u || port.atoms == nullptr) {
            return {binding_status_code_v1::invalid_argument, port_index};
        }
        for (std::uint64_t other = 0u; other < port_index; ++other) {
            const auto &identity = list.ports[other].port_identity;
            if (identity.low == port.port_identity.low &&
                identity.high == port.port_identity.high) {
                return {binding_status_code_v1::duplicate_port, port_index};
            }
        }
        for (std::uint64_t atom_index = 0u; atom_index < port.atom_count;
             ++atom_index) {
            const auto &atom = port.atoms[atom_index];
            if (!valid_identity_v1(atom.atom_identity)) {
                return {binding_status_code_v1::invalid_identity, atom_index};
            }
            if (atom.logical_extent == 0u ||
                atom.logical_begin > UINT64_MAX - atom.logical_extent) {
                return {binding_status_code_v1::invalid_extent, atom_index};
            }
            for (std::uint64_t other = 0u; other < atom_index; ++other) {
                const auto &identity = port.atoms[other].atom_identity;
                if (identity.low == atom.atom_identity.low &&
                    identity.high == atom.atom_identity.high) {
                    return {binding_status_code_v1::duplicate_atom, atom_index};
                }
            }
        }
    }
    return {};
}

constexpr bool power_of_two_v1(std::uint64_t value) noexcept {
    return value != 0u && (value & (value - 1u)) == 0u;
}

inline binding_status_v1 validate_multi_extent_physical_bindings_v1(
    const multi_extent_physical_binding_list_v1 &list) noexcept {
    if (list.port_count != 0u && list.ports == nullptr) {
        return {binding_status_code_v1::invalid_argument};
    }
    for (std::uint64_t port_index = 0u; port_index < list.port_count;
         ++port_index) {
        const auto &port = list.ports[port_index];
        if (!valid_identity_v1(port.port_identity)) {
            return {binding_status_code_v1::invalid_identity, port_index};
        }
        if (port.extent_count == 0u || port.extents == nullptr) {
            return {binding_status_code_v1::invalid_argument, port_index};
        }
        for (std::uint64_t extent_index = 0u;
             extent_index < port.extent_count; ++extent_index) {
            const auto &extent = port.extents[extent_index];
            if (!valid_identity_v1(extent.atom_identity)) {
                return {binding_status_code_v1::invalid_identity, extent_index};
            }
            if (extent.data == nullptr || extent.byte_count == 0u ||
                extent.element_count == 0u ||
                extent.element_stride_bytes == 0u ||
                !power_of_two_v1(extent.alignment_bytes) ||
                extent.element_count >
                    extent.byte_count / extent.element_stride_bytes) {
                return {binding_status_code_v1::invalid_extent, extent_index};
            }
            for (std::uint64_t other = 0u; other < extent_index; ++other) {
                const auto &identity = port.extents[other].atom_identity;
                if (identity.low == extent.atom_identity.low &&
                    identity.high == extent.atom_identity.high) {
                    return {binding_status_code_v1::duplicate_atom, extent_index};
                }
            }
        }
    }
    return {};
}

inline binding_status_v1 query_port_extent_requirements_v1(
    const multi_atom_port_binding_v1 &logical,
    const multi_extent_physical_binding_v1 &physical,
    const port_extent_requirement_v1 &requirement,
    port_extent_query_result_v1 *result) noexcept {
    if (result == nullptr ||
        logical.port_identity.low != physical.port_identity.low ||
        logical.port_identity.high != physical.port_identity.high ||
        logical.port_identity.low != requirement.port_identity.low ||
        logical.port_identity.high != requirement.port_identity.high) {
        return {binding_status_code_v1::invalid_argument};
    }
    *result = {};
    result->port_identity = logical.port_identity;
    result->extent_count = physical.extent_count;
    result->required_alignment_bytes = requirement.minimum_alignment_bytes;
    if (!power_of_two_v1(requirement.minimum_alignment_bytes) ||
        logical.atom_count != physical.extent_count ||
        physical.extents == nullptr || logical.atoms == nullptr) {
        return {binding_status_code_v1::incompatible_requirement};
    }
    bool compatible = physical.extent_count <= requirement.maximum_extent_count;
    std::uint64_t generation = physical.extent_count == 0u ? 0u :
        physical.extents[0].value_generation;
    for (std::uint64_t index = 0u; index < logical.atom_count; ++index) {
        const auto &atom = logical.atoms[index];
        const auto &extent = physical.extents[index];
        if (atom.atom_identity.low != extent.atom_identity.low ||
            atom.atom_identity.high != extent.atom_identity.high ||
            atom.logical_extent != extent.element_count ||
            extent.alignment_bytes < requirement.minimum_alignment_bytes ||
            (requirement.uniform_value_generation_required &&
                extent.value_generation != generation)) {
            compatible = false;
        }
        if (result->logical_element_count >
                UINT64_MAX - atom.logical_extent ||
            result->contiguous_bytes > UINT64_MAX - extent.byte_count) {
            return {binding_status_code_v1::invalid_extent, index};
        }
        result->logical_element_count += atom.logical_extent;
        result->contiguous_bytes += extent.byte_count;
    }
    result->assembly_required = requirement.contiguity ==
            contiguity_requirement_v1::contiguous_required &&
        physical.extent_count > 1u;
    result->directly_compatible = compatible && !result->assembly_required;
    return {};
}

inline binding_status_v1 compile_contiguous_assembly_v1(
    const multi_extent_physical_binding_v1 &physical,
    std::uint64_t destination_alignment_bytes,
    contiguous_assembly_segment_v1 *segments,
    std::uint64_t segment_capacity,
    contiguous_assembly_plan_v1 *plan) noexcept {
    if (plan == nullptr || !valid_identity_v1(physical.port_identity) ||
        !power_of_two_v1(destination_alignment_bytes) ||
        (physical.extent_count != 0u && physical.extents == nullptr)) {
        return {binding_status_code_v1::invalid_argument};
    }
    *plan = {};
    plan->port_identity = physical.port_identity;
    plan->destination_alignment_bytes = destination_alignment_bytes;
    plan->segment_count = physical.extent_count;
    if (segment_capacity < physical.extent_count ||
        (physical.extent_count != 0u && segments == nullptr)) {
        return {binding_status_code_v1::insufficient_capacity, 0u,
            physical.extent_count};
    }
    std::uint64_t offset = 0u;
    for (std::uint64_t index = 0u; index < physical.extent_count; ++index) {
        const auto &extent = physical.extents[index];
        if (extent.data == nullptr || extent.byte_count == 0u ||
            offset > UINT64_MAX - extent.byte_count) {
            return {binding_status_code_v1::invalid_extent, index};
        }
        segments[index] = {extent.data, extent.byte_count, offset};
        offset += extent.byte_count;
    }
    plan->segments = segments;
    plan->destination_bytes = offset;
    return {};
}

inline binding_status_v1 execute_contiguous_assembly_v1(
    const contiguous_assembly_plan_v1 &plan, void *destination,
    std::uint64_t destination_capacity) noexcept {
    if ((plan.destination_bytes != 0u && destination == nullptr) ||
        destination_capacity < plan.destination_bytes ||
        (plan.segment_count != 0u && plan.segments == nullptr)) {
        return {binding_status_code_v1::insufficient_capacity, 0u,
            plan.destination_bytes};
    }
    auto *bytes = static_cast<unsigned char *>(destination);
    for (std::uint64_t index = 0u; index < plan.segment_count; ++index) {
        const auto &segment = plan.segments[index];
        std::memcpy(bytes + segment.destination_offset_bytes,
            segment.source, static_cast<std::size_t>(segment.source_bytes));
    }
    return {};
}

inline binding_status_v1 validate_index_permutation_v1(
    const index_permutation_v1 &permutation) noexcept {
    if (permutation.element_count != 0u &&
        permutation.source_index_for_destination == nullptr) {
        return {binding_status_code_v1::invalid_argument};
    }
    for (std::uint64_t destination = 0u;
         destination < permutation.element_count; ++destination) {
        const auto source =
            permutation.source_index_for_destination[destination];
        if (source >= permutation.element_count) {
            return {binding_status_code_v1::invalid_extent, destination};
        }
        for (std::uint64_t prior = 0u; prior < destination; ++prior) {
            if (permutation.source_index_for_destination[prior] == source) {
                return {binding_status_code_v1::duplicate_atom, destination};
            }
        }
    }
    return {};
}

inline binding_status_v1 gather_permutation_v1(
    const void *source, void *destination, std::uint64_t element_bytes,
    const index_permutation_v1 &permutation) noexcept {
    const auto status = validate_index_permutation_v1(permutation);
    if (!status) {
        return status;
    }
    if (element_bytes == 0u ||
        (permutation.element_count != 0u &&
            (source == nullptr || destination == nullptr ||
                source == destination))) {
        return {binding_status_code_v1::invalid_argument};
    }
    const auto *source_bytes = static_cast<const unsigned char *>(source);
    auto *destination_bytes = static_cast<unsigned char *>(destination);
    for (std::uint64_t destination_index = 0u;
         destination_index < permutation.element_count; ++destination_index) {
        const auto source_index =
            permutation.source_index_for_destination[destination_index];
        std::memcpy(destination_bytes + destination_index * element_bytes,
            source_bytes + source_index * element_bytes,
            static_cast<std::size_t>(element_bytes));
    }
    return {};
}

inline binding_status_v1 scatter_permutation_v1(
    const void *source, void *destination, std::uint64_t element_bytes,
    const index_permutation_v1 &permutation) noexcept {
    const auto status = validate_index_permutation_v1(permutation);
    if (!status) {
        return status;
    }
    if (element_bytes == 0u ||
        (permutation.element_count != 0u &&
            (source == nullptr || destination == nullptr ||
                source == destination))) {
        return {binding_status_code_v1::invalid_argument};
    }
    const auto *source_bytes = static_cast<const unsigned char *>(source);
    auto *destination_bytes = static_cast<unsigned char *>(destination);
    for (std::uint64_t source_index = 0u;
         source_index < permutation.element_count; ++source_index) {
        const auto destination_index =
            permutation.source_index_for_destination[source_index];
        std::memcpy(destination_bytes + destination_index * element_bytes,
            source_bytes + source_index * element_bytes,
            static_cast<std::size_t>(element_bytes));
    }
    return {};
}

constexpr std::uint8_t residency_flag_v1(extent_residency_v1 residency) noexcept {
    switch (residency) {
        case extent_residency_v1::host:
            return host_extent_v1;
        case extent_residency_v1::device:
            return device_extent_v1;
        case extent_residency_v1::managed:
            return managed_extent_v1;
    }
    return 0u;
}

inline binding_status_v1 validate_direct_multi_extent_candidate_v1(
    const direct_multi_extent_candidate_v1 &candidate,
    const multi_extent_physical_binding_v1 &input) noexcept {
    if (!valid_identity_v1(candidate.candidate_identity) ||
        candidate.launch == nullptr ||
        candidate.requirements.maximum_extent_count == 0u ||
        !power_of_two_v1(candidate.requirements.minimum_alignment_bytes) ||
        candidate.requirements.element_stride_bytes == 0u) {
        return {binding_status_code_v1::invalid_argument};
    }
    const multi_extent_physical_binding_list_v1 input_list{&input, 1u};
    const auto input_status =
        validate_multi_extent_physical_bindings_v1(input_list);
    if (!input_status) {
        return input_status;
    }
    if (input.extent_count > candidate.requirements.maximum_extent_count) {
        return {binding_status_code_v1::incompatible_requirement,
            input.extent_count};
    }
    const auto generation = input.extents[0].value_generation;
    for (std::uint64_t index = 0u; index < input.extent_count; ++index) {
        const auto &extent = input.extents[index];
        if (extent.alignment_bytes <
                candidate.requirements.minimum_alignment_bytes ||
            extent.element_stride_bytes !=
                candidate.requirements.element_stride_bytes ||
            (candidate.requirements.accepted_residencies &
                residency_flag_v1(extent.residency)) == 0u ||
            (!candidate.requirements.accepts_mixed_value_generations &&
                extent.value_generation != generation)) {
            return {binding_status_code_v1::incompatible_requirement, index};
        }
    }
    return {};
}

static_assert(std::is_trivially_copyable_v<atom_port_binding_v1>);
static_assert(std::is_trivially_copyable_v<multi_atom_port_binding_v1>);
static_assert(std::is_trivially_copyable_v<multi_atom_port_binding_list_v1>);
static_assert(std::is_trivially_copyable_v<physical_extent_binding_v1>);
static_assert(std::is_trivially_copyable_v<multi_extent_physical_binding_v1>);
static_assert(
    std::is_trivially_copyable_v<multi_extent_physical_binding_list_v1>);
static_assert(std::is_trivially_copyable_v<port_extent_requirement_v1>);
static_assert(std::is_trivially_copyable_v<port_extent_query_result_v1>);
static_assert(std::is_trivially_copyable_v<contiguous_assembly_segment_v1>);
static_assert(std::is_trivially_copyable_v<contiguous_assembly_plan_v1>);
static_assert(std::is_trivially_copyable_v<index_permutation_v1>);
static_assert(std::is_trivially_copyable_v<
    direct_multi_extent_candidate_requirements_v1>);
static_assert(std::is_trivially_copyable_v<direct_multi_extent_candidate_v1>);

}  // namespace cellerator::execution::object_binding
