#pragma once

#include <Cellerator/execution/joint_compiler/atom_requirement_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::execution::joint_compiler {

inline constexpr std::uint32_t atom_affordance_schema_version_v1 = 1u;

enum atom_affordance_flag_v1 : std::uint32_t {
    multi_extent_legal_v1 = 1u << 0u,
    direct_gradient_support_v1 = 1u << 1u,
    direct_output_support_v1 = 1u << 2u,
    persistence_eligible_v1 = 1u << 3u,
    graph_stable_address_available_v1 = 1u << 4u
};

inline constexpr std::uint32_t known_atom_affordance_flags_v1 =
    multi_extent_legal_v1 | direct_gradient_support_v1
    | direct_output_support_v1 | persistence_eligible_v1
    | graph_stable_address_available_v1;

struct atom_plane_affordance_v1 {
    persistent_identity_v1 plane_identity{};
    order_id order{};
    numeric_type storage = numeric_type::invalid;
    numeric_type logical = numeric_type::invalid;
    mutability_requirement_v1 mutability = mutability_requirement_v1::immutable;
    std::uint8_t reserved = 0u;
    value_generation generation{};
};

// Describes portable semantic and physical capabilities without owning bytes
// or embedding provider callbacks. The physical IDs name source-linked,
// versioned registries; fused_transforms is sorted and caller-owned.
struct atom_affordance_v1 {
    std::uint32_t schema_version = atom_affordance_schema_version_v1;
    std::uint32_t record_bytes = sizeof(atom_affordance_v1);
    persistent_identity_v1 affordance_identity{};
    persistent_identity_v1 atom_species{};
    persistent_identity_v1 exact_coverage_identity{};
    persistent_identity_v1 physical_encoding{};
    persistent_identity_v1 local_projection_abi{};
    const atom_plane_affordance_v1 *planes = nullptr;
    std::uint64_t plane_count = 0u;
    std::uint32_t extent_count = 1u;
    std::uint32_t flags = 0u;
    const persistent_identity_v1 *fused_transforms = nullptr;
    std::uint64_t fused_transform_count = 0u;
};

enum class atom_affordance_validation_code_v1 : std::uint8_t {
    ok = 0u,
    unsupported_schema = 1u,
    invalid_record_bytes = 2u,
    invalid_affordance_identity = 3u,
    invalid_atom_species = 4u,
    invalid_coverage_identity = 5u,
    invalid_physical_encoding = 6u,
    invalid_projection_abi = 7u,
    missing_planes = 8u,
    invalid_plane_identity = 9u,
    duplicate_or_unordered_plane = 10u,
    invalid_plane_order = 11u,
    invalid_plane_numeric = 12u,
    invalid_plane_mutability = 13u,
    nonzero_reserved = 14u,
    invalid_plane_generation = 15u,
    invalid_extent_count = 16u,
    multi_extent_flag_missing = 17u,
    unknown_flag = 18u,
    invalid_fused_transform = 19u,
    duplicate_or_unordered_fused_transform = 20u,
    inconsistent_fused_transform_pointer = 21u
};

struct atom_affordance_validation_result_v1 {
    atom_affordance_validation_code_v1 code =
        atom_affordance_validation_code_v1::ok;
    std::uint64_t index = 0u;

    constexpr explicit operator bool() const noexcept {
        return code == atom_affordance_validation_code_v1::ok;
    }
};

atom_affordance_validation_result_v1 validate_atom_affordance_v1(
    const atom_affordance_v1 &affordance) noexcept;

static_assert(std::is_standard_layout_v<atom_plane_affordance_v1>);
static_assert(std::is_trivially_copyable_v<atom_plane_affordance_v1>);
static_assert(std::is_standard_layout_v<atom_affordance_v1>);
static_assert(std::is_trivially_copyable_v<atom_affordance_v1>);

}  // namespace cellerator::execution::joint_compiler
