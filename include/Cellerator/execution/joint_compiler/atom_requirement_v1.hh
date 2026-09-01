#pragma once

#include <Cellerator/execution/index_space/hierarchical_index_space_v1.hh>
#include <Cellerator/execution/joint_compiler/logical_coverage_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::execution::joint_compiler {

inline constexpr std::uint32_t atom_requirement_schema_version_v1 = 1u;

enum class contiguity_requirement_v1 : std::uint8_t {
    any = 1u,
    contiguous = 2u,
    regular_stride = 3u
};

enum class mutability_requirement_v1 : std::uint8_t {
    immutable = 1u,
    mutable_value_generation = 2u
};

enum class generation_requirement_v1 : std::uint8_t {
    any_current = 1u,
    exact = 2u,
    at_least = 3u
};

struct atom_numeric_requirement_v1 {
    numeric_type storage = numeric_type::invalid;
    numeric_type logical = numeric_type::invalid;
    numeric_type accumulation = numeric_type::invalid;
    std::uint8_t reserved = 0u;
};

// A cold, non-owning request describing what one local Cellerator candidate
// needs from an execution atom. Identity arrays are sorted and unique. Runtime
// addresses, device placement, and transform implementations are deliberately
// absent; transform_paths names portable, separately registered routes.
struct atom_requirement_v1 {
    std::uint32_t schema_version = atom_requirement_schema_version_v1;
    std::uint32_t record_bytes = sizeof(atom_requirement_v1);
    persistent_identity_v1 requirement_identity{};
    persistent_identity_v1 exact_coverage_identity{};
    const persistent_identity_v1 *accepted_atom_species = nullptr;
    std::uint64_t accepted_atom_species_count = 0u;
    const persistent_identity_v1 *required_planes = nullptr;
    std::uint64_t required_plane_count = 0u;
    atom_numeric_requirement_v1 numeric{};
    local_index_width_v1 index_width = local_index_width_v1::u32;
    std::uint8_t reserved0[3]{};
    order_id required_order{};
    std::uint64_t minimum_alignment = 1u;
    contiguity_requirement_v1 contiguity = contiguity_requirement_v1::any;
    mutability_requirement_v1 mutability =
        mutability_requirement_v1::immutable;
    generation_requirement_v1 generation_policy =
        generation_requirement_v1::any_current;
    bool graph_stable_address = false;
    std::uint32_t minimum_extent_count = 1u;
    std::uint32_t maximum_extent_count = 1u;
    value_generation required_generation{};
    const persistent_identity_v1 *transform_paths = nullptr;
    std::uint64_t transform_path_count = 0u;
};

enum class atom_requirement_validation_code_v1 : std::uint8_t {
    ok = 0u,
    unsupported_schema = 1u,
    invalid_record_bytes = 2u,
    nonzero_reserved = 3u,
    invalid_requirement_identity = 4u,
    invalid_coverage_identity = 5u,
    missing_atom_species = 6u,
    invalid_atom_species = 7u,
    duplicate_or_unordered_atom_species = 8u,
    missing_planes = 9u,
    invalid_plane = 10u,
    duplicate_or_unordered_plane = 11u,
    invalid_numeric = 12u,
    invalid_index_width = 13u,
    invalid_order = 14u,
    invalid_alignment = 15u,
    invalid_contiguity = 16u,
    invalid_mutability = 17u,
    invalid_generation_policy = 18u,
    invalid_generation = 19u,
    invalid_extent_count = 20u,
    missing_transform_paths = 21u,
    invalid_transform_path = 22u,
    duplicate_or_unordered_transform_path = 23u
};

struct atom_requirement_validation_result_v1 {
    atom_requirement_validation_code_v1 code =
        atom_requirement_validation_code_v1::ok;
    std::uint64_t index = 0u;

    constexpr explicit operator bool() const noexcept {
        return code == atom_requirement_validation_code_v1::ok;
    }
};

atom_requirement_validation_result_v1 validate_atom_requirement_v1(
    const atom_requirement_v1 &requirement) noexcept;

static_assert(std::is_standard_layout_v<atom_numeric_requirement_v1>);
static_assert(std::is_trivially_copyable_v<atom_numeric_requirement_v1>);
static_assert(std::is_standard_layout_v<atom_requirement_v1>);
static_assert(std::is_trivially_copyable_v<atom_requirement_v1>);

}  // namespace cellerator::execution::joint_compiler
