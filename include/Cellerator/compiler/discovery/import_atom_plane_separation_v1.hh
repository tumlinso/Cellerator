#pragma once

#include <Cellerator/compiler/discovery/import_the_common_jbc_atom_identity_adapters_v1.hh>

#include <cstdint>
#include <vector>

namespace Cellerator::compiler::discovery {

enum class atom_plane_kind_v1 : std::uint8_t {
    structure = 1,
    mutable_values,
    active_support,
    gradients,
    partials,
    physical_views,
    evidence,
    lineage,
};

[[nodiscard]] constexpr std::uint32_t atom_plane_bit_v1(
    atom_plane_kind_v1 kind) noexcept {
    return UINT32_C(1) << (static_cast<std::uint8_t>(kind) - 1);
}

struct separated_atom_plane_v1 {
    atom_plane_kind_v1 kind = atom_plane_kind_v1::structure;
    persistent_atom_identity_v1 plane_identity{};
    std::uint64_t generation = 0;
    std::uint32_t invalidated_by_plane_kinds = 0;
};

struct atom_plane_mutation_v1 {
    atom_plane_kind_v1 kind = atom_plane_kind_v1::structure;
    std::uint64_t previous_generation = 0;
    std::uint64_t new_generation = 0;
};

struct atom_plane_reuse_v1 {
    persistent_atom_identity_v1 plane_identity{};
    std::uint64_t generation = 0;
    bool reusable = false;
};

enum class atom_plane_separation_status_v1 : std::uint8_t {
    success = 0,
    missing_plane_kind,
    invalid_plane,
    unordered_or_duplicate_plane,
    invalid_mutation,
    duplicate_mutation,
    allocation_failure,
};

[[nodiscard]] atom_plane_separation_status_v1 evaluate_atom_plane_reuse_v1(
    const std::vector<separated_atom_plane_v1>& planes,
    const std::vector<atom_plane_mutation_v1>& mutations,
    std::vector<atom_plane_reuse_v1>* output) noexcept;

}  // namespace Cellerator::compiler::discovery
