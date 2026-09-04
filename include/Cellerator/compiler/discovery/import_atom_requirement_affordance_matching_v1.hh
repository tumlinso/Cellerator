#pragma once

#include <Cellerator/compiler/discovery/import_the_common_jbc_atom_identity_adapters_v1.hh>

#include <cstdint>
#include <vector>

namespace Cellerator::compiler::discovery {

enum class atom_generation_policy_v1 : std::uint8_t {
    any_current = 1,
    exact,
    at_least,
};

struct migrated_atom_requirement_v1 {
    persistent_atom_identity_v1 requirement_identity{};
    persistent_atom_identity_v1 exact_coverage_identity{};
    std::vector<persistent_atom_identity_v1> accepted_species;
    std::vector<persistent_atom_identity_v1> required_planes;
    persistent_atom_identity_v1 required_order_identity{};
    persistent_atom_identity_v1 required_projection_abi{};
    std::uint64_t required_generation = 0;
    std::uint64_t required_target_capabilities = 0;
    std::uint32_t minimum_extent_count = 1;
    std::uint32_t maximum_extent_count = 1;
    atom_generation_policy_v1 generation_policy =
        atom_generation_policy_v1::any_current;
};

struct migrated_plane_affordance_v1 {
    persistent_atom_identity_v1 plane_identity{};
    persistent_atom_identity_v1 order_identity{};
    std::uint64_t generation = 0;
};

struct migrated_atom_affordance_v1 {
    persistent_atom_identity_v1 affordance_identity{};
    persistent_atom_identity_v1 atom_identity{};
    persistent_atom_identity_v1 species_identity{};
    persistent_atom_identity_v1 exact_coverage_identity{};
    persistent_atom_identity_v1 projection_abi{};
    std::vector<migrated_plane_affordance_v1> planes;
    std::uint64_t target_capabilities = 0;
    std::uint32_t extent_count = 1;
};

enum class atom_match_status_v1 : std::uint8_t {
    matched = 0,
    invalid_requirement,
    invalid_affordance,
    species_mismatch,
    coverage_mismatch,
    projection_mismatch,
    target_capability_mismatch,
    extent_mismatch,
    missing_plane,
    order_mismatch,
    generation_mismatch,
};

struct atom_match_result_v1 {
    atom_match_status_v1 status = atom_match_status_v1::matched;
    std::uint64_t requirement_plane_index = 0;
    [[nodiscard]] constexpr bool matched() const noexcept {
        return status == atom_match_status_v1::matched;
    }
};

[[nodiscard]] atom_match_result_v1 match_migrated_atom_v1(
    const migrated_atom_requirement_v1& requirement,
    const migrated_atom_affordance_v1& affordance) noexcept;

}  // namespace Cellerator::compiler::discovery
