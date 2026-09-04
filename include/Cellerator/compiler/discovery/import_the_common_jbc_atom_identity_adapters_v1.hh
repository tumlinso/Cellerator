#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace Cellerator::compiler::discovery {

inline constexpr std::uint64_t cellerator_atom_provider_namespace_v1 = 1;

struct persistent_atom_identity_v1 {
    std::uint64_t producer_namespace = 0;
    std::uint64_t local_identity = 0;
};

struct cellshard_strong_id_view_v1 {
    std::uint64_t value = 0;
};

enum class atom_species_v1 : std::uint64_t {
    identity_spine = 1,
    support_signature,
    co_support_source,
    destination_convergence,
    divergence,
    motif,
    program,
    state_neighborhood,
    trajectory_prefix,
    trajectory_branch,
    trajectory_delta,
    multimodal,
    sequence,
    halo,
    stable_structure,
    stable_value,
    operation_polymorphic,
    segment,
    transform,
    partial,
    executable,
    superatom,
};

enum class atom_state_kind_v1 : std::uint32_t {
    cell_state = 1,
    biological_state,
    embedding,
    provider_defined,
};

struct atom_identity_contract_v1 {
    persistent_atom_identity_v1 atom{};
    persistent_atom_identity_v1 species{};
    atom_state_kind_v1 state = atom_state_kind_v1::cell_state;
};

enum class atom_identity_validation_code_v1 : std::uint8_t {
    success = 0,
    invalid_atom_identity,
    invalid_species_identity,
    invalid_state,
    invalid_legacy_identity,
    invalid_producer_namespace,
};

[[nodiscard]] constexpr bool valid_persistent_atom_identity_v1(
    persistent_atom_identity_v1 identity) noexcept {
    return identity.producer_namespace != 0 && identity.local_identity != 0;
}

[[nodiscard]] constexpr bool operator==(
    persistent_atom_identity_v1 left,
    persistent_atom_identity_v1 right) noexcept {
    return left.producer_namespace == right.producer_namespace &&
        left.local_identity == right.local_identity;
}

[[nodiscard]] constexpr bool operator!=(
    persistent_atom_identity_v1 left,
    persistent_atom_identity_v1 right) noexcept {
    return !(left == right);
}

[[nodiscard]] constexpr bool persistent_atom_identity_less_v1(
    persistent_atom_identity_v1 left,
    persistent_atom_identity_v1 right) noexcept {
    return left.producer_namespace < right.producer_namespace ||
        (left.producer_namespace == right.producer_namespace &&
         left.local_identity < right.local_identity);
}

[[nodiscard]] constexpr bool valid_atom_species_v1(atom_species_v1 species) noexcept {
    const auto value = static_cast<std::uint64_t>(species);
    return value >= 1 && value <= 22;
}

[[nodiscard]] constexpr bool valid_atom_state_kind_v1(
    atom_state_kind_v1 state) noexcept {
    const auto value = static_cast<std::uint32_t>(state);
    return value >= 1 && value <= 4;
}

[[nodiscard]] persistent_atom_identity_v1 make_cellerator_species_identity_v1(
    atom_species_v1 species) noexcept;

[[nodiscard]] persistent_atom_identity_v1 adapt_cellshard_strong_id_v1(
    std::uint64_t producer_namespace,
    cellshard_strong_id_view_v1 legacy_identity,
    atom_identity_validation_code_v1* status = nullptr) noexcept;

[[nodiscard]] atom_identity_validation_code_v1 validate_atom_identity_contract_v1(
    const atom_identity_contract_v1& contract) noexcept;

static_assert(sizeof(cellshard_strong_id_view_v1) == sizeof(std::uint64_t));
static_assert(std::is_standard_layout_v<persistent_atom_identity_v1>);
static_assert(std::is_trivially_copyable_v<persistent_atom_identity_v1>);
static_assert(sizeof(persistent_atom_identity_v1) == 2 * sizeof(std::uint64_t));

}  // namespace Cellerator::compiler::discovery
