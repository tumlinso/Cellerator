#pragma once

#include <Cellerator/compiler/ir/realization/implement_atom_and_extent_bindings_v1.hh>
#include <Cellerator/compiler/ir/realization/implement_target_and_capability_descriptions_v1.hh>

#include <cstdint>
#include <string>
#include <vector>

namespace cellerator::compiler::ir::realization::v1 {

enum class physical_plane_kind_v1 : std::uint8_t {
    structure = 1u,
    values,
    active_support,
    gradients,
    partials,
    workspace,
    metadata,
    generated_constants,
};

enum class plane_lifetime_v1 : std::uint8_t {
    module = 1u,
    structure_epoch,
    value_generation,
    invocation,
    stage,
};

struct physical_plane_v1 {
    stable_identity_v1 identity{};
    stable_identity_v1 artifact_identity{};
    stable_identity_v1 structure_identity{};
    std::uint64_t structure_epoch = 0u;
    std::uint64_t value_generation = 0u;
    physical_plane_kind_v1 kind = physical_plane_kind_v1::structure;
    plane_lifetime_v1 lifetime = plane_lifetime_v1::module;
    std::uint32_t residency_requirements = 0u;
    bool mutable_values = false;
};

struct physical_plane_set_v1 {
    stable_identity_v1 identity{};
    std::vector<physical_plane_v1> planes;
};

enum class physical_plane_status_v1 : std::uint8_t {
    valid = 0u,
    invalid_identity,
    duplicate_plane,
    invalid_epoch,
    invalid_generation,
    invalid_lifetime,
    invalid_residency,
    structure_mismatch,
};

[[nodiscard]] physical_plane_status_v1 validate_physical_plane_set_v1(
    const physical_plane_set_v1& planes,
    std::string* error = nullptr) noexcept;

[[nodiscard]] physical_plane_status_v1 advance_value_generation_v1(
    const physical_plane_set_v1& source,
    std::uint64_t generation,
    physical_plane_set_v1* output,
    std::string* error = nullptr) noexcept;

} // namespace cellerator::compiler::ir::realization::v1
