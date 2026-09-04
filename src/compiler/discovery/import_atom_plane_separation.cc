#include <Cellerator/compiler/discovery/import_atom_plane_separation_v1.hh>

#include <utility>

namespace Cellerator::compiler::discovery {
namespace {

bool valid_kind_v1(atom_plane_kind_v1 kind) noexcept {
    const auto value = static_cast<std::uint8_t>(kind);
    return value >= static_cast<std::uint8_t>(atom_plane_kind_v1::structure) &&
        value <= static_cast<std::uint8_t>(atom_plane_kind_v1::lineage);
}

}  // namespace

atom_plane_separation_status_v1 evaluate_atom_plane_reuse_v1(
    const std::vector<separated_atom_plane_v1>& planes,
    const std::vector<atom_plane_mutation_v1>& mutations,
    std::vector<atom_plane_reuse_v1>* output) noexcept {
    if (output == nullptr) {
        return atom_plane_separation_status_v1::allocation_failure;
    }
    std::uint32_t seen_kinds = 0;
    for (std::size_t index = 0; index < planes.size(); ++index) {
        const auto& plane = planes[index];
        if (!valid_kind_v1(plane.kind) ||
            !valid_persistent_atom_identity_v1(plane.plane_identity) ||
            plane.generation == 0 ||
            (plane.invalidated_by_plane_kinds & ~UINT32_C(0xff)) != 0) {
            return atom_plane_separation_status_v1::invalid_plane;
        }
        const auto bit = atom_plane_bit_v1(plane.kind);
        if ((seen_kinds & bit) != 0 ||
            (index != 0 && static_cast<std::uint8_t>(planes[index - 1].kind) >=
                               static_cast<std::uint8_t>(plane.kind))) {
            return atom_plane_separation_status_v1::unordered_or_duplicate_plane;
        }
        seen_kinds |= bit;
    }
    if (seen_kinds != UINT32_C(0xff)) {
        return atom_plane_separation_status_v1::missing_plane_kind;
    }

    std::uint32_t mutated_kinds = 0;
    for (const auto& mutation : mutations) {
        if (!valid_kind_v1(mutation.kind) || mutation.previous_generation == 0 ||
            mutation.new_generation <= mutation.previous_generation) {
            return atom_plane_separation_status_v1::invalid_mutation;
        }
        const auto bit = atom_plane_bit_v1(mutation.kind);
        if ((mutated_kinds & bit) != 0) {
            return atom_plane_separation_status_v1::duplicate_mutation;
        }
        mutated_kinds |= bit;
        const auto& plane = planes[static_cast<std::uint8_t>(mutation.kind) - 1];
        if (plane.generation != mutation.previous_generation) {
            return atom_plane_separation_status_v1::invalid_mutation;
        }
    }

    try {
        std::vector<atom_plane_reuse_v1> reuse;
        reuse.reserve(planes.size());
        for (const auto& plane : planes) {
            const auto self = atom_plane_bit_v1(plane.kind);
            const bool invalidated =
                (mutated_kinds & (self | plane.invalidated_by_plane_kinds)) != 0;
            reuse.push_back({plane.plane_identity, plane.generation, !invalidated});
        }
        *output = std::move(reuse);
        return atom_plane_separation_status_v1::success;
    } catch (...) {
        return atom_plane_separation_status_v1::allocation_failure;
    }
}

}  // namespace Cellerator::compiler::discovery
