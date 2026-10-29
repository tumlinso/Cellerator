#pragma once
#include <array>
#include <cstdint>
#include <string_view>
namespace Cellerator::compiler::migration {
enum class atom_owner_v1 : std::uint8_t { cellerator_compiler=1, cellshard_application };
enum class atom_level_v1 : std::uint8_t { candidate=1, certified, basis, super, physical, replica, partial, resident };
struct atom_level_owner_v1 { atom_level_v1 level; atom_owner_v1 owner; std::string_view contract; };
inline constexpr std::array<atom_level_owner_v1,8> atom_level_owners_v1{{
 {atom_level_v1::candidate,atom_owner_v1::cellerator_compiler,"uncertain proposal"},
 {atom_level_v1::certified,atom_owner_v1::cellerator_compiler,"exact coverage certificate"},
 {atom_level_v1::basis,atom_owner_v1::cellerator_compiler,"selected planning basis"},
 {atom_level_v1::super,atom_owner_v1::cellerator_compiler,"optional derived composition"},
 {atom_level_v1::physical,atom_owner_v1::cellerator_compiler,"realization requirement"},
 {atom_level_v1::replica,atom_owner_v1::cellerator_compiler,"logical replica requirement"},
 {atom_level_v1::partial,atom_owner_v1::cellerator_compiler,"partial algebra and legality"},
 {atom_level_v1::resident,atom_owner_v1::cellshard_application,"materialized resident instance"},
}};
[[nodiscard]] constexpr atom_owner_v1 owner_of(atom_level_v1 level) noexcept {
 for (auto row:atom_level_owners_v1) if(row.level==level) return row.owner;
 return atom_owner_v1::cellerator_compiler;
}
} // namespace Cellerator::compiler::migration
