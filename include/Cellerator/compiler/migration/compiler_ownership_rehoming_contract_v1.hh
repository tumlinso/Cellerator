#pragma once
#include <array>
#include <cstdint>
#include <string_view>
namespace Cellerator::compiler::migration {
enum class subsystem_owner_v1:std::uint8_t{cellerator=1,cellshard,split,temporary_adapter};
struct ownership_rehoming_row_v1{std::string_view old_cellshard_family,new_cellerator_target;subsystem_owner_v1 owner;};
inline constexpr std::array<ownership_rehoming_row_v1,12> compiler_ownership_rehoming_v1{{
 {"compiler/evidence + discovery","compiler/profile",subsystem_owner_v1::cellerator},
 {"compiler/certification","compiler/planning/certification",subsystem_owner_v1::cellerator},
 {"compiler/atom semantic states","compiler/ir/atom",subsystem_owner_v1::cellerator},
 {"compiler/composition + grammar","compiler/planning/extensions",subsystem_owner_v1::cellerator},
 {"compiler/basis","compiler/planner/basis",subsystem_owner_v1::cellerator},
 {"compiler/composition/superatom","compiler/planner/superatom",subsystem_owner_v1::cellerator},
 {"compiler/partial","compiler/partial + CellShard persistence",subsystem_owner_v1::split},
 {"compiler/graph","compiler/semantic + planning IR",subsystem_owner_v1::cellerator},
 {"compiler/schedule","compiler/planning + realization IR",subsystem_owner_v1::cellerator},
 {"atom-store + materialization + runtime","CellShard application",subsystem_owner_v1::cellshard},
 {"legacy compiler includes/tests","Cellerator public contracts",subsystem_owner_v1::temporary_adapter},
 {"compiled ruleset export","CellShard future consumer",subsystem_owner_v1::split},
}};
[[nodiscard]] constexpr bool complete_ownership_map_v1()noexcept{for(auto r:compiler_ownership_rehoming_v1)if(r.old_cellshard_family.empty()||r.new_cellerator_target.empty())return false;return true;}
} // namespace Cellerator::compiler::migration
