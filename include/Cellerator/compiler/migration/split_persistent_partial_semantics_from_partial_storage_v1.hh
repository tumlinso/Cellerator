#pragma once
#include <array>
#include <cstdint>
#include <string_view>
namespace Cellerator::compiler::migration {
enum class partial_owner_v1:std::uint8_t{cellerator=1,cellshard};
struct partial_interface_row_v1{std::string_view concern;partial_owner_v1 owner;std::string_view dependency;};
inline constexpr std::array<partial_interface_row_v1,8> partial_interface_v1{{
 {"merge algebra",partial_owner_v1::cellerator,"Cellerator"},{"dependency closure",partial_owner_v1::cellerator,"Cellerator"},{"numerical policy",partial_owner_v1::cellerator,"Cellerator"},{"persistence legality",partial_owner_v1::cellerator,"Cellerator"},
 {"payload bytes",partial_owner_v1::cellshard,"Cellerator ABI"},{"replication",partial_owner_v1::cellshard,"Cellerator ABI"},{"durable persistence",partial_owner_v1::cellshard,"Cellerator ABI"},{"recovery",partial_owner_v1::cellshard,"Cellerator ABI"},
}};
[[nodiscard]] constexpr bool acyclic_partial_dependency_v1()noexcept{for(auto row:partial_interface_v1)if(row.owner==partial_owner_v1::cellerator&&row.dependency!="Cellerator")return false;return true;}
} // namespace Cellerator::compiler::migration
