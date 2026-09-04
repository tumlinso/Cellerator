#pragma once
#include <array>
#include <string_view>
namespace Cellerator::compiler::migration {
struct retained_cellshard_concern_v1{std::string_view name;bool may_decide_compiler_semantics;};
inline constexpr std::array<retained_cellshard_concern_v1,10> retained_cellshard_boundary_v1{{
 {"atom-store containers",false},{"materialized instances",false},{"encoded replicas",false},{"file and object storage",false},{"staging",false},{"placement",false},{"residency",false},{"transport",false},{"leases",false},{"delivery",false},
}};
[[nodiscard]] constexpr bool application_only_boundary_v1()noexcept{for(auto r:retained_cellshard_boundary_v1)if(r.may_decide_compiler_semantics)return false;return true;}
} // namespace Cellerator::compiler::migration
