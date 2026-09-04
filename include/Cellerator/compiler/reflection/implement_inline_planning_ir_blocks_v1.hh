#pragma once
#include <Cellerator/compiler/reflection/implement_reflection_of_planning_ir_search_spaces_v1.hh>
#include <cstdint>
#include <string>
#include <vector>
namespace cellerator::compiler::reflection::v1 {
enum class planning_edit_kind_v1:std::uint8_t{add=1,remove,change_cost,replace_decomposition,offer_candidate,force_candidate,replace_planner};
struct inline_planning_edit_v1{planning_edit_kind_v1 kind=planning_edit_kind_v1::add;std::string subject,payload;std::vector<std::uint64_t>cost;bool unsafe_acknowledged=false;};
struct inline_planning_block_v1{std::string planning_point;std::vector<inline_planning_edit_v1>edits;};
enum class inline_planning_status_v1:std::uint8_t{valid=0,missing_point,invalid_subject,invalid_cost,force_not_acknowledged,conflicting_edit};
[[nodiscard]] inline_planning_status_v1 validate_inline_planning_block_v1(const inline_planning_block_v1&)noexcept;
[[nodiscard]] reflected_search_space_v1 apply_inline_planning_block_v1(const reflected_search_space_v1&,const inline_planning_block_v1&);
}
