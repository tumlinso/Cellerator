#pragma once
#include <Cellerator/compiler/ir/planning/freeze_planning_ir_module_and_decision_state_model_v1.hh>
#include <cstdint>
namespace cellerator::compiler::ir::planning::v1 {
enum class search_edit_kind_v1:std::uint8_t{add_candidate=0,remove_candidate,change_fact,change_objective,change_cost,replace_decomposition,force_selection,unsafe_assertion};
enum class edit_authority_v1:std::uint8_t{source_hint=0,compiler_pass,external_plan,user_force,unsafe_user_assertion};
struct search_space_edit_v1{planning_identity_v1 edit{},actor{},subject{},replacement{};search_edit_kind_v1 kind=search_edit_kind_v1::add_candidate;edit_authority_v1 authority=edit_authority_v1::source_hint;std::uint8_t unsafe_acknowledged=0,reserved8=0;std::uint64_t sequence=0;double value=0;};
enum class search_edit_status_v1:std::uint8_t{ok=0,invalid_identity,invalid_kind,invalid_authority,authority_mismatch,unsafe_not_acknowledged,stale_edit,nonzero_reserved};
search_edit_status_v1 validate_search_space_edit_v1(const search_space_edit_v1&,std::uint64_t minimum_sequence) noexcept;
bool edit_overrides_v1(const search_space_edit_v1&,const search_space_edit_v1&) noexcept;
}
