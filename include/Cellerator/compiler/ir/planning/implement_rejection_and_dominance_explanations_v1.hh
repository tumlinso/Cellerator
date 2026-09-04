#pragma once
#include <Cellerator/compiler/ir/planning/freeze_planning_ir_module_and_decision_state_model_v1.hh>
#include <cstddef>
#include <cstdint>
namespace cellerator::compiler::ir::planning::v1 {
enum class removal_reason_v1:std::uint8_t{correctness=0,capability,resource,numerical,profile,stale_evidence,cost,user_policy};
struct removal_explanation_v1{planning_identity_v1 candidate{},related_candidate{},evidence{};removal_reason_v1 reason=removal_reason_v1::correctness;std::uint8_t reserved8[7]{};double observed=0,limit=0;};
enum class explanation_status_v1:std::uint8_t{ok=0,invalid_argument,invalid_identity,invalid_reason,nonzero_reserved,insufficient_capacity};
explanation_status_v1 format_removal_explanation_v1(const removal_explanation_v1&,char*,std::size_t,std::size_t*) noexcept;
}
