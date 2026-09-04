#pragma once
#include <Cellerator/compiler/ir/planning/freeze_planning_ir_module_and_decision_state_model_v1.hh>
#include <Cellerator/planner/external_cost/complete_cost_v1.hh>
#include <Cellerator/profiling/joint_compiler/execution_export_v2.hh>
#include <cstdint>
namespace cellerator::compiler::ir::planning::v1 {
enum class planning_evidence_kind_v1:std::uint8_t{analytical=0,measured,external};
struct planning_evidence_v1{planning_identity_v1 evidence{},target{},toolchain{},build{},profile{},external_reference{};planning_evidence_kind_v1 kind=planning_evidence_kind_v1::analytical;std::uint8_t valid=0,contaminated=0,reserved8=0;std::uint32_t sample_count=0;std::uint64_t revision=0;double minimum=0,median=0,maximum=0,uncertainty=0,confidence=0,local_ns=0,external_ns=0,reuse_credit_ns=0;};
enum class planning_evidence_status_v1:std::uint8_t{ok=0,invalid_identity,invalid_kind,invalid_distribution,invalid_samples,invalid_confidence,nonzero_reserved};
planning_evidence_status_v1 validate_planning_evidence_v1(const planning_evidence_v1&) noexcept;
planning_evidence_v1 import_external_evidence_v1(planning_identity_v1,const cellerator::planner::external_cost::external_complete_cost_v1&,planning_identity_v1);
cellerator::planner::external_cost::external_complete_cost_v1 export_external_evidence_v1(const planning_evidence_v1&) noexcept;
planning_evidence_v1 import_execution_evidence_v1(planning_identity_v1,const cellerator::profiling::joint_compiler::performance_freshness_record_v2&);
}
