#pragma once

#include <Cellerator/compiler/ir/planning/first_inspectable_candidate_search_space_v1.hh>
#include <Cellerator/compiler/ir/planning/planning_ir_module_and_decision_state_model_v1.hh>
#include <Cellerator/compiler/ir/planning/analytical_measured_and_external_evidence_v1.hh>
#include <Cellerator/compiler/ir/planning/atom_requirement_and_affordance_nodes_v1.hh>
#include <Cellerator/compiler/ir/planning/candidate_family_and_provider_nodes_v1.hh>
#include <Cellerator/compiler/ir/planning/complete_cost_vectors_v1.hh>
#include <Cellerator/compiler/ir/planning/decomposition_alternative_nodes_v1.hh>
#include <Cellerator/compiler/ir/planning/exact_logical_coverage_nodes_v1.hh>
#include <Cellerator/compiler/ir/planning/partial_result_algebra_nodes_v1.hh>
#include <Cellerator/compiler/ir/planning/persistent_order_packing_alternatives_v1.hh>
#include <Cellerator/compiler/ir/planning/planning_ir_parser_printer_and_validator_v1.hh>
#include <Cellerator/compiler/ir/planning/planning_problems_and_operation_scopes_v1.hh>
#include <Cellerator/compiler/ir/planning/rejection_and_dominance_explanations_v1.hh>
#include <Cellerator/compiler/ir/planning/resource_and_stage_inventory_alternatives_v1.hh>
#include <Cellerator/compiler/ir/planning/semantic_to_planning_lowering_v1.hh>
#include <Cellerator/compiler/ir/planning/user_edits_and_authority_hierarchy_v1.hh>

namespace cellerator::compiler::ir::planning::v1 {

inline constexpr std::uint32_t planning_ir_contract_version_v1 = 1u;

}  // namespace cellerator::compiler::ir::planning::v1
