#pragma once

#include <Cellerator/compiler/ir/planning/deliver_the_first_inspectable_candidate_search_space_v1.hh>
#include <Cellerator/compiler/ir/planning/freeze_planning_ir_module_and_decision_state_model_v1.hh>
#include <Cellerator/compiler/ir/planning/implement_analytical_measured_and_external_evidence_v1.hh>
#include <Cellerator/compiler/ir/planning/implement_atom_requirement_and_affordance_nodes_v1.hh>
#include <Cellerator/compiler/ir/planning/implement_candidate_family_and_provider_nodes_v1.hh>
#include <Cellerator/compiler/ir/planning/implement_complete_cost_vectors_v1.hh>
#include <Cellerator/compiler/ir/planning/implement_decomposition_alternative_nodes_v1.hh>
#include <Cellerator/compiler/ir/planning/implement_exact_logical_coverage_nodes_v1.hh>
#include <Cellerator/compiler/ir/planning/implement_partial_result_algebra_nodes_v1.hh>
#include <Cellerator/compiler/ir/planning/implement_persistent_order_projection_and_packing_altern_v1.hh>
#include <Cellerator/compiler/ir/planning/implement_planning_ir_parser_printer_and_validator_v1.hh>
#include <Cellerator/compiler/ir/planning/implement_planning_problems_and_operation_scopes_v1.hh>
#include <Cellerator/compiler/ir/planning/implement_rejection_and_dominance_explanations_v1.hh>
#include <Cellerator/compiler/ir/planning/implement_resource_and_stage_inventory_alternatives_v1.hh>
#include <Cellerator/compiler/ir/planning/implement_semantic_to_planning_lowering_v1.hh>
#include <Cellerator/compiler/ir/planning/implement_user_edits_and_authority_hierarchy_v1.hh>

namespace cellerator::compiler::ir::planning::v1 {

inline constexpr std::uint32_t planning_ir_contract_version_v1 = 1u;

}  // namespace cellerator::compiler::ir::planning::v1
