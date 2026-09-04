#pragma once
#include <Cellerator/compiler/diagnostics/expose_structured_diagnostic_and_query_apis_v1.hh>
#include <Cellerator/compiler/diagnostics/freeze_validation_mode_semantics_v1.hh>
#include <Cellerator/compiler/diagnostics/implement_advisory_semantic_validators_v1.hh>
#include <Cellerator/compiler/diagnostics/implement_candidate_decision_reports_v1.hh>
#include <Cellerator/compiler/diagnostics/implement_crash_and_timeout_diagnostics_v1.hh>
#include <Cellerator/compiler/diagnostics/implement_exact_coverage_and_ownership_diagnostics_v1.hh>
#include <Cellerator/compiler/diagnostics/implement_optimization_remarks_and_missed_opportunity_di_v1.hh>
#include <Cellerator/compiler/diagnostics/implement_planning_barrier_explanations_v1.hh>
#include <Cellerator/compiler/diagnostics/implement_structural_impossibility_checks_v1.hh>
#include <Cellerator/compiler/diagnostics/implement_target_native_diagnostics_v1.hh>
namespace cellerator::compiler::diagnostics::v1 {inline constexpr unsigned diagnostics_contract_version=1;inline constexpr unsigned required_ast_version=1,required_planning_ir_version=1,required_realization_ir_version=1;}
