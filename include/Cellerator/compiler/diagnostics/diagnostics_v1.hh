#pragma once
#include <Cellerator/compiler/diagnostics/structured_diagnostic_queries_v1.hh>
#include <Cellerator/compiler/diagnostics/validation_mode_semantics_v1.hh>
#include <Cellerator/compiler/diagnostics/advisory_semantic_validators_v1.hh>
#include <Cellerator/compiler/diagnostics/candidate_decision_reports_v1.hh>
#include <Cellerator/compiler/diagnostics/crash_and_timeout_diagnostics_v1.hh>
#include <Cellerator/compiler/diagnostics/exact_coverage_and_ownership_diagnostics_v1.hh>
#include <Cellerator/compiler/diagnostics/optimization_opportunity_remarks_v1.hh>
#include <Cellerator/compiler/diagnostics/planning_barrier_explanations_v1.hh>
#include <Cellerator/compiler/diagnostics/structural_impossibility_checks_v1.hh>
#include <Cellerator/compiler/diagnostics/target_native_diagnostics_v1.hh>
namespace cellerator::compiler::diagnostics::v1 {inline constexpr unsigned diagnostics_contract_version=1;inline constexpr unsigned required_ast_version=1,required_planning_ir_version=1,required_realization_ir_version=1;}
