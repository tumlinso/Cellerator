#pragma once

// Stable public surface for execution-field semantics v1. The individual
// headers remain available for narrow consumers; this facade is the frozen
// interface consumed by downstream compiler stages.
#include <Cellerator/compiler/sema/field/define_execution_field_semantic_ownership_v1.hh>
#include <Cellerator/compiler/sema/field/deliver_the_first_profile_required_semantic_field_slice_v1.hh>
#include <Cellerator/compiler/sema/field/implement_automatic_lifetime_and_generation_transfer_v1.hh>
#include <Cellerator/compiler/sema/field/implement_conditional_profile_alternatives_and_joins_v1.hh>
#include <Cellerator/compiler/sema/field/implement_custom_candidate_and_forced_realization_contro_v1.hh>
#include <Cellerator/compiler/sema/field/implement_expected_data_state_transformation_hints_v1.hh>
#include <Cellerator/compiler/sema/field/implement_field_level_reflection_identity_v1.hh>
#include <Cellerator/compiler/sema/field/implement_hard_semantic_and_execution_constraints_v1.hh>
#include <Cellerator/compiler/sema/field/implement_missing_profile_failure_policy_v1.hh>
#include <Cellerator/compiler/sema/field/implement_named_representative_profile_binding_v1.hh>
#include <Cellerator/compiler/sema/field/implement_native_effect_contracts_v1.hh>
#include <Cellerator/compiler/sema/field/implement_opaque_native_call_barriers_v1.hh>
#include <Cellerator/compiler/sema/field/implement_persistence_and_reuse_facts_v1.hh>
#include <Cellerator/compiler/sema/field/implement_planning_facts_and_preferences_v1.hh>
#include <Cellerator/compiler/sema/field/implement_statement_ordering_and_observable_effects_v1.hh>
#include <Cellerator/compiler/sema/field/resolve_and_implement_nested_field_semantics_v1.hh>
