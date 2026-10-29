#pragma once

// Stable public surface for execution-field semantics v1. The individual
// headers remain available for narrow consumers; this facade is the frozen
// interface consumed by downstream compiler stages.
#include <Cellerator/compiler/sema/field/execution_field_semantic_ownership_v1.hh>
#include <Cellerator/compiler/sema/field/first_profile_required_semantic_field_slice_v1.hh>
#include <Cellerator/compiler/sema/field/automatic_lifetime_and_generation_transfer_v1.hh>
#include <Cellerator/compiler/sema/field/conditional_profile_alternatives_and_joins_v1.hh>
#include <Cellerator/compiler/sema/field/forced_realization_controls_v1.hh>
#include <Cellerator/compiler/sema/field/expected_data_state_transformation_hints_v1.hh>
#include <Cellerator/compiler/sema/field/field_level_reflection_identity_v1.hh>
#include <Cellerator/compiler/sema/field/hard_semantic_and_execution_constraints_v1.hh>
#include <Cellerator/compiler/sema/field/missing_profile_failure_policy_v1.hh>
#include <Cellerator/compiler/sema/field/named_representative_profile_binding_v1.hh>
#include <Cellerator/compiler/sema/field/native_effect_contracts_v1.hh>
#include <Cellerator/compiler/sema/field/opaque_native_call_barriers_v1.hh>
#include <Cellerator/compiler/sema/field/persistence_and_reuse_facts_v1.hh>
#include <Cellerator/compiler/sema/field/planning_facts_and_preferences_v1.hh>
#include <Cellerator/compiler/sema/field/statement_ordering_and_observable_effects_v1.hh>
#include <Cellerator/compiler/sema/field/nested_field_semantics_v1.hh>
