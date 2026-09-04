#pragma once

// Stable include surface for the source-level biological type system.  The
// implementation remains split by semantic concern so consumers need not
// depend on Sema's internal file organization.
#include <Cellerator/compiler/sema/freeze_compiler_semantic_type_categories_v1.hh>
#include <Cellerator/compiler/sema/implement_axis_semantics_v1.hh>
#include <Cellerator/compiler/sema/implement_domain_and_human_biological_tag_semantics_v1.hh>
#include <Cellerator/compiler/sema/implement_numerical_tuple_semantics_v1.hh>
#include <Cellerator/compiler/sema/implement_persistence_and_identity_typing_v1.hh>
#include <Cellerator/compiler/sema/implement_relation_endpoint_semantics_v1.hh>
#include <Cellerator/compiler/sema/implement_state_semantics_v1.hh>
#include <Cellerator/compiler/sema/implement_structure_value_and_support_generation_typing_v1.hh>
#include <Cellerator/compiler/sema/implement_support_and_logical_edge_identity_semantics_v1.hh>
