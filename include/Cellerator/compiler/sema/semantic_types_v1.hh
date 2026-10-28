#pragma once

// Stable include surface for the source-level biological type system.  The
// implementation remains split by semantic concern so consumers need not
// depend on Sema's internal file organization.
#include <Cellerator/compiler/sema/compiler_semantic_type_categories_v1.hh>
#include <Cellerator/compiler/sema/axis_semantics_v1.hh>
#include <Cellerator/compiler/sema/domain_and_human_biological_tag_semantics_v1.hh>
#include <Cellerator/compiler/sema/numerical_tuple_semantics_v1.hh>
#include <Cellerator/compiler/sema/persistence_and_identity_typing_v1.hh>
#include <Cellerator/compiler/sema/relation_endpoint_semantics_v1.hh>
#include <Cellerator/compiler/sema/state_semantics_v1.hh>
#include <Cellerator/compiler/sema/structure_value_and_support_generation_typing_v1.hh>
#include <Cellerator/compiler/sema/support_and_logical_edge_identity_semantics_v1.hh>
