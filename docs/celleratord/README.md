# Celleratord query contracts

These notes describe the semantic shape of query results and a recorded I02 acceptance surface. Individual query descriptions are contracts, not proof that each feature is exposed in the current server build. Verify current support in the [implementation record](../current_implementation.qmd), the [public tooling header](../../include/Cellerator/compiler/tooling/cellerator_queries_v1.hh), and the [source routing guide](../development/routing.md).

## Accepted surface and measurement

- [Query surface acceptance](query_surface_acceptance.md) records the frozen I02 fixture and its boundary.
- [Advanced semantic-query benchmark](benchmark_advanced_semantic_queries.md) records process-local latency evidence and responsiveness constraints.

## Query contract notes

- [Biological type and relation hover](biological_type_relation_hover.md)
- [Syntax completion](syntax_completion_contract.md)
- [Field ownership and effects](field_ownership_effect_views.md)
- [Generation and invalidation queries](generation_invalidation_queries.md)
- [Inline CEIR assistance](inline_ceir_assistance.md)
- [Opaque barrier actions](opaque_barrier_actions.md)
- [Planning IR and candidate views](planning_ir_candidate_views.md)
- [Profile state at cursor](profile_state_query.md)
- [Realization and decomposition view](realization_decomposition_view.md)
- [Semantic IR at cursor](semantic_ir_query.md)
- [Source-to-native navigation](source_to_native_navigation.md)
- [Candidate selection explanation](candidate_selection_explanation.md)
