# JBC no-code-loss migration matrix

This matrix gives every path in the raw CellShard JBC branch inventory exactly
one primary migration disposition. It covers the union of changes from common
base `7762a5925fe18b2ca45ab8a436f3461804ed2ad9` to all 24 `jbc/*` branch tips:
979 unique paths with sorted-list SHA-256
`af783b7c35be048289a8da5798e8b11c7895846f0d42d938dc6a235e73a5aee9`.
The branch tips are all reachable from CellShard main
`b9749ad3e5146a04f847533d8c6f1a54146aed20`.

A primary disposition states the lifecycle of a source path; it does not waive
the A02-009 provenance rules or the A02-010 license/include/build exceptions.
For example, `move` still requires namespace and include adaptation, and
`retire after replacement proof` requires export and target validation before
the source definition can be removed.

## Exact reconciliation

| Primary disposition | Paths | No-code-loss meaning |
|---|---:|---|
| preserve in place | 457 | CellShard-owned storage/materialization/runtime source and evidence, runtime/atom-store tests, runtime benchmarks, and historical ledgers remain reachable in CellShard. |
| move | 220 | Compiler-semantic source is exported and rehomed under Cellerator ownership with provenance and evidence. |
| adapt | 242 | Compiler tests and non-runtime benchmarks are recreated against Cellerator paths/contracts while preserving assertions, fixtures, and measured dispositions. |
| split | 52 | A file crosses the new ownership boundary or combines compiler and runtime validation; owner-specific pieces are separated under explicit integration scope. |
| wrap temporarily | 1 | The existing CellShard-to-Cellerator evidence adapter remains only until canonical consumers and compatibility proof are complete. |
| retain as compatibility | 4 | Frozen identity/coverage/evidence representations remain readable through narrow versioned adapters after Cellerator becomes semantic owner. |
| retire after replacement proof | 3 | Redundant graph/DAG definitions are removed only after canonical replacements, migrated tests, and old-artifact readers pass. |
| **Total** | 979 | Exactly the raw worktree inventory; no unclassified or multiply classified path. |

## Deterministic path rules

Rules are evaluated in the order below and produce exactly one disposition.

1. **Wrap temporarily:**
   `include/CellShard/interop/cellerator/evidence_adapter_v1.hh`.
2. **Split:** root `CMakeLists.txt`, `include/CellShard/CellShard.hh`, all 36
   `tests/jbc/validation/**` paths, and the 14 compiler headers whose include
   closure imports CellShard domain, identity/digest/strong-ID, or artifact
   image types:
   - `compiler/evidence/algorithm_provenance_v1.hh`;
   - `compiler/graph/operation_provider.hh`, `operation_node.hh`, and
     `physical_realization.hh`;
   - `compiler/atom/identity_classes_v1.hh` and `evidence_plane_v1.hh`;
   - `compiler/composition/coverage_v1.hh`, `relation_merge_v1.hh`,
     `production_identity_v1.hh`, `persistent_order_link_v1.hh`,
     `grammar_symbol_v1.hh`, `segment_alignment_v1.hh`,
     `identity_spine_join_v1.hh`, and `physical_view_addition_v1.hh`.
3. **Retain as compatibility:**
   `compiler/atom/persistent_identity_v1.hh`,
   `compiler/atom/logical_coverage_v1.hh`,
   `compiler/evidence/atom_evidence_record_v1.hh`, and
   `compiler/discovery/operation_trace/cellerator_identity_adapter_v1.hh`.
   These remain readers/adapters, not parallel semantic authorities.
4. **Retire after replacement proof:** the two distinct
   `compiler/{composition,grammar}/derivation_dag_v1.hh` definitions and
   `compiler/graph/graph_recipe.hh`. Their algorithms and tests move first;
   retirement follows a single canonical DAG/recipe implementation and
   compatibility validation.
5. **Move:** every remaining `include/CellShard/compiler/**` and
   `src/compiler/**` path, including evidence, discovery, certification, atom
   semantics, grammar/composition, basis, partial, graph, and schedule code.
6. **Preserve in place:** `tests/jbc/atom_store/**`, `tests/jbc/runtime/**`,
   `bench/jbc/runtime/**`, atom-store source/specification, runtime-v2 source,
   `docs/JBC/evidence/**`, Todo/history projections, and authority snapshots.
7. **Adapt:** every remaining `tests/**` or `bench/**` path. This covers the
   compiler-semantic test portfolio and non-runtime promotion/benchmark
   evidence after the retained/split cases above.

The focused A02-011 test reconstructs the branch-union directly from Git and
executes these rules, rather than trusting prose counts.

## Evidence movement rules

- The 430-file two-repository evidence index remains the behavior map. Source
  tests move or adapt with their subject, not in a later cleanup batch.
- Exact expected values, malformed-input cases, negative results, nulls,
  ablations, promotion dispositions, benchmark inputs, and raw result records
  are preserved even when target names and includes change.
- The 30 atom-store and 26 runtime test paths remain CellShard-owned. The 36
  integrated validation paths are split into Cellerator semantic validation and
  CellShard application/runtime consumer validation without weakening either.
- A benchmark result never becomes a performance claim merely because its code
  moves. Existing evaluated-not-promoted and correctness-only dispositions stay
  attached to their provenance.

## Replacement-proof boundary

No source or evidence path may be deleted, overwritten, or declared obsolete
until its applying Todo records the source blob and commit, lands the canonical
replacement, adapts or retains every mapped test, runs the required gates, and
proves all consumers use either the replacement or a versioned compatibility
reader. Branch reachability is preservation evidence, not permission to clean
the source worktrees or branches.
