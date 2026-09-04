# JBC Implementation Status by Subsystem

This source-backed receipt completes `CE-CCP1-A02-004`. Status is based on
production bodies, tests, branch reachability, aggregate build registration,
and the final integration receipts—not historical Todo titles.

## Status vocabulary

- **complete and tested**: production implementation is present, registered in
  an integrated build where applicable, and covered by executable validation.
- **complete but unintegrated**: implementation and focused tests exist but the
  producer tip is not integrated into the current mainline.
- **partial**: useful implementation or evidence exists, but its stated result
  or acceptance boundary is incomplete.
- **scaffold-only**: declarations or registration skeletons exist without the
  required mechanism body.
- **test-only**: a validation scenario/helper exists without a distinct
  production subsystem of its own.
- **design-only**: specification or planning exists without being executable
  implementation.
- **obsolete compatibility code**: retained code is superseded and has no
  forward implementation role beyond compatibility.

## Integration evidence

- Cellerator producer histories are retained by merge
  `82ccaf5`, registered by aggregate build commit `8267f41`, and both are
  ancestors of current Cellerator `main`.
- CellShard producer histories are retained by octopus merge `1efc4df`; current
  CellShard `main` is `b9749ad`.
- `components/CellShard/docs/JBC/evidence/integration_receipt.md` records 304
  host tests passing normally and under ASan+UBSan, four host benchmark/reference
  executables passing, six CUDA tests plus the dual-NUMA campaign building for
  `sm_70`, standalone install/consumer validation, and clean integration gates.
- `components/CellShard/docs/JBC/evidence/biological_novelty_readiness.md`
  explicitly withholds a biological-performance promotion because no new
  reserved biological dataset campaign was run during final integration.

## Cellerator subsystems

| Subsystem | Status | Production/source anchor | Test/evidence anchor | Finding |
|---|---|---|---|---|
| CE semantic interfaces | complete and tested | `include/Cellerator/execution/joint_compiler/` | `tests/jbc/interfaces/` | Nine public headers, twelve validator bodies, and twelve interface tests are integrated. |
| CE atom-fragment preparation | complete and tested | `src/execution/atom_fragment/` | `tests/jbc/fragment/` | Thirteen implementation bodies cover requirements, fallback, registry, binding, Pareto frontier, and preparation; fourteen tests exercise them. |
| CE decomposition catalog | complete and tested | `include/Cellerator/compute/decomposition/` | `tests/jbc/decomposition/` | Twenty contracts and seventeen bodies implement the decomposition vocabulary; eighteen tests include complete fallbacks and partial algebras. |
| CE atom/value planes | complete and tested | `include/Cellerator/execution/atom_plane/` | `tests/jbc/atom_plane/` | Structure, value, gradient, partial, lease, generation, and external mappings have paired bodies/tests. |
| CE multi-extent binding and candidate | complete and tested | `include/Cellerator/execution/object_binding/` | `tests/jbc/multi_extent/` | Multi-atom bindings, direct relation-apply candidate, acquisition, and assembly comparison are implemented and tested. |
| CE external complete-cost exchange | complete and tested | `include/Cellerator/planner/external_cost/` | `tests/jbc/external_cost/` | Six contracts and six bodies cover cost vectors, frontiers, geometry objective, pricing, and exchange; six tests pass through the aggregate target. |
| CE lowering resumption | complete and tested | `include/Cellerator/execution/lowering_resumption/resumption_v1.hh` | `tests/jbc/resumption/` | Canonical through executable-stage resumption and instrumentation have ten focused tests. |
| CE aggregate/package surface | complete and tested | commit `8267f41` (`CMakeLists.txt`) | `tests/jbc/verification/standalone_abi_gate_v1_test.cc` | `Cellerator::jbc_v1`, all JBC tests, benchmark, standalone ABI, and optional bridge gates are registered in mainline. |
| CE cross-operation validation scenarios | test-only | `tests/jbc/cross_operation/` | same | Eight scenarios validate reuse across existing production contracts; there is intentionally no separate cross-operation runtime subsystem. |
| CE independent verifier helpers | test-only | `tests/jbc/verification/atom_fragment_verifier_v1.hh` | `tests/jbc/verification/numerical_verifier_v1_test.cc` | Verification helpers remain test-owned and do not create production compiler authority. |

## CellShard-origin subsystems

These mechanisms are complete, tested, and integrated as preserved JBC input.
Their future ownership is governed separately by A02-003: compiler semantics
are rehoming candidates; concrete storage/materialization/transport remains
CellShard-owned.

| Subsystem | Status | Production/source anchor | Test/evidence anchor | Finding |
|---|---|---|---|---|
| CS atom model | complete and tested | `components/CellShard/include/CellShard/compiler/atom/` | `components/CellShard/tests/jbc/atom/` | Twenty atom contracts plus the common-atom body have twenty focused tests. |
| CS evidence atlas | complete and tested | `components/CellShard/include/CellShard/compiler/evidence/` | `components/CellShard/tests/jbc/evidence/` | Evidence records, provenance, atlas image/merge/query/statistics, and negative evidence are implemented and tested. |
| CS exact certification | complete and tested | `components/CellShard/include/CellShard/compiler/certification/` | `components/CellShard/tests/jbc/certification/` | Sixteen exact identity, coverage, ownership, residual, and independent-verifier contracts have sixteen tests. |
| CS support-signature discovery | complete and tested | `components/CellShard/include/CellShard/compiler/discovery/support_signature/` | `components/CellShard/tests/jbc/discovery/support_signature/` | Sampling/index/proposal and exact-rescan paths are present with ten tests. |
| CS co-support discovery | complete and tested | `components/CellShard/include/CellShard/compiler/discovery/co_support/` | `components/CellShard/tests/jbc/discovery/co_support/` | Raw/weighted/normalized evidence, sparse affinity, exact rescan, and stability are covered by eleven tests. |
| CS bicluster discovery | complete and tested | `components/CellShard/include/CellShard/compiler/discovery/bicluster/` | `components/CellShard/tests/jbc/discovery/bicluster/` | Baseline, expansion, overlap, spectral alternative, cost, and benchmark cases have eight tests. |
| CS overlap discovery | complete and tested | `components/CellShard/include/CellShard/compiler/discovery/overlap/` | `components/CellShard/tests/jbc/discovery/overlap/` | Disjoint baseline, bounded overlap, stability, certification, and promotion gate have six tests. |
| CS motif discovery | complete and tested | `components/CellShard/include/CellShard/compiler/discovery/motif/` | `components/CellShard/tests/jbc/discovery/motif/` | Typed motifs, recurrence, regulatory baseline, exact candidate, and frequent-fragment experiment have eight tests. |
| CS factor/topic discovery | complete and tested | `components/CellShard/include/CellShard/compiler/discovery/factor_topic/` | `components/CellShard/tests/jbc/discovery/factor_topic/` | External evidence, internal experiment, candidates, coverage, utility, and soft membership have six tests. |
| CS operation-trace discovery | complete and tested | `components/CellShard/include/CellShard/compiler/discovery/operation_trace/` | `components/CellShard/tests/jbc/discovery/operation_trace/` | Access/coaccess, graph family, partial recurrence, persistent order, and provenance comparison have eight tests. |
| CS trajectory discovery | complete and tested | `components/CellShard/include/CellShard/compiler/discovery/trajectory/` | `components/CellShard/tests/jbc/discovery/trajectory/` | Prefix, branch, delta, window, transition, prefetch, working-set, null, and promotion mechanisms have twelve tests. |
| CS multimodal discovery | complete and tested | `components/CellShard/include/CellShard/compiler/discovery/multimodal/` | `components/CellShard/tests/jbc/discovery/multimodal/` | Identity spine, overlays, missingness, cross-modal atoms, destination bundle, certification, and promotion have ten tests. |
| CS sequence compatibility discovery | complete and tested | `components/CellShard/include/CellShard/compiler/discovery/sequence_compat/` | `components/CellShard/tests/jbc/discovery/sequence_compat/` | Reference/strand identity, intervals, halo, coverage, mock provider, and long-range bridge have six tests. |
| CS composition | complete and tested | `components/CellShard/include/CellShard/compiler/composition/` | `components/CellShard/tests/jbc/composition/` | Coverage algebra, joins, overlays, bundles, ordering, parameter binding, and derivations are integrated with 24 focused tests. |
| CS explicit grammar | complete and tested | `components/CellShard/include/CellShard/compiler/grammar/` | `components/CellShard/tests/jbc/grammar/explicit/` | Typed symbols, productions, DAG, serialization, simplification, exact coverage, and flat fallback are implemented. |
| CS induced grammar experiment | complete and tested | `components/CellShard/src/compiler/grammar/induced/` | `components/CellShard/tests/jbc/grammar/induced/` | Candidate, bounds, MDL, stability, complete-cost comparison, and promotion evidence are executable; non-promotion remains valid. |
| CS basis selection | complete and tested | `components/CellShard/include/CellShard/compiler/basis/` | `components/CellShard/tests/jbc/basis/` | Baseline, greedy, multi-basis, overlap, exact oracle, alternatives, refinement, promotion, and manifest have seventeen tests. |
| CS superatoms | complete and tested | `components/CellShard/include/CellShard/compiler/composition/superatom/` | `components/CellShard/tests/jbc/superatom/` | Candidate, membership, cost, statistics, lifecycle, evolution, and benchmark mechanisms have eight tests; promotion is evidence-gated. |
| CS persistent partials | complete and tested | `components/CellShard/include/CellShard/compiler/partial/` | `components/CellShard/tests/jbc/partial/` | Algebra-specific partial states, images, promotion, dependency freshness, and merge trees have eighteen tests. |
| CS global graph and schedule | complete and tested | `components/CellShard/include/CellShard/compiler/graph/` | `components/CellShard/tests/jbc/global_ir/` | Operation nodes, graph recipe/family, dependencies, serialization, realizations, rewrites, replay, and distributed certificate have fourteen tests. |
| CS atom store | complete and tested | `components/CellShard/include/CellShard/artifact/atom_store/` | `components/CellShard/tests/jbc/atom_store/` | Format, arenas, frames, dictionary, publication, recovery, GC, codecs, caches, and linking have thirty tests. |
| CS runtime v2 | complete and tested | `components/CellShard/include/CellShard/runtime/v2/` | `components/CellShard/tests/jbc/runtime/` | Read planning, sources, materialization command IR, residency, topology, transport, staging, recovery, and CUDA worker paths have 24 tests. |
| CS integrated validation/package matrix | complete and tested | `components/CellShard/docs/JBC/evidence/integration_receipt.md` | `components/CellShard/tests/jbc/validation/` | The final integrated host, sanitizer, benchmark, CUDA-build, standalone, embedded, and package gates are recorded. |
| CS biological novelty campaign result | partial | `components/CellShard/docs/JBC/evidence/biological_novelty_readiness.md` | `components/CellShard/docs/JBC/evidence/metric_schema.md` | Technical campaign machinery is ready, but no new reserved biological run supports promotion; no performance claim is made. |

## Design package and empty classes

| Subsystem | Status | Source anchor | Finding |
|---|---|---|---|
| Original JBC pre-ledger package | design-only | `planning/jbc-preledger-v1/` | It is a historical design and execution plan. Current source bodies and receipts, not its proposed task titles, determine implementation status. |

There are no **complete but unintegrated** subsystems: every retained producer
tip is reachable from its repository mainline. There are no **scaffold-only**
subsystems in the inventoried JBC implementation. There is no **obsolete compatibility code**
subsystem yet: even superseded ownership
names contain useful mechanisms that Part One must preserve or rehome before a
later compatibility-only disposition can be justified.
