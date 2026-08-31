# Program architecture and invariants

## Purpose

This document freezes the implementation meaning of the joint compiler before Todo Orchestrator planning. It is not another research pass. The difficult architectural choices are resolved here so lower-reasoning implementation agents do not substitute a familiar generic distributed runtime.

## Compiler thesis

The joint compiler treats **recurrent biological organization as source code**.

Cellerator compiles the local mathematics of typed biological relations. CellShard discovers reusable execution atoms from repeated support, convergence/divergence, programs, trajectories, identities, modalities, segments, operation families, and mutation half-lives; composes them through an exact typed hierarchy; persists and specializes them; and lowers them into global schedules and runtime execution.

```text
Biological source IR
    typed domains, relations, values, strata, dynamics, workload families
        ↓
Atom evidence IR
    overlapping uncertain proposals with provenance and stability
        ↓
Certified atom IR
    exact coverage, identity, ports, planes, ownership, dependencies
        ↓
Composition/grammar IR
    typed productions, multi-parent derivations, superatoms
        ↓
Basis IR
    selected reusable atoms/productions for workload-family distributions
        ↓
Physical/lowering IR
    materializations, projections, partials, executable continuation points
        ↓
Global schedule IR
    decomposition, placement intent, partial trees, persistent orders
        ↓
Topology/runtime IR
    storage, NUMA, GPUs, routes, residency, CUDA Graphs, execution
```

## Atomicity model

An atom is atomic only relative to a compiler level and admissible composition algebra. It must be independently nameable, exactly related to biological coverage, independently materializable/bindable/movable/invalidatable where relevant, composable through declared operations, and consumable or producible by at least one operation. It may still be internally decomposable by Cellerator.

Required atomicity capabilities include semantic, ownership, materialization, transfer, cache/reuse, order, algebraic, executable, grammar, Cellerator-local, and CellShard-global atomicity. These capabilities may coexist.

## Three overlap domains

1. **Proposal overlap:** uncertain discovery memberships may overlap freely within explicit bounds.
2. **Physical overlap:** several views or replicas may cover the same biological data.
3. **Contribution overlap:** execution may overlap only when a declared partial-result algebra proves exact reconstruction.

A read-only halo or replica never becomes a contributor merely because it is present.

## Source-language recurrence that may generate atoms

- repeated typed sparse support and support signatures;
- co-support source groups and destination convergence/divergence;
- overlapping programs and relation motifs;
- repeated segment definitions;
- stable support with mutable values/activity/state/gradients;
- operation-polymorphic use of one support across apply, transpose, contraction, gating, normalization, moments, bundles, chains, and gradients;
- trajectory/state neighborhoods, common lineage prefixes, branch-local structure and deltas;
- shared multimodal identity spines with modality-specific overlays and typed cross-modal relations;
- future coordinate, strand, halo, hierarchy, and long-range sequence relationships;
- repeated operation, order, partial, and graph-family access patterns.

No property is assumed universally. Discovery records evidence and confidence; exact certification determines execution eligibility; complete cost determines selection.

## Preserved live Cellerator substrate

- include/Cellerator/geometry/support_atlas.hh
- include/Cellerator/geometry/rectangular_support.hh
- include/Cellerator/geometry/relation_cover.hh
- include/Cellerator/geometry/compiler/v2/
- include/Cellerator/geometry/optimizer/portfolio_v1.hh
- include/Cellerator/compute/operation/operation_core_v2/
- include/Cellerator/compute/operation/relation_algebra_v2/
- include/Cellerator/execution/projection_value_plane/
- include/Cellerator/execution/geometry_acquisition_v2/
- include/Cellerator/execution/index_space/hierarchical_index_space_v1.hh
- include/Cellerator/compute/operation/candidate_catalog_v3/
- include/Cellerator/execution/program/program_v2.h
- include/Cellerator/execution/training_program_v2/
- include/Cellerator/profiling/partition_export.h

## Cellerator compatibility/migration substrate

- include/Cellerator/interop/cellshard/access.cuh
- include/Cellerator/runtime/multi_gpu/fleet.cuh (standalone local support remains; global authority moves downstream)
- CPK1/CP-BP v1 semantics and wire bytes
- Cellerator examples/models directly consuming legacy CellShard sharded<T>

## Preserved live CellShard substrate

- include/CellShard/identity/strong_id.hh
- include/CellShard/domain/descriptor.hh
- include/CellShard/domain/partition.hh
- include/CellShard/artifact/image.hh
- include/CellShard/artifact/extent.hh
- include/CellShard/artifact/catalog.hh
- include/CellShard/artifact/snapshot.hh
- include/CellShard/runtime/source/payload_source.hh
- include/CellShard/runtime/source/local_file_source.hh
- include/CellShard/runtime/residency/host.hh
- include/CellShard/runtime/residency/device.cuh
- include/CellShard/io/pack/image_envelope.hh

## CellShard compatibility/replacement substrate

- include/CellShard/runtime/layout/sharded.cuh
- include/CellShard/runtime/distributed/distributed.cuh
- include/CellShard/runtime/device/sharded_device.cuh
- include/CellShard/runtime/storage/shard_storage.cuh
- include/CellShard/io/csh5/api.cuh
- include/CellShard/io/pack/execution_payload.cuh
- CSH5/CSPACK/CPEXEC01 direct hot execution paths

## Architectural invariants

- Biological organization generates the candidate atom vocabulary, composition search, basis candidates, decomposition dimensions, and persistent representations.
- Atomicity is relative to compiler level and interface; no single smallest universal atom exists.
- Candidate atom, certified atom, basis atom, physical instance, encoded replica, partial atom, superatom, topology realization, and resident instance are distinct records.
- CSG1 remains one exact selected semantic execution cover; overlapping atom evidence lives in a sibling CellShard evidence atlas.
- CPE2 remains a target-specific physical execution image and may be stored/delivered by CellShard without CellShard reinterpreting provider-private payloads.
- Cellerator owns local biological mathematics, exact local decomposition, partial algebra, projections, numerical policy, and prepared execution.
- CellShard owns cross-operation/cross-workload atom discovery, composition grammar, bases, persistence, global decomposition, placement, residency, transport, collection, and schedules.
- Standalone Cellerator has no required CellShard dependency; embedded CellShard may consume an explicit private compiler bridge.
- Approximate discovery proposes; independent exact certification constructs execution-eligible atoms.
- Proposal overlap, physical overlap, read-halo overlap, and execution-contribution overlap are modeled separately.
- Stable structure, mutable values, state, gradients, partial results, physical views, executable recipes, evidence, and lineage are separate atom planes.
- Every persistent structure has explicit correctness dependencies and separate preference/performance freshness.
- The execution hierarchy is a typed multi-parent derivation DAG; explicit grammar is required, induced grammar is experimental.
- A biological execution basis may be redundant, may have several workload-family variants, and may validly be empty/no-basis.
- Persistent partial computation requires exact dependency closure, algebra, generation, numerical policy, and positive amortized benefit.
- CellShard artifacts may resume Cellerator lowering at several stages; each stage states exactly what it bypasses and how it falls back.
- Cellerator can consume externally selected exact decomposition/order and atom bindings only after validation.
- Multi-extent direct execution is optional; explicit profiler-visible assembly is the complete fallback.
- One support family may have several operation-specific views and an optional generalized cross-operation family.
- Global storage/movement/reuse costs may influence Cellerator search through a generic external-cost interface, never through a CellShard dependency.
- Generic infrastructure—arenas, content stores, hypergraphs, schedulers, caches, CUDA Graphs, NCCL, numaBraid—is downstream machinery, not the biological novelty.
- No physical padding, storage frame, or transport batch acquires biological identity by convenience.
- No model objective, loss, epoch, sampler, optimizer policy, pseudotime inference, or biological causal interpretation enters CellShard or low-level Cellerator compiler ownership.
- No Baseplane implementation occurs; only provider-defined coordinate/strand/halo coverage and mock compatibility are planned.
- V100 and current topology choices remain physical specializations; CE-AMP is untouched.
- Hot and sealed runtime paths perform no atom discovery, geometry search, provider discovery, catalog parsing, global sorting, or hidden allocation.
- New public and hot interfaces are pointer-plus-count and explicit-storage first; benign STL remains acceptable in isolated cold tooling/tests where justified.
- Every experimental mechanism has a baseline, complete-cost benchmark, promotion gate, fallback, and valid evaluated-not-promoted result.
- Biological novelty must be isolatable with generic baselines and matched null transformations.
- The final source and ledger plan preserves two Git histories and two Todo authorities with explicit cross-project receipts and submodule integration order.

## Generic infrastructure versus novelty

The following are necessary infrastructure but are not, by themselves, the biological claim: immutable snapshots, content/action caches, arenas/frames, asynchronous I/O, graph/hypergraph partitioning, topology, NUMA, P2P, CUDA Graphs, NCCL, numaBraid, compression, leases, and recovery.

They become part of the biology-native compiler only when biology-generated atoms and compositions change what is materialized, reused, ordered, moved, combined, or avoided.

## Live contradictions converted into explicit implementation work

1. `components/README.md` currently states that components do not own native planning/runtime. The new charter must distinguish ordinary adapters from privileged compiler components; CellShard is the first privileged compiler component.
2. `components/CellShard/AGENTS.md` and `docs/FORMAT_ROLES.md` still center a storage/staging-only `.csh5 → .cspack → GPU` hot path. A versioned successor charter must supersede this while retaining compatibility behavior and history.
3. Cellerator semantic covers are exact and disjoint; atom discovery must overlap. This is resolved by a sibling CellShard evidence atlas, not by weakening CSG1.
4. Cellerator `program_v2` has a narrow local binding model. The resolved first step is a `prepared_atom_fragment_v1` wrapper, not an immediate program-v3 rewrite.
5. CellShard uses 64-bit strong IDs while current Cellerator operation/core compiler records use 128-bit stable IDs. The resolved seam is an explicit namespace-qualified 128-bit cross-project identity and adapters; content identity remains separate.
6. CellShard catalog/snapshot and some Cellerator portable-support validators contain linear-search or quadratic duplicate patterns acceptable for prior bounded work but unsuitable as the new atlas-scale certification path. New scalable validators are adjacent implementations.
7. The two Todo authorities have no native cross-project interface links in the current observation. The later bootstrap must use mirrored version/hash receipts rather than pretending query grouping creates dependency authority.
8. The historical CellShard access-adapter task has a raw-closed/effective-ready anomaly. It is evidence to reconcile, not implementation to duplicate.

None of these contradictions requires changing the compiler thesis.

## Completion meaning

Completion is not “distributed SpMM works.” Completion means the full exact atom hierarchy, evidence/certification split, composition/grammar, basis/no-basis paths, persistent partials, lowering-resumption artifacts, atom-aware Cellerator fragment compiler, global operation/schedule IR, atom-native persistence, topology/transport/residency/runtime, independent verification, and biology-isolating evidence hooks are integrated while standalone Cellerator remains valid and CE-AMP remains untouched.
