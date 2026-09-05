# CE-EXOP Program: Extended Operation Portfolio, Scalable Geometry, and Profiler Readiness

## Authority and execution boundary

CE-EXOP is the execution program for the next Volta-first Cellerator campaign. It is subordinate to [AGENTS.md](../AGENTS.md), [scope.md](../scope.md), [Architecture](architecture.qmd), [Current Implementation](current_implementation.qmd), [Migration Roadmap](migration_roadmap.qmd), and the completed [CE-GEO program](CE_GEO_PROGRAM.md). Those documents remain the architecture spine. This document supplies the campaign charter, work decomposition, integration rules, validation contract, and completion meaning needed by a fresh first-class Codex lane agent.

Implementation begins only through the configured Project Control front door after CE-EXOP-01 is claimed. First-class agents own serial lanes and managed isolated-merge workspaces. Local coding workers, if later delegated by a claimed parent, are subordinate candidate producers only: they do not claim CE-EXOP tasks, own lanes, publish interfaces, communicate cross-lane authority, join rendezvous, or complete the parent. Integration lanes alone accept cross-lane merges. This bootstrap creates no implementation worktree and claims no task.

The plan authority is `ce-exop-plan.json`, schema version 3. SQLite is the live task-graph authority; generated Markdown and snapshots are projections. This document explains the program but does not override live workflow state.

## Purpose and non-goals

CE-EXOP means **Extended Operation Portfolio, Scalable Geometry, and Profiler Readiness**. Its purpose is to turn the completed CE-GEO vertical slice and CE-PTR production substrate into a broad, correct, independently selectable, profiler-visible operation and geometry portfolio. It prepares the surface on which later measurement can decide winners. It does not promote candidates merely because they exist.

CE-EXOP is not CE-AMP, the full Cellerator plus CellShard Nsight campaign, a biological-data acquisition campaign, or a preprint benchmark campaign. It does not require an A5000 or B200 implementation, CellShard integration, a new dataset, exhaustive Nsight sweeps, or a real-data speedup. It does not add trainers, losses, epochs, optimizer policy, model schedulers, or a general tensor framework to Cellerator.

## Inherited completed foundation

CE-GEO established portable CSG1 semantic geometry, CPE2 physical projections, exact semantic and physical covers, architecture capability records, source-linked providers, candidate catalog v2, executable program v2, a correct sm_70 N=64 MMA plus residual vertical slice, typed relation-algebra vocabulary, and strong correctness/evidence infrastructure. Its performance promotion remains limited to the exact measured N=64 regime. Whole-exchange fusion and other variants remain non-promoted evidence.

CE-PTR subsequently established pointer-first production data structures, explicit and caller-owned storage, static plans, device-first preparation, graph-native execution, compact views, 64-bit-safe extents in its owned paths, and a strict no-new-inappropriate-STL policy. CE-EXOP consumes those results and does not recreate either campaign.

The live bootstrap audit found an additive authority gap: CE-PTR-15 is terminal and its final acceptance evidence is authoritative, but no `CE-PTR-COMPLETE` checkpoint existed. The schema-v3 plan therefore adds one evidence-only terminal bootstrap record, `CE-EXOP-PTR-BASELINE`, whose reached checkpoint records that prerequisite without changing CE-PTR-15, its result, or the historical CE-PTR graph.

## Live implementation gaps at bootstrap

At source commit `d7dc09bc39a758ffca581b6b1f47d1ffcce2ec6d`, the important gaps are:

1. Target refinement selects preconstructed proposals rather than maintaining a genuinely interacting source by destination optimization state.
2. The mature provider path remains centered on a narrow synthetic N=64 hybrid regime. Other widths and WMMA/PTX mechanisms are incomplete or calibration-only.
3. Five relation-algebra catalog entries encode out-of-range operation kinds and deliberately reject preparation pending a real operation-core schema v2.
4. Transpose, contraction, gradients, segments, bundles, exchange, and residual mechanisms exist unevenly and are not a complete catalog/planner portfolio.
5. Mutable trainable values are logical-primary; projection-primary value ownership, generation publication, and direct projection-order gradients do not exist.
6. Value packing rewrites broad buffers, including physical holes, rather than offering occupied-slot, dirty-subset, and direct-producer routes.
7. `executable_program_v2` exists, but v1 still carries a canonical five-entry identity rule and projection switch; program capacity remains fixed.
8. Training is hard-coded to `native_feature_major_n16` rather than planner-selected prepared forward, transpose, and edge-gradient programs.
9. `CELLERATOR_CUDA_MODE` still drives broad CUDA policy. The provider-selection, tuning-profile, experimental-candidate, and marker controls documented by CE-GEO are absent; root CMake applies global fast math and cache policy.
10. Candidate discovery and planner storage retain fixed capacities (`operation_candidate_capacity` is 64 and `maximum_planner_candidates` is 32) instead of caller-owned cold workspace.
11. Geometry workload profiles do not represent a caller-owned mixture of operations, widths, reuse, value modes, layout continuity, segment work, fusion, graph capture, and memory budgets.
12. Source audits expose quadratic duplicate scans and repeated global scans in catalog validation, geometry builders/validators, and refinement paths.
13. Central paths reject aggregate counts above 32-bit limits, including MMA cover validation, physical payload construction, value packing, relation apply, transpose, edge gradients, and some segment paths.
14. CPE2 payload builders still assume individual physical arrays fit 32-bit element-size fields; no general chunk manifest joins bounded local sections into a 64-bit aggregate relation.
15. Profiling identity, stage manifests, resource receipts, mechanism statistics, stable per-stage names, and disabled-marker hot-path proofs are piecemeal.

CE-EXOP-01 records exact current paths and line identities in a machine-readable preflight artifact before implementation. This list is a bootstrap route, not a substitute for that source-identity record.

## Settled ownership boundary

Cellerator owns typed biological relation mathematics, biological axes and identity, semantic geometry, target-specific projections, low-level operations, structure-sensitive composition, complete-cost planning, prepared GPU execution, persistent order, value-generation correctness, and profiler-visible mechanism descriptions.

GlassHelix owns developmental and dynamical theory, model architecture, objectives, losses, model-specific interpretation, and training policy. CellShard owns distributed placement, global sharding, transport, distributed accumulation, global dataset hierarchy, and later joint profiling orchestration. BioPrep owns parsing, QC, normalization policy, conventional preprocessing, and dataset workflow. Cellerator remains fully usable on one GPU without CellShard and may export only generic partition, order, locality, stage, and resource descriptors.

No CE-EXOP production path imports CellShard headers, invokes CellShard callbacks or schedulers, or creates a privileged CellShard route. No public “attention” abstraction is introduced; sparse biological exchange remains a composition of typed relation primitives.

## Settled geometry and scale model

The authoritative pipeline remains:

```text
typed biological relation
    -> portable semantic support evidence
    -> portable semantic geometry (CSG1)
    -> target-specific exact mechanism cover
    -> architecture-specific physical projection (CPE2)
    -> candidate catalog
    -> complete-cost planner
    -> prepared executable program
```

CP-BP/CPK1 remains a sibling sparse mechanism. CP-BP is not converted into the MMA optimizer. Semantic components are not physical tiles. GPU generation and WMMA shapes never enter portable biological identity. Pure-sparse fallbacks remain complete, and no logical edge may be deleted to increase Tensor Core occupancy.

Hierarchical index spaces use:

```text
global biological identity/count       uint64
semantic partition identity            stable uint64 identity
component-local index                  smallest valid u16/u32/u64 width
physical kernel-local index            u32 when the component fits
aggregate relation                     arbitrary bounded local components
```

The public pointer-first view records global extent, partition identity, local extent and width, local-to-global mapping, and optional global-identity sidecar. A projection may be split into independently indexable components. Kernels receive one compact local component plus global recovery metadata. Aggregate logical counts do not impose 64-bit arithmetic on every kernel lane, and no relation is rejected solely because the aggregate edge count exceeds `UINT32_MAX`.

CSG1/CPE2 extension sections should encode a small chunk manifest pointing to independently bounded payload sections. CPE3 is not created merely because one physical array exceeds 4 GiB. A new top-level artifact version requires a source audit proving existing extension machinery cannot safely encode the chunk table.

Potentially large relation, support, work-window, value, and validation paths must be linear or near-linear. Use exact inverse maps, generation marks, radix/sort workspace, bounded prepared hash tables, streaming checks, and count/scan/fill construction; never repeated global scans or quadratic duplicate search.

## Operation-core schema v2

Operation-core v2 defines real semantics for:

- `relation_apply`;
- `relation_apply_transpose`;
- `contract_on_support`;
- `segment_reduce`;
- `segment_normalize`;
- `edge_map_or_gate`;
- `relation_bundle_apply`;
- `sparse_axis_update`.

The enum stays mathematically compact. Fused candidates are prepared stage graphs with stable composition IDs, not new global operation kinds. A v2 problem expresses typed source/destination axes, orientation, logical edge identity, segment/result axes, numerical and output-effect policy, persistent order, value mode, gate/bundle/update operands, structure and value generations, required gradients, and stable operation identity. Explicit v1 adapters may remain, but compatibility cannot preserve a second internal engine or rejected placeholder enum values.

Every primitive has a direct low-level callable surface, one or more prepared candidates, catalog and planner visibility, mechanism/resource metadata, explicit numerics, exact order and identity recovery, focused correctness tests, a synthetic profiler fixture, and legal fallback behavior.

## Operation portfolio

### Relation apply and residual

The sm_70 inventory contains independently requestable, uniquely named candidates for N=16; both plausible N=32 CTA organizations; N=64 direct-global, shared-A, and software-pipelined variants; disjoint N>64 panels; WMMA 16x16x16, 8x32x16, and 32x8x16; an isolated PTX `mma.sync m8n8k4` experiment where feasible; pure sparse; hybrid MMA plus residual; canonical input; and persistent physical-order input.

Residual execution contains scalable row-owned, warp-per-row, CTA-per-high-degree-row, degree-bucketed mixed, pinned feature-major, same-output-owner fused, separate same-output, and optional separate-buffer concurrent plus explicit combine candidates. None is canonical before later measurement.

### Transpose and contraction

Transpose uses an explicit cover and source-owned output schedule, sparse and geometry-legal MMA candidates, direct projection-order value maps, compact local indices with global edge recovery, and exact dense-input gradients. It does not scan every forward edge per output, reconstruct CSR implicitly, or require forward and transpose to share a target cover.

`contract_on_support` includes scalar/thread-per-edge, cooperative group, warp-per-edge, rectangular MMA score-tile, and exact residual candidates. Outputs may be logical-edge order or projection-native edge order; gradients may be emitted directly in projection order. Width-dependent D regimes are explicit.

### Segments, maps, gates, and sparse updates

Segment reduction supports sum, mean, minimum, maximum, sum of squares, first/second moments, and variance-ready paired statistics through differentiated warp, CTA, and large-segment mechanisms.

Segment normalization supports log-sum-exp, softmax, log-softmax, L1, L2, and RMS normalization plus required backward operations. It offers logical-edge-order, projection-order segmented, and cover-native MMA/residual-partition paths.

Edge maps and gates support arbitrary per-edge maps and multiplicative gates, per-source, per-destination, per-component/module, factorized source by destination, predicate, and dynamic active-support masks over a stable support superset. Dynamic gating changes value state, not topology identity, unless the support really changes.

Sparse axis update supplies typed assign, add, subtract, multiply, and numerically valid maximum primitives. These are low-level scatter/update operations; optimizer policy remains with the caller.

### Bundles, chains, hierarchy, and moments

Relation bundles provide sequential, grouped-launch, and shared-destination-owner candidates over multiple typed source domains, distinct value planes, and independent projections with one destination epilogue where legal.

Relation chains provide materialized two-hop execution, persistent-order two-hop execution without forced canonicalization, hierarchy pool/broadcast, and stage descriptions without mandatory adjacency-matrix materialization.

Relation moments add a profiler-visible paired traversal such as `P X` and `P (X * X)` where legal. It remains reusable mathematics, not a velocity or model API.

## Fused and unfused composition inventory

Every fused candidate preserves independently callable unfused stages. Initial compositions are experimental, `requires_measurement`, explicitly selectable, and never auto-promoted:

- value generation -> projection value pack;
- value pack -> relation apply;
- MMA contribution -> same-owner residual;
- relation apply -> epilogue;
- contraction -> edge map;
- contraction -> segment maximum/sum;
- normalization -> relation apply;
- contraction -> map/gate -> normalization -> relation apply;
- relation bundle -> shared destination accumulation;
- relation moments pair.

The four-stage sparse biological exchange is not named “attention” in the public API.

## Mutable-value model

Logical-primary mode stores logical edge-order values and explicitly packs a selected projection. It remains appropriate for inference, static/slow values, and interchange.

Projection-primary mode stores the MMA physical value plane, one or more residual physical value planes, and stable logical-edge maps as one logical relation generation. Ownership is exact and disjoint. Padding holes have no biological identity and are never trainable. All components share one structure ID, structure epoch, and value generation; publication occurs only after every required component is ready. Gradients may be written and updated directly in projection order. Import/export, checkpoint conversion, and architecture migration use explicit logical maps. Dirty subsets do not rewrite physical holes, and projection-native producers may bypass packing. SGD, Adam, and all optimizer policy remain outside the abstraction.

## Geometry optimizer portfolio

Portable semantic geometry strategies and target-specific physical-cover strategies have separate source-linked registries and data-only problem/solution contracts. One serializable optimizer envelope carries either stage using an explicit tag.

The deterministic joint greedy strategy maintains real interacting source groups, destination groups, rectangles, residual fragmentation, input/output order cost, and operation-mixture state. Agglomeration, construction, moves, swaps, splits, merges, rectangle add/remove, and admissible work-item exchange update every affected exact rectangle and cost.

The multilevel strategy builds a sparse affinity hierarchy, solves a coarse grouping and cover, uncoarsens, locally refines, preserves provenance, uses bounded memory, and remains deterministic under operation mixtures and work windows.

The bounded-overlap strategy owns an explicit source-group dictionary and charges repeated source state, dense-input movement, value maps, gradient reconciliation, persistent bytes, construction, and canonical recovery. Logical contribution ownership remains unique. Zero overlap is equivalent to the disjoint baseline.

The exact small-problem oracle uses exhaustive or branch-and-bound search to certify planted fixtures, objective deltas, regression quality, and distance from optimum. It need not scale to atlases.

The portfolio runs several strategies and emits a deduplicated Pareto frontier over predicted latency, preparation, persistent/transient bytes, value updates, layout/canonicalization, forward/transpose/contraction quality, and reuse. Incremental work-window realization reuses a dataset-level source skeleton. The optional GPU-assisted cold backend evaluates batched rectangle scores, exact census changes, and proposal deltas asynchronously with CPU-identical results and its own resource receipt. A documented non-adoption is acceptable and never places scoring work in steady-state operation execution.

## Workload profile v2 and planner portfolio

Workload profile v2 is a pointer-plus-count view over caller-owned components. Each component records operation, orientation, dense-width range/bucket, frequency, repetitions, structure/projection/value/dense-layout/work-window reuse, logical- or projection-primary values, static/dynamic values, packed-output permission, canonical-output requirement, graph-capture requirement, segment operations, fusion opportunities, and persistent/transient budgets. It describes expected computation, not epochs, losses, labels, model quality, or training policy.

Candidate catalog v3 adds provider/device/projection/capability/numerical/width/operation identity; experimental classification; variable resource, mechanism-statistics, stage-graph, profiling, and persistent/transient queries; stable kernel-stage identities; empirical-measurement requirements; and explicit user selection. Cold discovery uses caller-owned pointer-plus-count workspaces and may exceed 32 candidates. The selected prepared program remains compact.

The planner performs compatibility and numerical filtering, coarse analytical filtering, Pareto pruning, configurable shortlisting, freshness checks, user-forced selection, experimental inclusion/exclusion, complete stage costing, layout continuity, value-mode economics, and connected-operation/fusion selection. No kernel self-promotes.

Executable program v2 becomes the sole internal production engine. V1 may remain only as a thin deprecated source adapter. Active production removes the five-entry pointer identity assumption, central projection-type switch, v1-only discovery, and hard-coded typed preparation.

## Training program v2

Training program v2 is a generic planner-selected prepared stage graph over forward relation apply, transpose relation apply, logical-edge gradient/contraction, optional sparse update, generation publication, persistent order, and graph capture. The caller owns optimizer policy. Profiler-visible modes include logical-primary per-generation packing, projection-primary direct gradients, persistent physical source order, explicit canonicalization, graph replay, unfused stages, and obvious fused stages.

## Geometry acquisition v2

The low-level callback routes remain available, but a two-pass requirements façade assembles compiled providers and catalogs, validates or compiles semantic geometry, realizes/enumerates physical projections, invokes the planner, prepares program v2, consumes caller/session buffers, and returns complete diagnostics without hidden allocation.

Routes include compile now, load CSG1, load compatible CPE2, adapt CPK1, explicit CPE2-to-embedded-CSG1 fallback, multiple projection candidates, chunked physical payloads, and both value modes.

## Provider build controls

CE-EXOP implements:

- `CELLERATOR_CUDA_PROVIDERS` (`generic`, `generic;sm70`, or `sm70`);
- `CELLERATOR_CUDA_TUNING_PROFILE`;
- `CELLERATOR_CUDA_PRIMARY_PROVIDER`;
- `CELLERATOR_ENABLE_EXPERIMENTAL_CANDIDATES`;
- `CELLERATOR_ENABLE_PROFILING_MARKERS`;
- existing `CMAKE_CUDA_ARCHITECTURES` and `CELLERATOR_ENABLE_CUDA_LINEINFO`.

Common, generic, and sm70 provider sources become distinct targets and generate a compiled-provider manifest. Hardware discovery may suggest defaults but does not redefine binary content. Provider-local helpers separate precise/approximate numerics, tuning, line information, markers, register limits, launch bounds, and cache policy. `--use_fast_math`, `-dlcm=ca`, register caps, and launch bounds are never silently global. No sm86 implementation is added.

## Profiler-readiness contract

Every prepared candidate exposes a cold pointer-first static mechanism manifest containing operation/candidate/provider/capability/projection/geometry IDs; value and order modes; stable stage IDs and static names; stage kinds and launch count; logical, physical, useful, and padded work; relation, dense-input, output, and value-pack bytes; residual edges; group/tile and owner-work statistics; persistent/transient bytes; shared memory; threads/warps per CTA; graph-capture status; numerics; and `requires_measurement` state.

A cold resource query may incorporate build-specific `cudaFuncGetAttributes` data. It never runs in the hot path. Optional profiling markers are disabled by default, allocate nothing, contain no dynamic strings, use stable correlation IDs and candidate/stage names, and add no measurable launch-path work when disabled beyond already-required prepared-state dereferences. Kernel variants have distinct human-readable symbols.

The generic CellShard-facing export is Cellerator-owned and CellShard-independent. It includes semantic partition and local/global index descriptions, input/output order, stage graph, candidate/provider/capability IDs, memory/transform costs, graph compatibility, device requirements, and optional communication-boundary descriptors—but no transport implementation.

## Lane and integration topology

The run contains 31 serial first-class lanes: one coordinator; eight core-contract lanes; five optimizer lanes; ten value/operation/training lanes; three validation/profiler-fixture lanes; and four integration lanes. Lanes are parallel across narrow file roots. Shared root/subsystem CMake, package exports, umbrella headers, central manifests/catalogs, and documentation integration belong only to designated integration tasks.

After architecture and baseline freeze, safe lane heads exist for BUILD, SCALE, OPCORE, CATALOG, PLANNER, ACQUISITION, PROFILE, GEOMETRY, all five optimizer lanes, VALUE, APPLY, RESIDUAL, TRANSPOSE, CONTRACT, SEGMENT, GATE-UPDATE, BUNDLE-CHAIN, FUSION, TRAINING, VALID-HOST, VALID-CUDA, and PROFILER-FIXTURES. Private fixtures allow algorithm, CUDA, validation, and profiling work to begin before public interfaces freeze. Integration waits on frozen contracts and real fan-in barriers.

Core integration owns provider/build, scale, operation core, catalog/program, planner, acquisition, profiling, and geometry-contract fan-in. Geometry integration owns optimizer/value/cover convergence. Operation integration owns apply/residual/transpose/contraction, segments/gates/updates/bundles/chains, fusion, and training. Final integration owns the complete package surface and evidence generation.

Only two rendezvous exist: core ABI and optimizer fan-in. The seven barriers in `ce-exop-plan.json` represent real integration joins. No campaign-wide lock exists. Narrow locks protect root build, scale schema, operation core, catalog/program, planner, persistence, profile schema, docs, final integration, and the CE-AMP interlock. Comparable timing reuses `cuda-benchmark-mutex`.

## Resource and workspace policy

Host work requires no accelerator. CUDA correctness gates request `accelerator:any`, verify sm_70 when required, and release immediately. Independent V100 correctness tasks may run concurrently on separate leased devices. Timing smoke additionally holds `cuda-benchmark-mutex`. No task hard-codes an ordinal/UUID, leases the A5000, or runs CE-AMP code.

Write lanes use `isolated_merge` workspaces tied to their integration task. Coordinator and final integrators use exclusive authority where appropriate. Bootstrap creates none of these worktrees; they are provisioned only when later Project Control assignment claims a writable lane.

## Validation and acceptance

Scale validation uses synthetic identities beyond 2^32 without impossible allocations. It proves chunked aggregate structures, compact local projections, exact recovery, overflow-safe sizes/grids/conversions, and non-quadratic operation-counter growth.

Optimizer validation proves deterministic replay, exact cover/admissibility, unique contributions, evaluator-matching deltas, greedy/multilevel quality versus the exact oracle, overlap accounting, zero-MMA/all-residual legality, Pareto deduplication, and external snapshot validation.

Operation validation covers empty/singleton/high-degree rows and segments; 15/16/17 and 31/32/33 tails; N/D ranges; logical/physical order; static/dynamic values and gates; alpha/beta; NaN/Inf policy; forward/backward; fused/unfused equivalence; generation changes; >2^32 identities; and chunk boundaries.

After preparation, runtime validation proves no allocation, provider discovery, image parsing, strategy execution, host sorting, dynamic strings, manifest construction, device-wide synchronization, hidden canonicalization, or hot resource query. CUDA validation includes focused referees, Compute Sanitizer, applicable race/init checks, external streams, graph replay, concurrent plans, repeated generations, and fused/unfused comparisons.

Profiler fixtures use synthetic controlled widths, degrees, occupancy, residual fractions, value modes, work-window groups, segment sizes, and fusion choices. CE-EXOP may perform tiny timing, launch sanity, resource queries, symbol/lineinfo checks, and one or two captures proving stage correlation. It must not perform exhaustive Nsight sweeps or deep microtuning.

## Deferred work

`CE-EXOP-DEEP-PROFILING` is a human-controlled decision with values `deferred` and `authorized`, initially `deferred`. No CE-EXOP task requires authorization. The following remain explicit follow-up work:

- deep Nsight Systems and Nsight Compute profiling;
- joint Cellerator plus CellShard topology/profiling orchestration;
- extensive biological dataset search or download;
- embryo, heart, regulatory, perturbation, multiome, and trajectory campaigns;
- complete cost-surface calibration and performance promotion;
- preprint figures, evidence, and claims;
- CE-AMP.

CE-EXOP-287 publishes `docs/CE_EXOP_PROFILING_HANDOFF.md` with the actual remaining questions and exact profiler-ready identities.

## Completion semantics

CE-EXOP completes when the required contracts and portfolios are implemented and integrated; operation primitives and fused/unfused candidates are independently callable, planner-visible, correct, and profiler-visible; program v2 is the sole internal production engine; global-64/local-compact scaling and chunking remove practical aggregate-u32 ceilings; no potentially large path is quadratic; provider build policy is local; mechanism/resource/stage metadata is cold; synthetic/adversarial host and Volta acceptance passes; profiler fixtures prove stable source-correlated identities; the documentation spine records actual behavior; and the deferred campaign handoff is published.

Completion does not assert a universal winner, real-data speedup, exhaustive profile, preprint claim, distributed integration, or Ampere implementation. Experimental and correct candidates may complete with `requires_measurement`. Optional PTX and GPU-assisted optimizer experiments use the workflow’s built-in terminal dispositions: `implemented` means the requested experimental implementation was retained; `evaluated_not_promoted` means it was evaluated and not retained/adopted; `failed` remains explicit. These map to the program terms `implemented_experimental`, `evaluated_not_retained`, and `evaluated_not_adopted` without extending the authority’s fixed disposition enum.

`CE-EXOP-COMPLETE` is reached only by CE-EXOP-290 after final audit and CE-AMP interlock verification. Every CE-AMP task requires both `CE-EXOP-COMPLETE` and `CE-AMP-PERMISSION == granted`, in addition to its existing prerequisites. CE-AMP permission remains `not_granted`; CE-EXOP completion does not authorize it.

## Stable invariants

The schema-v3 plan stores error-severity invariants for preservation of user work; additive graph changes; CE-GEO/CE-PTR inheritance; semantic/physical and two-cover separation; planner authority; portfolios rather than canonical winners; profiler-readiness without promotion; independent primitives and fused/unfused coexistence; explicit experimental visibility and unique kernel identities; cold profiling and no hot instrumentation; global-64/local-compact chunking; no aggregate-u32 ceiling or quadratic scale path; CellShard independence; real operation-core v2; program-v2 sole ownership and thin v1; provider-local builds and no global fast math; projection-primary identity, permanent holes, and direct gradients; persistent order and dynamic-support separation; no model/framework or attention API; pure sparse fallback and no edge pruning; Volta breadth without deep tuning; deferred data/profiling/preprint work; CE-AMP interlock; narrow scopes; first-class lanes; and documentation-spine authority.
