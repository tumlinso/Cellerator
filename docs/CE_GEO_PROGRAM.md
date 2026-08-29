# CE-GEO Program: Biological Geometry Compilation, Volta MMA, and Relation Algebra

## Authority and purpose

CE-GEO is the execution program for the Cellerator geometry and Volta campaign. It is subordinate to [AGENTS.md](../AGENTS.md), [scope.md](../scope.md), [Architecture](architecture.qmd), [Current Implementation](current_implementation.qmd), and [Migration Roadmap](migration_roadmap.qmd). Those documents define repository architecture and ownership; this document defines the settled campaign decomposition, interfaces, gates, empirical questions, and completion semantics. It does not reopen CE-ARCH, CE-LIVE, CP-BP, CPK1, CPE2, or planner-v2 decisions.

The campaign is Volta-first and additive. Ordinary implementation begins only through the configured `coding-workflow` front door. First-class agents own serial run lanes. Local workers may return bounded candidate findings to one claimed parent but never own CE-GEO tasks, lanes, interfaces, decisions, rendezvous, or completion.

## Objective and system boundary

CE-GEO makes biologically natural computation fast without forcing higher layers to deform biological structure around ordinary tensor machinery. Cellerator is a biology and omics accelerator and execution substrate, not a general tensor framework, model framework, generic autograd engine, trainer, optimizer, sampler, loss owner, storage owner, universal sparse library, universal matrix-format doctrine, or mandate to use Tensor Cores.

Ownership remains explicit:

- Cellerator owns reusable operations whose biological identity, typed relations, sparse support, hierarchy, order, repeated structure, trajectory structure, or value lifetime creates a distinct mathematical or execution opportunity.
- GlassHelix owns model architecture, dynamical and developmental theory, latent-state semantics, losses, training behavior, model-specific exchange, and trajectory interpretation.
- PyTorch or libtorch may supply ordinary dense operations when biology does not materially change execution.
- CellShard owns storage, transport, sharding, upload, and opaque artifact delivery.
- BioPrep owns conventional preprocessing, normalization policy, QC, workflow policy, and dataset orchestration.
- Baseplane owns sequence-specialized representations and primitives; Cellerator owns public-contract sequence-to-state relation execution and planning.
- CelleraTorch is a thin adapter and owns no native allocator, planner, structure, math, or runtime.

## Settled execution architecture

The campaign implements this fixed pipeline:

```text
typed biological relation + mutable value planes
  + bounded work window + caller admissibility + workload profile
    -> portable semantic geometry acquisition
       (registered strategy, CSG1 load, or CP-BP/CPK1 adaptation)
    -> independently validated portable semantic geometry
       (work layout, exact logical-edge ownership, recovery maps,
        optional support atlas and rectangular evidence)
    -> architecture-provider target refinement
       (generic, nvidia_sm70, later permission-gated nvidia_sm86)
    -> exact device-specific physical contribution cover
       (MMA regions, exact residual regions, padding, schedules, value maps)
    -> CPE2 physical execution image
    -> assembled candidate catalog v2
    -> existing end-to-end planner
    -> prepared executable program
    -> sealed allocation-free steady-state execution
```

The existing planner is the sole final candidate-selection authority. Every MMA or fusion candidate begins empirical-required; a correct fast kernel is not promoted when complete execution is slower.

## Exact covers and identity

CSG1 owns the portable semantic cover: an exact disjoint partition of the logical relation into hardware-neutral components, each owning exact logical edge IDs. A semantic component may describe rectangular or hierarchical organization but never means a WMMA tile.

An architecture provider refines each semantic component into a physical contribution cover containing MMA and one or more residual regions. Every logical edge has exactly one physical contributing owner. Read replication is permitted; duplicate logical contribution is forbidden. Padding has no biological identity and is never a work item or edge. Residual is a role, not a synonym for CSR; row-owned CSR is only the first residual realization.

Shape never proves biological equivalence. Domain, order, geometry, partition, structure identity and epoch, and value generation remain explicit. Immutable topology and mutable values have independent lifetimes and invalidation.

## Portable CSG1 artifact

CSG1 is fixed-width, field-encoded, little-endian, checksummed, aligned, sectioned, pointer-free, relocatable, independently validated, and extensible through optional sections. Mandatory content is relation and structure identity; source and destination domain and portable order; work-window identity; exact semantic work layout and cover; logical edge ownership; and canonical recovery maps.

Optional content includes rectangular support, support affinity, prevalence, raw and normalized co-support, multiresolution communities, work signatures, biological strata, resampling stability, strategy provenance, sampling parameters, and validation summaries.

CSG1 contains no GPU model, compute capability, architecture class, provider kernel, WMMA/MMA instruction or tile identity, stream, runtime handle, device pointer, mutable generation, provider shared-memory layout, or fragment layout. Its identity hashes what the geometry is, not why it was selected.

## Device-specific CPE2 artifact

CPE2 remains the architecture-specific executable image and embeds the exact CSG1 bytes in its semantic-geometry section. Existing wire records remain unchanged: the header is 256 bytes, section entries are 64 bytes, and projection entries are 64 bytes. The existing projection `capability_section` references a typed device capability manifest. A compatible prebound projection v2 exposes capability bytes while v1 readers and views remain valid.

A projection may contain target groups, exact MMA and residual ownership, physical padding, masks, compact logical-edge maps, projection-local value maps, schedules, provider/projection ABI identity, numerical tuple, and required engine capability.

CPK1 bytes and v1 semantics never change. CP-BP remains a successful sparse-oriented strategy and execution path. Its width-16 blocks are not universal semantic groups. A compatibility adapter exposes existing semantic information without thawing, rebuilding, or reinterpreting CPK1. The row-masked path and historical V100 dense-fragment experiment remain intact; the latter stays a negative control outside the normal catalog unless future complete-cost evidence supports a separate production candidate.

## Work window and admissibility

Cellerator introduces no universal ML batch. A bounded work window binds exactly one axis: relation-row work, dense-column work, or grouped-operation-instance work. The caller chooses membership. Within the window Cellerator may reorder, regroup across caller batch boundaries, choose physical groups, group by support signature, combine compatible independent work, and preserve noncanonical output order.

Cold axis-qualified admissibility records may impose fixed position, fixed original-group membership, must-link, cannot-share-group, precedence, partition barrier, or bounded exchange window. The zero-constraint permissive route remains cheap. Constraint graphs are compiled during preparation and never traversed in kernels. Portable work layouts contain real work items only as exact invertible permutations; provider layouts may add invalid-sentinel padding.

## Hardware, capabilities, and providers

One canonical cold `runtime::device_descriptor_v1` is populated during session initialization and is the source for derived runtime, execution, and planner compatibility views. It records vendor, ordinal, compute capability and architecture class, multiprocessors, warp and thread limits, residency limits where available, register and shared-memory limits including opt-in, global memory, L2, hardware compatibility identity, and performance-class identity. Hardware identity is separate from runtime/kernel build identity. No device query occurs after sealing.

Architecture-neutral implemented capability records describe source-linked instruction capabilities: provider, compatible compute capability range, instruction family and collective scope, threads, exact M/N/K, A/B/C/D types and layouts, accumulation/output layouts, dense or structured sparsity, structured operand and group semantics, fragment opacity, convergence, and optional memory-interface contract. They advertise implemented code, not hardware documentation.

Memory-interface records separately describe base alignment, stride multiples, address spaces, and interface flags. WMMA memory loads are not conflated with register-level `mma.sync` capability.

Providers are explicitly source-linked (`generic`, `nvidia_sm70`, optional `nvidia_sm86`, future providers). They have immutable identity, active-device predicate, implemented capabilities, catalog fragment, projection requirement query, host realization and validation, and device activation. Candidate-specific preparation remains on candidates. There are no global constructors, `dlopen`, stable external binary plugin ABI, hot-path provider lookup, or WMMA fragments in provider contracts.

Build inclusion and tuning policy are distinct through `CMAKE_CUDA_ARCHITECTURES`, `CELLERATOR_CUDA_PROVIDERS`, `CELLERATOR_CUDA_TUNING_PROFILE`, and `CELLERATOR_CUDA_PRIMARY_PROVIDER`, with current modes retained as compatibility aliases. Provider helpers do not silently impose fast math, cache policy, global register caps, or launch bounds.

## Candidate catalog and executable program

Candidate catalog v2 is one cold assembled authority combining the current five candidates through a compatibility fragment, active provider fragments, and explicitly linked biological-operation fragments. Its POD descriptor carries candidate and provider IDs; operation, projection, backend, view, ABI, schema and variant; required capability; device and numerical predicates; width range; capability/planner flags; preparation requirements; caller-owned prepared-state size and alignment; variable resource query; erased typed preparation adapter; mechanism statistics; and empirical-measurement requirement.

The compact existing `operation_candidate` may remain. Provider-erased activated projection references carry stable provider, view, ABI, schema, and capability identities. Each catalog entry owns its preparation adapter. The v2 path removes pointer-equality closure around the five-entry built-in array and the central activated-projection switch. The v1 executable-program API is a compatibility wrapper over the single v2 engine.

## Geometry strategy and support evidence

The only semantic optimizer extension point is a source-linked data contract:

```text
geometry problem + search policy + caller workspace + caller output buffers
    -> geometry solution data
```

The problem carries typed structures, primary relation and axes, edge count and input view, work window, admissibility, workload profile, and optional portable support evidence. Search tiers are instant, bounded, offline, and external. Bypass is an acquisition route, not a tier. The public compile path validates the problem, resolves and queries a strategy, invokes it, independently validates work layout and exact semantic cover, derives identity, and returns validated data. Foundation strategies are identity/full-relation and CP-BP v1 compatibility; rectangular affinity follows later. Future external optimizers exchange versioned data and pass the same validators, without a frozen compiler-dependent binary ABI.

The portable support atlas avoids all-pairs feature matrices. It records prevalence, destination degree, sampled and weighted co-support, normalized association, sparse top-L affinity, multiresolution communities, work signatures, strata, resampling stability, exact rescan, and deterministic provenance. High-degree rows use bounded pair sampling. Target complexity is approximately `O(E + S log L)`. Sampling proposes; one exact relation pass decides occupancy, ownership, and cost. Support is computational evidence, not a causal biological claim.

## First sm_70 refinement and physical projection

The understandable replaceable first solver builds disjoint source groups of at most 16, derives destination signatures, groups compatible destinations up to 16, performs an exact `O(E)` rectangle census, evaluates complete marginal cost, assigns profitable edges to MMA, assigns every remainder to residual, performs bounded local refinement, and emits pure-sparse, conservative-hybrid, and aggressive-hybrid complete covers. Disjoint source groups give one source order, contiguous panels, no source-state replication, simpler edge identity and gradients, and cheaper reuse. Overlap remains empirical.

Local refinement may move or swap source/destination members, split or merge groups, add/remove rectangles, and exchange rows across caller groups when admissible. No fixed density threshold substitutes for complete marginal cost.

The `architecture_specific` MMA hybrid projection is distinct from the historical `dense_fragment`. It carries source/destination maps, sentinel padding, destination-to-tile offsets, tile descriptors, 256-bit occupancy masks, compact occupied slots, width-tagged logical edge IDs, residual descriptors, schedules, projection-local value maps, exact cover validation, and mutable value-pack state outside immutable CPE2 structure. The first residual is row-owned CSR in the same physical order. Value changes repack without rebuilding structure; widths may select different prepared candidates; precision may select another projection variant.

## Volta execution

The initial implemented capability is sm_70 WMMA, FP16 relation and dense input, FP32 accumulation and output, `16x16x16`. WMMA fragments remain private to sm_70 translation units.

The primary N=64 kernel is output-owned: one CTA owns one 16-row destination group by one 64-column panel. Four warps own column ranges 0-15, 16-31, 32-47, and 48-63, traverse the same source groups, keep FP32 accumulators resident, and store once after all K contributions. No per-tile output atomics are permitted.

N=32 evaluates two warps/one destination group against four warps/two compatible groups. N=16 evaluates one warp/group with several groups per CTA. N>64 uses disjoint CTA column panels. N<16 retains sparse paths unless evidence wins; N=1 remains specialized sparse.

The hybrid sequence is optional value pack, output-owned MMA contribution, row-owned exact sparse residual, and one final epilogue, sequential on one stream without output atomics. Alpha and beta apply exactly once. Preallocated MMA and residual value buffers support asynchronous generation-aware packing and CUDA Graph capture after stable addressing; no value update triggers semantic search or projection reconstruction.

## Biology-centered relation algebra

CE-GEO delivers reusable typed operations, not only SpMM:

- `relation_apply`: typed relation times dense state; the main MMA target.
- `relation_apply_transpose`: reverse-domain operation and gradient path, with independent target cover where useful.
- `contract_on_support`: logical-edge-only `dot(Q_i, K_j)`, with sm_70 tiled path, residual, and stable logical-edge output; it is not a public attention abstraction.
- `segment_reduce`: sum and maximum with explicit axis/segment identity.
- `segment_normalize`: log-sum-exp and softmax, empty/singleton rules, FP32 reduction policy, and required backward primitives.
- `edge_map_or_gate`: composable projection-aware edge-value transforms without model interpretation.
- `relation_bundle_apply`: typed accumulation of relations sharing a destination; sequential constituents are acceptable until fusion evidence wins.

Incidence pool/broadcast compose relation apply and transpose. Sparse biological exchange composes contraction, edge transform, normalization, and relation apply as separate callable operations. Cellerator does not publish an attention abstraction or generic autograd engine.

Public examples cover sparse cell-state embedding, regulatory propagation, transition/transport, hierarchy incidence pool/broadcast, multimodal typed relations, and perturbation-delta propagation without privileged sibling-repository internals.

## Numerical policy

The initial sm_70 tuple is FP16 relation storage, FP16 dense input, Tensor Core FP16 multiply, FP32 accumulation, and FP32 output. There is no saturation, silent quantization, silent FP32-to-FP16 conversion, or global fast-math assumption.

Two referees are mandatory: one after exact conversion to candidate operand precision and one full logical FP32/FP64 scientific referee. Evidence reports maximum absolute error, relative L2 and Frobenius error, mixed absolute/relative pass, and error versus row degree and accumulation depth. Bitwise equality across architectures, builds, and sparse/MMA candidates is not promised. `candidate_deterministic` means the same prepared candidate, device/build, and fixed schedule unless the existing contract is stronger.

## Complete-cost evidence

Planner v2 remains in place. Costs map as follows:

- support extraction, semantic search, and CSG1 validation -> `semantic_packing_ns`;
- provider refinement, projection build/upload, and CPE2 prebind -> `projection_construction_ns`;
- candidate preparation -> `backend_prepare_ns`;
- mutable relation packing -> `static_value_pack_ns`;
- per-use dense permutation -> `dynamic_input_pack_ns`;
- MMA and residual -> `kernel_ns`;
- activation/normalization -> `epilogue_ns`;
- canonicalization/order conversion -> `order_transform_ns`.

Persistent projection upload is not per-use H2D. Diagnostics record structure, semantic-geometry, projection, value-generation, dense-layout, work-window, prepared-program, and graph-replay reuse without immediately expanding planner reuse dimensions.

Permanent evidence spans `N={1,4,8,16,32,64,128,256,512}`, `D={16,32,64,128,256,512}`, reuse `{1,4,16,64,256,1000+}`, and work windows `{1,4,16,64}` original groups where practical. It records every cold/warm phase, complete latency and throughput, useful/executed work, occupancy, residual fraction/fragmentation, artifact sizes, break-even reuse, Tensor Core and memory behavior, registers, stalls, launches, atomics, and numerical error.

Promotion requires independent structural and numerical validation, a defined complete-cost winning regime against the best local sparse baseline, reported preparation/break-even, no unrelated sparse regression, stable repeated selection, profiler confirmation, included mapping/artifact costs, and automatic fallback. Negative and non-promotion results are durable valid outcomes.

Real-data evidence includes PBMC3K as negative control, available developmental embryo data, at least one heart-relevant relation or dataset, synthetic controlled structures, and additional perturbation/multiome/regulatory/trajectory structures where checked manifests exist. Parsing and storage remain outside core. Required ablations cover isolated and joint reorder/grouping, batch constraints, complete-cost versus density, support priors, refinement, persistent order, repeated canonicalization, value mutability, sparse/dense/partial covers, residual choices, and shared versus operation-specific covers.

All comparable timing and profiler work leases `accelerator:any` and holds the existing `cuda-benchmark-mutex`. Evidence captures `CUDA_VISIBLE_DEVICES`, source, device/topology, toolchain, CUDA/driver/libraries, build flags, warmup/repeats, cold/warm costs, contamination, and spread. Correctness leases accelerators only while executing. No fixed GPU index or UUID appears in the plan.

## STL coexistence

New CE-GEO production interfaces and implementations use POD views, pointer-plus-count spans, caller/session-owned buffers, fixed-capacity records, explicit requirements queries, compact arrays, and explicit lifetimes. They introduce no STL ownership or allocator-heavy graphs. Tests and offline benchmarks may use bounded host STL when it does not leak into production ABI.

CE-GEO neither duplicates nor certifies the separate CE-PTR STL-removal campaign and does not depend on its completion. Agents revalidate interfaces and workflow deltas when CE-PTR changes shared neighbors. Ownership remains file-level or narrow-module; root manifests and documentation spine are integration-owned.

## Task graph and integration

The persistent schema-v3 graph contains one Volta run with coordinator, hardware, catalog, semantic, persistence, support, projection, sm_70, biology, advanced-MMA, validation, benchmark-infrastructure, microbenchmark, biological-benchmark, and integration lanes. Lanes are serial internally and parallel across lanes. The single foundational rendezvous joins hardware, catalog, semantic, and persistence lanes at the foundation fan-in. Other synchronization uses interface/checkpoint dependencies and five real fan-in barriers.

The integration sequence is:

1. `CE-GEO-120` integrates frozen provider, catalog, semantic/CSG1, and CPE2/acquisition foundations.
2. `CE-GEO-121` integrates support evidence, target refinement, physical projection, N=64 execution, residual, planner, and preparation.
3. `CE-GEO-122` integrates relation algebra, transpose, contraction, segment operations, gradients, bundles, examples, and optional evaluated fusion.
4. `CE-GEO-123` owns final shared build/export/catalog/program wiring.
5. Independent validation, microarchitecture evidence, biological/preprint evidence, and integration feed documentation and final acceptance.

Interfaces freeze only after their required gates pass. Consumers depend on frozen checkpoints/interfaces, not task prose. Shared CMake, exports, central catalog assembly, and documentation are deferred to integration/designated documentation tasks.

## Gates and allowed outcomes

Gates use current focused-binary conventions and permanent scripts created by their owning tasks. Compatibility covers CPK1, CPE2, session, program/catalog, sparse candidates, transpose, and the historical experimental WMMA path; CPK1 golden bytes and CPE2 record sizes cannot change. The build matrix covers normal native, generic provider, sm_70 provider, supported multi-provider linking, and relevant explicit compatibility mode.

Structural gates cover semantic and physical exact covers, invertible maps, padding, duplicate policy, residual recovery, identities and generations, and artifact corruption. Runtime gates cover post-seal allocation/query/parsing/provider search/synchronization, streams, graph addresses, repeated generations, and concurrent plans. Numerical gates cover both referees, gradients, empty/singleton/high-degree/tail cases, alpha/beta, NaN/Inf, and tolerances. Static audits are scoped only to CE-GEO production paths. Compute Sanitizer covers new device views, rebind, packing, MMA, residual, segments, and gradients.

Tasks with empirical outcomes explicitly allow `evaluated_not_promoted`; failures remain explicit and cannot be disguised as promotion. Every empirical branch receives a terminal disposition before closure.

## Completion semantics

CE-GEO is complete when all required Volta-capable foundation, implementation, operation, validation, benchmark, documentation, and integration tasks are terminal in allowed dispositions; mandatory interfaces are frozen; mandatory checkpoints are reached; correctness/compatibility gates pass; every benchmark-driven candidate has a recorded disposition including negative results; `CE-GEO-VOLTA-COMPLETE` is reached; the optional Ampere graph is loaded and proven dormant; and `CE-GEO-COMPLETE` is reached.

CE-GEO completion does not require CE-AMP execution.

## Permission-gated CE-AMP extension

`CE-AMP-RUN-V1` is loaded at bootstrap as a subordinate, separately runnable extension. Its mutable decision is `CE-AMP-PERMISSION`, allowed values `not_granted` and `granted`, initially `not_granted`. Only explicit human authorization may change it. Hardware discovery, branches, benchmark results, inferred intent, gates, or agents may not do so.

Every CE-AMP lane head requires both checkpoint `CE-GEO-COMPLETE` and decision `CE-AMP-PERMISSION == granted`. Serial lane order carries that interlock forward, with explicit cross-lane dependencies where needed. CE-AMP consumes frozen CSG1/provider/catalog/projection/relation-algebra contracts, preserves semantic identity, and is outside the CE-GEO final-evidence barrier.

After permission, CE-AMP revalidates live A5000/toolchain state and same CSG1 fixtures; adds only source-linked sm_86 `m16n8k16` FP16 and BF16 to FP32 capabilities; creates an independent Ampere physical realization; evaluates provider-private `cp.async`, `ldmatrix`, and `mma.sync`; adds BF16 policy, transpose/contraction parity, correctness/sanitizer and local baselines; and performs a cross-architecture experiment requiring identical CSG1 identity, independent V100/A5000 projections, exact logical reconstruction, and declared numerical agreement. It compares physical relowering of the Volta target cover with fresh Ampere refinement. It does not prune biological edges for 2:4. TF32 and structured sparsity remain optional benchmark-gated candidates.

## Stable campaign invariants

The schema-v3 plan stores the following error-severity invariants as executable coordination policy: preserve user work; additive graph; biological identity; structure/value separation; semantic/physical separation; exact two-level covers; padding non-identity; portable CSG1; physical and compatible CPE2; preserved CPK1 and sibling CP-BP; planner authority and complete cost; pure sparse fallback; measurement-gated promotion and negative evidence; bounded work window and default freedom; no model scheduler; persistent execution order; cold provider truth; no WMMA ABI; output ownership; sealed hot path; explicit numerics; no edge pruning or global fast math; no new STL and STL-campaign independence; one external optimizer seam; generic interop; GlassHelix boundary; Volta-first ordering; human Ampere permission; narrow ownership; and documentation-spine authority.

## Bootstrap and continuation

The bootstrap creates this document and `ce-geo-plan.json`, validates and semantically diffs the additive plan, applies it transactionally, verifies the runs/lanes/interfaces/decisions/barriers/checkpoints/locks/resources and representative readiness, and stops. It does not claim `CE-GEO-01`, begin source implementation, or begin CE-AMP. A later implementation agent starts through `coding-workflow next_task` and follows the bounded run charter and task packet.
