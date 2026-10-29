# Cellerator JBC source transition map

## Authority cursor

This map is the `CE-JBC-B02` source observation. It changes no runtime or
compiler behavior and does not reopen any historical program.

- Git cursor: Cellerator `f27bddad348f9366ade2b3a9772de584b8f1f530`;
  registered CellShard submodule
  `5f6a502b4355732c4ed3cc873a25b8aec66d8338`.
- Todo cursor: Cellerator revision `3606`, semantic fingerprint
  `36f3eb5f04f8fa9896650716782bc1e57890f86454151aa189cc654d7d6de6b7`.
- Observation time: `2026-09-01T05:49:26Z`.

The Git and Todo cursors are intentionally recorded separately. Project
Control observations are not globally atomic. The dirty Cellerator observation
at this cursor consists of the active JBC claim projections and the registered
CellShard worktree being advanced under its separately authorized bootstrap
lane; it is not normalized into either source identity.

## Classification vocabulary

- **Preserve**: the contract already has the correct owner and semantic role.
- **Adjacent extension**: add a versioned or sibling facility without changing
  the existing contract.
- **Generalize**: retain the implementation but widen its bounded problem or
  portfolio model under the same owner.
- **Compatibility-only**: keep an adapter or historical surface for existing
  consumers; do not give it new authority.
- **Migrate**: move callers to the already-defined owning boundary while the old
  route remains available.
- **Retire after gate**: deletion is allowed only after an explicit JBC gate
  proves replacement coverage and compatibility disposition.

## Live source map

| Capability | Exact live source | Classification | JBC transition and boundary |
| --- | --- | --- | --- |
| Support evidence | `include/Cellerator/geometry/support_atlas.hh`; `src/geometry/support_atlas.cc`; `src/geometry/strategy/support_affinity.cc`; `src/geometry/strategy/support_multiresolution.cc`; `src/geometry/strategy/exact_relation_rescan.cc` | **Preserve** | Keep sampled evidence, deterministic provenance, biological strata, stability, and exact-rescan summaries as proposal evidence. Approximation may propose geometry but never certifies logical-edge coverage or causal biology. |
| Portable semantic geometry and exact cover | `include/Cellerator/geometry/compiler/v2/solution.hh`; `include/Cellerator/geometry/relation_cover.hh`; `include/Cellerator/geometry/work_layout_v2/work_layout_v2.hh`; `include/Cellerator/geometry/persistence/semantic_geometry_image_v1.hh`; corresponding `src/geometry/compiler/` and `src/geometry/persistence/` sources | **Preserve** | CSG1 remains the portable, pointer-free semantic artifact. Exact logical-edge ownership and canonical recovery are the execution admission boundary; device tiles, ordinals, routes, and pointers remain outside portable identity. |
| Geometry strategy/compiler facade | `include/Cellerator/geometry/compiler/v2/strategy_registry.hh`; `include/Cellerator/geometry/compiler/v2/exact_evaluator.hh`; `src/geometry/compiler/v2/`; `src/geometry/compiler/compile_geometry.cc` | **Adjacent extension** | Add JBC strategies through the registry and exact evaluator. Do not replace CSG1 or reinterpret CP-BP blocks as universal biological modules. Cold builders must query caller-owned capacities and report bounded work and peak storage. |
| Optimizer portfolios | `include/Cellerator/geometry/optimizer/portfolio_v1.hh`; `include/Cellerator/geometry/optimizer/{greedy,multilevel,oracle,overlap}/`; `src/geometry/compiler/optimizer_portfolio.cc`; `src/geometry/optimizer/` | **Generalize** | Retain greedy, multilevel, exact-oracle, overlap, and device-assisted implementations as a measured portfolio. Generalization must accept operation/reuse/stratum evidence, bound candidate growth, preserve deterministic fallback, and separate proposal overlap from exact contribution ownership. |
| CP-BP v1 geometry and CPK1 | `include/Cellerator/geometry/CP_BP_V1_COMPATIBILITY.md`; `include/Cellerator/geometry/semantic_geometry.hh`; `include/Cellerator/geometry/strategy/cpbp_v1_compatibility.hh`; `src/geometry/compiler/cpbp_v1_compatibility.cc`; `src/geometry/persistence/execution_image_v2_cpk1.cc` | **Compatibility-only** | Preserve exact bytes, recovery maps, and the proven direct consumer. New compiler ownership does not accrue to CPK1, its width-16 grouping, or the `cellpack` compatibility namespace. Any successor is adjacent and versioned. |
| Operation core v2 | `include/Cellerator/compute/operation/operation_core_v2.hh`; `include/Cellerator/compute/operation/operation_core_v2/schema.hh`; `src/compute/operation/operation_core_v2/` | **Preserve** | This remains the operation-contract and preparation authority. JBC work binds explicit domains, orders, structure epochs, value generations, capacities, and launch state here instead of creating another runtime or prepared-plan lifetime. |
| Relation algebra v2 | `include/Cellerator/compute/operation/relation_algebra_v2/`; `src/compute/operation/operation_core_v2/relation_algebra.cc`; `src/compute/operation/relation_algebra_assembly.cc` | **Generalize** | Extend typed relation composition, segment operations, gates, gradients, and projection-aware values for compiler-produced programs. Keep reusable mathematics in Cellerator and model, loss, optimizer, and workflow policy outside it. |
| Projection value planes | `include/Cellerator/execution/projection_value_plane/`; `src/execution/projection_value_plane/`; `src/compute/architecture/providers/nvidia/sm70/value_pack/` | **Adjacent extension** | Preserve logical-primary values, independent mutable generations, explicit logical/physical maps, publication, and validation. Add value-pack alternatives as measured portfolios; value changes must not reconstruct immutable geometry. |
| Geometry acquisition | `include/Cellerator/execution/geometry_acquisition_v2/`; `src/execution/geometry_acquisition_v2/`; `src/execution/projection_activation_v2.cc` | **Adjacent extension** | Keep unified CSG1/CPE2/CPK1 acquisition and explicit projection activation. A JBC artifact enters through a versioned external-payload or assembly path, with allocation, transfer, validation, and fallback costs reported to the existing planner. |
| Prepared execution programs | `include/Cellerator/execution/program/program_v2.h`; `src/execution/program/program_v2.cc`; `include/Cellerator/execution/training_program_v2/`; `src/execution/training_program_v2/` | **Generalize** | Widen prepared stage graphs for compiler-produced relation programs while keeping current pointers, streams, values, and transient workspace launch-bound. Cold discovery, parsing, sorting, and topology search remain forbidden in steady-state execution. |
| Candidate catalog | `include/Cellerator/compute/operation/candidate_catalog_v3/`; `src/compute/operation/candidate_catalog/`; `src/compute/operation/ce_geo_catalog_assembly.cc`; `src/compute/operation/ce_exop_operation_portfolio.cc` | **Generalize** | Register JBC candidates with stable provider, device, projection, operation, numerical, resource, and stage identities. The existing end-to-end planner remains the sole promotion authority; experimental candidates require complete-cost measurement and may validly end in non-promotion. |
| Profiling and receipts | `include/Cellerator/profiling/`; `src/profiling/` | **Adjacent extension** | Add JBC mechanism, partition, resource, and static-marker records through the frozen cold-path surface. Profiling must expose build, allocation, packing, assembly, transfer, synchronization, launch, and canonicalization costs without authorizing promotion by itself. |
| Retained CP-Math v1 and legacy sparse implementations | `compat/cp_math_v1/`; `compat/legacy_sparse/`; `include/Cellerator/compute/math/operation_core/` | **Compatibility-only** | Keep reference paths, baselines, and forwarding includes. Do not add ownership to the old request, `PreparedExecution`, `DeviceMathContext`, structural planner, or a global sparse-format default. |
| Cellerator-to-CellShard matrix adapters | `include/Cellerator/interop/cellshard/access.cuh`; `include/Cellerator/interop/cellshard/matrix.cuh`; forwarding headers `include/Cellerator/interop/cellshard.cuh` and `include/Cellerator/interop/cellshard_access.cuh` | **Migrate** | Existing dense/CSR/SELL/Blocked-ELL/quantized matrix bindings remain usable while JBC consumers move to opaque, identity- and generation-validated execution-envelope delivery. These adapters do not grant CellShard geometry, projection, planner, or numerical ownership. |
| Legacy matrix-adapter retirement | The same `include/Cellerator/interop/cellshard/` surface and its `CellShard/access/adapter.cuh` dependency | **Retire after gate** | Retire only after a JBC integration gate proves every registered consumer uses the opaque artifact route, frozen-format readers remain covered, standalone and embedded builds pass, and an explicit compatibility disposition is recorded. Until then, no deletion or semantic wire change is authorized. |
| CellShard opaque artifact delivery | `components/CellShard/include/CellShard/interop/cellerator/execution_payload.hh`; `components/CellShard/include/CellShard/io/pack/execution_payload.cuh`; `components/CellShard/include/CellShard/io/pack/image_envelope.hh` | **Preserve** | CellShard owns persistence, validation, staging, and transport of opaque compiler/execution bytes. Cellerator owns the biological meaning, exact cover, projections, candidates, and execution; neither side recreates the other's semantics. |

## Data flow and lifetime result

The source-backed transition is therefore additive:

```text
support evidence + typed biological identity + caller policy
  -> bounded cold geometry strategies and optimizer portfolio
  -> independently certified exact semantic cover (CSG1)
  -> explicit physical projection and mutable value-plane alternatives
  -> candidate catalog and complete-cost planner
  -> reusable prepared program + launch bindings
  -> opaque CellShard persistence/transport where requested
```

Cold compilation may allocate only when the allocation and peak bytes are
declared; public execution views remain non-owning pointer-plus-count records.
At scale, proposal generation must use bounded top-L, streaming, sparse,
count/scan/fill, radix/sort, or caller-owned workspaces. Unbounded all-pairs and
unrestricted subgraph enumeration are rejected except in an explicitly small
exact oracle. Steady-state execution performs no discovery, catalog parsing,
hidden allocation, global sorting, structure hashing, or topology search.

## Non-transitions

This map does not modify CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics; does
not activate `CE-AMP`; does not reopen CE-GEO, CE-EXOP, CE-PTR, or CE-AMP; and
does not move mathematical semantics into CellShard or global
placement/storage policy into standalone Cellerator. A statistical cluster is
evidence for candidate work, never a causal biological module.
