# CE-LIVE program

Status: complete through CE-LIVE-45 final audit
Authority: Cellerator source and architecture spine, with todo-orchestrator schema v2 as execution authority
Program root: `CE-LIVE-00`
Bootstrap owner: `CE-LIVE-01`

## 1. Current baseline and CE-ARCH boundary

The bootstrap source baseline is clean commit
`0062596dce9fee4d7c39f34f682ad848a0ca6c68` at todo revision 1422 on
2026-08-26. Project-control reported no active Cellerator claims, no ready work,
and CE-ARCH terminal. The native configured build is
`build-dissolution-smoke`, with `CELLERATOR_ENABLE_TORCH_MODELS=OFF` and
`CMAKE_CUDA_ARCHITECTURES=70`. Four idle Tesla V100-SXM2-16GB devices were
visible. The local-worker supervisor and CUDA background-evidence database were
unavailable; both are optional and neither blocks foreground CE-LIVE work.

CE-ARCH is complete historical evidence, not pickup guidance. Its completed
source establishes biological identities and lifetimes, CPE2, operation-core
candidates, the bounded end-to-end and connected-operation planners, the sole
execution session, native forward/training slices, sequence integration, and
real-regime V100 evidence. CE-LIVE activates those pieces into one quantitative
planner-backed program. The stale “Current Implementation Status” text in
`AGENTS.md` is deliberately left for CE-LIVE-15.

The authoritative plan is `ce-live-plan.json`, schema version 2. Its bootstrap
digest is historical; the tracked plan and generated projections are reconciled
to the final ledger by CE-LIVE-45. It contains 32 tasks, 14 error-severity
invariants, 9 interfaces, 29 checkpoints, and 273 gates.

## 2. Resolved decisions

1. **CE-LIVE-D1 — forward relation direction.** CP-BP matrix-like forward
   relations are feature or gene source to row, module, observation, or cell
   destination. The compact decoder maps each value to a feature and row.
   Transpose and backward reuse the same logical edge identity through an
   explicit transpose projection and exact value-position map.
2. **CE-LIVE-D2 — quantitative PBMC3K fixture.**
   `data/test/reference/pbmc3k_raw.h5ad`, source
   `pbmc3k-raw-local`, is eligible only as a checksum-pinned computational
   correctness and performance fixture after validating its exact matrix/value
   properties. Missing donor, sample, chemistry, species, normalization, and
   other provenance remains visible.
3. **CE-LIVE-D3 — executable program ownership.** The executable program is a
   host orchestration layer over the biological ABI, CPE2 typed activation,
   operation core, planner, sole execution session, prepared operations, and
   launch bindings. It is not a second runtime, framework, storage layer, model
   layer, or planner.
4. **CE-LIVE-D4 — planner scope.** CE-LIVE v1 uses the existing bounded
   single-operation and linear connected-operation planners. It does not add a
   general DAG planner.
5. **CE-LIVE-D5 — conventional fallback.** The live path includes a strong,
   session-integrated cuSPARSE CSR SpMV/SpMM candidate. Projection,
   descriptors, preprocessing, packing, order work, synchronization, and reuse
   costs remain explicit. Native paths receive no preference without evidence.
6. **CE-LIVE-D6 — value readiness.** Numerical generation and runtime readiness
   are separate. Runtime readiness supports same-stream chaining and explicit
   cross-stream event ordering, publishes no generation on enqueue failure,
   performs no device-wide synchronization, and uses no pointer identity.
7. **CE-LIVE-D7 — Tensor Core posture.** Tensor Core work is one optional
   sm_70 physical candidate using explicit FP16 inputs and FP32 accumulation.
   Conversion, densification, fragments, padding, tails, output conversion, and
   amortization are priced. CE-LIVE-32 succeeds on either measured promotion or
   measured non-promotion.
8. **CE-LIVE-D8 — CelleraTorch ordering.** Broad adapter work begins only after
   `CELLERATOR_CELLERATORCH_ENTRY_V1`. CelleraTorch owns Torch build glue,
   non-owning views, current-stream adaptation, custom ops, autograd, and module
   wrappers; Cellerator retains parameters, structures, planning, resources,
   generations, and reusable math.
9. **CE-LIVE-D9 — CellShard boundary.** The first live replay uses existing
   opaque `CPEXEC01` compatibility and does not wait for CellShard CS-FOUND.
   `CPEXEC02` source/residency integration remains a later sibling-repository
   vertical slice.
10. **CE-LIVE-D10 — Baseplane boundary.** CE-LIVE imports frozen
    predicate/event contracts and preserves the Cellerator-side proof. It does
    not modify Baseplane or repair its independent ledger.

These decisions are closed for bootstrap and implementation planning.

## 3. CE-LIVE invariants

| ID | Required rule |
|---|---|
| CE-LIVE-IDENTITY | Shape never establishes compatibility; domain, exact order, geometry, partition, structure identity/epoch, and value generation stay explicit. |
| CE-LIVE-ORIENTATION | Forward CP-BP is feature/gene source to row/cell destination; transpose/backward uses the same logical edges and explicit orientation. |
| CE-LIVE-LIFETIME | Pointer, scalar, stream, workspace, allocation-address, or value-generation changes do not rebuild immutable topology. |
| CE-LIVE-NATIVE | Repeated native execution never silently reconstructs CSR, BELL, COO, dense tensors, or canonical order. |
| CE-LIVE-RUNTIME | Steady state has no hidden allocation, device selection, descriptor creation, structural hashing, host round trip, or device-wide sync. |
| CE-LIVE-PLANNER | Selection uses complete cost, reuse, independent correctness, and a strong conventional fallback. |
| CE-LIVE-FIXTURE | PBMC3K is computational evidence only; no scientific inference from missing provenance. |
| CE-LIVE-STORAGE | Cellerator does not become an H5AD parser, archive/source/network service, or CellShard replacement. |
| CE-LIVE-CPE2 | CPE2 remains pointer-free; activation is non-owning; CellShard transports opaque bytes without kernel semantics. |
| CE-LIVE-BASEPLANE | Baseplane stays separate and owns its general sequence engine. |
| CE-LIVE-TORCH | CelleraTorch is never canonical allocator, parameter owner, planner, runtime, or reusable-math owner. |
| CE-LIVE-TENSOR | Tensor Core state is architecture-specific candidate state, never biological identity. |
| CE-LIVE-EVIDENCE | Correctness and adversarial validation precede timing; microkernel timing cannot promote. |
| CE-LIVE-OWNERSHIP | Every mutation has a matching claim/scope; only fan-ins edit integration files. |

All are ledger invariants with severity `error`.

## 4. Authoritative bounded source map

The bootstrap used separate bounded contexts; they were not merged into a
repository dump. ctxpp routing was available in degraded lexical mode because
the configured build does not export a compilation database. Ambiguous
semantic-slice identifiers were resolved by bounded canonical inspection, as
required by the ctxpp fallback contract.

### Biological identity, lifetime, relation orientation, and output order

- `include/Cellerator/execution/identity.hh`: strong handles and
  `value_generation`.
- `include/Cellerator/execution/lifetimes.hh`: `relation_structure`,
  structure epoch, value plane/binding, and source/destination validation.
- `include/Cellerator/execution/execution_order.hh`: output-axis contracts,
  order transitions, and logical value maps.
- `include/Cellerator/execution/launch_bindings.hh`: launch-time structure,
  value, dense operand, output effect, and order checks.
- `include/Cellerator/execution/operands.hh`: dense and relation operand views.
- `components/CellPack/include/CellPack/semantic_geometry.hh` and
  `components/CellPack/src/semantic_geometry.cc`: CP-BP v1 compatibility
  relation construction.

Archaeology found the one production mismatch:
`build_cp_bp_v1_compatibility_adapter_host` currently assigns
`source_axis = row_axis` and `destination_axis = feature_axis`. In contrast,
row-masked, feature-major, and CSR candidate run paths require source=feature
and destination=row, and transpose tests preserve that logical relation.
Therefore CE-LIVE-11 exclusively adds
`components/CellPack/include/CellPack/semantic_geometry.hh`,
`components/CellPack/src/semantic_geometry.cc`, and
`components/CellPack/tests/semantic_geometry_adapter_test.cc` to its initial
scope. No other production CP-BP/CPK1/FMP1/CSR/CTP1 adapter constructs an
affected relation.

### CPK1, CPE2, projection payloads, and typed physical views

- `components/CellPack/include/CellPack/persistent_packing_payload.hh`:
  pointer-free CPK1 view.
- `components/CellPack/include/CellPack/persistence/execution_image_v2.hh` and
  `components/CellPack/src/persistence/execution_image_v2.cc`: CPE2 header,
  section/projection directories, axis records, validation, and device prebind.
- `include/Cellerator/compute/math/native_tile_view.hh` and
  `src/compute/math/native_tile_view.cc`: CPK1 non-owning tile view and exact
  value coordinate decoder.
- `include/Cellerator/compute/math/physical_csr.hh`,
  `physical_feature_major.hh`, and `physical_transpose.hh`: CSR, FMP1, and
  CTP1 typed physical payloads/views.
- `src/compute/math/physical_csr.cc`,
  `physical_feature_major.cc`, and `physical_transpose.cc`: current
  construction and validation bridges.

### Operation core, candidates, preparation, and planning

- `include/Cellerator/compute/math/operation_core/operation_core.hh`:
  `operation_problem`, `prepared_operation`, and fixed
  `operation_candidate` contract.
- `src/compute/math/operation_core/row_masked_n1_candidate.cu`:
  feature→row CPK1 N=1.
- `src/compute/math/operation_core/csr_fallback_candidate.cu`:
  feature→row custom CSR N=1, including the current per-run device-selection
  cleanup target.
- `src/compute/math/operation_core/feature_major_small_n_candidate.cu`:
  feature→row warp/CTA paths.
- `src/compute/math/operation_core/transpose_backward_candidate.cu`:
  transpose projection and backward value-position map.
- `include/Cellerator/planner/end_to_end_planner.hh` and
  `src/planner/end_to_end_planner.cc`: bounded total-cost selection and
  connected-operation planning.
- `include/Cellerator/planner/candidate_measurement.hh` and
  `src/planner/candidate_measurement.cu`: empirical candidate evidence.

### Runtime session, memory, streams, graph stability, and values

- `include/Cellerator/runtime/session.cuh` and `src/runtime/runtime.cu`:
  sole execution session, persistent allocation/cache records, stream slots,
  launch bindings, accounting, and library handles.
- `include/Cellerator/execution/launch_bindings.hh`: transient stream,
  pointer, scalar, workspace, structure, and generation bindings.
- `tests/runtime/execution_session_test.cu`: resource ownership and runtime
  acceptance.
- `tests/math_core/value_generation_reuse_test.cu`: immutable prepared state
  across relocated value buffers and generations.

The static scan shows legacy and test uses of `cudaSetDevice`,
`cudaMalloc/cudaFree`, library creation, and synchronization. CE-LIVE gates
classify preparation/test-only use separately from repeated live-run use.

### Quantitative fixture, referee, and evidence

- `data/test/reference/pbmc3k_raw.h5ad`: local source fixture; never committed
  again or interpreted scientifically.
- `data/manifests/architecture_evidence`: current source IDs, checksums,
  structure-only traces, forbidden-use fields, and evidence manifests.
- `include/Cellerator/compute/math/referee.hh` and
  `src/compute/math/referee.cc`: independent numerical comparison.
- `bench/architecture_evidence/real_regime_bench.cu`: CE-ARCH-92 real-regime
  options and complete-candidate campaign machinery.
- `bench/architecture_evidence/ce_arch_92_v100_summary.json`: historical
  V100 calibration evidence, not a new performance claim.

### CelleraTorch parameters and tensor boundaries

- `include/Cellerator/parameters.hh`: canonical
  `parameter_descriptor`/`parameter_view` boundary.
- `components/CelleraTorch/include/CelleraTorch/bindings.hh`: existing
  non-owning/export boundary.
- `components/CelleraTorch/src/bindings.cc`: copied compatibility exporter.
- `components/CelleraTorch/CMakeLists.txt`: Torch-only target wiring.
- `components/CelleraTorch/AGENTS.md`: adapter-only ownership.

### Configured target map for later gates

The current Torch-off sm_70 build exposes these relevant targets:

- ABI/runtime: `celleratorBiologicalAbiHostTest`,
  `celleratorBiologicalAbiCudaCompileTest`,
  `celleratorIdentityRegistryTest`, `celleratorExecutionSessionTest`,
  `celleratorOperationCoreTest`.
- CellPack/CPE2: `cellPackSemanticGeometryAdapterTest`,
  `cellPackExecutionImageV2Test`, `cellPackExecutionImageV2DeviceTest`,
  `celleratorOpaqueExecutionArtifactTest`.
- Candidates/training: `celleratorRowMaskedN1CandidateTest`,
  `celleratorCsrFallbackCandidateTest`,
  `celleratorFeatureMajorSmallNCandidateTest`,
  `celleratorTransposeBackwardCandidateTest`,
  `celleratorNativeTrainingSliceTest`,
  `celleratorValueGenerationReuseTest`.
- Planner/evidence: `celleratorPlannerV1Test`,
  `celleratorConnectedOperationPlannerTest`,
  `celleratorObjectiveV2CalibrationTest`,
  `celleratorCandidateMeasurementTest`,
  `celleratorCeArch76CandidateBench`,
  `celleratorCeArch92RealRegimeBench`.
- Historical EXCLUDE_FROM_ALL evidence:
  `cpMathOperationContractTest`, `cpMathRefereeFoundationTest`,
  `cpMathNativeTileAdapterTest`, `cpMathBellLoweringTest`,
  `cpMathExecutionCsrTest`.
- CelleraTorch targets exist only when explicitly enabled:
  `celleraTorchBindingsCompileTest`,
  `celleraTorchDenseReduceCompileTest`,
  `celleraTorchQuantizePrimitiveTest`, and
  `celleraTorchModelCustomOpsTest`.

Leaf work uses bounded standalone or existing-target gates. CE-LIVE-19,
CE-LIVE-29, CE-LIVE-37, CE-LIVE-43, and CE-LIVE-45 own official root fan-in
wiring.

## 5. Task graph

| Task | Dependency set | Ownership/result |
|---|---|---|
| CE-LIVE-00 | — | Serial epic root |
| CE-LIVE-01 | — | Project-exclusive bootstrap |
| CE-LIVE-10,11,12 | CE-LIVE-01 | Floating validation plus two fork entries |
| CE-LIVE-13 | CE-LIVE-11 | Fork A candidate/projection inventory |
| CE-LIVE-16 | CE-LIVE-13 | Fork A Tensor Core contract |
| CE-LIVE-14 | CE-LIVE-12 | Fork B value-readiness contract |
| CE-LIVE-15 | CE-LIVE-14 | Fork B authority/build guidance |
| CE-LIVE-19 | 10,11,12,13,14,15,16 | Foundation integration fan-in |
| CE-LIVE-20 | 11,13,19 | CPE2 typed activation |
| CE-LIVE-21 | 11,13,19 | Built-in catalog |
| CE-LIVE-22 | 10,11,20 | cuSPARSE CSR fallback |
| CE-LIVE-23 | 20,21 | Preparation factory |
| CE-LIVE-24 | 11,12,19 | Quantitative native adapter |
| CE-LIVE-25 | 11,14,19 | Training/readiness integration |
| CE-LIVE-26 | 12,13,19 | Planner inputs |
| CE-LIVE-29 | 20,21,22,23,24,25,26 | Executable-core fan-in |
| CE-LIVE-30 | 22,23,24,25,26,29 | Deliberate serial program bridge |
| CE-LIVE-31 | 30 | Quantitative forward |
| CE-LIVE-32 | 16,20,21,24,30 | Optional Tensor Core decision |
| CE-LIVE-33 | 25,30 | Native training executable |
| CE-LIVE-34 | 30,31 | CPE2/CPEXEC01 replay |
| CE-LIVE-35 | 30,33 | Streams/graphs/hot-path validation |
| CE-LIVE-36 | 10,31,33,34,35 | Serial complete-cost evidence |
| CE-LIVE-37 | 36 | Project-exclusive live/CelleraTorch entry audit |
| CE-LIVE-40,41,42 | 37 | Three parallel-safe CelleraTorch adapters |
| CE-LIVE-43 | 40,41,42 | Adapter build/package fan-in |
| CE-LIVE-44 | 31,33,43 | Serial Torch quantitative validation |
| CE-LIVE-45 | 32,44 | Project-exclusive final audit |

## 6. Parallel execution map

### Wave A

After CE-LIVE-01, Wave A is two long-lived serialized forks plus one floating
validation task. The forks remain separate until the CE-LIVE-19 rendezvous.

```text
Fork A:
CE-LIVE-11  Freeze logical relation orientation and edge identity
    -> CE-LIVE-13  Built-in candidate and projection activation inventory
    -> CE-LIVE-16  Bounded Tensor Core feasibility and candidate contract

Fork B:
CE-LIVE-12  Quantitative biological fixture, provenance, and referee contract
    -> CE-LIVE-14  Value readiness and asynchronous generation contract
    -> CE-LIVE-15  Refresh architecture authority and quarantine legacy default builds

Floating:
CE-LIVE-10  Current-head build, correctness, sanitizer, and evidence rebaseline

Rendezvous:
10 + 11 + 12 + 13 + 14 + 15 + 16 -> CE-LIVE-19
```

Fork A exclusively owns relation-orientation implementation/tests/docs,
candidate inventory documentation and benchmark artifacts, and Tensor Core
contract/evidence-design paths. It publishes
`CELLERATOR_LOGICAL_EDGE_ORIENTATION_V1_READY`, then
`CE_LIVE_CANDIDATE_INVENTORY_READY`, then
`CELLERATOR_TENSOR_CORE_CONTRACT_READY`. Fork A may assume finalized forward and
transpose semantics only after CE-LIVE-11 is terminal, and the Tensor Core
contract may assume the actual candidate inventory only after CE-LIVE-13 is
terminal.

Fork B exclusively owns fixture manifests/tools/tests/docs, runtime value
readiness implementation/tests/docs, and architecture/build guidance. It
publishes `CELLERATOR_QUANTITATIVE_FIXTURE_V1_READY`, then
`CELLERATOR_VALUE_READINESS_V1_READY`, then `CE_LIVE_AUTHORITY_REFRESHED`.
Fork B may assume the quantitative fixture contract only after CE-LIVE-12 is
terminal, and authority/build guidance may assume the readiness contract only
after CE-LIVE-14 is terminal.

CE-LIVE-10 remains unassigned floating validation work. Neither fork owns
CE-LIVE-19, `docs/CE_LIVE_PROGRAM.md`, root integration CMake seams, or shared
live integration tests. CE-LIVE-19 is integration-exclusive and receives one
explicit owner only after all seven Wave A tasks are terminal.

### Foundation fan-in

```text
CE-LIVE-10 ─────┐
CE-LIVE-11 ─────┤
CE-LIVE-12 ─────┤
CE-LIVE-13 ─────┤
CE-LIVE-14 ─────┤
CE-LIVE-15 ─────┤
CE-LIVE-16 ─────┘
                 → CE-LIVE-19
```

CE-LIVE-19 owns the single Wave A fan-in. Its native integration seam links the
runtime-only value-readiness implementation into `Cellerator::runtime`, exposes
the focused `celleratorValueReadinessTest` target, and audits the frozen
orientation, quantitative fixture, candidate inventory, authority/build, and
Tensor Core design-only contracts together. The rendezvous publishes
`CE_LIVE_FOUNDATIONS_READY` only after the native build, focused host/CUDA
correctness, readiness sanitizer, todo reconciliation, and frontier audit all
pass. It does not implement the Wave B catalog, projection activation, or
preparation factory.

### Wave B

Immediately after CE-LIVE-19:

```text
CE-LIVE-20  projection activation
CE-LIVE-21  built-in catalog
CE-LIVE-24  quantitative native adapter
CE-LIVE-25  training/readiness integration
CE-LIVE-26  planner inputs

CE-LIVE-20 → CE-LIVE-22
CE-LIVE-20 + CE-LIVE-21 → CE-LIVE-23
20 + 21 + 22 + 23 + 24 + 25 + 26 → CE-LIVE-29
```

The two serialized implementation forks completed as:

```text
Fork A: CE-LIVE-20 → CE-LIVE-22 → CE-LIVE-23
Fork B: CE-LIVE-21 → CE-LIVE-24 → CE-LIVE-26 → CE-LIVE-25
Cross-fork: CE-LIVE-21 → CE-LIVE-23
```

CE-LIVE-29 is the single-owner implementation rendezvous. Its root build seam
publishes `Cellerator::executable_core` as an aggregate of non-owning CPE2 typed
activation, the immutable host candidate catalog, the strong session-backed
cuSPARSE candidates, the typed preparation factory, and the bounded native
training slice. The native training target now links the existing runtime
readiness implementation; event/stream state remains launch-time state.

Task-owned fixture and planner-input libraries remain validation/benchmark
adapters rather than production storage or a second planner. Focused targets
cover every Wave B leaf, and
`celleratorExecutableCoreIntegrationTest` verifies that the catalog plus the
two conventional cuSPARSE candidates share one registry, typed activation and
preparation link together, the quantitative relation produces factored planner
keys, and readiness stays outside persistent parameter identity.

This fan-in freezes only the minimum executable-core build interfaces. It does
not create the executable program API, select a Tensor Core implementation, or
start the quantitative vertical slice; those remain CE-LIVE-30 and later work.

### Executable bridge

```text
CE-LIVE-29 → CE-LIVE-30
```

CE-LIVE-30 is deliberately serial. No competing program API may be developed
in parallel.

### Wave C

```text
CE-LIVE-30 → CE-LIVE-31  quantitative forward
CE-LIVE-30 → CE-LIVE-32  Tensor Core candidate
CE-LIVE-30 → CE-LIVE-33  native training executable

CE-LIVE-31 → CE-LIVE-34
CE-LIVE-33 → CE-LIVE-35
CE-LIVE-31 + 33 + 34 + 35 → CE-LIVE-36 → CE-LIVE-37
```

CE-LIVE-32 may continue independently and does not block CelleraTorch entry.

### Wave D

```text
CE-LIVE-37 → CE-LIVE-40  zero-copy native views
CE-LIVE-37 → CE-LIVE-41  forward custom op
CE-LIVE-37 → CE-LIVE-42  autograd/readiness adapter

40 + 41 + 42 → 43 → 44
32 + 44 → 45
```

## 7. Interfaces and checkpoints

All interfaces begin in `draft`; only their owner freezes them after gates.

| Draft interface | Owner | Publication checkpoint |
|---|---|---|
| cellerator-logical-edge-orientation-v1 | CE-LIVE-11 | CELLERATOR_LOGICAL_EDGE_ORIENTATION_V1_READY |
| cellerator-quantitative-fixture-v1 | CE-LIVE-12 | CELLERATOR_QUANTITATIVE_FIXTURE_V1_READY |
| cellerator-value-readiness-v1 | CE-LIVE-14 | CELLERATOR_VALUE_READINESS_V1_READY |
| cellerator-projection-activation-v1 | CE-LIVE-20 | CELLERATOR_PROJECTION_ACTIVATION_V1_READY |
| cellerator-builtin-candidate-catalog-v1 | CE-LIVE-21 | CELLERATOR_BUILTIN_CANDIDATE_CATALOG_V1_READY |
| cellerator-preparation-factory-v1 | CE-LIVE-23 | CELLERATOR_PREPARATION_FACTORY_V1_READY |
| cellerator-executable-program-v1 | CE-LIVE-30 | CELLERATOR_EXECUTABLE_PROGRAM_V1_READY |
| cellerator-native-training-program-v1 | CE-LIVE-33 | CELLERATOR_LIVE_TRAINING_V1 |
| cellerator-cellera-torch-entry-v1 | CE-LIVE-37 | CELLERATOR_CELLERATORCH_ENTRY_V1 |

There is no stable Tensor Core semantic interface.

The CE-LIVE checkpoints are:
`CE_LIVE_BOOTSTRAP_READY`, `CE_LIVE_CURRENT_HEAD_BASELINED`,
`CELLERATOR_LOGICAL_EDGE_ORIENTATION_V1_READY`,
`CELLERATOR_QUANTITATIVE_FIXTURE_V1_READY`,
`CE_LIVE_CANDIDATE_INVENTORY_READY`,
`CELLERATOR_VALUE_READINESS_V1_READY`,
`CE_LIVE_AUTHORITY_REFRESHED`,
`CELLERATOR_TENSOR_CORE_CONTRACT_READY`, `CE_LIVE_FOUNDATIONS_READY`,
`CELLERATOR_PROJECTION_ACTIVATION_V1_READY`,
`CELLERATOR_BUILTIN_CANDIDATE_CATALOG_V1_READY`,
`CELLERATOR_CONVENTIONAL_CSR_V1_READY`,
`CELLERATOR_PREPARATION_FACTORY_V1_READY`,
`CELLERATOR_QUANTITATIVE_RELATION_READY`,
`CELLERATOR_NATIVE_TRAINING_READINESS_READY`,
`CELLERATOR_LIVE_PLANNER_INPUTS_READY`,
`CELLERATOR_EXECUTABLE_CORE_READY`,
`CELLERATOR_EXECUTABLE_PROGRAM_V1_READY`,
`CELLERATOR_LIVE_FORWARD_V1`,
`CELLERATOR_TENSOR_CORE_DECISION_V1`,
`CELLERATOR_LIVE_TRAINING_V1`, `CELLERATOR_LIVE_REPLAY_V1`,
`CELLERATOR_LIVE_CONCURRENCY_V1`, `CELLERATOR_LIVE_EVIDENCE_V1`,
`CELLERATOR_LIVE_BIOLOGICAL_V1`,
`CELLERATOR_CELLERATORCH_ENTRY_V1`,
`CELLERATOR_CELLERATORCH_ADAPTER_READY`,
`CELLERATOR_CELLERATORCH_LIVE_V1`, and `CE_LIVE_COMPLETE`.

## 8. External repository boundaries

### Baseplane

The lower-case `baseplane` workspace is clean at
`840abb262c357d59ce7b3110aa89ce79342fa1fa`, todo revision 130.
Predicate/event ABI foundations are complete; general scalar/CUDA BitOp
execution is not complete. `CE-BITOP-40` is terminally superseded and its old
DeviceMathContext targets/gates must not run. The cross-authority warning is
preserved as historical inconsistency; Cellerator does not mutate or repair the
Baseplane ledger. CE-LIVE consumes the frozen Baseplane contracts and current
Cellerator-side proof without waiting for the broader BitOp roadmap.

### CellShard

The lower-case `cellshard` workspace is clean at
`3745d66e344f8fac3c39cd92f110ca479aa4555d`, todo revision 90.
CS-FOUND-01 through 03 are complete. CS-FOUND-04 (domain/partition),
CS-FOUND-05 (projection/image), and CS-FOUND-06 (extent/source) are independently
ready with no claims. The catalog, CPEXEC02 codec, residency, and opaque vertical
slice are incomplete. CE-LIVE-34 uses the existing CPEXEC01 compatibility path.
CPEXEC02 integration follows only after CellShard reaches its image/residency
checkpoint. No CellShard file is owned by CE-LIVE.

Cellerator, Baseplane, and CellShard are independent sibling repositories. The
former CellStack wrapper is neither recreated nor used.

## 9. Fixture scientific limitation

The fixture manifest must preserve source checksum, exact H5AD matrix path or
layer, stored dtype, orientation, selected row identities, complete
feature-index bytes or digest, deterministic seed/mapping, extracted-value
checksum, and derived axis/order/geometry/partition/structure/generation
identities. The full stored distribution must be checked for finite,
non-negative, integral, minimum/maximum, and execution-type representability
properties. Generation 1 uses the validated stored values with an explicit
cast. Generation 2 is deterministic numerical stress over identical support
and is not normalization or biological transformation. Raw H5AD and large
derived matrices stay local and ignored.

No donor, sample, chemistry, species, normalization, biological-comparison, or
scientific-interpretation claim may be made from this fixture.

## 10. Tensor Core decision rule

CE-LIVE-32 may implement at most one sm_70 dense-fragment/WMMA candidate.
Promotion requires a complete-cost real-fixture win against feature-major warp,
feature-major CTA, CSR/cuSPARSE, and dense cuBLAS where effectively dense.
Correctness, packing, conversions, projection construction, tails, epilogue,
output order, synchronization, and reuse amortization are included. A
microkernel win is insufficient. Reproducible measured non-promotion, with the
candidate unregistered by default, is a successful CE-LIVE result.

## 11. Native Cellerator live definition

`CELLERATOR_LIVE_BIOLOGICAL_V1` means that a checksum-pinned quantitative
fixture can:

1. preserve exact biological axis/order/geometry/partition/structure/value
   identities;
2. compile CP-BP geometry and build typed CPE2 projections;
3. enumerate native and strong conventional candidates;
4. select by complete measured cost and declared reuse;
5. prepare through the sole execution session;
6. rebind pointers, streams, workspace, and value generations without topology
   reconstruction;
7. execute forward and the bounded N=16 training slice with independent
   numerical referees;
8. report output order explicitly and recover canonical values only when a
   consumer asks;
9. replay through CPE2 and existing opaque CPEXEC01 compatibility;
10. pass concurrency, stale-identity, graph, sanitizer, and forbidden-hot-path
    acceptance; and
11. publish reproducible evidence and planner regret.

Unsupported shapes, numeric tuples, graphs, projections, and training regimes
remain explicit.

## 12. CelleraTorch entry definition

`CELLERATOR_CELLERATORCH_ENTRY_V1` is reached only after the native live audit
has frozen the minimal parameter, executable-program, training-program, and
value-readiness entry. CelleraTorch then provides non-owning views, current
Torch stream adaptation, custom operations, autograd wrappers, and framework
modules. It does not own Cellerator parameters, structures, planners, sessions,
generations, or reusable kernels. The copied CPU CSR exporter remains an
explicit debug/compatibility path, never the repeated primary path.

### CE-LIVE-37 activation audit

The native activation audit accepts the following bounded surface for
`CELLERATOR_LIVE_BIOLOGICAL_V1`:

| Surface | Accepted result | Explicit boundary |
|---|---|---|
| Quantitative forward | PBMC3K computational fixture, widths 1, 16, 17, 31, 32, 48, and 64; reuse 1, 8, and 1024; two value generations; independent referee | The built-in catalog exposes one legal FMP1 SpMM schedule per tested request, so zero regret is over the legal set and is not a universal cuSPARSE comparison. |
| Native training | Five-stage forward/epilogue/explicit-transpose/update pipeline with readiness publication | N=16 small-module slice only; not a general-N training engine. |
| Persistence | Pointer-free CPE2 reload through existing opaque CellShard CPEXEC01 transport | No CPEXEC02 claim and no CellShard semantic ownership. |
| Runtime | Caller-stream execution, same/cross-stream readiness, fixed-transition CUDA Graph capture, pointer relocation, stale identity rejection, and sanitizer coverage | No hidden allocation, device selection, device synchronization, or persistent stream/event identity. |
| Tensor Core | One bounded V100 candidate received a measured non-promotion decision | No registered Tensor Core default and no universal dense-fragment policy. |

The reproducible Wave C evidence is
`bench/ce_live/evidence/ce_live_evidence_v1.json`; its interpretation and exact
foreground controller command are recorded in `docs/CE_LIVE_EVIDENCE.md`.
The fixture supports computational validation only and makes no biological or
scientific claim.

The frozen `cellerator-cellera-torch-entry-v1` interface consists only of the
existing native contracts in `include/Cellerator/parameters.hh`,
`include/Cellerator/execution/program.hh`,
`include/Cellerator/execution/training_program.hh`, and
`include/Cellerator/runtime/value_readiness.cuh`, plus this ownership contract.
Wave D may adapt those contracts using non-owning tensor views and the current
Torch stream. It may not move parameter, structure, planner, execution-session,
value-generation, or reusable-kernel ownership into CelleraTorch. No
CelleraTorch implementation is claimed by this checkpoint.

## 13. Non-goals

CE-LIVE does not implement:

- general DAG planning;
- general dataset/H5AD parsing;
- network or service infrastructure;
- CellShard CS-FOUND completion;
- Baseplane BitOp completion;
- rewrites of all historical Cellerator models;
- deletion of compatibility systems;
- a multi-node runtime;
- universal Tensor Core conversion;
- universal dense fragments;
- broad precision redesign; or
- hidden canonicalization, transfer, or sparse-format conversion.

## 14. Validation and evidence policy

Every implementation task has an artifact-existence gate, focused build gate,
focused correctness gate, applicable negative identity/order/generation/capacity
checks, `git diff --check`, scope audit, and coding-workflow finish/handoff
gate. Fan-ins additionally require todo audit, reconcile/export,
project-control overview/frontier, cross-authority warning review, affected
target fan-in, documentation update, and exact next-frontier validation.

CUDA work commits controller specifications and runs correctness before timing.
Device and benchmark leases and the shared benchmark mutex are mandatory for
performance. Use one GPU unless the task declares otherwise. New pointer
rebinding, device views, projections, and kernels require Compute Sanitizer.
Evidence records exact source and binary identities, compiler/CUDA/driver and
libraries, device, command, data shape/distribution, dtype/accumulation,
warmups/repeats, tolerance, setup/transfer/conversion/sync/output-order phase
costs, reuse, baselines, variance, and contamination. Missing background
evidence storage does not block foreground work.

The CUDA-heavy tasks are CE-LIVE-10, CE-LIVE-22, CE-LIVE-31, CE-LIVE-32,
CE-LIVE-33, CE-LIVE-35, CE-LIVE-36, and CE-LIVE-44. No benchmark is launched
during bootstrap.

Safe local-worker lanes are annotated for host inventory, fixture tooling,
candidate inventory, projection activation, catalog work, and isolated
CelleraTorch views. Local workers never own architecture decisions or
integration fan-ins and remain inactive after bootstrap.

## 15. Historical bootstrap frontier

After CE-LIVE-01 reaches `CE_LIVE_BOOTSTRAP_READY` and closes, the exact
frontier is:

- CE-LIVE-10 — current-head evidence
- CE-LIVE-11 — relation orientation
- CE-LIVE-12 — quantitative fixture
- CE-LIVE-13 — candidate inventory
- CE-LIVE-14 — value readiness
- CE-LIVE-15 — authority/build defaults
- CE-LIVE-16 — Tensor Core contract

All seven have non-overlapping exclusive scopes and no active claims. No later
task is ready before its declared dependency set. No CE-LIVE implementation
task is begun by the bootstrap.

## 16. Final closure state

CE-LIVE-45 closes the program after every implementation and validation leaf,
the Tensor Core decision, all rendezvous tasks, and the CelleraTorch
quantitative validation are terminal. `CE_LIVE_COMPLETE` records this bounded
result.

The final accepted system includes:

- planner-backed quantitative Cellerator forward execution over the pinned
  PBMC3K computational fixture at widths 1, 16, 17, 31, 32, 48, and 64;
- immutable-topology reuse across mutable value generations;
- bounded N=16 native training with explicit transpose identity and readiness;
- CPE2 reload through opaque CellShard CPEXEC01 compatibility transport;
- caller-stream, cross-stream readiness, CUDA Graph, stale-identity, and
  hot-path acceptance;
- a measured V100 Tensor Core non-promotion decision;
- lifetime-bound CelleraTorch native views, current-stream forward dispatch,
  native autograd/readiness adaptation, and quantitative adapter evidence.

The result is intentionally bounded. The quantitative built-in catalog exposes
one legal FMP1 schedule per tested SpMM request, the native training program is
N=16 only, the experimental Tensor Core candidate is not registered, CPEXEC02
remains CellShard work, broader Baseplane execution remains in its repository,
and distributed/general-DAG execution is later work. Exact evidence and
limitations are consolidated in `docs/CE_LIVE_FINAL_AUDIT.md`.
