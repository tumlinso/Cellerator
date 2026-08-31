# Cellerator atomized implementation Todo plan

This document contains the 100 proposed Cellerator implementation Todos. They preserve current CSG1/CPE2, operation-core v2, relation-algebra v2, candidate-catalog v3, program-v2, acquisition-v2, hierarchical index spaces, projection-value planes, and planner authority. New work is generic enough for standalone Cellerator; embedded CellShard consumes the privileged bridge without entering Cellerator's required dependency graph.

# Workstream CEBOOT: Cellerator baseline, charter, and source ownership

**Repository:** Cellerator  
**Suggested lane:** `CE-JBC-L-BOOTSTRAP`  
**Barrier:** `JBC-G0-LIVE-BASELINE`

## Workstream design

Record both Git and Todo cursors separately; Project Control observations are not globally atomic. Classify each live subsystem as preserve, adjacent extension, generalize, compatibility-only, migrate, or retire-after-gate.

## Existing live source extended

- AGENTS.md
- scope.md
- CMakeLists.txt
- docs/CE_GEO_PROGRAM.md
- docs/CE_EXOP_PROGRAM.md
- ce-exop-plan.json
- components/CellShard

## Likely source scope

- [proposed] docs/JBC/
- [proposed] planning/jbc/
- components/README.md

## Proposed Todos (6)

## CE-JBC-B01 — Revalidate live Cellerator, submodule, and Todo cursors

**Repository / subsystem / lane:** Cellerator · `program/bootstrap` · `CE-JBC-L-BOOTSTRAP`  
**Classification:** validation/evidence  
**Parallelism:** Serial within CE-JBC-L-BOOTSTRAP; parallel with other provider/workstream lanes after JBC-G0-LIVE-BASELINE.

### Why

**Purpose.** Revalidate live Cellerator, submodule, and Todo cursors. Deliver this as one isolated, reviewable step in the Cellerator baseline, charter, and source ownership workstream.

**Biological motivation.** A source-backed charter prevents later agents from reducing the integrated compiler to generic distribution and preserves the biological-performance thesis already embodied in Cellerator.

**Compiler-architectural reason.** This work freezes only the current baseline, ownership, and compatibility map. It creates the stable context from which all implementation lanes can proceed without reinterpreting legacy boundaries.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] docs/JBC/
- [proposed] planning/jbc/
- components/README.md

**Existing code and permitted read scope:**

- AGENTS.md
- scope.md
- CMakeLists.txt
- docs/CE_GEO_PROGRAM.md
- docs/CE_EXOP_PROGRAM.md
- ce-exop-plan.json
- components/CellShard

**Explicitly out of scope / forbidden shortcuts:**

- Do not alter runtime or compiler behavior in baseline tasks.
- Do not close, reopen, or absorb historical CE-GEO, CE-EXOP, CE-PTR, or CE-AMP work.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- None.

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- Focused tests and adversarial fixtures demonstrating both success and explicit rejection paths.

### Concrete mechanism

Read Cellerator and nested CellShard commits, cleanliness, worktrees, Todo revisions/fingerprints, active runs, interfaces, and CE-AMP interlock through Project Control. Record two separately timestamped observations and refuse to normalize cross-authority skew.

Workstream mechanism: Record both Git and Todo cursors separately; Project Control observations are not globally atomic. Classify each live subsystem as preserve, adjacent extension, generalize, compatibility-only, migrate, or retire-after-gate.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Dirty worktrees, authority skew, unresolved submodule commit, or conflicting active claims block bootstrap completion.
- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Re-run standalone and embedded configure/build smoke commands and record exact commands and commits.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Preserve current performance evidence without reinterpreting unprofiled behavior.
- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-BOOTSTRAP; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-B02 — Produce the source-backed Cellerator transition map

**Repository / subsystem / lane:** Cellerator · `program/bootstrap` · `CE-JBC-L-BOOTSTRAP`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-BOOTSTRAP; parallel with other provider/workstream lanes after JBC-G0-LIVE-BASELINE.

### Why

**Purpose.** Produce the source-backed Cellerator transition map. Deliver this as one isolated, reviewable step in the Cellerator baseline, charter, and source ownership workstream.

**Biological motivation.** A source-backed charter prevents later agents from reducing the integrated compiler to generic distribution and preserves the biological-performance thesis already embodied in Cellerator.

**Compiler-architectural reason.** This work freezes only the current baseline, ownership, and compatibility map. It creates the stable context from which all implementation lanes can proceed without reinterpreting legacy boundaries.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] docs/JBC/
- [proposed] planning/jbc/
- components/README.md

**Existing code and permitted read scope:**

- AGENTS.md
- scope.md
- CMakeLists.txt
- docs/CE_GEO_PROGRAM.md
- docs/CE_EXOP_PROGRAM.md
- ce-exop-plan.json
- components/CellShard

**Explicitly out of scope / forbidden shortcuts:**

- Do not alter runtime or compiler behavior in baseline tasks.
- Do not close, reopen, or absorb historical CE-GEO, CE-EXOP, CE-PTR, or CE-AMP work.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-B01

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Inspect the exact live paths for support evidence, semantic geometry, optimizer portfolios, operation core, relation algebra, projection values, acquisition, programs, candidate catalog, profiling, and legacy CellShard adapters. Classify each as preserve, adjacent extension, generalize, compatibility-only, migrate, or retire-after-gate.

Workstream mechanism: Record both Git and Todo cursors separately; Project Control observations are not globally atomic. Classify each live subsystem as preserve, adjacent extension, generalize, compatibility-only, migrate, or retire-after-gate.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Dirty worktrees, authority skew, unresolved submodule commit, or conflicting active claims block bootstrap completion.
- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Re-run standalone and embedded configure/build smoke commands and record exact commands and commits.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Preserve current performance evidence without reinterpreting unprofiled behavior.
- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-BOOTSTRAP; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-B03 — Freeze the privileged compiler-component charter

**Repository / subsystem / lane:** Cellerator · `program/bootstrap` · `CE-JBC-L-BOOTSTRAP`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-BOOTSTRAP; parallel with other provider/workstream lanes after JBC-G0-LIVE-BASELINE.

### Why

**Purpose.** Freeze the privileged compiler-component charter. Deliver this as one isolated, reviewable step in the Cellerator baseline, charter, and source ownership workstream.

**Biological motivation.** A source-backed charter prevents later agents from reducing the integrated compiler to generic distribution and preserves the biological-performance thesis already embodied in Cellerator.

**Compiler-architectural reason.** This work freezes only the current baseline, ownership, and compatibility map. It creates the stable context from which all implementation lanes can proceed without reinterpreting legacy boundaries.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] docs/JBC/
- [proposed] planning/jbc/
- components/README.md

**Existing code and permitted read scope:**

- AGENTS.md
- scope.md
- CMakeLists.txt
- docs/CE_GEO_PROGRAM.md
- docs/CE_EXOP_PROGRAM.md
- ce-exop-plan.json
- components/CellShard

**Explicitly out of scope / forbidden shortcuts:**

- Do not alter runtime or compiler behavior in baseline tasks.
- Do not close, reopen, or absorb historical CE-GEO, CE-EXOP, CE-PTR, or CE-AMP work.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-B02

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Replace the blanket component rule with two categories: framework adapters and privileged compiler components. State that CellShard may own higher-level planning/runtime above Cellerator while libCellerator remains independent and has no required CellShard dependency.

Workstream mechanism: Record both Git and Todo cursors separately; Project Control observations are not globally atomic. Classify each live subsystem as preserve, adjacent extension, generalize, compatibility-only, migrate, or retire-after-gate.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Dirty worktrees, authority skew, unresolved submodule commit, or conflicting active claims block bootstrap completion.
- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Re-run standalone and embedded configure/build smoke commands and record exact commands and commits.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Preserve current performance evidence without reinterpreting unprofiled behavior.
- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-BOOTSTRAP; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-B04 — Record standalone and embedded build baselines

**Repository / subsystem / lane:** Cellerator · `program/bootstrap` · `CE-JBC-L-BOOTSTRAP`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-BOOTSTRAP; parallel with other provider/workstream lanes after JBC-G0-LIVE-BASELINE.

### Why

**Purpose.** Record standalone and embedded build baselines. Deliver this as one isolated, reviewable step in the Cellerator baseline, charter, and source ownership workstream.

**Biological motivation.** A source-backed charter prevents later agents from reducing the integrated compiler to generic distribution and preserves the biological-performance thesis already embodied in Cellerator.

**Compiler-architectural reason.** This work freezes only the current baseline, ownership, and compatibility map. It creates the stable context from which all implementation lanes can proceed without reinterpreting legacy boundaries.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] docs/JBC/
- [proposed] planning/jbc/
- components/README.md

**Existing code and permitted read scope:**

- AGENTS.md
- scope.md
- CMakeLists.txt
- docs/CE_GEO_PROGRAM.md
- docs/CE_EXOP_PROGRAM.md
- ce-exop-plan.json
- components/CellShard

**Explicitly out of scope / forbidden shortcuts:**

- Do not alter runtime or compiler behavior in baseline tasks.
- Do not close, reopen, or absorb historical CE-GEO, CE-EXOP, CE-PTR, or CE-AMP work.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-B03

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.
- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Run and record configure/build/test smoke for CELLERATOR_ENABLE_CELLSHARD=OFF and ON without changing implementation. Capture target graph and prove canonical Cellerator::runtime remains unchanged.

Workstream mechanism: Record both Git and Todo cursors separately; Project Control observations are not globally atomic. Classify each live subsystem as preserve, adjacent extension, generalize, compatibility-only, migrate, or retire-after-gate.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Dirty worktrees, authority skew, unresolved submodule commit, or conflicting active claims block bootstrap completion.
- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Re-run standalone and embedded configure/build smoke commands and record exact commands and commits.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Preserve current performance evidence without reinterpreting unprofiled behavior.
- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-BOOTSTRAP; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-B05 — Record frozen ABI and compatibility baseline

**Repository / subsystem / lane:** Cellerator · `program/bootstrap` · `CE-JBC-L-BOOTSTRAP`  
**Classification:** compatibility/migration  
**Parallelism:** Serial within CE-JBC-L-BOOTSTRAP; parallel with other provider/workstream lanes after JBC-G0-LIVE-BASELINE.

### Why

**Purpose.** Record frozen ABI and compatibility baseline. Deliver this as one isolated, reviewable step in the Cellerator baseline, charter, and source ownership workstream.

**Biological motivation.** A source-backed charter prevents later agents from reducing the integrated compiler to generic distribution and preserves the biological-performance thesis already embodied in Cellerator.

**Compiler-architectural reason.** This work freezes only the current baseline, ownership, and compatibility map. It creates the stable context from which all implementation lanes can proceed without reinterpreting legacy boundaries.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] docs/JBC/
- [proposed] planning/jbc/
- components/README.md

**Existing code and permitted read scope:**

- AGENTS.md
- scope.md
- CMakeLists.txt
- docs/CE_GEO_PROGRAM.md
- docs/CE_EXOP_PROGRAM.md
- ce-exop-plan.json
- components/CellShard

**Explicitly out of scope / forbidden shortcuts:**

- Do not alter runtime or compiler behavior in baseline tasks.
- Do not close, reopen, or absorb historical CE-GEO, CE-EXOP, CE-PTR, or CE-AMP work.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-B04

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.

### Concrete mechanism

Snapshot operation_core_v2, relation_algebra_v2, CSG1, CPE2, CPK1, projection-value-plane v1, candidate-catalog v3, program_v2, training-program v2, acquisition-v2, and hierarchical-index interfaces; identify only adjacent extension points.

Workstream mechanism: Record both Git and Todo cursors separately; Project Control observations are not globally atomic. Classify each live subsystem as preserve, adjacent extension, generalize, compatibility-only, migrate, or retire-after-gate.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Dirty worktrees, authority skew, unresolved submodule commit, or conflicting active claims block bootstrap completion.
- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Re-run standalone and embedded configure/build smoke commands and record exact commands and commits.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Preserve current performance evidence without reinterpreting unprofiled behavior.
- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-BOOTSTRAP; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-B06 — Reserve Cellerator integration paths and registry ownership

**Repository / subsystem / lane:** Cellerator · `program/bootstrap` · `CE-JBC-L-BOOTSTRAP`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-BOOTSTRAP; parallel with other provider/workstream lanes after JBC-G0-LIVE-BASELINE.

### Why

**Purpose.** Reserve Cellerator integration paths and registry ownership. Deliver this as one isolated, reviewable step in the Cellerator baseline, charter, and source ownership workstream.

**Biological motivation.** A source-backed charter prevents later agents from reducing the integrated compiler to generic distribution and preserves the biological-performance thesis already embodied in Cellerator.

**Compiler-architectural reason.** This work freezes only the current baseline, ownership, and compatibility map. It creates the stable context from which all implementation lanes can proceed without reinterpreting legacy boundaries.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] docs/JBC/
- [proposed] planning/jbc/
- components/README.md

**Existing code and permitted read scope:**

- AGENTS.md
- scope.md
- CMakeLists.txt
- docs/CE_GEO_PROGRAM.md
- docs/CE_EXOP_PROGRAM.md
- ce-exop-plan.json
- components/CellShard

**Explicitly out of scope / forbidden shortcuts:**

- Do not alter runtime or compiler behavior in baseline tasks.
- Do not close, reopen, or absorb historical CE-GEO, CE-EXOP, CE-PTR, or CE-AMP work.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-B05

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A bounded implementation unit, focused tests, mechanism statistics, and an integration receipt.

### Concrete mechanism

Publish the exact list of root CMake, umbrella headers, provider/catalog aggregators, package exports, and component documentation that only integration lanes may write. Require source-linked fragments from provider lanes.

Workstream mechanism: Record both Git and Todo cursors separately; Project Control observations are not globally atomic. Classify each live subsystem as preserve, adjacent extension, generalize, compatibility-only, migrate, or retire-after-gate.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Dirty worktrees, authority skew, unresolved submodule commit, or conflicting active claims block bootstrap completion.
- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Re-run standalone and embedded configure/build smoke commands and record exact commands and commits.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Preserve current performance evidence without reinterpreting unprofiled behavior.
- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-BOOTSTRAP; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


# Workstream CEIF: Cellerator-owned joint-compiler thin-waist interfaces

**Repository:** Cellerator  
**Suggested lane:** `CE-JBC-L-INTERFACES`  
**Barrier:** `JBC-G0-LIVE-BASELINE`

## Workstream design

Use explicit standard-layout C++17 records with schema_version/record_bytes where evolution is expected. Use namespace-qualified 128-bit semantic identities, pointer-plus-count arrays, caller-owned buffers, and source-linked provider tables. Persist provider IDs and schemas, never function pointers.

## Existing live source extended

- include/Cellerator/execution/identity.hh
- include/Cellerator/execution/biological_abi.hh
- include/Cellerator/execution/lifetimes.hh
- include/Cellerator/execution/index_space/hierarchical_index_space_v1.hh
- include/Cellerator/compute/operation/operation_core_v2/schema.hh
- include/Cellerator/profiling/partition_export.h

## Likely source scope

- [proposed] include/Cellerator/execution/joint_compiler/
- [proposed] include/Cellerator/compute/decomposition/
- [proposed] include/Cellerator/profiling/joint_compiler/
- [proposed] src/execution/joint_compiler/
- [proposed] tests/jbc/interfaces/

## Proposed Todos (12)

## CE-JBC-I01 — Define a namespace-qualified persistent identity bridge

**Repository / subsystem / lane:** Cellerator · `execution/identity, coverage, decomposition, fragment ABI` · `CE-JBC-L-INTERFACES`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-INTERFACES; parallel with other provider/workstream lanes after JBC-G0-LIVE-BASELINE.

### Why

**Purpose.** Define a namespace-qualified persistent identity bridge. Deliver this as one isolated, reviewable step in the Cellerator-owned joint-compiler thin-waist interfaces workstream.

**Biological motivation.** Typed biological axes, exact relation coverage, persistent order, and numerical partial semantics must cross the compiler boundary without being flattened to anonymous tensors.

**Compiler-architectural reason.** Cellerator owns the generic contracts that state what local biological computation means. CellShard may consume them deeply, but standalone Cellerator must remain fully usable without CellShard types or callbacks.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/execution/joint_compiler/
- [proposed] include/Cellerator/compute/decomposition/
- [proposed] include/Cellerator/profiling/joint_compiler/
- [proposed] src/execution/joint_compiler/
- [proposed] tests/jbc/interfaces/

**Existing code and permitted read scope:**

- include/Cellerator/execution/identity.hh
- include/Cellerator/execution/biological_abi.hh
- include/Cellerator/execution/lifetimes.hh
- include/Cellerator/execution/index_space/hierarchical_index_space_v1.hh
- include/Cellerator/compute/operation/operation_core_v2/schema.hh
- include/Cellerator/profiling/partition_export.h

**Explicitly out of scope / forbidden shortcuts:**

- No CellShard include may enter a required Cellerator public-core header.
- Do not mutate operation_core_v2, CSG1, CPE2, or program_v2 merely to add the new contracts; add adjacent versions or wrappers first.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-B06

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.
- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Add a 128-bit identity represented explicitly as producer/namespace plus local identity; adapt operation_core_v2::stable_id and CellShard 64-bit strong IDs without pointer hashing or silently mixing legacy counters. Keep content digests separate.

Workstream mechanism: Use explicit standard-layout C++17 records with schema_version/record_bytes where evolution is expected. Use namespace-qualified 128-bit semantic identities, pointer-plus-count arrays, caller-owned buffers, and source-linked provider tables. Persist provider IDs and schemas, never function pointers.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Static-assert trivial-copy/standard-layout properties and add cross-version rejection tests.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-INTERFACES; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-I02 — Define exact logical coverage view v1

**Repository / subsystem / lane:** Cellerator · `execution/identity, coverage, decomposition, fragment ABI` · `CE-JBC-L-INTERFACES`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-INTERFACES; parallel with other provider/workstream lanes after JBC-G0-LIVE-BASELINE.

### Why

**Purpose.** Define exact logical coverage view v1. Deliver this as one isolated, reviewable step in the Cellerator-owned joint-compiler thin-waist interfaces workstream.

**Biological motivation.** Typed biological axes, exact relation coverage, persistent order, and numerical partial semantics must cross the compiler boundary without being flattened to anonymous tensors.

**Compiler-architectural reason.** Cellerator owns the generic contracts that state what local biological computation means. CellShard may consume them deeply, but standalone Cellerator must remain fully usable without CellShard types or callbacks.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/execution/joint_compiler/
- [proposed] include/Cellerator/compute/decomposition/
- [proposed] include/Cellerator/profiling/joint_compiler/
- [proposed] src/execution/joint_compiler/
- [proposed] tests/jbc/interfaces/

**Existing code and permitted read scope:**

- include/Cellerator/execution/identity.hh
- include/Cellerator/execution/biological_abi.hh
- include/Cellerator/execution/lifetimes.hh
- include/Cellerator/execution/index_space/hierarchical_index_space_v1.hh
- include/Cellerator/compute/operation/operation_core_v2/schema.hh
- include/Cellerator/profiling/partition_export.h

**Explicitly out of scope / forbidden shortcuts:**

- No CellShard include may enter a required Cellerator public-core header.
- Do not mutate operation_core_v2, CSG1, CPE2, or program_v2 merely to add the new contracts; add adjacent versions or wrappers first.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-I01

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.

### Concrete mechanism

Represent coverage kind, role, source/destination axes, structure and epoch, logical count, payload schema, and pointer-plus-count membership. Support canonical intervals, explicit IDs, relation-edge IDs, semantic components, segment sets, unions, and provider-defined payloads.

Workstream mechanism: Use explicit standard-layout C++17 records with schema_version/record_bytes where evolution is expected. Use namespace-qualified 128-bit semantic identities, pointer-plus-count arrays, caller-owned buffers, and source-linked provider tables. Persist provider IDs and schemas, never function pointers.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Static-assert trivial-copy/standard-layout properties and add cross-version rejection tests.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-INTERFACES; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-I03 — Define ownership, proposal, replica, and halo coverage roles

**Repository / subsystem / lane:** Cellerator · `execution/identity, coverage, decomposition, fragment ABI` · `CE-JBC-L-INTERFACES`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-INTERFACES; parallel with other provider/workstream lanes after JBC-G0-LIVE-BASELINE.

### Why

**Purpose.** Define ownership, proposal, replica, and halo coverage roles. Deliver this as one isolated, reviewable step in the Cellerator-owned joint-compiler thin-waist interfaces workstream.

**Biological motivation.** Typed biological axes, exact relation coverage, persistent order, and numerical partial semantics must cross the compiler boundary without being flattened to anonymous tensors.

**Compiler-architectural reason.** Cellerator owns the generic contracts that state what local biological computation means. CellShard may consume them deeply, but standalone Cellerator must remain fully usable without CellShard types or callbacks.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/execution/joint_compiler/
- [proposed] include/Cellerator/compute/decomposition/
- [proposed] include/Cellerator/profiling/joint_compiler/
- [proposed] src/execution/joint_compiler/
- [proposed] tests/jbc/interfaces/

**Existing code and permitted read scope:**

- include/Cellerator/execution/identity.hh
- include/Cellerator/execution/biological_abi.hh
- include/Cellerator/execution/lifetimes.hh
- include/Cellerator/execution/index_space/hierarchical_index_space_v1.hh
- include/Cellerator/compute/operation/operation_core_v2/schema.hh
- include/Cellerator/profiling/partition_export.h

**Explicitly out of scope / forbidden shortcuts:**

- No CellShard include may enter a required Cellerator public-core header.
- Do not mutate operation_core_v2, CSG1, CPE2, or program_v2 merely to add the new contracts; add adjacent versions or wrappers first.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-I02

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.

### Concrete mechanism

Add explicit role flags/records for approximate proposal membership, exact read requirement, read-only halo, physical replica, exclusive output owner, and partial contribution owner. Validation must reject role combinations that create ambiguous contribution semantics.

Workstream mechanism: Use explicit standard-layout C++17 records with schema_version/record_bytes where evolution is expected. Use namespace-qualified 128-bit semantic identities, pointer-plus-count arrays, caller-owned buffers, and source-linked provider tables. Persist provider IDs and schemas, never function pointers.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Static-assert trivial-copy/standard-layout properties and add cross-version rejection tests.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-INTERFACES; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-I04 — Define atom requirement descriptor v1

**Repository / subsystem / lane:** Cellerator · `execution/identity, coverage, decomposition, fragment ABI` · `CE-JBC-L-INTERFACES`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-INTERFACES; parallel with other provider/workstream lanes after JBC-G0-LIVE-BASELINE.

### Why

**Purpose.** Define atom requirement descriptor v1. Deliver this as one isolated, reviewable step in the Cellerator-owned joint-compiler thin-waist interfaces workstream.

**Biological motivation.** Typed biological axes, exact relation coverage, persistent order, and numerical partial semantics must cross the compiler boundary without being flattened to anonymous tensors.

**Compiler-architectural reason.** Cellerator owns the generic contracts that state what local biological computation means. CellShard may consume them deeply, but standalone Cellerator must remain fully usable without CellShard types or callbacks.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/execution/joint_compiler/
- [proposed] include/Cellerator/compute/decomposition/
- [proposed] include/Cellerator/profiling/joint_compiler/
- [proposed] src/execution/joint_compiler/
- [proposed] tests/jbc/interfaces/

**Existing code and permitted read scope:**

- include/Cellerator/execution/identity.hh
- include/Cellerator/execution/biological_abi.hh
- include/Cellerator/execution/lifetimes.hh
- include/Cellerator/execution/index_space/hierarchical_index_space_v1.hh
- include/Cellerator/compute/operation/operation_core_v2/schema.hh
- include/Cellerator/profiling/partition_export.h

**Explicitly out of scope / forbidden shortcuts:**

- No CellShard include may enter a required Cellerator public-core header.
- Do not mutate operation_core_v2, CSG1, CPE2, or program_v2 merely to add the new contracts; add adjacent versions or wrappers first.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-I03

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.

### Concrete mechanism

Describe what a local Cellerator candidate requires: exact coverage, accepted atom species/planes, numeric type, index width, order, alignment, contiguity, extent count, mutability, generation, graph-stable address, and allowed transform paths.

Workstream mechanism: Use explicit standard-layout C++17 records with schema_version/record_bytes where evolution is expected. Use namespace-qualified 128-bit semantic identities, pointer-plus-count arrays, caller-owned buffers, and source-linked provider tables. Persist provider IDs and schemas, never function pointers.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Static-assert trivial-copy/standard-layout properties and add cross-version rejection tests.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-INTERFACES; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-I05 — Define atom affordance descriptor v1

**Repository / subsystem / lane:** Cellerator · `execution/identity, coverage, decomposition, fragment ABI` · `CE-JBC-L-INTERFACES`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-INTERFACES; parallel with other provider/workstream lanes after JBC-G0-LIVE-BASELINE.

### Why

**Purpose.** Define atom affordance descriptor v1. Deliver this as one isolated, reviewable step in the Cellerator-owned joint-compiler thin-waist interfaces workstream.

**Biological motivation.** Typed biological axes, exact relation coverage, persistent order, and numerical partial semantics must cross the compiler boundary without being flattened to anonymous tensors.

**Compiler-architectural reason.** Cellerator owns the generic contracts that state what local biological computation means. CellShard may consume them deeply, but standalone Cellerator must remain fully usable without CellShard types or callbacks.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/execution/joint_compiler/
- [proposed] include/Cellerator/compute/decomposition/
- [proposed] include/Cellerator/profiling/joint_compiler/
- [proposed] src/execution/joint_compiler/
- [proposed] tests/jbc/interfaces/

**Existing code and permitted read scope:**

- include/Cellerator/execution/identity.hh
- include/Cellerator/execution/biological_abi.hh
- include/Cellerator/execution/lifetimes.hh
- include/Cellerator/execution/index_space/hierarchical_index_space_v1.hh
- include/Cellerator/compute/operation/operation_core_v2/schema.hh
- include/Cellerator/profiling/partition_export.h

**Explicitly out of scope / forbidden shortcuts:**

- No CellShard include may enter a required Cellerator public-core header.
- Do not mutate operation_core_v2, CSG1, CPE2, or program_v2 merely to add the new contracts; add adjacent versions or wrappers first.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-I04

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.

### Concrete mechanism

Describe what an input or output atom provides: coverage, planes, order, physical encoding, local projection ABI, multi-extent legality, direct gradient/output support, persistence eligibility, and fused transform opportunities.

Workstream mechanism: Use explicit standard-layout C++17 records with schema_version/record_bytes where evolution is expected. Use namespace-qualified 128-bit semantic identities, pointer-plus-count arrays, caller-owned buffers, and source-linked provider tables. Persist provider IDs and schemas, never function pointers.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Static-assert trivial-copy/standard-layout properties and add cross-version rejection tests.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-INTERFACES; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-I06 — Define partial-result algebra ABI v1

**Repository / subsystem / lane:** Cellerator · `execution/identity, coverage, decomposition, fragment ABI` · `CE-JBC-L-INTERFACES`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-INTERFACES; parallel with other provider/workstream lanes after JBC-G0-LIVE-BASELINE.

### Why

**Purpose.** Define partial-result algebra ABI v1. Deliver this as one isolated, reviewable step in the Cellerator-owned joint-compiler thin-waist interfaces workstream.

**Biological motivation.** Typed biological axes, exact relation coverage, persistent order, and numerical partial semantics must cross the compiler boundary without being flattened to anonymous tensors.

**Compiler-architectural reason.** Cellerator owns the generic contracts that state what local biological computation means. CellShard may consume them deeply, but standalone Cellerator must remain fully usable without CellShard types or callbacks.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/execution/joint_compiler/
- [proposed] include/Cellerator/compute/decomposition/
- [proposed] include/Cellerator/profiling/joint_compiler/
- [proposed] src/execution/joint_compiler/
- [proposed] tests/jbc/interfaces/

**Existing code and permitted read scope:**

- include/Cellerator/execution/identity.hh
- include/Cellerator/execution/biological_abi.hh
- include/Cellerator/execution/lifetimes.hh
- include/Cellerator/execution/index_space/hierarchical_index_space_v1.hh
- include/Cellerator/compute/operation/operation_core_v2/schema.hh
- include/Cellerator/profiling/partition_export.h

**Explicitly out of scope / forbidden shortcuts:**

- No CellShard include may enter a required Cellerator public-core header.
- Do not mutate operation_core_v2, CSG1, CPE2, or program_v2 merely to add the new contracts; add adjacent versions or wrappers first.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-I05

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.

### Concrete mechanism

Create a data-only algebra descriptor naming state layout, neutral element, merge/finalize operations, associativity, commutativity, idempotence, ordered-only constraints, deterministic-tree requirements, and numerical policy. Persist operation identities, not function pointers.

Workstream mechanism: Use explicit standard-layout C++17 records with schema_version/record_bytes where evolution is expected. Use namespace-qualified 128-bit semantic identities, pointer-plus-count arrays, caller-owned buffers, and source-linked provider tables. Persist provider IDs and schemas, never function pointers.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Static-assert trivial-copy/standard-layout properties and add cross-version rejection tests.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-INTERFACES; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-I07 — Define operation decomposition alternative ABI v1

**Repository / subsystem / lane:** Cellerator · `execution/identity, coverage, decomposition, fragment ABI` · `CE-JBC-L-INTERFACES`  
**Classification:** experimental candidate; baseline and negative result required  
**Parallelism:** Serial within CE-JBC-L-INTERFACES; parallel with other provider/workstream lanes after JBC-G0-LIVE-BASELINE.

### Why

**Purpose.** Define operation decomposition alternative ABI v1. Deliver this as one isolated, reviewable step in the Cellerator-owned joint-compiler thin-waist interfaces workstream.

**Biological motivation.** Typed biological axes, exact relation coverage, persistent order, and numerical partial semantics must cross the compiler boundary without being flattened to anonymous tensors.

**Compiler-architectural reason.** Cellerator owns the generic contracts that state what local biological computation means. CellShard may consume them deeply, but standalone Cellerator must remain fully usable without CellShard types or callbacks.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/execution/joint_compiler/
- [proposed] include/Cellerator/compute/decomposition/
- [proposed] include/Cellerator/profiling/joint_compiler/
- [proposed] src/execution/joint_compiler/
- [proposed] tests/jbc/interfaces/

**Existing code and permitted read scope:**

- include/Cellerator/execution/identity.hh
- include/Cellerator/execution/biological_abi.hh
- include/Cellerator/execution/lifetimes.hh
- include/Cellerator/execution/index_space/hierarchical_index_space_v1.hh
- include/Cellerator/compute/operation/operation_core_v2/schema.hh
- include/Cellerator/profiling/partition_export.h

**Explicitly out of scope / forbidden shortcuts:**

- No CellShard include may enter a required Cellerator public-core header.
- Do not mutate operation_core_v2, CSG1, CPE2, or program_v2 merely to add the new contracts; add adjacent versions or wrappers first.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-I06

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.

### Concrete mechanism

Represent split axis, exact required input coverage, exact final or partial output coverage, required replication/halos, persistent-order constraints, partial algebra, numerical consequences, candidate family, and legality flags.

Workstream mechanism: Use explicit standard-layout C++17 records with schema_version/record_bytes where evolution is expected. Use namespace-qualified 128-bit semantic identities, pointer-plus-count arrays, caller-owned buffers, and source-linked provider tables. Persist provider IDs and schemas, never function pointers.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Static-assert trivial-copy/standard-layout properties and add cross-version rejection tests.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-INTERFACES; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-I08 — Define atom-fragment request ABI v1

**Repository / subsystem / lane:** Cellerator · `execution/identity, coverage, decomposition, fragment ABI` · `CE-JBC-L-INTERFACES`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-INTERFACES; parallel with other provider/workstream lanes after JBC-G0-LIVE-BASELINE.

### Why

**Purpose.** Define atom-fragment request ABI v1. Deliver this as one isolated, reviewable step in the Cellerator-owned joint-compiler thin-waist interfaces workstream.

**Biological motivation.** Typed biological axes, exact relation coverage, persistent order, and numerical partial semantics must cross the compiler boundary without being flattened to anonymous tensors.

**Compiler-architectural reason.** Cellerator owns the generic contracts that state what local biological computation means. CellShard may consume them deeply, but standalone Cellerator must remain fully usable without CellShard types or callbacks.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/execution/joint_compiler/
- [proposed] include/Cellerator/compute/decomposition/
- [proposed] include/Cellerator/profiling/joint_compiler/
- [proposed] src/execution/joint_compiler/
- [proposed] tests/jbc/interfaces/

**Existing code and permitted read scope:**

- include/Cellerator/execution/identity.hh
- include/Cellerator/execution/biological_abi.hh
- include/Cellerator/execution/lifetimes.hh
- include/Cellerator/execution/index_space/hierarchical_index_space_v1.hh
- include/Cellerator/compute/operation/operation_core_v2/schema.hh
- include/Cellerator/profiling/partition_export.h

**Explicitly out of scope / forbidden shortcuts:**

- No CellShard include may enter a required Cellerator public-core header.
- Do not mutate operation_core_v2, CSG1, CPE2, or program_v2 merely to add the new contracts; add adjacent versions or wrappers first.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-I07

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.

### Concrete mechanism

Wrap an operation_core_v2 problem with exact atom coverages, local index spaces, external orders/decomposition, atom bindings, global costs, target profile, desired output affordance, and lowering-resumption stage.

Workstream mechanism: Use explicit standard-layout C++17 records with schema_version/record_bytes where evolution is expected. Use namespace-qualified 128-bit semantic identities, pointer-plus-count arrays, caller-owned buffers, and source-linked provider tables. Persist provider IDs and schemas, never function pointers.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Static-assert trivial-copy/standard-layout properties and add cross-version rejection tests.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-INTERFACES; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-I09 — Define atom-fragment candidate and result ABI v1

**Repository / subsystem / lane:** Cellerator · `execution/identity, coverage, decomposition, fragment ABI` · `CE-JBC-L-INTERFACES`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-INTERFACES; parallel with other provider/workstream lanes after JBC-G0-LIVE-BASELINE.

### Why

**Purpose.** Define atom-fragment candidate and result ABI v1. Deliver this as one isolated, reviewable step in the Cellerator-owned joint-compiler thin-waist interfaces workstream.

**Biological motivation.** Typed biological axes, exact relation coverage, persistent order, and numerical partial semantics must cross the compiler boundary without being flattened to anonymous tensors.

**Compiler-architectural reason.** Cellerator owns the generic contracts that state what local biological computation means. CellShard may consume them deeply, but standalone Cellerator must remain fully usable without CellShard types or callbacks.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/execution/joint_compiler/
- [proposed] include/Cellerator/compute/decomposition/
- [proposed] include/Cellerator/profiling/joint_compiler/
- [proposed] src/execution/joint_compiler/
- [proposed] tests/jbc/interfaces/

**Existing code and permitted read scope:**

- include/Cellerator/execution/identity.hh
- include/Cellerator/execution/biological_abi.hh
- include/Cellerator/execution/lifetimes.hh
- include/Cellerator/execution/index_space/hierarchical_index_space_v1.hh
- include/Cellerator/compute/operation/operation_core_v2/schema.hh
- include/Cellerator/profiling/partition_export.h

**Explicitly out of scope / forbidden shortcuts:**

- No CellShard include may enter a required Cellerator public-core header.
- Do not mutate operation_core_v2, CSG1, CPE2, or program_v2 merely to add the new contracts; add adjacent versions or wrappers first.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-I08

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.

### Concrete mechanism

Return a bounded candidate frontier with exact local cover, required atom inputs, program/projection requirements, output/partial affordances, persistent orders, resource and complete-cost vectors, empirical status, and validation receipt.

Workstream mechanism: Use explicit standard-layout C++17 records with schema_version/record_bytes where evolution is expected. Use namespace-qualified 128-bit semantic identities, pointer-plus-count arrays, caller-owned buffers, and source-linked provider tables. Persist provider IDs and schemas, never function pointers.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Static-assert trivial-copy/standard-layout properties and add cross-version rejection tests.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-INTERFACES; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-I10 — Define multi-extent external binding ABI v1

**Repository / subsystem / lane:** Cellerator · `execution/identity, coverage, decomposition, fragment ABI` · `CE-JBC-L-INTERFACES`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-INTERFACES; parallel with other provider/workstream lanes after JBC-G0-LIVE-BASELINE.

### Why

**Purpose.** Define multi-extent external binding ABI v1. Deliver this as one isolated, reviewable step in the Cellerator-owned joint-compiler thin-waist interfaces workstream.

**Biological motivation.** Typed biological axes, exact relation coverage, persistent order, and numerical partial semantics must cross the compiler boundary without being flattened to anonymous tensors.

**Compiler-architectural reason.** Cellerator owns the generic contracts that state what local biological computation means. CellShard may consume them deeply, but standalone Cellerator must remain fully usable without CellShard types or callbacks.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/execution/joint_compiler/
- [proposed] include/Cellerator/compute/decomposition/
- [proposed] include/Cellerator/profiling/joint_compiler/
- [proposed] src/execution/joint_compiler/
- [proposed] tests/jbc/interfaces/

**Existing code and permitted read scope:**

- include/Cellerator/execution/identity.hh
- include/Cellerator/execution/biological_abi.hh
- include/Cellerator/execution/lifetimes.hh
- include/Cellerator/execution/index_space/hierarchical_index_space_v1.hh
- include/Cellerator/compute/operation/operation_core_v2/schema.hh
- include/Cellerator/profiling/partition_export.h

**Explicitly out of scope / forbidden shortcuts:**

- No CellShard include may enter a required Cellerator public-core header.
- Do not mutate operation_core_v2, CSG1, CPE2, or program_v2 merely to add the new contracts; add adjacent versions or wrappers first.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-I09

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.
- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Represent one atom plane as a checked list of extents with address space, bytes, offset, alignment, order, generation, readiness token, and lease token. Runtime handles remain opaque and nonpersistent.

Workstream mechanism: Use explicit standard-layout C++17 records with schema_version/record_bytes where evolution is expected. Use namespace-qualified 128-bit semantic identities, pointer-plus-count arrays, caller-owned buffers, and source-linked provider tables. Persist provider IDs and schemas, never function pointers.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Static-assert trivial-copy/standard-layout properties and add cross-version rejection tests.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-INTERFACES; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-I11 — Define lowering-resumption ABI v1

**Repository / subsystem / lane:** Cellerator · `execution/identity, coverage, decomposition, fragment ABI` · `CE-JBC-L-INTERFACES`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-INTERFACES; parallel with other provider/workstream lanes after JBC-G0-LIVE-BASELINE.

### Why

**Purpose.** Define lowering-resumption ABI v1. Deliver this as one isolated, reviewable step in the Cellerator-owned joint-compiler thin-waist interfaces workstream.

**Biological motivation.** Typed biological axes, exact relation coverage, persistent order, and numerical partial semantics must cross the compiler boundary without being flattened to anonymous tensors.

**Compiler-architectural reason.** Cellerator owns the generic contracts that state what local biological computation means. CellShard may consume them deeply, but standalone Cellerator must remain fully usable without CellShard types or callbacks.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/execution/joint_compiler/
- [proposed] include/Cellerator/compute/decomposition/
- [proposed] include/Cellerator/profiling/joint_compiler/
- [proposed] src/execution/joint_compiler/
- [proposed] tests/jbc/interfaces/

**Existing code and permitted read scope:**

- include/Cellerator/execution/identity.hh
- include/Cellerator/execution/biological_abi.hh
- include/Cellerator/execution/lifetimes.hh
- include/Cellerator/execution/index_space/hierarchical_index_space_v1.hh
- include/Cellerator/compute/operation/operation_core_v2/schema.hh
- include/Cellerator/profiling/partition_export.h

**Explicitly out of scope / forbidden shortcuts:**

- No CellShard include may enter a required Cellerator public-core header.
- Do not mutate operation_core_v2, CSG1, CPE2, or program_v2 merely to add the new contracts; add adjacent versions or wrappers first.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-I10

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.

### Concrete mechanism

Enumerate canonical, evidence, semantic atom/basis, target-cover, physical projection, packed operand/value, executable recipe, topology-linked, and resident stages; define compatibility, validation, fallback, and phases-bypassed records.

Workstream mechanism: Use explicit standard-layout C++17 records with schema_version/record_bytes where evolution is expected. Use namespace-qualified 128-bit semantic identities, pointer-plus-count arrays, caller-owned buffers, and source-linked provider tables. Persist provider IDs and schemas, never function pointers.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Static-assert trivial-copy/standard-layout properties and add cross-version rejection tests.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-INTERFACES; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-I12 — Extend generic execution export to atom-aware v2

**Repository / subsystem / lane:** Cellerator · `execution/identity, coverage, decomposition, fragment ABI` · `CE-JBC-L-INTERFACES`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-INTERFACES; parallel with other provider/workstream lanes after JBC-G0-LIVE-BASELINE.

### Why

**Purpose.** Extend generic execution export to atom-aware v2. Deliver this as one isolated, reviewable step in the Cellerator-owned joint-compiler thin-waist interfaces workstream.

**Biological motivation.** Typed biological axes, exact relation coverage, persistent order, and numerical partial semantics must cross the compiler boundary without being flattened to anonymous tensors.

**Compiler-architectural reason.** Cellerator owns the generic contracts that state what local biological computation means. CellShard may consume them deeply, but standalone Cellerator must remain fully usable without CellShard types or callbacks.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/execution/joint_compiler/
- [proposed] include/Cellerator/compute/decomposition/
- [proposed] include/Cellerator/profiling/joint_compiler/
- [proposed] src/execution/joint_compiler/
- [proposed] tests/jbc/interfaces/

**Existing code and permitted read scope:**

- include/Cellerator/execution/identity.hh
- include/Cellerator/execution/biological_abi.hh
- include/Cellerator/execution/lifetimes.hh
- include/Cellerator/execution/index_space/hierarchical_index_space_v1.hh
- include/Cellerator/compute/operation/operation_core_v2/schema.hh
- include/Cellerator/profiling/partition_export.h

**Explicitly out of scope / forbidden shortcuts:**

- No CellShard include may enter a required Cellerator public-core header.
- Do not mutate operation_core_v2, CSG1, CPE2, or program_v2 merely to add the new contracts; add adjacent versions or wrappers first.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-I11

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A bounded implementation unit, focused tests, mechanism statistics, and an integration receipt.

### Concrete mechanism

Add exact coverage, decomposition, atom requirements/affordances, partial algebra, persistent orders, local candidate frontier, stage graph, complete cost, correctness compatibility, and performance freshness to the existing partition_export concept while retaining v1 compatibility.

Workstream mechanism: Use explicit standard-layout C++17 records with schema_version/record_bytes where evolution is expected. Use namespace-qualified 128-bit semantic identities, pointer-plus-count arrays, caller-owned buffers, and source-linked provider tables. Persist provider IDs and schemas, never function pointers.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Static-assert trivial-copy/standard-layout properties and add cross-version rejection tests.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-INTERFACES; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


# Workstream CEDEC: Biological operation decomposition and partial algebra

**Repository:** Cellerator  
**Suggested lane:** `CE-JBC-L-DECOMPOSITION`  
**Barrier:** `JBC-G1-ATOM-THIN-WAIST`

## Workstream design

Represent each decomposition as a data-only alternative naming split axis, exact input coverage, output or partial coverage, replication/halo requirements, persistent-order constraints, partial algebra, numerical/determinism consequences, and candidate family. Every operation keeps a complete unsplit fallback.

## Existing live source extended

- include/Cellerator/compute/operation/operation_core_v2/schema.hh
- include/Cellerator/compute/operation/relation_algebra_v2/
- include/Cellerator/geometry/relation_cover.hh
- include/Cellerator/execution/index_space/hierarchical_index_space_v1.hh

## Likely source scope

- [proposed] include/Cellerator/compute/decomposition/
- [proposed] src/compute/decomposition/
- [proposed] tests/jbc/decomposition/

## Proposed Todos (18)

## CE-JBC-D01 — Define split-axis and decomposition vocabulary

**Repository / subsystem / lane:** Cellerator · `compute/decomposition` · `CE-JBC-L-DECOMPOSITION`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-DECOMPOSITION; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Define split-axis and decomposition vocabulary. Deliver this as one isolated, reviewable step in the Biological operation decomposition and partial algebra workstream.

**Biological motivation.** Biological relations permit meaningful split dimensions—destination programs, source regulators, support edges, segments, relation types, and trajectories—that a generic scheduler cannot infer safely.

**Compiler-architectural reason.** CellShard may choose global decomposition only from alternatives whose mathematical legality and partial-result reconstruction are declared by Cellerator or another operation provider.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/compute/decomposition/
- [proposed] src/compute/decomposition/
- [proposed] tests/jbc/decomposition/

**Existing code and permitted read scope:**

- include/Cellerator/compute/operation/operation_core_v2/schema.hh
- include/Cellerator/compute/operation/relation_algebra_v2/
- include/Cellerator/geometry/relation_cover.hh
- include/Cellerator/execution/index_space/hierarchical_index_space_v1.hh

**Explicitly out of scope / forbidden shortcuts:**

- Do not add distributed placement or topology policy to Cellerator.
- Do not infer associative or commutative semantics from a kernel name or destination_update flag.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-I12

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.

### Concrete mechanism

Create stable split-axis identities for destination, source/reduction K, dense width, logical edge, segment, relation bundle/type, work item, sequence/provider-defined axes, and unsplit fallback. Keep operation kinds unchanged.

Workstream mechanism: Represent each decomposition as a data-only alternative naming split axis, exact input coverage, output or partial coverage, replication/halo requirements, persistent-order constraints, partial algebra, numerical/determinism consequences, and candidate family. Every operation keeps a complete unsplit fallback.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- For every split, reconstruct the unsplit CPU/GPU reference and test empty, singleton, tail, duplicate, and nonfinite cases.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-DECOMPOSITION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-D02 — Implement destination-disjoint relation-apply decomposition

**Repository / subsystem / lane:** Cellerator · `compute/decomposition` · `CE-JBC-L-DECOMPOSITION`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-DECOMPOSITION; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Implement destination-disjoint relation-apply decomposition. Deliver this as one isolated, reviewable step in the Biological operation decomposition and partial algebra workstream.

**Biological motivation.** Biological relations permit meaningful split dimensions—destination programs, source regulators, support edges, segments, relation types, and trajectories—that a generic scheduler cannot infer safely.

**Compiler-architectural reason.** CellShard may choose global decomposition only from alternatives whose mathematical legality and partial-result reconstruction are declared by Cellerator or another operation provider.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/compute/decomposition/
- [proposed] src/compute/decomposition/
- [proposed] tests/jbc/decomposition/

**Existing code and permitted read scope:**

- include/Cellerator/compute/operation/operation_core_v2/schema.hh
- include/Cellerator/compute/operation/relation_algebra_v2/
- include/Cellerator/geometry/relation_cover.hh
- include/Cellerator/execution/index_space/hierarchical_index_space_v1.hh

**Explicitly out of scope / forbidden shortcuts:**

- Do not add distributed placement or topology policy to Cellerator.
- Do not infer associative or commutative semantics from a kernel name or destination_update flag.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-D01

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

For Y=AX, partition destination rows/semantic destination atoms. Each fragment reads its exact relation edges and required source state, owns disjoint Y coverage, and requires no global combine.

Workstream mechanism: Represent each decomposition as a data-only alternative naming split axis, exact input coverage, output or partial coverage, replication/halo requirements, persistent-order constraints, partial algebra, numerical/determinism consequences, and candidate family. Every operation keeps a complete unsplit fallback.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- For every split, reconstruct the unsplit CPU/GPU reference and test empty, singleton, tail, duplicate, and nonfinite cases.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-DECOMPOSITION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-D03 — Implement source/K relation-apply decomposition

**Repository / subsystem / lane:** Cellerator · `compute/decomposition` · `CE-JBC-L-DECOMPOSITION`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-DECOMPOSITION; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Implement source/K relation-apply decomposition. Deliver this as one isolated, reviewable step in the Biological operation decomposition and partial algebra workstream.

**Biological motivation.** Biological relations permit meaningful split dimensions—destination programs, source regulators, support edges, segments, relation types, and trajectories—that a generic scheduler cannot infer safely.

**Compiler-architectural reason.** CellShard may choose global decomposition only from alternatives whose mathematical legality and partial-result reconstruction are declared by Cellerator or another operation provider.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/compute/decomposition/
- [proposed] src/compute/decomposition/
- [proposed] tests/jbc/decomposition/

**Existing code and permitted read scope:**

- include/Cellerator/compute/operation/operation_core_v2/schema.hh
- include/Cellerator/compute/operation/relation_algebra_v2/
- include/Cellerator/geometry/relation_cover.hh
- include/Cellerator/execution/index_space/hierarchical_index_space_v1.hh

**Explicitly out of scope / forbidden shortcuts:**

- Do not add distributed placement or topology policy to Cellerator.
- Do not infer associative or commutative semantics from a kernel name or destination_update flag.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-D02

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Partition source entities or exact edge contributions while preserving destination coverage. Emit additive partial Y atoms with an explicit sum algebra and record source-state replication versus reduction costs.

Workstream mechanism: Represent each decomposition as a data-only alternative naming split axis, exact input coverage, output or partial coverage, replication/halo requirements, persistent-order constraints, partial algebra, numerical/determinism consequences, and candidate family. Every operation keeps a complete unsplit fallback.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- For every split, reconstruct the unsplit CPU/GPU reference and test empty, singleton, tail, duplicate, and nonfinite cases.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-DECOMPOSITION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-D04 — Implement dense-width relation-apply decomposition

**Repository / subsystem / lane:** Cellerator · `compute/decomposition` · `CE-JBC-L-DECOMPOSITION`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-DECOMPOSITION; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Implement dense-width relation-apply decomposition. Deliver this as one isolated, reviewable step in the Biological operation decomposition and partial algebra workstream.

**Biological motivation.** Biological relations permit meaningful split dimensions—destination programs, source regulators, support edges, segments, relation types, and trajectories—that a generic scheduler cannot infer safely.

**Compiler-architectural reason.** CellShard may choose global decomposition only from alternatives whose mathematical legality and partial-result reconstruction are declared by Cellerator or another operation provider.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/compute/decomposition/
- [proposed] src/compute/decomposition/
- [proposed] tests/jbc/decomposition/

**Existing code and permitted read scope:**

- include/Cellerator/compute/operation/operation_core_v2/schema.hh
- include/Cellerator/compute/operation/relation_algebra_v2/
- include/Cellerator/geometry/relation_cover.hh
- include/Cellerator/execution/index_space/hierarchical_index_space_v1.hh

**Explicitly out of scope / forbidden shortcuts:**

- Do not add distributed placement or topology policy to Cellerator.
- Do not infer associative or commutative semantics from a kernel name or destination_update flag.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-D03

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Partition the dense output N axis into disjoint panels. Each fragment shares relation structure but owns nonoverlapping output columns and requires no combine; record relation replication and panel order.

Workstream mechanism: Represent each decomposition as a data-only alternative naming split axis, exact input coverage, output or partial coverage, replication/halo requirements, persistent-order constraints, partial algebra, numerical/determinism consequences, and candidate family. Every operation keeps a complete unsplit fallback.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- For every split, reconstruct the unsplit CPU/GPU reference and test empty, singleton, tail, duplicate, and nonfinite cases.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-DECOMPOSITION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-D05 — Implement edge/component relation-apply decomposition

**Repository / subsystem / lane:** Cellerator · `compute/decomposition` · `CE-JBC-L-DECOMPOSITION`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-DECOMPOSITION; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Implement edge/component relation-apply decomposition. Deliver this as one isolated, reviewable step in the Biological operation decomposition and partial algebra workstream.

**Biological motivation.** Biological relations permit meaningful split dimensions—destination programs, source regulators, support edges, segments, relation types, and trajectories—that a generic scheduler cannot infer safely.

**Compiler-architectural reason.** CellShard may choose global decomposition only from alternatives whose mathematical legality and partial-result reconstruction are declared by Cellerator or another operation provider.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/compute/decomposition/
- [proposed] src/compute/decomposition/
- [proposed] tests/jbc/decomposition/

**Existing code and permitted read scope:**

- include/Cellerator/compute/operation/operation_core_v2/schema.hh
- include/Cellerator/compute/operation/relation_algebra_v2/
- include/Cellerator/geometry/relation_cover.hh
- include/Cellerator/execution/index_space/hierarchical_index_space_v1.hh

**Explicitly out of scope / forbidden shortcuts:**

- Do not add distributed placement or topology policy to Cellerator.
- Do not infer associative or commutative semantics from a kernel name or destination_update flag.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-D04

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Partition by exact logical edge blocks or CSG1 semantic components. Produce additive destination partials, preserve logical edge identities, and expose residual fragmentation and destination-owner requirements.

Workstream mechanism: Represent each decomposition as a data-only alternative naming split axis, exact input coverage, output or partial coverage, replication/halo requirements, persistent-order constraints, partial algebra, numerical/determinism consequences, and candidate family. Every operation keeps a complete unsplit fallback.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- For every split, reconstruct the unsplit CPU/GPU reference and test empty, singleton, tail, duplicate, and nonfinite cases.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-DECOMPOSITION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-D06 — Implement relation-bundle/type decomposition

**Repository / subsystem / lane:** Cellerator · `compute/decomposition` · `CE-JBC-L-DECOMPOSITION`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-DECOMPOSITION; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Implement relation-bundle/type decomposition. Deliver this as one isolated, reviewable step in the Biological operation decomposition and partial algebra workstream.

**Biological motivation.** Biological relations permit meaningful split dimensions—destination programs, source regulators, support edges, segments, relation types, and trajectories—that a generic scheduler cannot infer safely.

**Compiler-architectural reason.** CellShard may choose global decomposition only from alternatives whose mathematical legality and partial-result reconstruction are declared by Cellerator or another operation provider.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/compute/decomposition/
- [proposed] src/compute/decomposition/
- [proposed] tests/jbc/decomposition/

**Existing code and permitted read scope:**

- include/Cellerator/compute/operation/operation_core_v2/schema.hh
- include/Cellerator/compute/operation/relation_algebra_v2/
- include/Cellerator/geometry/relation_cover.hh
- include/Cellerator/execution/index_space/hierarchical_index_space_v1.hh

**Explicitly out of scope / forbidden shortcuts:**

- Do not add distributed placement or topology policy to Cellerator.
- Do not infer associative or commutative semantics from a kernel name or destination_update flag.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-D05

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Split a relation_bundle_apply by typed relation or modality while retaining one destination axis. Declare additive/affine combine semantics and shared-destination physical-order opportunities.

Workstream mechanism: Represent each decomposition as a data-only alternative naming split axis, exact input coverage, output or partial coverage, replication/halo requirements, persistent-order constraints, partial algebra, numerical/determinism consequences, and candidate family. Every operation keeps a complete unsplit fallback.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- For every split, reconstruct the unsplit CPU/GPU reference and test empty, singleton, tail, duplicate, and nonfinite cases.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-DECOMPOSITION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-D07 — Implement transpose decomposition and source partials

**Repository / subsystem / lane:** Cellerator · `compute/decomposition` · `CE-JBC-L-DECOMPOSITION`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-DECOMPOSITION; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Implement transpose decomposition and source partials. Deliver this as one isolated, reviewable step in the Biological operation decomposition and partial algebra workstream.

**Biological motivation.** Biological relations permit meaningful split dimensions—destination programs, source regulators, support edges, segments, relation types, and trajectories—that a generic scheduler cannot infer safely.

**Compiler-architectural reason.** CellShard may choose global decomposition only from alternatives whose mathematical legality and partial-result reconstruction are declared by Cellerator or another operation provider.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/compute/decomposition/
- [proposed] src/compute/decomposition/
- [proposed] tests/jbc/decomposition/

**Existing code and permitted read scope:**

- include/Cellerator/compute/operation/operation_core_v2/schema.hh
- include/Cellerator/compute/operation/relation_algebra_v2/
- include/Cellerator/geometry/relation_cover.hh
- include/Cellerator/execution/index_space/hierarchical_index_space_v1.hh

**Explicitly out of scope / forbidden shortcuts:**

- Do not add distributed placement or topology policy to Cellerator.
- Do not infer associative or commutative semantics from a kernel name or destination_update flag.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-D06

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

For A^T Y, provide source-disjoint alternatives and forward-destination partitions that emit additive source partials. Do not assume forward geometry or placement is optimal for transpose.

Workstream mechanism: Represent each decomposition as a data-only alternative naming split axis, exact input coverage, output or partial coverage, replication/halo requirements, persistent-order constraints, partial algebra, numerical/determinism consequences, and candidate family. Every operation keeps a complete unsplit fallback.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- For every split, reconstruct the unsplit CPU/GPU reference and test empty, singleton, tail, duplicate, and nonfinite cases.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-DECOMPOSITION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-D08 — Implement support-contraction destination/source decomposition

**Repository / subsystem / lane:** Cellerator · `compute/decomposition` · `CE-JBC-L-DECOMPOSITION`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-DECOMPOSITION; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Implement support-contraction destination/source decomposition. Deliver this as one isolated, reviewable step in the Biological operation decomposition and partial algebra workstream.

**Biological motivation.** Biological relations permit meaningful split dimensions—destination programs, source regulators, support edges, segments, relation types, and trajectories—that a generic scheduler cannot infer safely.

**Compiler-architectural reason.** CellShard may choose global decomposition only from alternatives whose mathematical legality and partial-result reconstruction are declared by Cellerator or another operation provider.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/compute/decomposition/
- [proposed] src/compute/decomposition/
- [proposed] tests/jbc/decomposition/

**Existing code and permitted read scope:**

- include/Cellerator/compute/operation/operation_core_v2/schema.hh
- include/Cellerator/compute/operation/relation_algebra_v2/
- include/Cellerator/geometry/relation_cover.hh
- include/Cellerator/execution/index_space/hierarchical_index_space_v1.hh

**Explicitly out of scope / forbidden shortcuts:**

- Do not add distributed placement or topology policy to Cellerator.
- Do not infer associative or commutative semantics from a kernel name or destination_update flag.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-D07

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.
- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Partition Q/K state and exact support by destination or source groups. Record which state plane must be replicated and whether outputs are disjoint edge sets.

Workstream mechanism: Represent each decomposition as a data-only alternative naming split axis, exact input coverage, output or partial coverage, replication/halo requirements, persistent-order constraints, partial algebra, numerical/determinism consequences, and candidate family. Every operation keeps a complete unsplit fallback.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- For every split, reconstruct the unsplit CPU/GPU reference and test empty, singleton, tail, duplicate, and nonfinite cases.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-DECOMPOSITION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-D09 — Implement support-contraction edge/rectangle decomposition

**Repository / subsystem / lane:** Cellerator · `compute/decomposition` · `CE-JBC-L-DECOMPOSITION`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-DECOMPOSITION; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Implement support-contraction edge/rectangle decomposition. Deliver this as one isolated, reviewable step in the Biological operation decomposition and partial algebra workstream.

**Biological motivation.** Biological relations permit meaningful split dimensions—destination programs, source regulators, support edges, segments, relation types, and trajectories—that a generic scheduler cannot infer safely.

**Compiler-architectural reason.** CellShard may choose global decomposition only from alternatives whose mathematical legality and partial-result reconstruction are declared by Cellerator or another operation provider.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/compute/decomposition/
- [proposed] src/compute/decomposition/
- [proposed] tests/jbc/decomposition/

**Existing code and permitted read scope:**

- include/Cellerator/compute/operation/operation_core_v2/schema.hh
- include/Cellerator/compute/operation/relation_algebra_v2/
- include/Cellerator/geometry/relation_cover.hh
- include/Cellerator/execution/index_space/hierarchical_index_space_v1.hh

**Explicitly out of scope / forbidden shortcuts:**

- Do not add distributed placement or topology policy to Cellerator.
- Do not infer associative or commutative semantics from a kernel name or destination_update flag.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-D08

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.
- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Partition contract_on_support by exact edge blocks or certified source×destination rectangles; output exact logical-edge scores without overlap unless explicitly combined.

Workstream mechanism: Represent each decomposition as a data-only alternative naming split axis, exact input coverage, output or partial coverage, replication/halo requirements, persistent-order constraints, partial algebra, numerical/determinism consequences, and candidate family. Every operation keeps a complete unsplit fallback.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- For every split, reconstruct the unsplit CPU/GPU reference and test empty, singleton, tail, duplicate, and nonfinite cases.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-DECOMPOSITION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-D10 — Implement support-contraction embedding-dimension decomposition

**Repository / subsystem / lane:** Cellerator · `compute/decomposition` · `CE-JBC-L-DECOMPOSITION`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-DECOMPOSITION; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Implement support-contraction embedding-dimension decomposition. Deliver this as one isolated, reviewable step in the Biological operation decomposition and partial algebra workstream.

**Biological motivation.** Biological relations permit meaningful split dimensions—destination programs, source regulators, support edges, segments, relation types, and trajectories—that a generic scheduler cannot infer safely.

**Compiler-architectural reason.** CellShard may choose global decomposition only from alternatives whose mathematical legality and partial-result reconstruction are declared by Cellerator or another operation provider.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/compute/decomposition/
- [proposed] src/compute/decomposition/
- [proposed] tests/jbc/decomposition/

**Existing code and permitted read scope:**

- include/Cellerator/compute/operation/operation_core_v2/schema.hh
- include/Cellerator/compute/operation/relation_algebra_v2/
- include/Cellerator/geometry/relation_cover.hh
- include/Cellerator/execution/index_space/hierarchical_index_space_v1.hh

**Explicitly out of scope / forbidden shortcuts:**

- Do not add distributed placement or topology policy to Cellerator.
- Do not infer associative or commutative semantics from a kernel name or destination_update flag.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-D09

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.
- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Split dot-product dimension D into partial dot products keyed by exact edge IDs. Emit additive partial-edge states with explicit accumulation precision and tree constraints.

Workstream mechanism: Represent each decomposition as a data-only alternative naming split axis, exact input coverage, output or partial coverage, replication/halo requirements, persistent-order constraints, partial algebra, numerical/determinism consequences, and candidate family. Every operation keeps a complete unsplit fallback.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- For every split, reconstruct the unsplit CPU/GPU reference and test empty, singleton, tail, duplicate, and nonfinite cases.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-DECOMPOSITION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-D11 — Implement segment-disjoint decomposition

**Repository / subsystem / lane:** Cellerator · `compute/decomposition` · `CE-JBC-L-DECOMPOSITION`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-DECOMPOSITION; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Implement segment-disjoint decomposition. Deliver this as one isolated, reviewable step in the Biological operation decomposition and partial algebra workstream.

**Biological motivation.** Biological relations permit meaningful split dimensions—destination programs, source regulators, support edges, segments, relation types, and trajectories—that a generic scheduler cannot infer safely.

**Compiler-architectural reason.** CellShard may choose global decomposition only from alternatives whose mathematical legality and partial-result reconstruction are declared by Cellerator or another operation provider.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/compute/decomposition/
- [proposed] src/compute/decomposition/
- [proposed] tests/jbc/decomposition/

**Existing code and permitted read scope:**

- include/Cellerator/compute/operation/operation_core_v2/schema.hh
- include/Cellerator/compute/operation/relation_algebra_v2/
- include/Cellerator/geometry/relation_cover.hh
- include/Cellerator/execution/index_space/hierarchical_index_space_v1.hh

**Explicitly out of scope / forbidden shortcuts:**

- Do not add distributed placement or topology policy to Cellerator.
- Do not infer associative or commutative semantics from a kernel name or destination_update flag.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-D10

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Assign whole segments to fragments so each output segment is independently owned; preserve empty/singleton semantics and segment-order identity.

Workstream mechanism: Represent each decomposition as a data-only alternative naming split axis, exact input coverage, output or partial coverage, replication/halo requirements, persistent-order constraints, partial algebra, numerical/determinism consequences, and candidate family. Every operation keeps a complete unsplit fallback.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- For every split, reconstruct the unsplit CPU/GPU reference and test empty, singleton, tail, duplicate, and nonfinite cases.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-DECOMPOSITION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-D12 — Implement split-segment reduction decomposition

**Repository / subsystem / lane:** Cellerator · `compute/decomposition` · `CE-JBC-L-DECOMPOSITION`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-DECOMPOSITION; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Implement split-segment reduction decomposition. Deliver this as one isolated, reviewable step in the Biological operation decomposition and partial algebra workstream.

**Biological motivation.** Biological relations permit meaningful split dimensions—destination programs, source regulators, support edges, segments, relation types, and trajectories—that a generic scheduler cannot infer safely.

**Compiler-architectural reason.** CellShard may choose global decomposition only from alternatives whose mathematical legality and partial-result reconstruction are declared by Cellerator or another operation provider.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/compute/decomposition/
- [proposed] src/compute/decomposition/
- [proposed] tests/jbc/decomposition/

**Existing code and permitted read scope:**

- include/Cellerator/compute/operation/operation_core_v2/schema.hh
- include/Cellerator/compute/operation/relation_algebra_v2/
- include/Cellerator/geometry/relation_cover.hh
- include/Cellerator/execution/index_space/hierarchical_index_space_v1.hh

**Explicitly out of scope / forbidden shortcuts:**

- Do not add distributed placement or topology policy to Cellerator.
- Do not infer associative or commutative semantics from a kernel name or destination_update flag.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-D11

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Allow one segment to span fragments and emit mergeable partial states. Declare the required algebra, finalization, deterministic tree, and exact member coverage.

Workstream mechanism: Represent each decomposition as a data-only alternative naming split axis, exact input coverage, output or partial coverage, replication/halo requirements, persistent-order constraints, partial algebra, numerical/determinism consequences, and candidate family. Every operation keeps a complete unsplit fallback.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- For every split, reconstruct the unsplit CPU/GPU reference and test empty, singleton, tail, duplicate, and nonfinite cases.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-DECOMPOSITION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-D13 — Implement mergeable log-sum-exp and softmax state

**Repository / subsystem / lane:** Cellerator · `compute/decomposition` · `CE-JBC-L-DECOMPOSITION`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-DECOMPOSITION; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Implement mergeable log-sum-exp and softmax state. Deliver this as one isolated, reviewable step in the Biological operation decomposition and partial algebra workstream.

**Biological motivation.** Biological relations permit meaningful split dimensions—destination programs, source regulators, support edges, segments, relation types, and trajectories—that a generic scheduler cannot infer safely.

**Compiler-architectural reason.** CellShard may choose global decomposition only from alternatives whose mathematical legality and partial-result reconstruction are declared by Cellerator or another operation provider.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/compute/decomposition/
- [proposed] src/compute/decomposition/
- [proposed] tests/jbc/decomposition/

**Existing code and permitted read scope:**

- include/Cellerator/compute/operation/operation_core_v2/schema.hh
- include/Cellerator/compute/operation/relation_algebra_v2/
- include/Cellerator/geometry/relation_cover.hh
- include/Cellerator/execution/index_space/hierarchical_index_space_v1.hh

**Explicitly out of scope / forbidden shortcuts:**

- Do not add distributed placement or topology policy to Cellerator.
- Do not infer associative or commutative semantics from a kernel name or destination_update flag.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-D12

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Define state (m,s), merge m=max(m1,m2), s=s1*exp(m1-m)+s2*exp(m2-m), and finalization. Specify empty/singleton/nonfinite and FP32/FP64 reference behavior.

Workstream mechanism: Represent each decomposition as a data-only alternative naming split axis, exact input coverage, output or partial coverage, replication/halo requirements, persistent-order constraints, partial algebra, numerical/determinism consequences, and candidate family. Every operation keeps a complete unsplit fallback.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- For every split, reconstruct the unsplit CPU/GPU reference and test empty, singleton, tail, duplicate, and nonfinite cases.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-DECOMPOSITION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-D14 — Implement mergeable moments state

**Repository / subsystem / lane:** Cellerator · `compute/decomposition` · `CE-JBC-L-DECOMPOSITION`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-DECOMPOSITION; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Implement mergeable moments state. Deliver this as one isolated, reviewable step in the Biological operation decomposition and partial algebra workstream.

**Biological motivation.** Biological relations permit meaningful split dimensions—destination programs, source regulators, support edges, segments, relation types, and trajectories—that a generic scheduler cannot infer safely.

**Compiler-architectural reason.** CellShard may choose global decomposition only from alternatives whose mathematical legality and partial-result reconstruction are declared by Cellerator or another operation provider.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/compute/decomposition/
- [proposed] src/compute/decomposition/
- [proposed] tests/jbc/decomposition/

**Existing code and permitted read scope:**

- include/Cellerator/compute/operation/operation_core_v2/schema.hh
- include/Cellerator/compute/operation/relation_algebra_v2/
- include/Cellerator/geometry/relation_cover.hh
- include/Cellerator/execution/index_space/hierarchical_index_space_v1.hh

**Explicitly out of scope / forbidden shortcuts:**

- Do not add distributed placement or topology policy to Cellerator.
- Do not infer associative or commutative semantics from a kernel name or destination_update flag.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-D13

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Define count, weighted count where needed, sum, sum-of-squares or stable pairwise moments, merge/finalize, and variance-ready outputs under explicit numerical policy.

Workstream mechanism: Represent each decomposition as a data-only alternative naming split axis, exact input coverage, output or partial coverage, replication/halo requirements, persistent-order constraints, partial algebra, numerical/determinism consequences, and candidate family. Every operation keeps a complete unsplit fallback.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- For every split, reconstruct the unsplit CPU/GPU reference and test empty, singleton, tail, duplicate, and nonfinite cases.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-DECOMPOSITION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-D15 — Define gate-dependent input and replication semantics

**Repository / subsystem / lane:** Cellerator · `compute/decomposition` · `CE-JBC-L-DECOMPOSITION`  
**Classification:** validation/evidence  
**Parallelism:** Serial within CE-JBC-L-DECOMPOSITION; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Define gate-dependent input and replication semantics. Deliver this as one isolated, reviewable step in the Biological operation decomposition and partial algebra workstream.

**Biological motivation.** Biological relations permit meaningful split dimensions—destination programs, source regulators, support edges, segments, relation types, and trajectories—that a generic scheduler cannot infer safely.

**Compiler-architectural reason.** CellShard may choose global decomposition only from alternatives whose mathematical legality and partial-result reconstruction are declared by Cellerator or another operation provider.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/compute/decomposition/
- [proposed] src/compute/decomposition/
- [proposed] tests/jbc/decomposition/

**Existing code and permitted read scope:**

- include/Cellerator/compute/operation/operation_core_v2/schema.hh
- include/Cellerator/compute/operation/relation_algebra_v2/
- include/Cellerator/geometry/relation_cover.hh
- include/Cellerator/execution/index_space/hierarchical_index_space_v1.hh

**Explicitly out of scope / forbidden shortcuts:**

- Do not add distributed placement or topology policy to Cellerator.
- Do not infer associative or commutative semantics from a kernel name or destination_update flag.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-D14

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.
- A machine-readable evidence record containing the baseline, candidate, complete-cost metrics, environment identity, and promotion disposition.

### Concrete mechanism

For per-edge/source/destination/component/factorized/predicate gates, state exact additional atom coverages, legal replication, value-generation dependencies, and whether output remains edge-disjoint.

Workstream mechanism: Represent each decomposition as a data-only alternative naming split axis, exact input coverage, output or partial coverage, replication/halo requirements, persistent-order constraints, partial algebra, numerical/determinism consequences, and candidate family. Every operation keeps a complete unsplit fallback.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- For every split, reconstruct the unsplit CPU/GPU reference and test empty, singleton, tail, duplicate, and nonfinite cases.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-DECOMPOSITION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-D16 — Define sparse-update conflict and ordering algebra

**Repository / subsystem / lane:** Cellerator · `compute/decomposition` · `CE-JBC-L-DECOMPOSITION`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-DECOMPOSITION; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Define sparse-update conflict and ordering algebra. Deliver this as one isolated, reviewable step in the Biological operation decomposition and partial algebra workstream.

**Biological motivation.** Biological relations permit meaningful split dimensions—destination programs, source regulators, support edges, segments, relation types, and trajectories—that a generic scheduler cannot infer safely.

**Compiler-architectural reason.** CellShard may choose global decomposition only from alternatives whose mathematical legality and partial-result reconstruction are declared by Cellerator or another operation provider.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/compute/decomposition/
- [proposed] src/compute/decomposition/
- [proposed] tests/jbc/decomposition/

**Existing code and permitted read scope:**

- include/Cellerator/compute/operation/operation_core_v2/schema.hh
- include/Cellerator/compute/operation/relation_algebra_v2/
- include/Cellerator/geometry/relation_cover.hh
- include/Cellerator/execution/index_space/hierarchical_index_space_v1.hh

**Explicitly out of scope / forbidden shortcuts:**

- Do not add distributed placement or topology policy to Cellerator.
- Do not infer associative or commutative semantics from a kernel name or destination_update flag.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-D15

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.

### Concrete mechanism

Classify unique, additive, maximum/minimum, ordered assign, and noncommutative updates. Export legal reorder/parallelization, identity element, and deterministic serialization requirements.

Workstream mechanism: Represent each decomposition as a data-only alternative naming split axis, exact input coverage, output or partial coverage, replication/halo requirements, persistent-order constraints, partial algebra, numerical/determinism consequences, and candidate family. Every operation keeps a complete unsplit fallback.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- For every split, reconstruct the unsplit CPU/GPU reference and test empty, singleton, tail, duplicate, and nonfinite cases.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-DECOMPOSITION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-D17 — Define training and gradient decomposition contracts

**Repository / subsystem / lane:** Cellerator · `compute/decomposition` · `CE-JBC-L-DECOMPOSITION`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-DECOMPOSITION; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Define training and gradient decomposition contracts. Deliver this as one isolated, reviewable step in the Biological operation decomposition and partial algebra workstream.

**Biological motivation.** Biological relations permit meaningful split dimensions—destination programs, source regulators, support edges, segments, relation types, and trajectories—that a generic scheduler cannot infer safely.

**Compiler-architectural reason.** CellShard may choose global decomposition only from alternatives whose mathematical legality and partial-result reconstruction are declared by Cellerator or another operation provider.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/compute/decomposition/
- [proposed] src/compute/decomposition/
- [proposed] tests/jbc/decomposition/

**Existing code and permitted read scope:**

- include/Cellerator/compute/operation/operation_core_v2/schema.hh
- include/Cellerator/compute/operation/relation_algebra_v2/
- include/Cellerator/geometry/relation_cover.hh
- include/Cellerator/execution/index_space/hierarchical_index_space_v1.hh

**Explicitly out of scope / forbidden shortcuts:**

- Do not add distributed placement or topology policy to Cellerator.
- Do not infer associative or commutative semantics from a kernel name or destination_update flag.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-D16

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.

### Concrete mechanism

Describe source gradients, destination gradients, logical-edge gradients, projection-order gradients, sparse updates, and value-generation publication without embedding optimizer policy.

Workstream mechanism: Represent each decomposition as a data-only alternative naming split axis, exact input coverage, output or partial coverage, replication/halo requirements, persistent-order constraints, partial algebra, numerical/determinism consequences, and candidate family. Every operation keeps a complete unsplit fallback.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- For every split, reconstruct the unsplit CPU/GPU reference and test empty, singleton, tail, duplicate, and nonfinite cases.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-DECOMPOSITION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-D18 — Register and independently validate decomposition providers

**Repository / subsystem / lane:** Cellerator · `compute/decomposition` · `CE-JBC-L-DECOMPOSITION`  
**Classification:** validation/evidence  
**Parallelism:** Serial within CE-JBC-L-DECOMPOSITION; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Register and independently validate decomposition providers. Deliver this as one isolated, reviewable step in the Biological operation decomposition and partial algebra workstream.

**Biological motivation.** Biological relations permit meaningful split dimensions—destination programs, source regulators, support edges, segments, relation types, and trajectories—that a generic scheduler cannot infer safely.

**Compiler-architectural reason.** CellShard may choose global decomposition only from alternatives whose mathematical legality and partial-result reconstruction are declared by Cellerator or another operation provider.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/compute/decomposition/
- [proposed] src/compute/decomposition/
- [proposed] tests/jbc/decomposition/

**Existing code and permitted read scope:**

- include/Cellerator/compute/operation/operation_core_v2/schema.hh
- include/Cellerator/compute/operation/relation_algebra_v2/
- include/Cellerator/geometry/relation_cover.hh
- include/Cellerator/execution/index_space/hierarchical_index_space_v1.hh

**Explicitly out of scope / forbidden shortcuts:**

- Do not add distributed placement or topology policy to Cellerator.
- Do not infer associative or commutative semantics from a kernel name or destination_update flag.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-D17

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- Focused tests and adversarial fixtures demonstrating both success and explicit rejection paths.

### Concrete mechanism

Assemble source-linked provider fragments; validate stable IDs, exact coverages, partial algebras, operation compatibility, and complete unsplit fallback. Avoid a central type switch.

Workstream mechanism: Represent each decomposition as a data-only alternative naming split axis, exact input coverage, output or partial coverage, replication/halo requirements, persistent-order constraints, partial algebra, numerical/determinism consequences, and candidate family. Every operation keeps a complete unsplit fallback.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- For every split, reconstruct the unsplit CPU/GPU reference and test empty, singleton, tail, duplicate, and nonfinite cases.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-DECOMPOSITION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


# Workstream CEFRAG: Atom-aware Cellerator fragment compiler

**Repository:** Cellerator  
**Suggested lane:** `CE-JBC-L-FRAGMENT`  
**Barrier:** `JBC-G1-ATOM-THIN-WAIST`

## Workstream design

Use a two-pass requirements/query API. Adapt operation_core_v2 into an atom-fragment request, build compact local index spaces, validate the external cover and order, discover local candidates, retain nondominated alternatives, and prepare the chosen program_v2 inside a prepared_atom_fragment wrapper. No hidden allocation occurs after requirements are returned.

## Existing live source extended

- include/Cellerator/compute/operation/candidate_catalog_v3/
- include/Cellerator/execution/program/program_v2.h
- include/Cellerator/execution/geometry_acquisition_v2/
- include/Cellerator/geometry/compiler/v2/
- include/Cellerator/planner/

## Likely source scope

- [proposed] include/Cellerator/execution/atom_fragment/
- [proposed] src/execution/atom_fragment/
- [proposed] tests/jbc/fragment/

## Proposed Todos (14)

## CE-JBC-F01 — Implement atom-fragment requirements query

**Repository / subsystem / lane:** Cellerator · `execution/atom_fragment` · `CE-JBC-L-FRAGMENT`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-FRAGMENT; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Implement atom-fragment requirements query. Deliver this as one isolated, reviewable step in the Atom-aware Cellerator fragment compiler workstream.

**Biological motivation.** A reusable CellShard atom is only useful if Cellerator can compile the exact biological coverage, order, mutable planes, and partial output that atom represents.

**Compiler-architectural reason.** The fragment compiler is the primary local lowering boundary. It wraps rather than replaces program_v2, validates externally proposed structure, and emits a bounded Pareto frontier rather than a single locally optimal answer.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/execution/atom_fragment/
- [proposed] src/execution/atom_fragment/
- [proposed] tests/jbc/fragment/

**Existing code and permitted read scope:**

- include/Cellerator/compute/operation/candidate_catalog_v3/
- include/Cellerator/execution/program/program_v2.h
- include/Cellerator/execution/geometry_acquisition_v2/
- include/Cellerator/geometry/compiler/v2/
- include/Cellerator/planner/

**Explicitly out of scope / forbidden shortcuts:**

- Do not create a second operation core or planner.
- Do not force a program_v3 before the wrapper is proven insufficient.
- Do not accept unvalidated CellShard decomposition as authoritative.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-D18

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Given a fragment request, return exact capacities and alignments for local indexes, projections, candidate workspace, prepared program, bindings, diagnostics, and transient storage before any allocation.

Workstream mechanism: Use a two-pass requirements/query API. Adapt operation_core_v2 into an atom-fragment request, build compact local index spaces, validate the external cover and order, discover local candidates, retain nondominated alternatives, and prepare the chosen program_v2 inside a prepared_atom_fragment wrapper. No hidden allocation occurs after requirements are returned.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Use an independent fragment verifier and a canonical relation-apply vertical smoke before distributed integration.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-FRAGMENT; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-F02 — Adapt operation_core_v2 problems to fragment compilation

**Repository / subsystem / lane:** Cellerator · `execution/atom_fragment` · `CE-JBC-L-FRAGMENT`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-FRAGMENT; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Adapt operation_core_v2 problems to fragment compilation. Deliver this as one isolated, reviewable step in the Atom-aware Cellerator fragment compiler workstream.

**Biological motivation.** A reusable CellShard atom is only useful if Cellerator can compile the exact biological coverage, order, mutable planes, and partial output that atom represents.

**Compiler-architectural reason.** The fragment compiler is the primary local lowering boundary. It wraps rather than replaces program_v2, validates externally proposed structure, and emits a bounded Pareto frontier rather than a single locally optimal answer.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/execution/atom_fragment/
- [proposed] src/execution/atom_fragment/
- [proposed] tests/jbc/fragment/

**Existing code and permitted read scope:**

- include/Cellerator/compute/operation/candidate_catalog_v3/
- include/Cellerator/execution/program/program_v2.h
- include/Cellerator/execution/geometry_acquisition_v2/
- include/Cellerator/geometry/compiler/v2/
- include/Cellerator/planner/

**Explicitly out of scope / forbidden shortcuts:**

- Do not create a second operation core or planner.
- Do not force a program_v3 before the wrapper is proven insufficient.
- Do not accept unvalidated CellShard decomposition as authoritative.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-F01

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Preserve operation/numerical/output/determinism semantics while restricting exact coverage and work items to the requested atom fragment. Reject incompatible axis or generation reductions.

Workstream mechanism: Use a two-pass requirements/query API. Adapt operation_core_v2 into an atom-fragment request, build compact local index spaces, validate the external cover and order, discover local candidates, retain nondominated alternatives, and prepare the chosen program_v2 inside a prepared_atom_fragment wrapper. No hidden allocation occurs after requirements are returned.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Use an independent fragment verifier and a canonical relation-apply vertical smoke before distributed integration.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-FRAGMENT; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-F03 — Build atom-local hierarchical index spaces

**Repository / subsystem / lane:** Cellerator · `execution/atom_fragment` · `CE-JBC-L-FRAGMENT`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-FRAGMENT; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Build atom-local hierarchical index spaces. Deliver this as one isolated, reviewable step in the Atom-aware Cellerator fragment compiler workstream.

**Biological motivation.** A reusable CellShard atom is only useful if Cellerator can compile the exact biological coverage, order, mutable planes, and partial output that atom represents.

**Compiler-architectural reason.** The fragment compiler is the primary local lowering boundary. It wraps rather than replaces program_v2, validates externally proposed structure, and emits a bounded Pareto frontier rather than a single locally optimal answer.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/execution/atom_fragment/
- [proposed] src/execution/atom_fragment/
- [proposed] tests/jbc/fragment/

**Existing code and permitted read scope:**

- include/Cellerator/compute/operation/candidate_catalog_v3/
- include/Cellerator/execution/program/program_v2.h
- include/Cellerator/execution/geometry_acquisition_v2/
- include/Cellerator/geometry/compiler/v2/
- include/Cellerator/planner/

**Explicitly out of scope / forbidden shortcuts:**

- Do not create a second operation core or planner.
- Do not force a program_v3 before the wrapper is proven insufficient.
- Do not accept unvalidated CellShard decomposition as authoritative.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-F02

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Map global 64-bit biological identities to smallest valid u16/u32/u64 local widths, preserve local-to-global sidecars, and split aggregate relations into bounded local components rather than rejecting large counts.

Workstream mechanism: Use a two-pass requirements/query API. Adapt operation_core_v2 into an atom-fragment request, build compact local index spaces, validate the external cover and order, discover local candidates, retain nondominated alternatives, and prepare the chosen program_v2 inside a prepared_atom_fragment wrapper. No hidden allocation occurs after requirements are returned.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Use an independent fragment verifier and a canonical relation-apply vertical smoke before distributed integration.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-FRAGMENT; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-F04 — Validate externally supplied exact decomposition

**Repository / subsystem / lane:** Cellerator · `execution/atom_fragment` · `CE-JBC-L-FRAGMENT`  
**Classification:** validation/evidence  
**Parallelism:** Serial within CE-JBC-L-FRAGMENT; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Validate externally supplied exact decomposition. Deliver this as one isolated, reviewable step in the Atom-aware Cellerator fragment compiler workstream.

**Biological motivation.** A reusable CellShard atom is only useful if Cellerator can compile the exact biological coverage, order, mutable planes, and partial output that atom represents.

**Compiler-architectural reason.** The fragment compiler is the primary local lowering boundary. It wraps rather than replaces program_v2, validates externally proposed structure, and emits a bounded Pareto frontier rather than a single locally optimal answer.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/execution/atom_fragment/
- [proposed] src/execution/atom_fragment/
- [proposed] tests/jbc/fragment/

**Existing code and permitted read scope:**

- include/Cellerator/compute/operation/candidate_catalog_v3/
- include/Cellerator/execution/program/program_v2.h
- include/Cellerator/execution/geometry_acquisition_v2/
- include/Cellerator/geometry/compiler/v2/
- include/Cellerator/planner/

**Explicitly out of scope / forbidden shortcuts:**

- Do not create a second operation core or planner.
- Do not force a program_v3 before the wrapper is proven insufficient.
- Do not accept unvalidated CellShard decomposition as authoritative.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-F03

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- Focused tests and adversarial fixtures demonstrating both success and explicit rejection paths.

### Concrete mechanism

Check every supplied local coverage against the parent operation, relation structure, edge spine, partial algebra, and output ownership before geometry or candidate search.

Workstream mechanism: Use a two-pass requirements/query API. Adapt operation_core_v2 into an atom-fragment request, build compact local index spaces, validate the external cover and order, discover local candidates, retain nondominated alternatives, and prepare the chosen program_v2 inside a prepared_atom_fragment wrapper. No hidden allocation occurs after requirements are returned.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Use an independent fragment verifier and a canonical relation-apply vertical smoke before distributed integration.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-FRAGMENT; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-F05 — Validate externally supplied persistent order

**Repository / subsystem / lane:** Cellerator · `execution/atom_fragment` · `CE-JBC-L-FRAGMENT`  
**Classification:** validation/evidence  
**Parallelism:** Serial within CE-JBC-L-FRAGMENT; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Validate externally supplied persistent order. Deliver this as one isolated, reviewable step in the Atom-aware Cellerator fragment compiler workstream.

**Biological motivation.** A reusable CellShard atom is only useful if Cellerator can compile the exact biological coverage, order, mutable planes, and partial output that atom represents.

**Compiler-architectural reason.** The fragment compiler is the primary local lowering boundary. It wraps rather than replaces program_v2, validates externally proposed structure, and emits a bounded Pareto frontier rather than a single locally optimal answer.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/execution/atom_fragment/
- [proposed] src/execution/atom_fragment/
- [proposed] tests/jbc/fragment/

**Existing code and permitted read scope:**

- include/Cellerator/compute/operation/candidate_catalog_v3/
- include/Cellerator/execution/program/program_v2.h
- include/Cellerator/execution/geometry_acquisition_v2/
- include/Cellerator/geometry/compiler/v2/
- include/Cellerator/planner/

**Explicitly out of scope / forbidden shortcuts:**

- Do not create a second operation core or planner.
- Do not force a program_v3 before the wrapper is proven insufficient.
- Do not accept unvalidated CellShard decomposition as authoritative.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-F04

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.
- Focused tests and adversarial fixtures demonstrating both success and explicit rejection paths.

### Concrete mechanism

Verify order identity, bijection over exact local membership, canonical recovery, producer/consumer axis compatibility, and generation. Reject shape-only matches.

Workstream mechanism: Use a two-pass requirements/query API. Adapt operation_core_v2 into an atom-fragment request, build compact local index spaces, validate the external cover and order, discover local candidates, retain nondominated alternatives, and prepare the chosen program_v2 inside a prepared_atom_fragment wrapper. No hidden allocation occurs after requirements are returned.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Use an independent fragment verifier and a canonical relation-apply vertical smoke before distributed integration.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-FRAGMENT; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-F06 — Extract atom requirements from local candidates

**Repository / subsystem / lane:** Cellerator · `execution/atom_fragment` · `CE-JBC-L-FRAGMENT`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-FRAGMENT; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Extract atom requirements from local candidates. Deliver this as one isolated, reviewable step in the Atom-aware Cellerator fragment compiler workstream.

**Biological motivation.** A reusable CellShard atom is only useful if Cellerator can compile the exact biological coverage, order, mutable planes, and partial output that atom represents.

**Compiler-architectural reason.** The fragment compiler is the primary local lowering boundary. It wraps rather than replaces program_v2, validates externally proposed structure, and emits a bounded Pareto frontier rather than a single locally optimal answer.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/execution/atom_fragment/
- [proposed] src/execution/atom_fragment/
- [proposed] tests/jbc/fragment/

**Existing code and permitted read scope:**

- include/Cellerator/compute/operation/candidate_catalog_v3/
- include/Cellerator/execution/program/program_v2.h
- include/Cellerator/execution/geometry_acquisition_v2/
- include/Cellerator/geometry/compiler/v2/
- include/Cellerator/planner/

**Explicitly out of scope / forbidden shortcuts:**

- Do not create a second operation core or planner.
- Do not force a program_v3 before the wrapper is proven insufficient.
- Do not accept unvalidated CellShard decomposition as authoritative.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-F05

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A bounded implementation unit, focused tests, mechanism statistics, and an integration receipt.

### Concrete mechanism

Translate candidate projection, order, value, alignment, contiguity, index, and graph-stability needs into atom requirement descriptors before CellShard materialization.

Workstream mechanism: Use a two-pass requirements/query API. Adapt operation_core_v2 into an atom-fragment request, build compact local index spaces, validate the external cover and order, discover local candidates, retain nondominated alternatives, and prepare the chosen program_v2 inside a prepared_atom_fragment wrapper. No hidden allocation occurs after requirements are returned.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Use an independent fragment verifier and a canonical relation-apply vertical smoke before distributed integration.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-FRAGMENT; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-F07 — Discover atom-bound local candidates

**Repository / subsystem / lane:** Cellerator · `execution/atom_fragment` · `CE-JBC-L-FRAGMENT`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-FRAGMENT; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Discover atom-bound local candidates. Deliver this as one isolated, reviewable step in the Atom-aware Cellerator fragment compiler workstream.

**Biological motivation.** A reusable CellShard atom is only useful if Cellerator can compile the exact biological coverage, order, mutable planes, and partial output that atom represents.

**Compiler-architectural reason.** The fragment compiler is the primary local lowering boundary. It wraps rather than replaces program_v2, validates externally proposed structure, and emits a bounded Pareto frontier rather than a single locally optimal answer.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/execution/atom_fragment/
- [proposed] src/execution/atom_fragment/
- [proposed] tests/jbc/fragment/

**Existing code and permitted read scope:**

- include/Cellerator/compute/operation/candidate_catalog_v3/
- include/Cellerator/execution/program/program_v2.h
- include/Cellerator/execution/geometry_acquisition_v2/
- include/Cellerator/geometry/compiler/v2/
- include/Cellerator/planner/

**Explicitly out of scope / forbidden shortcuts:**

- Do not create a second operation core or planner.
- Do not force a program_v3 before the wrapper is proven insufficient.
- Do not accept unvalidated CellShard decomposition as authoritative.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-F06

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Use candidate-catalog v3 and geometry compiler v2 to enumerate pure sparse, physical projection, packed-order, partial-output, and legal experimental alternatives for the exact fragment.

Workstream mechanism: Use a two-pass requirements/query API. Adapt operation_core_v2 into an atom-fragment request, build compact local index spaces, validate the external cover and order, discover local candidates, retain nondominated alternatives, and prepare the chosen program_v2 inside a prepared_atom_fragment wrapper. No hidden allocation occurs after requirements are returned.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Use an independent fragment verifier and a canonical relation-apply vertical smoke before distributed integration.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-FRAGMENT; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-F08 — Retain a bounded local Pareto frontier

**Repository / subsystem / lane:** Cellerator · `execution/atom_fragment` · `CE-JBC-L-FRAGMENT`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-FRAGMENT; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Retain a bounded local Pareto frontier. Deliver this as one isolated, reviewable step in the Atom-aware Cellerator fragment compiler workstream.

**Biological motivation.** A reusable CellShard atom is only useful if Cellerator can compile the exact biological coverage, order, mutable planes, and partial output that atom represents.

**Compiler-architectural reason.** The fragment compiler is the primary local lowering boundary. It wraps rather than replaces program_v2, validates externally proposed structure, and emits a bounded Pareto frontier rather than a single locally optimal answer.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/execution/atom_fragment/
- [proposed] src/execution/atom_fragment/
- [proposed] tests/jbc/fragment/

**Existing code and permitted read scope:**

- include/Cellerator/compute/operation/candidate_catalog_v3/
- include/Cellerator/execution/program/program_v2.h
- include/Cellerator/execution/geometry_acquisition_v2/
- include/Cellerator/geometry/compiler/v2/
- include/Cellerator/planner/

**Explicitly out of scope / forbidden shortcuts:**

- Do not create a second operation core or planner.
- Do not force a program_v3 before the wrapper is proven insufficient.
- Do not accept unvalidated CellShard decomposition as authoritative.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-F07

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A bounded implementation unit, focused tests, mechanism statistics, and an integration receipt.

### Concrete mechanism

Pareto-prune by complete local latency, preparation, persistent/transient bytes, output order, numerical mode, partial form, and empirical confidence; preserve caller-selected caps and canonical fallback.

Workstream mechanism: Use a two-pass requirements/query API. Adapt operation_core_v2 into an atom-fragment request, build compact local index spaces, validate the external cover and order, discover local candidates, retain nondominated alternatives, and prepare the chosen program_v2 inside a prepared_atom_fragment wrapper. No hidden allocation occurs after requirements are returned.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Use an independent fragment verifier and a canonical relation-apply vertical smoke before distributed integration.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-FRAGMENT; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-F09 — Wrap program_v2 as prepared atom fragment

**Repository / subsystem / lane:** Cellerator · `execution/atom_fragment` · `CE-JBC-L-FRAGMENT`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-FRAGMENT; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Wrap program_v2 as prepared atom fragment. Deliver this as one isolated, reviewable step in the Atom-aware Cellerator fragment compiler workstream.

**Biological motivation.** A reusable CellShard atom is only useful if Cellerator can compile the exact biological coverage, order, mutable planes, and partial output that atom represents.

**Compiler-architectural reason.** The fragment compiler is the primary local lowering boundary. It wraps rather than replaces program_v2, validates externally proposed structure, and emits a bounded Pareto frontier rather than a single locally optimal answer.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/execution/atom_fragment/
- [proposed] src/execution/atom_fragment/
- [proposed] tests/jbc/fragment/

**Existing code and permitted read scope:**

- include/Cellerator/compute/operation/candidate_catalog_v3/
- include/Cellerator/execution/program/program_v2.h
- include/Cellerator/execution/geometry_acquisition_v2/
- include/Cellerator/geometry/compiler/v2/
- include/Cellerator/planner/

**Explicitly out of scope / forbidden shortcuts:**

- Do not create a second operation core or planner.
- Do not force a program_v3 before the wrapper is proven insufficient.
- Do not accept unvalidated CellShard decomposition as authoritative.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-F08

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A bounded implementation unit, focused tests, mechanism statistics, and an integration receipt.

### Concrete mechanism

Associate one prepared program_v2 with exact fragment identity, atom ports, binding slots, output/partial affordance, resource receipt, and validation certificate. Do not rewrite program_v2 yet.

Workstream mechanism: Use a two-pass requirements/query API. Adapt operation_core_v2 into an atom-fragment request, build compact local index spaces, validate the external cover and order, discover local candidates, retain nondominated alternatives, and prepare the chosen program_v2 inside a prepared_atom_fragment wrapper. No hidden allocation occurs after requirements are returned.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Use an independent fragment verifier and a canonical relation-apply vertical smoke before distributed integration.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-FRAGMENT; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-F10 — Bind external atom planes and lease tokens

**Repository / subsystem / lane:** Cellerator · `execution/atom_fragment` · `CE-JBC-L-FRAGMENT`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-FRAGMENT; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Bind external atom planes and lease tokens. Deliver this as one isolated, reviewable step in the Atom-aware Cellerator fragment compiler workstream.

**Biological motivation.** A reusable CellShard atom is only useful if Cellerator can compile the exact biological coverage, order, mutable planes, and partial output that atom represents.

**Compiler-architectural reason.** The fragment compiler is the primary local lowering boundary. It wraps rather than replaces program_v2, validates externally proposed structure, and emits a bounded Pareto frontier rather than a single locally optimal answer.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/execution/atom_fragment/
- [proposed] src/execution/atom_fragment/
- [proposed] tests/jbc/fragment/

**Existing code and permitted read scope:**

- include/Cellerator/compute/operation/candidate_catalog_v3/
- include/Cellerator/execution/program/program_v2.h
- include/Cellerator/execution/geometry_acquisition_v2/
- include/Cellerator/geometry/compiler/v2/
- include/Cellerator/planner/

**Explicitly out of scope / forbidden shortcuts:**

- Do not create a second operation core or planner.
- Do not force a program_v3 before the wrapper is proven insufficient.
- Do not accept unvalidated CellShard decomposition as authoritative.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-F09

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Validate each multi-extent atom binding, readiness token, lease lifetime, generation, order, and address-space requirement, then populate prepared launch bindings without catalog lookup in the hot path.

Workstream mechanism: Use a two-pass requirements/query API. Adapt operation_core_v2 into an atom-fragment request, build compact local index spaces, validate the external cover and order, discover local candidates, retain nondominated alternatives, and prepare the chosen program_v2 inside a prepared_atom_fragment wrapper. No hidden allocation occurs after requirements are returned.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Use an independent fragment verifier and a canonical relation-apply vertical smoke before distributed integration.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-FRAGMENT; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-F11 — Describe output atom and partial affordances

**Repository / subsystem / lane:** Cellerator · `execution/atom_fragment` · `CE-JBC-L-FRAGMENT`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-FRAGMENT; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Describe output atom and partial affordances. Deliver this as one isolated, reviewable step in the Atom-aware Cellerator fragment compiler workstream.

**Biological motivation.** A reusable CellShard atom is only useful if Cellerator can compile the exact biological coverage, order, mutable planes, and partial output that atom represents.

**Compiler-architectural reason.** The fragment compiler is the primary local lowering boundary. It wraps rather than replaces program_v2, validates externally proposed structure, and emits a bounded Pareto frontier rather than a single locally optimal answer.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/execution/atom_fragment/
- [proposed] src/execution/atom_fragment/
- [proposed] tests/jbc/fragment/

**Existing code and permitted read scope:**

- include/Cellerator/compute/operation/candidate_catalog_v3/
- include/Cellerator/execution/program/program_v2.h
- include/Cellerator/execution/geometry_acquisition_v2/
- include/Cellerator/geometry/compiler/v2/
- include/Cellerator/planner/

**Explicitly out of scope / forbidden shortcuts:**

- Do not create a second operation core or planner.
- Do not force a program_v3 before the wrapper is proven insufficient.
- Do not accept unvalidated CellShard decomposition as authoritative.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-F10

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A bounded implementation unit, focused tests, mechanism statistics, and an integration receipt.

### Concrete mechanism

Emit exact output coverage, order, plane kind, ownership, partial algebra, persistence eligibility, and canonical recovery so CellShard can route, combine, persist, or discard directly.

Workstream mechanism: Use a two-pass requirements/query API. Adapt operation_core_v2 into an atom-fragment request, build compact local index spaces, validate the external cover and order, discover local candidates, retain nondominated alternatives, and prepare the chosen program_v2 inside a prepared_atom_fragment wrapper. No hidden allocation occurs after requirements are returned.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Use an independent fragment verifier and a canonical relation-apply vertical smoke before distributed integration.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-FRAGMENT; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-F12 — Implement canonical fallback and diagnostics

**Repository / subsystem / lane:** Cellerator · `execution/atom_fragment` · `CE-JBC-L-FRAGMENT`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-FRAGMENT; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Implement canonical fallback and diagnostics. Deliver this as one isolated, reviewable step in the Atom-aware Cellerator fragment compiler workstream.

**Biological motivation.** A reusable CellShard atom is only useful if Cellerator can compile the exact biological coverage, order, mutable planes, and partial output that atom represents.

**Compiler-architectural reason.** The fragment compiler is the primary local lowering boundary. It wraps rather than replaces program_v2, validates externally proposed structure, and emits a bounded Pareto frontier rather than a single locally optimal answer.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/execution/atom_fragment/
- [proposed] src/execution/atom_fragment/
- [proposed] tests/jbc/fragment/

**Existing code and permitted read scope:**

- include/Cellerator/compute/operation/candidate_catalog_v3/
- include/Cellerator/execution/program/program_v2.h
- include/Cellerator/execution/geometry_acquisition_v2/
- include/Cellerator/geometry/compiler/v2/
- include/Cellerator/planner/

**Explicitly out of scope / forbidden shortcuts:**

- Do not create a second operation core or planner.
- Do not force a program_v3 before the wrapper is proven insufficient.
- Do not accept unvalidated CellShard decomposition as authoritative.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-F11

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

When external atoms or decompositions are incompatible, select explicit canonical assembly/compile routes or reject according to policy; record the failed predicate and work required.

Workstream mechanism: Use a two-pass requirements/query API. Adapt operation_core_v2 into an atom-fragment request, build compact local index spaces, validate the external cover and order, discover local candidates, retain nondominated alternatives, and prepare the chosen program_v2 inside a prepared_atom_fragment wrapper. No hidden allocation occurs after requirements are returned.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Use an independent fragment verifier and a canonical relation-apply vertical smoke before distributed integration.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-FRAGMENT; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-F13 — Register source-linked fragment compilers

**Repository / subsystem / lane:** Cellerator · `execution/atom_fragment` · `CE-JBC-L-FRAGMENT`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-FRAGMENT; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Register source-linked fragment compilers. Deliver this as one isolated, reviewable step in the Atom-aware Cellerator fragment compiler workstream.

**Biological motivation.** A reusable CellShard atom is only useful if Cellerator can compile the exact biological coverage, order, mutable planes, and partial output that atom represents.

**Compiler-architectural reason.** The fragment compiler is the primary local lowering boundary. It wraps rather than replaces program_v2, validates externally proposed structure, and emits a bounded Pareto frontier rather than a single locally optimal answer.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/execution/atom_fragment/
- [proposed] src/execution/atom_fragment/
- [proposed] tests/jbc/fragment/

**Existing code and permitted read scope:**

- include/Cellerator/compute/operation/candidate_catalog_v3/
- include/Cellerator/execution/program/program_v2.h
- include/Cellerator/execution/geometry_acquisition_v2/
- include/Cellerator/geometry/compiler/v2/
- include/Cellerator/planner/

**Explicitly out of scope / forbidden shortcuts:**

- Do not create a second operation core or planner.
- Do not force a program_v3 before the wrapper is proven insufficient.
- Do not accept unvalidated CellShard decomposition as authoritative.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-F12

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Expose count/query/fill provider fragments by operation family; keep central aggregation integration-only and permit external mock providers in tests.

Workstream mechanism: Use a two-pass requirements/query API. Adapt operation_core_v2 into an atom-fragment request, build compact local index spaces, validate the external cover and order, discover local candidates, retain nondominated alternatives, and prepare the chosen program_v2 inside a prepared_atom_fragment wrapper. No hidden allocation occurs after requirements are returned.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Use an independent fragment verifier and a canonical relation-apply vertical smoke before distributed integration.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-FRAGMENT; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-F14 — Execute canonical relation-apply atom-fragment smoke

**Repository / subsystem / lane:** Cellerator · `execution/atom_fragment` · `CE-JBC-L-FRAGMENT`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-FRAGMENT; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Execute canonical relation-apply atom-fragment smoke. Deliver this as one isolated, reviewable step in the Atom-aware Cellerator fragment compiler workstream.

**Biological motivation.** A reusable CellShard atom is only useful if Cellerator can compile the exact biological coverage, order, mutable planes, and partial output that atom represents.

**Compiler-architectural reason.** The fragment compiler is the primary local lowering boundary. It wraps rather than replaces program_v2, validates externally proposed structure, and emits a bounded Pareto frontier rather than a single locally optimal answer.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/execution/atom_fragment/
- [proposed] src/execution/atom_fragment/
- [proposed] tests/jbc/fragment/

**Existing code and permitted read scope:**

- include/Cellerator/compute/operation/candidate_catalog_v3/
- include/Cellerator/execution/program/program_v2.h
- include/Cellerator/execution/geometry_acquisition_v2/
- include/Cellerator/geometry/compiler/v2/
- include/Cellerator/planner/

**Explicitly out of scope / forbidden shortcuts:**

- Do not create a second operation core or planner.
- Do not force a program_v3 before the wrapper is proven insufficient.
- Do not accept unvalidated CellShard decomposition as authoritative.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-F13

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A bounded implementation unit, focused tests, mechanism statistics, and an integration receipt.

### Concrete mechanism

Compile one exact certified relation atom, bind canonical input/value/output planes, run through an execution session, recover canonical output, and prove no distributed or CellShard dependency is required.

Workstream mechanism: Use a two-pass requirements/query API. Adapt operation_core_v2 into an atom-fragment request, build compact local index spaces, validate the external cover and order, discover local candidates, retain nondominated alternatives, and prepare the chosen program_v2 inside a prepared_atom_fragment wrapper. No hidden allocation occurs after requirements are returned.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Use an independent fragment verifier and a canonical relation-apply vertical smoke before distributed integration.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-FRAGMENT; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


# Workstream CEMULTI: Multi-atom and multi-extent Cellerator operands

**Repository:** Cellerator  
**Suggested lane:** `CE-JBC-L-MULTIATOM`  
**Barrier:** `JBC-G3-CELLERATOR-FRAGMENT`

## Workstream design

Define lists of atom ports and extents, query contiguity/alignment/index requirements, compile an explicit gather/scatter assembly baseline, and add direct multi-extent candidates behind measurement gates. A candidate claiming direct execution may not perform full hidden assembly.

## Existing live source extended

- include/Cellerator/execution/geometry_acquisition_v2/external_payload.hh
- include/Cellerator/execution/geometry_acquisition_v2/schema.hh
- include/Cellerator/execution/program/program_v2.h
- include/Cellerator/execution/index_space/hierarchical_index_space_v1.hh

## Likely source scope

- [proposed] include/Cellerator/execution/object_binding/
- [proposed] src/execution/object_binding/
- [proposed] src/compute/candidate/jbc_multi_extent/
- [proposed] tests/jbc/multi_extent/
- [proposed] bench/jbc/multi_extent/

## Proposed Todos (10)

## CE-JBC-M01 — Define multi-atom port binding list

**Repository / subsystem / lane:** Cellerator · `execution/object_binding and candidate adapters` · `CE-JBC-L-MULTIATOM`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-MULTIATOM; parallel with other provider/workstream lanes after JBC-G3-CELLERATOR-FRAGMENT.

### Why

**Purpose.** Define multi-atom port binding list. Deliver this as one isolated, reviewable step in the Multi-atom and multi-extent Cellerator operands workstream.

**Biological motivation.** Biologically reusable atoms may remain physically separate—stable support plus mutable values, MMA cover plus residual, trajectory base plus branch delta, or several modalities—yet jointly form one legal operand.

**Compiler-architectural reason.** The compiler must compare explicit assembly with direct multi-extent execution rather than forcing every atom portfolio into a monolithic matrix.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/execution/object_binding/
- [proposed] src/execution/object_binding/
- [proposed] src/compute/candidate/jbc_multi_extent/
- [proposed] tests/jbc/multi_extent/
- [proposed] bench/jbc/multi_extent/

**Existing code and permitted read scope:**

- include/Cellerator/execution/geometry_acquisition_v2/external_payload.hh
- include/Cellerator/execution/geometry_acquisition_v2/schema.hh
- include/Cellerator/execution/program/program_v2.h
- include/Cellerator/execution/index_space/hierarchical_index_space_v1.hh

**Explicitly out of scope / forbidden shortcuts:**

- Do not make scatter-gather mandatory for kernels that require contiguous storage.
- Do not allow pointer arrays to substitute for exact atom identity or coverage.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-F14

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.
- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Allow one logical operand port to be satisfied by an ordered list of certified atoms with explicit coverage union, plane roles, generations, and order compatibility.

Workstream mechanism: Define lists of atom ports and extents, query contiguity/alignment/index requirements, compile an explicit gather/scatter assembly baseline, and add direct multi-extent candidates behind measurement gates. A candidate claiming direct execution may not perform full hidden assembly.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Compare direct and assembled paths against the same canonical output and inspect preparation bytes and hidden allocations.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-MULTIATOM; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-M02 — Define multi-extent physical binding list

**Repository / subsystem / lane:** Cellerator · `execution/object_binding and candidate adapters` · `CE-JBC-L-MULTIATOM`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-MULTIATOM; parallel with other provider/workstream lanes after JBC-G3-CELLERATOR-FRAGMENT.

### Why

**Purpose.** Define multi-extent physical binding list. Deliver this as one isolated, reviewable step in the Multi-atom and multi-extent Cellerator operands workstream.

**Biological motivation.** Biologically reusable atoms may remain physically separate—stable support plus mutable values, MMA cover plus residual, trajectory base plus branch delta, or several modalities—yet jointly form one legal operand.

**Compiler-architectural reason.** The compiler must compare explicit assembly with direct multi-extent execution rather than forcing every atom portfolio into a monolithic matrix.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/execution/object_binding/
- [proposed] src/execution/object_binding/
- [proposed] src/compute/candidate/jbc_multi_extent/
- [proposed] tests/jbc/multi_extent/
- [proposed] bench/jbc/multi_extent/

**Existing code and permitted read scope:**

- include/Cellerator/execution/geometry_acquisition_v2/external_payload.hh
- include/Cellerator/execution/geometry_acquisition_v2/schema.hh
- include/Cellerator/execution/program/program_v2.h
- include/Cellerator/execution/index_space/hierarchical_index_space_v1.hh

**Explicitly out of scope / forbidden shortcuts:**

- Do not make scatter-gather mandatory for kernels that require contiguous storage.
- Do not allow pointer arrays to substitute for exact atom identity or coverage.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-M01

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.
- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Represent each atom plane by one or more checked host/device/peer extents with offsets, bytes, alignment, readiness, and lease tokens; no raw persistent pointers.

Workstream mechanism: Define lists of atom ports and extents, query contiguity/alignment/index requirements, compile an explicit gather/scatter assembly baseline, and add direct multi-extent candidates behind measurement gates. A candidate claiming direct execution may not perform full hidden assembly.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Compare direct and assembled paths against the same canonical output and inspect preparation bytes and hidden allocations.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-MULTIATOM; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-M03 — Query contiguity, alignment, and extent requirements

**Repository / subsystem / lane:** Cellerator · `execution/object_binding and candidate adapters` · `CE-JBC-L-MULTIATOM`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-MULTIATOM; parallel with other provider/workstream lanes after JBC-G3-CELLERATOR-FRAGMENT.

### Why

**Purpose.** Query contiguity, alignment, and extent requirements. Deliver this as one isolated, reviewable step in the Multi-atom and multi-extent Cellerator operands workstream.

**Biological motivation.** Biologically reusable atoms may remain physically separate—stable support plus mutable values, MMA cover plus residual, trajectory base plus branch delta, or several modalities—yet jointly form one legal operand.

**Compiler-architectural reason.** The compiler must compare explicit assembly with direct multi-extent execution rather than forcing every atom portfolio into a monolithic matrix.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/execution/object_binding/
- [proposed] src/execution/object_binding/
- [proposed] src/compute/candidate/jbc_multi_extent/
- [proposed] tests/jbc/multi_extent/
- [proposed] bench/jbc/multi_extent/

**Existing code and permitted read scope:**

- include/Cellerator/execution/geometry_acquisition_v2/external_payload.hh
- include/Cellerator/execution/geometry_acquisition_v2/schema.hh
- include/Cellerator/execution/program/program_v2.h
- include/Cellerator/execution/index_space/hierarchical_index_space_v1.hh

**Explicitly out of scope / forbidden shortcuts:**

- Do not make scatter-gather mandatory for kernels that require contiguous storage.
- Do not allow pointer arrays to substitute for exact atom identity or coverage.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-M02

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

For each local candidate, report maximum extent count, required contiguous spans, accepted scatter-gather forms, index widths, alignment, scratch, and graph-stable address requirements.

Workstream mechanism: Define lists of atom ports and extents, query contiguity/alignment/index requirements, compile an explicit gather/scatter assembly baseline, and add direct multi-extent candidates behind measurement gates. A candidate claiming direct execution may not perform full hidden assembly.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Compare direct and assembled paths against the same canonical output and inspect preparation bytes and hidden allocations.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-MULTIATOM; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-M04 — Compile the explicit contiguous-assembly baseline

**Repository / subsystem / lane:** Cellerator · `execution/object_binding and candidate adapters` · `CE-JBC-L-MULTIATOM`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-MULTIATOM; parallel with other provider/workstream lanes after JBC-G3-CELLERATOR-FRAGMENT.

### Why

**Purpose.** Compile the explicit contiguous-assembly baseline. Deliver this as one isolated, reviewable step in the Multi-atom and multi-extent Cellerator operands workstream.

**Biological motivation.** Biologically reusable atoms may remain physically separate—stable support plus mutable values, MMA cover plus residual, trajectory base plus branch delta, or several modalities—yet jointly form one legal operand.

**Compiler-architectural reason.** The compiler must compare explicit assembly with direct multi-extent execution rather than forcing every atom portfolio into a monolithic matrix.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/execution/object_binding/
- [proposed] src/execution/object_binding/
- [proposed] src/compute/candidate/jbc_multi_extent/
- [proposed] tests/jbc/multi_extent/
- [proposed] bench/jbc/multi_extent/

**Existing code and permitted read scope:**

- include/Cellerator/execution/geometry_acquisition_v2/external_payload.hh
- include/Cellerator/execution/geometry_acquisition_v2/schema.hh
- include/Cellerator/execution/program/program_v2.h
- include/Cellerator/execution/index_space/hierarchical_index_space_v1.hh

**Explicitly out of scope / forbidden shortcuts:**

- Do not make scatter-gather mandatory for kernels that require contiguous storage.
- Do not allow pointer arrays to substitute for exact atom identity or coverage.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-M03

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Create a visible transform plan that gathers selected atom extents into one Cellerator-owned/caller-provided contiguous buffer, records bytes and order, and supports inverse scatter when required.

Workstream mechanism: Define lists of atom ports and extents, query contiguity/alignment/index requirements, compile an explicit gather/scatter assembly baseline, and add direct multi-extent candidates behind measurement gates. A candidate claiming direct execution may not perform full hidden assembly.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Compare direct and assembled paths against the same canonical output and inspect preparation bytes and hidden allocations.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-MULTIATOM; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-M05 — Implement reusable gather, scatter, and permutation operations

**Repository / subsystem / lane:** Cellerator · `execution/object_binding and candidate adapters` · `CE-JBC-L-MULTIATOM`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-MULTIATOM; parallel with other provider/workstream lanes after JBC-G3-CELLERATOR-FRAGMENT.

### Why

**Purpose.** Implement reusable gather, scatter, and permutation operations. Deliver this as one isolated, reviewable step in the Multi-atom and multi-extent Cellerator operands workstream.

**Biological motivation.** Biologically reusable atoms may remain physically separate—stable support plus mutable values, MMA cover plus residual, trajectory base plus branch delta, or several modalities—yet jointly form one legal operand.

**Compiler-architectural reason.** The compiler must compare explicit assembly with direct multi-extent execution rather than forcing every atom portfolio into a monolithic matrix.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/execution/object_binding/
- [proposed] src/execution/object_binding/
- [proposed] src/compute/candidate/jbc_multi_extent/
- [proposed] tests/jbc/multi_extent/
- [proposed] bench/jbc/multi_extent/

**Existing code and permitted read scope:**

- include/Cellerator/execution/geometry_acquisition_v2/external_payload.hh
- include/Cellerator/execution/geometry_acquisition_v2/schema.hh
- include/Cellerator/execution/program/program_v2.h
- include/Cellerator/execution/index_space/hierarchical_index_space_v1.hh

**Explicitly out of scope / forbidden shortcuts:**

- Do not make scatter-gather mandatory for kernels that require contiguous storage.
- Do not allow pointer arrays to substitute for exact atom identity or coverage.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-M04

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Add Cellerator-native low-level operations for atom assembly and recovery using compact local maps, explicit streams, caller workspace, persistent order, and profiler-visible bytes.

Workstream mechanism: Define lists of atom ports and extents, query contiguity/alignment/index requirements, compile an explicit gather/scatter assembly baseline, and add direct multi-extent candidates behind measurement gates. A candidate claiming direct execution may not perform full hidden assembly.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Compare direct and assembled paths against the same canonical output and inspect preparation bytes and hidden allocations.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-MULTIATOM; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-M06 — Define the direct multi-extent candidate contract

**Repository / subsystem / lane:** Cellerator · `execution/object_binding and candidate adapters` · `CE-JBC-L-MULTIATOM`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-MULTIATOM; parallel with other provider/workstream lanes after JBC-G3-CELLERATOR-FRAGMENT.

### Why

**Purpose.** Define the direct multi-extent candidate contract. Deliver this as one isolated, reviewable step in the Multi-atom and multi-extent Cellerator operands workstream.

**Biological motivation.** Biologically reusable atoms may remain physically separate—stable support plus mutable values, MMA cover plus residual, trajectory base plus branch delta, or several modalities—yet jointly form one legal operand.

**Compiler-architectural reason.** The compiler must compare explicit assembly with direct multi-extent execution rather than forcing every atom portfolio into a monolithic matrix.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/execution/object_binding/
- [proposed] src/execution/object_binding/
- [proposed] src/compute/candidate/jbc_multi_extent/
- [proposed] tests/jbc/multi_extent/
- [proposed] bench/jbc/multi_extent/

**Existing code and permitted read scope:**

- include/Cellerator/execution/geometry_acquisition_v2/external_payload.hh
- include/Cellerator/execution/geometry_acquisition_v2/schema.hh
- include/Cellerator/execution/program/program_v2.h
- include/Cellerator/execution/index_space/hierarchical_index_space_v1.hh

**Explicitly out of scope / forbidden shortcuts:**

- Do not make scatter-gather mandatory for kernels that require contiguous storage.
- Do not allow pointer arrays to substitute for exact atom identity or coverage.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-M05

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.

### Concrete mechanism

Specify pointer-table/segment descriptors, supported operations, maximum extents, memory-space constraints, and output semantics. Mark direct execution experimental and empirical-required.

Workstream mechanism: Define lists of atom ports and extents, query contiguity/alignment/index requirements, compile an explicit gather/scatter assembly baseline, and add direct multi-extent candidates behind measurement gates. A candidate claiming direct execution may not perform full hidden assembly.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Compare direct and assembled paths against the same canonical output and inspect preparation bytes and hidden allocations.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-MULTIATOM; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-M07 — Implement an experimental multi-extent relation-apply candidate

**Repository / subsystem / lane:** Cellerator · `execution/object_binding and candidate adapters` · `CE-JBC-L-MULTIATOM`  
**Classification:** experimental candidate; baseline and negative result required  
**Parallelism:** Serial within CE-JBC-L-MULTIATOM; parallel with other provider/workstream lanes after JBC-G3-CELLERATOR-FRAGMENT.

### Why

**Purpose.** Implement an experimental multi-extent relation-apply candidate. Deliver this as one isolated, reviewable step in the Multi-atom and multi-extent Cellerator operands workstream.

**Biological motivation.** Biologically reusable atoms may remain physically separate—stable support plus mutable values, MMA cover plus residual, trajectory base plus branch delta, or several modalities—yet jointly form one legal operand.

**Compiler-architectural reason.** The compiler must compare explicit assembly with direct multi-extent execution rather than forcing every atom portfolio into a monolithic matrix.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/execution/object_binding/
- [proposed] src/execution/object_binding/
- [proposed] src/compute/candidate/jbc_multi_extent/
- [proposed] tests/jbc/multi_extent/
- [proposed] bench/jbc/multi_extent/

**Existing code and permitted read scope:**

- include/Cellerator/execution/geometry_acquisition_v2/external_payload.hh
- include/Cellerator/execution/geometry_acquisition_v2/schema.hh
- include/Cellerator/execution/program/program_v2.h
- include/Cellerator/execution/index_space/hierarchical_index_space_v1.hh

**Explicitly out of scope / forbidden shortcuts:**

- Do not make scatter-gather mandatory for kernels that require contiguous storage.
- Do not allow pointer arrays to substitute for exact atom identity or coverage.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-M06

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Consume several source/support/value extents without full assembly, using grouped launches or kernel-side segment traversal while preserving exact logical edge and output ownership.

Workstream mechanism: Define lists of atom ports and extents, query contiguity/alignment/index requirements, compile an explicit gather/scatter assembly baseline, and add direct multi-extent candidates behind measurement gates. A candidate claiming direct execution may not perform full hidden assembly.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Compare direct and assembled paths against the same canonical output and inspect preparation bytes and hidden allocations.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-MULTIATOM; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-M08 — Bind a structural atom plus mutable value overlay

**Repository / subsystem / lane:** Cellerator · `execution/object_binding and candidate adapters` · `CE-JBC-L-MULTIATOM`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-MULTIATOM; parallel with other provider/workstream lanes after JBC-G3-CELLERATOR-FRAGMENT.

### Why

**Purpose.** Bind a structural atom plus mutable value overlay. Deliver this as one isolated, reviewable step in the Multi-atom and multi-extent Cellerator operands workstream.

**Biological motivation.** Biologically reusable atoms may remain physically separate—stable support plus mutable values, MMA cover plus residual, trajectory base plus branch delta, or several modalities—yet jointly form one legal operand.

**Compiler-architectural reason.** The compiler must compare explicit assembly with direct multi-extent execution rather than forcing every atom portfolio into a monolithic matrix.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/execution/object_binding/
- [proposed] src/execution/object_binding/
- [proposed] src/compute/candidate/jbc_multi_extent/
- [proposed] tests/jbc/multi_extent/
- [proposed] bench/jbc/multi_extent/

**Existing code and permitted read scope:**

- include/Cellerator/execution/geometry_acquisition_v2/external_payload.hh
- include/Cellerator/execution/geometry_acquisition_v2/schema.hh
- include/Cellerator/execution/program/program_v2.h
- include/Cellerator/execution/index_space/hierarchical_index_space_v1.hh

**Explicitly out of scope / forbidden shortcuts:**

- Do not make scatter-gather mandatory for kernels that require contiguous storage.
- Do not allow pointer arrays to substitute for exact atom identity or coverage.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-M07

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Demonstrate a support/projection plane in one atom and current values/gates in another, validated by shared structure/epoch/edge spine and independent value generation.

Workstream mechanism: Define lists of atom ports and extents, query contiguity/alignment/index requirements, compile an explicit gather/scatter assembly baseline, and add direct multi-extent candidates behind measurement gates. A candidate claiming direct execution may not perform full hidden assembly.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Compare direct and assembled paths against the same canonical output and inspect preparation bytes and hidden allocations.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-MULTIATOM; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-M09 — Integrate chunk-native projection acquisition

**Repository / subsystem / lane:** Cellerator · `execution/object_binding and candidate adapters` · `CE-JBC-L-MULTIATOM`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-MULTIATOM; parallel with other provider/workstream lanes after JBC-G3-CELLERATOR-FRAGMENT.

### Why

**Purpose.** Integrate chunk-native projection acquisition. Deliver this as one isolated, reviewable step in the Multi-atom and multi-extent Cellerator operands workstream.

**Biological motivation.** Biologically reusable atoms may remain physically separate—stable support plus mutable values, MMA cover plus residual, trajectory base plus branch delta, or several modalities—yet jointly form one legal operand.

**Compiler-architectural reason.** The compiler must compare explicit assembly with direct multi-extent execution rather than forcing every atom portfolio into a monolithic matrix.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/execution/object_binding/
- [proposed] src/execution/object_binding/
- [proposed] src/compute/candidate/jbc_multi_extent/
- [proposed] tests/jbc/multi_extent/
- [proposed] bench/jbc/multi_extent/

**Existing code and permitted read scope:**

- include/Cellerator/execution/geometry_acquisition_v2/external_payload.hh
- include/Cellerator/execution/geometry_acquisition_v2/schema.hh
- include/Cellerator/execution/program/program_v2.h
- include/Cellerator/execution/index_space/hierarchical_index_space_v1.hh

**Explicitly out of scope / forbidden shortcuts:**

- Do not make scatter-gather mandatory for kernels that require contiguous storage.
- Do not allow pointer arrays to substitute for exact atom identity or coverage.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-M08

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Permit acquisition/resumption to bind bounded projection chunks and exact maps directly, while retaining a contiguous projection fallback and CPE2 semantics.

Workstream mechanism: Define lists of atom ports and extents, query contiguity/alignment/index requirements, compile an explicit gather/scatter assembly baseline, and add direct multi-extent candidates behind measurement gates. A candidate claiming direct execution may not perform full hidden assembly.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Compare direct and assembled paths against the same canonical output and inspect preparation bytes and hidden allocations.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-MULTIATOM; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-M10 — Benchmark assembly versus direct multi-extent execution

**Repository / subsystem / lane:** Cellerator · `execution/object_binding and candidate adapters` · `CE-JBC-L-MULTIATOM`  
**Classification:** validation/evidence  
**Parallelism:** Serial within CE-JBC-L-MULTIATOM; parallel with other provider/workstream lanes after JBC-G3-CELLERATOR-FRAGMENT.

### Why

**Purpose.** Benchmark assembly versus direct multi-extent execution. Deliver this as one isolated, reviewable step in the Multi-atom and multi-extent Cellerator operands workstream.

**Biological motivation.** Biologically reusable atoms may remain physically separate—stable support plus mutable values, MMA cover plus residual, trajectory base plus branch delta, or several modalities—yet jointly form one legal operand.

**Compiler-architectural reason.** The compiler must compare explicit assembly with direct multi-extent execution rather than forcing every atom portfolio into a monolithic matrix.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/execution/object_binding/
- [proposed] src/execution/object_binding/
- [proposed] src/compute/candidate/jbc_multi_extent/
- [proposed] tests/jbc/multi_extent/
- [proposed] bench/jbc/multi_extent/

**Existing code and permitted read scope:**

- include/Cellerator/execution/geometry_acquisition_v2/external_payload.hh
- include/Cellerator/execution/geometry_acquisition_v2/schema.hh
- include/Cellerator/execution/program/program_v2.h
- include/Cellerator/execution/index_space/hierarchical_index_space_v1.hh

**Explicitly out of scope / forbidden shortcuts:**

- Do not make scatter-gather mandatory for kernels that require contiguous storage.
- Do not allow pointer arrays to substitute for exact atom identity or coverage.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-M09

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A machine-readable evidence record containing the baseline, candidate, complete-cost metrics, environment identity, and promotion disposition.

### Concrete mechanism

Measure preparation bytes, pointer/descriptor overhead, kernel efficiency, cache/TLB behavior, launches, memory, and reuse break-even; promote only complete-cost wins.

Workstream mechanism: Define lists of atom ports and extents, query contiguity/alignment/index requirements, compile an explicit gather/scatter assembly baseline, and add direct multi-extent candidates behind measurement gates. A candidate claiming direct execution may not perform full hidden assembly.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Compare direct and assembled paths against the same canonical output and inspect preparation bytes and hidden allocations.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-MULTIATOM; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


# Workstream CEPLANE: Cellerator atom planes, mutable values, gradients, and atom outputs

**Repository:** Cellerator  
**Suggested lane:** `CE-JBC-L-PLANES`  
**Barrier:** `JBC-G1-ATOM-THIN-WAIST`

## Workstream design

Create neutral atom-plane bindings over existing structure/value/state/gradient storage, retain exact primary ownership and mirror semantics, bind external readiness/lease tokens, and allow Cellerator stages to emit persistent-order final or partial atoms. Runtime ownership stays external.

## Existing live source extended

- include/Cellerator/execution/projection_value_plane/value_plane_v1.hh
- include/Cellerator/execution/projection_value_plane/composite_plane_v1.hh
- include/Cellerator/execution/projection_value_plane/generation_publication_v1.hh
- include/Cellerator/compute/operation/relation_algebra_v2/relation_algebra.hh

## Likely source scope

- [proposed] include/Cellerator/execution/atom_plane/
- [proposed] src/execution/atom_plane/
- include/Cellerator/execution/projection_value_plane/
- src/execution/projection_value_plane/
- [proposed] tests/jbc/atom_plane/

## Proposed Todos (10)

## CE-JBC-P01 — Map CellShard atom planes to projection_value_plane_v1

**Repository / subsystem / lane:** Cellerator · `execution/atom_plane and projection_value_plane adapters` · `CE-JBC-L-PLANES`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-PLANES; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Map CellShard atom planes to projection_value_plane_v1. Deliver this as one isolated, reviewable step in the Cellerator atom planes, mutable values, gradients, and atom outputs workstream.

**Biological motivation.** Stable biological support with changing activity, state, parameters, and gradients is a dominant recurring pattern and must be represented without rebuilding structure.

**Compiler-architectural reason.** Projection-value-plane v1 already supplies exact primary ownership, alternate mirrors, permanent holes, direct gradients, and generation publication. The atom plane model should adapt and extend it rather than duplicate it.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/execution/atom_plane/
- [proposed] src/execution/atom_plane/
- include/Cellerator/execution/projection_value_plane/
- src/execution/projection_value_plane/
- [proposed] tests/jbc/atom_plane/

**Existing code and permitted read scope:**

- include/Cellerator/execution/projection_value_plane/value_plane_v1.hh
- include/Cellerator/execution/projection_value_plane/composite_plane_v1.hh
- include/Cellerator/execution/projection_value_plane/generation_publication_v1.hh
- include/Cellerator/compute/operation/relation_algebra_v2/relation_algebra.hh

**Explicitly out of scope / forbidden shortcuts:**

- No physical padding hole becomes a parameter, edge, or biological identity.
- Do not publish a value generation until every required primary component is ready.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-I12

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Create a neutral adapter from external atom-plane descriptors to existing logical/projection-primary component views without changing v1 ownership semantics.

Workstream mechanism: Create neutral atom-plane bindings over existing structure/value/state/gradient storage, retain exact primary ownership and mirror semantics, bind external readiness/lease tokens, and allow Cellerator stages to emit persistent-order final or partial atoms. Runtime ownership stays external.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Exercise logical-primary and projection-primary modes, direct gradients, dirty subsets, failed publication, and canonical export.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-PLANES; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-P02 — Define structural atom-plane binding

**Repository / subsystem / lane:** Cellerator · `execution/atom_plane and projection_value_plane adapters` · `CE-JBC-L-PLANES`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-PLANES; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Define structural atom-plane binding. Deliver this as one isolated, reviewable step in the Cellerator atom planes, mutable values, gradients, and atom outputs workstream.

**Biological motivation.** Stable biological support with changing activity, state, parameters, and gradients is a dominant recurring pattern and must be represented without rebuilding structure.

**Compiler-architectural reason.** Projection-value-plane v1 already supplies exact primary ownership, alternate mirrors, permanent holes, direct gradients, and generation publication. The atom plane model should adapt and extend it rather than duplicate it.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/execution/atom_plane/
- [proposed] src/execution/atom_plane/
- include/Cellerator/execution/projection_value_plane/
- src/execution/projection_value_plane/
- [proposed] tests/jbc/atom_plane/

**Existing code and permitted read scope:**

- include/Cellerator/execution/projection_value_plane/value_plane_v1.hh
- include/Cellerator/execution/projection_value_plane/composite_plane_v1.hh
- include/Cellerator/execution/projection_value_plane/generation_publication_v1.hh
- include/Cellerator/compute/operation/relation_algebra_v2/relation_algebra.hh

**Explicitly out of scope / forbidden shortcuts:**

- No physical padding hole becomes a parameter, edge, or biological identity.
- Do not publish a value generation until every required primary component is ready.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-P01

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.
- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Bind immutable relation support, physical projection maps, orders, and structure epoch separately from all mutable values.

Workstream mechanism: Create neutral atom-plane bindings over existing structure/value/state/gradient storage, retain exact primary ownership and mirror semantics, bind external readiness/lease tokens, and allow Cellerator stages to emit persistent-order final or partial atoms. Runtime ownership stays external.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Exercise logical-primary and projection-primary modes, direct gradients, dirty subsets, failed publication, and canonical export.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-PLANES; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-P03 — Define mutable relation-value atom plane

**Repository / subsystem / lane:** Cellerator · `execution/atom_plane and projection_value_plane adapters` · `CE-JBC-L-PLANES`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-PLANES; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Define mutable relation-value atom plane. Deliver this as one isolated, reviewable step in the Cellerator atom planes, mutable values, gradients, and atom outputs workstream.

**Biological motivation.** Stable biological support with changing activity, state, parameters, and gradients is a dominant recurring pattern and must be represented without rebuilding structure.

**Compiler-architectural reason.** Projection-value-plane v1 already supplies exact primary ownership, alternate mirrors, permanent holes, direct gradients, and generation publication. The atom plane model should adapt and extend it rather than duplicate it.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/execution/atom_plane/
- [proposed] src/execution/atom_plane/
- include/Cellerator/execution/projection_value_plane/
- src/execution/projection_value_plane/
- [proposed] tests/jbc/atom_plane/

**Existing code and permitted read scope:**

- include/Cellerator/execution/projection_value_plane/value_plane_v1.hh
- include/Cellerator/execution/projection_value_plane/composite_plane_v1.hh
- include/Cellerator/execution/projection_value_plane/generation_publication_v1.hh
- include/Cellerator/compute/operation/relation_algebra_v2/relation_algebra.hh

**Explicitly out of scope / forbidden shortcuts:**

- No physical padding hole becomes a parameter, edge, or biological identity.
- Do not publish a value generation until every required primary component is ready.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-P02

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.

### Concrete mechanism

Bind logical-primary or projection-primary edge values with exact primary-component ownership, alternate mirrors, numeric type, and value generation.

Workstream mechanism: Create neutral atom-plane bindings over existing structure/value/state/gradient storage, retain exact primary ownership and mirror semantics, bind external readiness/lease tokens, and allow Cellerator stages to emit persistent-order final or partial atoms. Runtime ownership stays external.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Exercise logical-primary and projection-primary modes, direct gradients, dirty subsets, failed publication, and canonical export.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-PLANES; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-P04 — Define active-support overlay atom plane

**Repository / subsystem / lane:** Cellerator · `execution/atom_plane and projection_value_plane adapters` · `CE-JBC-L-PLANES`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-PLANES; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Define active-support overlay atom plane. Deliver this as one isolated, reviewable step in the Cellerator atom planes, mutable values, gradients, and atom outputs workstream.

**Biological motivation.** Stable biological support with changing activity, state, parameters, and gradients is a dominant recurring pattern and must be represented without rebuilding structure.

**Compiler-architectural reason.** Projection-value-plane v1 already supplies exact primary ownership, alternate mirrors, permanent holes, direct gradients, and generation publication. The atom plane model should adapt and extend it rather than duplicate it.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/execution/atom_plane/
- [proposed] src/execution/atom_plane/
- include/Cellerator/execution/projection_value_plane/
- src/execution/projection_value_plane/
- [proposed] tests/jbc/atom_plane/

**Existing code and permitted read scope:**

- include/Cellerator/execution/projection_value_plane/value_plane_v1.hh
- include/Cellerator/execution/projection_value_plane/composite_plane_v1.hh
- include/Cellerator/execution/projection_value_plane/generation_publication_v1.hh
- include/Cellerator/compute/operation/relation_algebra_v2/relation_algebra.hh

**Explicitly out of scope / forbidden shortcuts:**

- No physical padding hole becomes a parameter, edge, or biological identity.
- Do not publish a value generation until every required primary component is ready.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-P03

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.

### Concrete mechanism

Represent dynamic masks over a stable exact support superset; zero/inactive state changes values, not edge identity or structure epoch, unless topology actually changes.

Workstream mechanism: Create neutral atom-plane bindings over existing structure/value/state/gradient storage, retain exact primary ownership and mirror semantics, bind external readiness/lease tokens, and allow Cellerator stages to emit persistent-order final or partial atoms. Runtime ownership stays external.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Exercise logical-primary and projection-primary modes, direct gradients, dirty subsets, failed publication, and canonical export.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-PLANES; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-P05 — Define mutable state atom plane

**Repository / subsystem / lane:** Cellerator · `execution/atom_plane and projection_value_plane adapters` · `CE-JBC-L-PLANES`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-PLANES; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Define mutable state atom plane. Deliver this as one isolated, reviewable step in the Cellerator atom planes, mutable values, gradients, and atom outputs workstream.

**Biological motivation.** Stable biological support with changing activity, state, parameters, and gradients is a dominant recurring pattern and must be represented without rebuilding structure.

**Compiler-architectural reason.** Projection-value-plane v1 already supplies exact primary ownership, alternate mirrors, permanent holes, direct gradients, and generation publication. The atom plane model should adapt and extend it rather than duplicate it.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/execution/atom_plane/
- [proposed] src/execution/atom_plane/
- include/Cellerator/execution/projection_value_plane/
- src/execution/projection_value_plane/
- [proposed] tests/jbc/atom_plane/

**Existing code and permitted read scope:**

- include/Cellerator/execution/projection_value_plane/value_plane_v1.hh
- include/Cellerator/execution/projection_value_plane/composite_plane_v1.hh
- include/Cellerator/execution/projection_value_plane/generation_publication_v1.hh
- include/Cellerator/compute/operation/relation_algebra_v2/relation_algebra.hh

**Explicitly out of scope / forbidden shortcuts:**

- No physical padding hole becomes a parameter, edge, or biological identity.
- Do not publish a value generation until every required primary component is ready.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-P04

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.

### Concrete mechanism

Bind dense or sparse biological state keyed by persistent axes and order, with explicit generation and producer/consumer affordances.

Workstream mechanism: Create neutral atom-plane bindings over existing structure/value/state/gradient storage, retain exact primary ownership and mirror semantics, bind external readiness/lease tokens, and allow Cellerator stages to emit persistent-order final or partial atoms. Runtime ownership stays external.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Exercise logical-primary and projection-primary modes, direct gradients, dirty subsets, failed publication, and canonical export.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-PLANES; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-P06 — Define gradient atom plane

**Repository / subsystem / lane:** Cellerator · `execution/atom_plane and projection_value_plane adapters` · `CE-JBC-L-PLANES`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-PLANES; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Define gradient atom plane. Deliver this as one isolated, reviewable step in the Cellerator atom planes, mutable values, gradients, and atom outputs workstream.

**Biological motivation.** Stable biological support with changing activity, state, parameters, and gradients is a dominant recurring pattern and must be represented without rebuilding structure.

**Compiler-architectural reason.** Projection-value-plane v1 already supplies exact primary ownership, alternate mirrors, permanent holes, direct gradients, and generation publication. The atom plane model should adapt and extend it rather than duplicate it.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/execution/atom_plane/
- [proposed] src/execution/atom_plane/
- include/Cellerator/execution/projection_value_plane/
- src/execution/projection_value_plane/
- [proposed] tests/jbc/atom_plane/

**Existing code and permitted read scope:**

- include/Cellerator/execution/projection_value_plane/value_plane_v1.hh
- include/Cellerator/execution/projection_value_plane/composite_plane_v1.hh
- include/Cellerator/execution/projection_value_plane/generation_publication_v1.hh
- include/Cellerator/compute/operation/relation_algebra_v2/relation_algebra.hh

**Explicitly out of scope / forbidden shortcuts:**

- No physical padding hole becomes a parameter, edge, or biological identity.
- Do not publish a value generation until every required primary component is ready.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-P05

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.

### Concrete mechanism

Bind logical or projection-order gradients to exact logical entities/edges, distinguish primary ownership from mirrors, and preserve accumulation algebra.

Workstream mechanism: Create neutral atom-plane bindings over existing structure/value/state/gradient storage, retain exact primary ownership and mirror semantics, bind external readiness/lease tokens, and allow Cellerator stages to emit persistent-order final or partial atoms. Runtime ownership stays external.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Exercise logical-primary and projection-primary modes, direct gradients, dirty subsets, failed publication, and canonical export.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-PLANES; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-P07 — Bind generation_publication_v1 to atom generations

**Repository / subsystem / lane:** Cellerator · `execution/atom_plane and projection_value_plane adapters` · `CE-JBC-L-PLANES`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-PLANES; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Bind generation_publication_v1 to atom generations. Deliver this as one isolated, reviewable step in the Cellerator atom planes, mutable values, gradients, and atom outputs workstream.

**Biological motivation.** Stable biological support with changing activity, state, parameters, and gradients is a dominant recurring pattern and must be represented without rebuilding structure.

**Compiler-architectural reason.** Projection-value-plane v1 already supplies exact primary ownership, alternate mirrors, permanent holes, direct gradients, and generation publication. The atom plane model should adapt and extend it rather than duplicate it.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/execution/atom_plane/
- [proposed] src/execution/atom_plane/
- include/Cellerator/execution/projection_value_plane/
- src/execution/projection_value_plane/
- [proposed] tests/jbc/atom_plane/

**Existing code and permitted read scope:**

- include/Cellerator/execution/projection_value_plane/value_plane_v1.hh
- include/Cellerator/execution/projection_value_plane/composite_plane_v1.hh
- include/Cellerator/execution/projection_value_plane/generation_publication_v1.hh
- include/Cellerator/compute/operation/relation_algebra_v2/relation_algebra.hh

**Explicitly out of scope / forbidden shortcuts:**

- No physical padding hole becomes a parameter, edge, or biological identity.
- Do not publish a value generation until every required primary component is ready.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-P06

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Publish a new atom value generation only after every required primary plane is ready; propagate failure without partial visibility.

Workstream mechanism: Create neutral atom-plane bindings over existing structure/value/state/gradient storage, retain exact primary ownership and mirror semantics, bind external readiness/lease tokens, and allow Cellerator stages to emit persistent-order final or partial atoms. Runtime ownership stays external.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Exercise logical-primary and projection-primary modes, direct gradients, dirty subsets, failed publication, and canonical export.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-PLANES; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-P08 — Emit persistent-order dense result atoms

**Repository / subsystem / lane:** Cellerator · `execution/atom_plane and projection_value_plane adapters` · `CE-JBC-L-PLANES`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-PLANES; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Emit persistent-order dense result atoms. Deliver this as one isolated, reviewable step in the Cellerator atom planes, mutable values, gradients, and atom outputs workstream.

**Biological motivation.** Stable biological support with changing activity, state, parameters, and gradients is a dominant recurring pattern and must be represented without rebuilding structure.

**Compiler-architectural reason.** Projection-value-plane v1 already supplies exact primary ownership, alternate mirrors, permanent holes, direct gradients, and generation publication. The atom plane model should adapt and extend it rather than duplicate it.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/execution/atom_plane/
- [proposed] src/execution/atom_plane/
- include/Cellerator/execution/projection_value_plane/
- src/execution/projection_value_plane/
- [proposed] tests/jbc/atom_plane/

**Existing code and permitted read scope:**

- include/Cellerator/execution/projection_value_plane/value_plane_v1.hh
- include/Cellerator/execution/projection_value_plane/composite_plane_v1.hh
- include/Cellerator/execution/projection_value_plane/generation_publication_v1.hh
- include/Cellerator/compute/operation/relation_algebra_v2/relation_algebra.hh

**Explicitly out of scope / forbidden shortcuts:**

- No physical padding hole becomes a parameter, edge, or biological identity.
- Do not publish a value generation until every required primary component is ready.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-P07

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Allow local operations to write output in a declared noncanonical persistent order with exact inverse mapping and no implicit canonicalization.

Workstream mechanism: Create neutral atom-plane bindings over existing structure/value/state/gradient storage, retain exact primary ownership and mirror semantics, bind external readiness/lease tokens, and allow Cellerator stages to emit persistent-order final or partial atoms. Runtime ownership stays external.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Exercise logical-primary and projection-primary modes, direct gradients, dirty subsets, failed publication, and canonical export.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-PLANES; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-P09 — Emit partial-result atoms

**Repository / subsystem / lane:** Cellerator · `execution/atom_plane and projection_value_plane adapters` · `CE-JBC-L-PLANES`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-PLANES; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Emit partial-result atoms. Deliver this as one isolated, reviewable step in the Cellerator atom planes, mutable values, gradients, and atom outputs workstream.

**Biological motivation.** Stable biological support with changing activity, state, parameters, and gradients is a dominant recurring pattern and must be represented without rebuilding structure.

**Compiler-architectural reason.** Projection-value-plane v1 already supplies exact primary ownership, alternate mirrors, permanent holes, direct gradients, and generation publication. The atom plane model should adapt and extend it rather than duplicate it.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/execution/atom_plane/
- [proposed] src/execution/atom_plane/
- include/Cellerator/execution/projection_value_plane/
- src/execution/projection_value_plane/
- [proposed] tests/jbc/atom_plane/

**Existing code and permitted read scope:**

- include/Cellerator/execution/projection_value_plane/value_plane_v1.hh
- include/Cellerator/execution/projection_value_plane/composite_plane_v1.hh
- include/Cellerator/execution/projection_value_plane/generation_publication_v1.hh
- include/Cellerator/compute/operation/relation_algebra_v2/relation_algebra.hh

**Explicitly out of scope / forbidden shortcuts:**

- No physical padding hole becomes a parameter, edge, or biological identity.
- Do not publish a value generation until every required primary component is ready.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-P08

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Allow local operations to write additive, extrema, moments, log-sum-exp, or provider-defined partial state with exact coverage and algebra identity.

Workstream mechanism: Create neutral atom-plane bindings over existing structure/value/state/gradient storage, retain exact primary ownership and mirror semantics, bind external readiness/lease tokens, and allow Cellerator stages to emit persistent-order final or partial atoms. Runtime ownership stays external.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Exercise logical-primary and projection-primary modes, direct gradients, dirty subsets, failed publication, and canonical export.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-PLANES; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-P10 — Define neutral ready-event and lease binding

**Repository / subsystem / lane:** Cellerator · `execution/atom_plane and projection_value_plane adapters` · `CE-JBC-L-PLANES`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-PLANES; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Define neutral ready-event and lease binding. Deliver this as one isolated, reviewable step in the Cellerator atom planes, mutable values, gradients, and atom outputs workstream.

**Biological motivation.** Stable biological support with changing activity, state, parameters, and gradients is a dominant recurring pattern and must be represented without rebuilding structure.

**Compiler-architectural reason.** Projection-value-plane v1 already supplies exact primary ownership, alternate mirrors, permanent holes, direct gradients, and generation publication. The atom plane model should adapt and extend it rather than duplicate it.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/execution/atom_plane/
- [proposed] src/execution/atom_plane/
- include/Cellerator/execution/projection_value_plane/
- src/execution/projection_value_plane/
- [proposed] tests/jbc/atom_plane/

**Existing code and permitted read scope:**

- include/Cellerator/execution/projection_value_plane/value_plane_v1.hh
- include/Cellerator/execution/projection_value_plane/composite_plane_v1.hh
- include/Cellerator/execution/projection_value_plane/generation_publication_v1.hh
- include/Cellerator/compute/operation/relation_algebra_v2/relation_algebra.hh

**Explicitly out of scope / forbidden shortcuts:**

- No physical padding hole becomes a parameter, edge, or biological identity.
- Do not publish a value generation until every required primary component is ready.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-P09

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.
- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Represent external residency readiness and lifetime through opaque tokens/callback-free launch bindings; CellShard owns resolution, Cellerator validates and consumes the already-resolved view.

Workstream mechanism: Create neutral atom-plane bindings over existing structure/value/state/gradient storage, retain exact primary ownership and mirror semantics, bind external readiness/lease tokens, and allow Cellerator stages to emit persistent-order final or partial atoms. Runtime ownership stays external.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Exercise logical-primary and projection-primary modes, direct gradients, dirty subsets, failed publication, and canonical export.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-PLANES; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


# Workstream CERESUME: Cellerator lowering-resumption contracts

**Repository:** Cellerator  
**Suggested lane:** `CE-JBC-L-RESUMPTION`  
**Barrier:** `JBC-G3-CELLERATOR-FRAGMENT`

## Workstream design

Add a sibling lowering-resumption facade around geometry_acquisition_v2. For each stage, query requirements, validate producer ABI/identities/generations/target, bind the payload, record exactly which phases were bypassed, and fall back to the earliest embedded compatible stage when policy permits.

## Existing live source extended

- include/Cellerator/execution/geometry_acquisition_v2/schema.hh
- include/Cellerator/execution/geometry_acquisition_v2/external_payload.hh
- include/Cellerator/execution/geometry_acquisition_v2/assembly.hh
- include/Cellerator/geometry/persistence/semantic_geometry_image_v1.hh
- include/Cellerator/geometry/persistence/execution_image_v2.hh

## Likely source scope

- [proposed] include/Cellerator/execution/lowering_resumption/
- [proposed] src/execution/lowering_resumption/
- include/Cellerator/execution/geometry_acquisition_v2/
- src/execution/geometry_acquisition_v2/
- [proposed] tests/jbc/resumption/

## Proposed Todos (10)

## CE-JBC-R01 — Define lowering-stage and compatibility status taxonomy

**Repository / subsystem / lane:** Cellerator · `execution/lowering_resumption and acquisition adapters` · `CE-JBC-L-RESUMPTION`  
**Classification:** compatibility/migration  
**Parallelism:** Serial within CE-JBC-L-RESUMPTION; parallel with other provider/workstream lanes after JBC-G3-CELLERATOR-FRAGMENT.

### Why

**Purpose.** Define lowering-stage and compatibility status taxonomy. Deliver this as one isolated, reviewable step in the Cellerator lowering-resumption contracts workstream.

**Biological motivation.** CellShard artifacts should preserve progressively compiled biological structure so repeated workloads can resume Cellerator after evidence, semantic geometry, target-cover, projection, packing, or preparation work.

**Compiler-architectural reason.** Each resumption stage is a typed continuation contract with explicit validity and fallback, not a generic cached byte blob.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/execution/lowering_resumption/
- [proposed] src/execution/lowering_resumption/
- include/Cellerator/execution/geometry_acquisition_v2/
- src/execution/geometry_acquisition_v2/
- [proposed] tests/jbc/resumption/

**Existing code and permitted read scope:**

- include/Cellerator/execution/geometry_acquisition_v2/schema.hh
- include/Cellerator/execution/geometry_acquisition_v2/external_payload.hh
- include/Cellerator/execution/geometry_acquisition_v2/assembly.hh
- include/Cellerator/geometry/persistence/semantic_geometry_image_v1.hh
- include/Cellerator/geometry/persistence/execution_image_v2.hh

**Explicitly out of scope / forbidden shortcuts:**

- Do not persist CUDA pointers, streams, events, graph instances, or live provider state.
- Do not claim a phase was bypassed without instrumentation proving it did not execute.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-F14

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.

### Concrete mechanism

Create stable stages and statuses: invalid, incompatible, compatible, compatible-but-stale-costed, requires-earlier-stage, and preferred. Separate correctness from performance freshness.

Workstream mechanism: Add a sibling lowering-resumption facade around geometry_acquisition_v2. For each stage, query requirements, validate producer ABI/identities/generations/target, bind the payload, record exactly which phases were bypassed, and fall back to the earliest embedded compatible stage when policy permits.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Use phase counters and fault injection to prove bypass, rejection, and fallback behavior.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-RESUMPTION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-R02 — Resume from canonical source input

**Repository / subsystem / lane:** Cellerator · `execution/lowering_resumption and acquisition adapters` · `CE-JBC-L-RESUMPTION`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-RESUMPTION; parallel with other provider/workstream lanes after JBC-G3-CELLERATOR-FRAGMENT.

### Why

**Purpose.** Resume from canonical source input. Deliver this as one isolated, reviewable step in the Cellerator lowering-resumption contracts workstream.

**Biological motivation.** CellShard artifacts should preserve progressively compiled biological structure so repeated workloads can resume Cellerator after evidence, semantic geometry, target-cover, projection, packing, or preparation work.

**Compiler-architectural reason.** Each resumption stage is a typed continuation contract with explicit validity and fallback, not a generic cached byte blob.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/execution/lowering_resumption/
- [proposed] src/execution/lowering_resumption/
- include/Cellerator/execution/geometry_acquisition_v2/
- src/execution/geometry_acquisition_v2/
- [proposed] tests/jbc/resumption/

**Existing code and permitted read scope:**

- include/Cellerator/execution/geometry_acquisition_v2/schema.hh
- include/Cellerator/execution/geometry_acquisition_v2/external_payload.hh
- include/Cellerator/execution/geometry_acquisition_v2/assembly.hh
- include/Cellerator/geometry/persistence/semantic_geometry_image_v1.hh
- include/Cellerator/geometry/persistence/execution_image_v2.hh

**Explicitly out of scope / forbidden shortcuts:**

- Do not persist CUDA pointers, streams, events, graph instances, or live provider state.
- Do not claim a phase was bypassed without instrumentation proving it did not execute.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-R01

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Route through full semantic evidence, geometry, target cover, projection, candidate, and preparation; use as the universal fallback and accounting reference.

Workstream mechanism: Add a sibling lowering-resumption facade around geometry_acquisition_v2. For each stage, query requirements, validate producer ABI/identities/generations/target, bind the payload, record exactly which phases were bypassed, and fall back to the earliest embedded compatible stage when policy permits.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Use phase counters and fault injection to prove bypass, rejection, and fallback behavior.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-RESUMPTION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-R03 — Resume from atom-evidence artifact

**Repository / subsystem / lane:** Cellerator · `execution/lowering_resumption and acquisition adapters` · `CE-JBC-L-RESUMPTION`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-RESUMPTION; parallel with other provider/workstream lanes after JBC-G3-CELLERATOR-FRAGMENT.

### Why

**Purpose.** Resume from atom-evidence artifact. Deliver this as one isolated, reviewable step in the Cellerator lowering-resumption contracts workstream.

**Biological motivation.** CellShard artifacts should preserve progressively compiled biological structure so repeated workloads can resume Cellerator after evidence, semantic geometry, target-cover, projection, packing, or preparation work.

**Compiler-architectural reason.** Each resumption stage is a typed continuation contract with explicit validity and fallback, not a generic cached byte blob.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/execution/lowering_resumption/
- [proposed] src/execution/lowering_resumption/
- include/Cellerator/execution/geometry_acquisition_v2/
- src/execution/geometry_acquisition_v2/
- [proposed] tests/jbc/resumption/

**Existing code and permitted read scope:**

- include/Cellerator/execution/geometry_acquisition_v2/schema.hh
- include/Cellerator/execution/geometry_acquisition_v2/external_payload.hh
- include/Cellerator/execution/geometry_acquisition_v2/assembly.hh
- include/Cellerator/geometry/persistence/semantic_geometry_image_v1.hh
- include/Cellerator/geometry/persistence/execution_image_v2.hh

**Explicitly out of scope / forbidden shortcuts:**

- Do not persist CUDA pointers, streams, events, graph instances, or live provider state.
- Do not claim a phase was bypassed without instrumentation proving it did not execute.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-R02

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Validate evidence identities/provenance/strata and bypass only evidence extraction; still perform exact atom selection/certification and all later lowering.

Workstream mechanism: Add a sibling lowering-resumption facade around geometry_acquisition_v2. For each stage, query requirements, validate producer ABI/identities/generations/target, bind the payload, record exactly which phases were bypassed, and fall back to the earliest embedded compatible stage when policy permits.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Use phase counters and fault injection to prove bypass, rejection, and fallback behavior.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-RESUMPTION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-R04 — Resume from semantic atom or basis artifact

**Repository / subsystem / lane:** Cellerator · `execution/lowering_resumption and acquisition adapters` · `CE-JBC-L-RESUMPTION`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-RESUMPTION; parallel with other provider/workstream lanes after JBC-G3-CELLERATOR-FRAGMENT.

### Why

**Purpose.** Resume from semantic atom or basis artifact. Deliver this as one isolated, reviewable step in the Cellerator lowering-resumption contracts workstream.

**Biological motivation.** CellShard artifacts should preserve progressively compiled biological structure so repeated workloads can resume Cellerator after evidence, semantic geometry, target-cover, projection, packing, or preparation work.

**Compiler-architectural reason.** Each resumption stage is a typed continuation contract with explicit validity and fallback, not a generic cached byte blob.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/execution/lowering_resumption/
- [proposed] src/execution/lowering_resumption/
- include/Cellerator/execution/geometry_acquisition_v2/
- src/execution/geometry_acquisition_v2/
- [proposed] tests/jbc/resumption/

**Existing code and permitted read scope:**

- include/Cellerator/execution/geometry_acquisition_v2/schema.hh
- include/Cellerator/execution/geometry_acquisition_v2/external_payload.hh
- include/Cellerator/execution/geometry_acquisition_v2/assembly.hh
- include/Cellerator/geometry/persistence/semantic_geometry_image_v1.hh
- include/Cellerator/geometry/persistence/execution_image_v2.hh

**Explicitly out of scope / forbidden shortcuts:**

- Do not persist CUDA pointers, streams, events, graph instances, or live provider state.
- Do not claim a phase was bypassed without instrumentation proving it did not execute.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-R03

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Validate certified atoms, grammar/basis identities, exact coverage, orders, and dependencies; bypass proposal discovery/basis construction but still select target realization.

Workstream mechanism: Add a sibling lowering-resumption facade around geometry_acquisition_v2. For each stage, query requirements, validate producer ABI/identities/generations/target, bind the payload, record exactly which phases were bypassed, and fall back to the earliest embedded compatible stage when policy permits.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Use phase counters and fault injection to prove bypass, rejection, and fallback behavior.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-RESUMPTION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-R05 — Resume from target-cover artifact

**Repository / subsystem / lane:** Cellerator · `execution/lowering_resumption and acquisition adapters` · `CE-JBC-L-RESUMPTION`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-RESUMPTION; parallel with other provider/workstream lanes after JBC-G3-CELLERATOR-FRAGMENT.

### Why

**Purpose.** Resume from target-cover artifact. Deliver this as one isolated, reviewable step in the Cellerator lowering-resumption contracts workstream.

**Biological motivation.** CellShard artifacts should preserve progressively compiled biological structure so repeated workloads can resume Cellerator after evidence, semantic geometry, target-cover, projection, packing, or preparation work.

**Compiler-architectural reason.** Each resumption stage is a typed continuation contract with explicit validity and fallback, not a generic cached byte blob.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/execution/lowering_resumption/
- [proposed] src/execution/lowering_resumption/
- include/Cellerator/execution/geometry_acquisition_v2/
- src/execution/geometry_acquisition_v2/
- [proposed] tests/jbc/resumption/

**Existing code and permitted read scope:**

- include/Cellerator/execution/geometry_acquisition_v2/schema.hh
- include/Cellerator/execution/geometry_acquisition_v2/external_payload.hh
- include/Cellerator/execution/geometry_acquisition_v2/assembly.hh
- include/Cellerator/geometry/persistence/semantic_geometry_image_v1.hh
- include/Cellerator/geometry/persistence/execution_image_v2.hh

**Explicitly out of scope / forbidden shortcuts:**

- Do not persist CUDA pointers, streams, events, graph instances, or live provider state.
- Do not claim a phase was bypassed without instrumentation proving it did not execute.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-R04

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Validate device capability class, exact physical contribution cover, residual, operation mixture, and semantic parent; bypass target-cover search while rebuilding projection bytes if needed.

Workstream mechanism: Add a sibling lowering-resumption facade around geometry_acquisition_v2. For each stage, query requirements, validate producer ABI/identities/generations/target, bind the payload, record exactly which phases were bypassed, and fall back to the earliest embedded compatible stage when policy permits.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Use phase counters and fault injection to prove bypass, rejection, and fallback behavior.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-RESUMPTION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-R06 — Resume from physical projection or CPE2 artifact

**Repository / subsystem / lane:** Cellerator · `execution/lowering_resumption and acquisition adapters` · `CE-JBC-L-RESUMPTION`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-RESUMPTION; parallel with other provider/workstream lanes after JBC-G3-CELLERATOR-FRAGMENT.

### Why

**Purpose.** Resume from physical projection or CPE2 artifact. Deliver this as one isolated, reviewable step in the Cellerator lowering-resumption contracts workstream.

**Biological motivation.** CellShard artifacts should preserve progressively compiled biological structure so repeated workloads can resume Cellerator after evidence, semantic geometry, target-cover, projection, packing, or preparation work.

**Compiler-architectural reason.** Each resumption stage is a typed continuation contract with explicit validity and fallback, not a generic cached byte blob.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/execution/lowering_resumption/
- [proposed] src/execution/lowering_resumption/
- include/Cellerator/execution/geometry_acquisition_v2/
- src/execution/geometry_acquisition_v2/
- [proposed] tests/jbc/resumption/

**Existing code and permitted read scope:**

- include/Cellerator/execution/geometry_acquisition_v2/schema.hh
- include/Cellerator/execution/geometry_acquisition_v2/external_payload.hh
- include/Cellerator/execution/geometry_acquisition_v2/assembly.hh
- include/Cellerator/geometry/persistence/semantic_geometry_image_v1.hh
- include/Cellerator/geometry/persistence/execution_image_v2.hh

**Explicitly out of scope / forbidden shortcuts:**

- Do not persist CUDA pointers, streams, events, graph instances, or live provider state.
- Do not claim a phase was bypassed without instrumentation proving it did not execute.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-R05

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Use existing acquisition-v2 validation and embedded semantic fallback; bypass projection construction but retain provider/numeric/structure checks.

Workstream mechanism: Add a sibling lowering-resumption facade around geometry_acquisition_v2. For each stage, query requirements, validate producer ABI/identities/generations/target, bind the payload, record exactly which phases were bypassed, and fall back to the earliest embedded compatible stage when policy permits.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Use phase counters and fault injection to prove bypass, rejection, and fallback behavior.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-RESUMPTION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-R07 — Resume from packed value or operand artifact

**Repository / subsystem / lane:** Cellerator · `execution/lowering_resumption and acquisition adapters` · `CE-JBC-L-RESUMPTION`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-RESUMPTION; parallel with other provider/workstream lanes after JBC-G3-CELLERATOR-FRAGMENT.

### Why

**Purpose.** Resume from packed value or operand artifact. Deliver this as one isolated, reviewable step in the Cellerator lowering-resumption contracts workstream.

**Biological motivation.** CellShard artifacts should preserve progressively compiled biological structure so repeated workloads can resume Cellerator after evidence, semantic geometry, target-cover, projection, packing, or preparation work.

**Compiler-architectural reason.** Each resumption stage is a typed continuation contract with explicit validity and fallback, not a generic cached byte blob.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/execution/lowering_resumption/
- [proposed] src/execution/lowering_resumption/
- include/Cellerator/execution/geometry_acquisition_v2/
- src/execution/geometry_acquisition_v2/
- [proposed] tests/jbc/resumption/

**Existing code and permitted read scope:**

- include/Cellerator/execution/geometry_acquisition_v2/schema.hh
- include/Cellerator/execution/geometry_acquisition_v2/external_payload.hh
- include/Cellerator/execution/geometry_acquisition_v2/assembly.hh
- include/Cellerator/geometry/persistence/semantic_geometry_image_v1.hh
- include/Cellerator/geometry/persistence/execution_image_v2.hh

**Explicitly out of scope / forbidden shortcuts:**

- Do not persist CUDA pointers, streams, events, graph instances, or live provider state.
- Do not claim a phase was bypassed without instrumentation proving it did not execute.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-R06

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Validate structure/value generations, order, numeric type, physical slots, holes, and target ABI; bypass value pack/gather only when exact.

Workstream mechanism: Add a sibling lowering-resumption facade around geometry_acquisition_v2. For each stage, query requirements, validate producer ABI/identities/generations/target, bind the payload, record exactly which phases were bypassed, and fall back to the earliest embedded compatible stage when policy permits.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Use phase counters and fault injection to prove bypass, rejection, and fallback behavior.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-RESUMPTION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-R08 — Resume from executable atom recipe

**Repository / subsystem / lane:** Cellerator · `execution/lowering_resumption and acquisition adapters` · `CE-JBC-L-RESUMPTION`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-RESUMPTION; parallel with other provider/workstream lanes after JBC-G3-CELLERATOR-FRAGMENT.

### Why

**Purpose.** Resume from executable atom recipe. Deliver this as one isolated, reviewable step in the Cellerator lowering-resumption contracts workstream.

**Biological motivation.** CellShard artifacts should preserve progressively compiled biological structure so repeated workloads can resume Cellerator after evidence, semantic geometry, target-cover, projection, packing, or preparation work.

**Compiler-architectural reason.** Each resumption stage is a typed continuation contract with explicit validity and fallback, not a generic cached byte blob.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/execution/lowering_resumption/
- [proposed] src/execution/lowering_resumption/
- include/Cellerator/execution/geometry_acquisition_v2/
- src/execution/geometry_acquisition_v2/
- [proposed] tests/jbc/resumption/

**Existing code and permitted read scope:**

- include/Cellerator/execution/geometry_acquisition_v2/schema.hh
- include/Cellerator/execution/geometry_acquisition_v2/external_payload.hh
- include/Cellerator/execution/geometry_acquisition_v2/assembly.hh
- include/Cellerator/geometry/persistence/semantic_geometry_image_v1.hh
- include/Cellerator/geometry/persistence/execution_image_v2.hh

**Explicitly out of scope / forbidden shortcuts:**

- Do not persist CUDA pointers, streams, events, graph instances, or live provider state.
- Do not claim a phase was bypassed without instrumentation proving it did not execute.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-R07

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Validate candidate/provider/build IDs, stage graph, resource requirements, mutable parameter layout, and graph-capture recipe; reprepare runtime handles and addresses.

Workstream mechanism: Add a sibling lowering-resumption facade around geometry_acquisition_v2. For each stage, query requirements, validate producer ABI/identities/generations/target, bind the payload, record exactly which phases were bypassed, and fall back to the earliest embedded compatible stage when policy permits.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Use phase counters and fault injection to prove bypass, rejection, and fallback behavior.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-RESUMPTION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-R09 — Validate topology-linked local realization

**Repository / subsystem / lane:** Cellerator · `execution/lowering_resumption and acquisition adapters` · `CE-JBC-L-RESUMPTION`  
**Classification:** validation/evidence  
**Parallelism:** Serial within CE-JBC-L-RESUMPTION; parallel with other provider/workstream lanes after JBC-G3-CELLERATOR-FRAGMENT.

### Why

**Purpose.** Validate topology-linked local realization. Deliver this as one isolated, reviewable step in the Cellerator lowering-resumption contracts workstream.

**Biological motivation.** CellShard artifacts should preserve progressively compiled biological structure so repeated workloads can resume Cellerator after evidence, semantic geometry, target-cover, projection, packing, or preparation work.

**Compiler-architectural reason.** Each resumption stage is a typed continuation contract with explicit validity and fallback, not a generic cached byte blob.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/execution/lowering_resumption/
- [proposed] src/execution/lowering_resumption/
- include/Cellerator/execution/geometry_acquisition_v2/
- src/execution/geometry_acquisition_v2/
- [proposed] tests/jbc/resumption/

**Existing code and permitted read scope:**

- include/Cellerator/execution/geometry_acquisition_v2/schema.hh
- include/Cellerator/execution/geometry_acquisition_v2/external_payload.hh
- include/Cellerator/execution/geometry_acquisition_v2/assembly.hh
- include/Cellerator/geometry/persistence/semantic_geometry_image_v1.hh
- include/Cellerator/geometry/persistence/execution_image_v2.hh

**Explicitly out of scope / forbidden shortcuts:**

- Do not persist CUDA pointers, streams, events, graph instances, or live provider state.
- Do not claim a phase was bypassed without instrumentation proving it did not execute.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-R08

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- Focused tests and adversarial fixtures demonstrating both success and explicit rejection paths.

### Concrete mechanism

Accept topology-linked metadata only as constraints on the local fragment—device capability, selected projection/candidate, memory tier—without making Cellerator own global topology.

Workstream mechanism: Add a sibling lowering-resumption facade around geometry_acquisition_v2. For each stage, query requirements, validate producer ABI/identities/generations/target, bind the payload, record exactly which phases were bypassed, and fall back to the earliest embedded compatible stage when policy permits.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Use phase counters and fault injection to prove bypass, rejection, and fallback behavior.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-RESUMPTION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-R10 — Instrument phase bypass and earliest-compatible fallback

**Repository / subsystem / lane:** Cellerator · `execution/lowering_resumption and acquisition adapters` · `CE-JBC-L-RESUMPTION`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-RESUMPTION; parallel with other provider/workstream lanes after JBC-G3-CELLERATOR-FRAGMENT.

### Why

**Purpose.** Instrument phase bypass and earliest-compatible fallback. Deliver this as one isolated, reviewable step in the Cellerator lowering-resumption contracts workstream.

**Biological motivation.** CellShard artifacts should preserve progressively compiled biological structure so repeated workloads can resume Cellerator after evidence, semantic geometry, target-cover, projection, packing, or preparation work.

**Compiler-architectural reason.** Each resumption stage is a typed continuation contract with explicit validity and fallback, not a generic cached byte blob.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/execution/lowering_resumption/
- [proposed] src/execution/lowering_resumption/
- include/Cellerator/execution/geometry_acquisition_v2/
- src/execution/geometry_acquisition_v2/
- [proposed] tests/jbc/resumption/

**Existing code and permitted read scope:**

- include/Cellerator/execution/geometry_acquisition_v2/schema.hh
- include/Cellerator/execution/geometry_acquisition_v2/external_payload.hh
- include/Cellerator/execution/geometry_acquisition_v2/assembly.hh
- include/Cellerator/geometry/persistence/semantic_geometry_image_v1.hh
- include/Cellerator/geometry/persistence/execution_image_v2.hh

**Explicitly out of scope / forbidden shortcuts:**

- Do not persist CUDA pointers, streams, events, graph instances, or live provider state.
- Do not claim a phase was bypassed without instrumentation proving it did not execute.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-R09

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A bounded implementation unit, focused tests, mechanism statistics, and an integration receipt.

### Concrete mechanism

Expose per-phase counters and diagnostics; tests must prove skipped phases do not execute and incompatible stages fall back exactly as policy declares.

Workstream mechanism: Add a sibling lowering-resumption facade around geometry_acquisition_v2. For each stage, query requirements, validate producer ABI/identities/generations/target, bind the payload, record exactly which phases were bypassed, and fall back to the earliest embedded compatible stage when policy permits.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Use phase counters and fault injection to prove bypass, rejection, and fallback behavior.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-RESUMPTION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


# Workstream CEXOP: Cross-operation Cellerator projection families

**Repository:** Cellerator  
**Suggested lane:** `CE-JBC-L-CROSSOP`  
**Barrier:** `JBC-G3-CELLERATOR-FRAGMENT`

## Workstream design

Name one semantic support-family identity and attach several independently target-specific physical views sharing the canonical edge spine. Compare per-operation specialists with generalized families under operation-mixture and transition costs; retain a Pareto frontier.

## Existing live source extended

- include/Cellerator/geometry/relation_cover.hh
- include/Cellerator/execution/projection_value_plane/
- include/Cellerator/compute/operation/candidate_catalog_v3/
- include/Cellerator/compute/operation/relation_algebra_v2/

## Likely source scope

- [proposed] include/Cellerator/compute/projection_family/
- [proposed] src/compute/projection_family/
- [proposed] tests/jbc/cross_operation/
- [proposed] bench/jbc/cross_operation/

## Proposed Todos (8)

## CE-JBC-X01 — Define operation-polymorphic support-family identity

**Repository / subsystem / lane:** Cellerator · `compute/projection_family` · `CE-JBC-L-CROSSOP`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-CROSSOP; parallel with other provider/workstream lanes after JBC-G3-CELLERATOR-FRAGMENT.

### Why

**Purpose.** Define operation-polymorphic support-family identity. Deliver this as one isolated, reviewable step in the Cross-operation Cellerator projection families workstream.

**Biological motivation.** The same biological support is repeatedly traversed by forward apply, transpose, support contraction, gating, normalization, moments, bundles, and gradients.

**Compiler-architectural reason.** The local compiler must expose specialized and operation-polymorphic physical-view families so CellShard can trade a small local loss for major cross-operation storage, preparation, and movement savings.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/compute/projection_family/
- [proposed] src/compute/projection_family/
- [proposed] tests/jbc/cross_operation/
- [proposed] bench/jbc/cross_operation/

**Existing code and permitted read scope:**

- include/Cellerator/geometry/relation_cover.hh
- include/Cellerator/execution/projection_value_plane/
- include/Cellerator/compute/operation/candidate_catalog_v3/
- include/Cellerator/compute/operation/relation_algebra_v2/

**Explicitly out of scope / forbidden shortcuts:**

- Do not force one physical projection to serve every operation.
- Do not merge physical mirrors into primary contribution ownership.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-F14
- CE-JBC-P10

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.

### Concrete mechanism

Name one exact biological relation/support family independently of any single physical projection or operation and attach its canonical edge spine and axes.

Workstream mechanism: Name one semantic support-family identity and attach several independently target-specific physical views sharing the canonical edge spine. Compare per-operation specialists with generalized families under operation-mixture and transition costs; retain a Pareto frontier.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Validate every view against the same logical edge spine and operation-specific canonical referee.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-CROSSOP; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-X02 — Build the forward relation-apply physical view

**Repository / subsystem / lane:** Cellerator · `compute/projection_family` · `CE-JBC-L-CROSSOP`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-CROSSOP; parallel with other provider/workstream lanes after JBC-G3-CELLERATOR-FRAGMENT.

### Why

**Purpose.** Build the forward relation-apply physical view. Deliver this as one isolated, reviewable step in the Cross-operation Cellerator projection families workstream.

**Biological motivation.** The same biological support is repeatedly traversed by forward apply, transpose, support contraction, gating, normalization, moments, bundles, and gradients.

**Compiler-architectural reason.** The local compiler must expose specialized and operation-polymorphic physical-view families so CellShard can trade a small local loss for major cross-operation storage, preparation, and movement savings.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/compute/projection_family/
- [proposed] src/compute/projection_family/
- [proposed] tests/jbc/cross_operation/
- [proposed] bench/jbc/cross_operation/

**Existing code and permitted read scope:**

- include/Cellerator/geometry/relation_cover.hh
- include/Cellerator/execution/projection_value_plane/
- include/Cellerator/compute/operation/candidate_catalog_v3/
- include/Cellerator/compute/operation/relation_algebra_v2/

**Explicitly out of scope / forbidden shortcuts:**

- Do not force one physical projection to serve every operation.
- Do not merge physical mirrors into primary contribution ownership.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-X01

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.
- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Create or reference the best forward-oriented local projection/order and advertise its atom requirements and output affordance.

Workstream mechanism: Name one semantic support-family identity and attach several independently target-specific physical views sharing the canonical edge spine. Compare per-operation specialists with generalized families under operation-mixture and transition costs; retain a Pareto frontier.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Validate every view against the same logical edge spine and operation-specific canonical referee.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-CROSSOP; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-X03 — Build the transpose physical view

**Repository / subsystem / lane:** Cellerator · `compute/projection_family` · `CE-JBC-L-CROSSOP`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-CROSSOP; parallel with other provider/workstream lanes after JBC-G3-CELLERATOR-FRAGMENT.

### Why

**Purpose.** Build the transpose physical view. Deliver this as one isolated, reviewable step in the Cross-operation Cellerator projection families workstream.

**Biological motivation.** The same biological support is repeatedly traversed by forward apply, transpose, support contraction, gating, normalization, moments, bundles, and gradients.

**Compiler-architectural reason.** The local compiler must expose specialized and operation-polymorphic physical-view families so CellShard can trade a small local loss for major cross-operation storage, preparation, and movement savings.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/compute/projection_family/
- [proposed] src/compute/projection_family/
- [proposed] tests/jbc/cross_operation/
- [proposed] bench/jbc/cross_operation/

**Existing code and permitted read scope:**

- include/Cellerator/geometry/relation_cover.hh
- include/Cellerator/execution/projection_value_plane/
- include/Cellerator/compute/operation/candidate_catalog_v3/
- include/Cellerator/compute/operation/relation_algebra_v2/

**Explicitly out of scope / forbidden shortcuts:**

- Do not force one physical projection to serve every operation.
- Do not merge physical mirrors into primary contribution ownership.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-X02

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.
- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Create an independently optimized transpose view sharing the semantic edge spine but not forced to share forward grouping or schedule.

Workstream mechanism: Name one semantic support-family identity and attach several independently target-specific physical views sharing the canonical edge spine. Compare per-operation specialists with generalized families under operation-mixture and transition costs; retain a Pareto frontier.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Validate every view against the same logical edge spine and operation-specific canonical referee.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-CROSSOP; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-X04 — Build the support-contraction physical view

**Repository / subsystem / lane:** Cellerator · `compute/projection_family` · `CE-JBC-L-CROSSOP`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-CROSSOP; parallel with other provider/workstream lanes after JBC-G3-CELLERATOR-FRAGMENT.

### Why

**Purpose.** Build the support-contraction physical view. Deliver this as one isolated, reviewable step in the Cross-operation Cellerator projection families workstream.

**Biological motivation.** The same biological support is repeatedly traversed by forward apply, transpose, support contraction, gating, normalization, moments, bundles, and gradients.

**Compiler-architectural reason.** The local compiler must expose specialized and operation-polymorphic physical-view families so CellShard can trade a small local loss for major cross-operation storage, preparation, and movement savings.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/compute/projection_family/
- [proposed] src/compute/projection_family/
- [proposed] tests/jbc/cross_operation/
- [proposed] bench/jbc/cross_operation/

**Existing code and permitted read scope:**

- include/Cellerator/geometry/relation_cover.hh
- include/Cellerator/execution/projection_value_plane/
- include/Cellerator/compute/operation/candidate_catalog_v3/
- include/Cellerator/compute/operation/relation_algebra_v2/

**Explicitly out of scope / forbidden shortcuts:**

- Do not force one physical projection to serve every operation.
- Do not merge physical mirrors into primary contribution ownership.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-X03

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.
- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Expose edge/rectangle and state-order requirements for contract_on_support while sharing exact support identity and value maps.

Workstream mechanism: Name one semantic support-family identity and attach several independently target-specific physical views sharing the canonical edge spine. Compare per-operation specialists with generalized families under operation-mixture and transition costs; retain a Pareto frontier.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Validate every view against the same logical edge spine and operation-specific canonical referee.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-CROSSOP; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-X05 — Build segment and gate physical views

**Repository / subsystem / lane:** Cellerator · `compute/projection_family` · `CE-JBC-L-CROSSOP`  
**Classification:** validation/evidence  
**Parallelism:** Serial within CE-JBC-L-CROSSOP; parallel with other provider/workstream lanes after JBC-G3-CELLERATOR-FRAGMENT.

### Why

**Purpose.** Build segment and gate physical views. Deliver this as one isolated, reviewable step in the Cross-operation Cellerator projection families workstream.

**Biological motivation.** The same biological support is repeatedly traversed by forward apply, transpose, support contraction, gating, normalization, moments, bundles, and gradients.

**Compiler-architectural reason.** The local compiler must expose specialized and operation-polymorphic physical-view families so CellShard can trade a small local loss for major cross-operation storage, preparation, and movement savings.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/compute/projection_family/
- [proposed] src/compute/projection_family/
- [proposed] tests/jbc/cross_operation/
- [proposed] bench/jbc/cross_operation/

**Existing code and permitted read scope:**

- include/Cellerator/geometry/relation_cover.hh
- include/Cellerator/execution/projection_value_plane/
- include/Cellerator/compute/operation/candidate_catalog_v3/
- include/Cellerator/compute/operation/relation_algebra_v2/

**Explicitly out of scope / forbidden shortcuts:**

- Do not force one physical projection to serve every operation.
- Do not merge physical mirrors into primary contribution ownership.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-X04

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.
- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.
- A machine-readable evidence record containing the baseline, candidate, complete-cost metrics, environment identity, and promotion disposition.

### Concrete mechanism

Attach persistent segment maps and gate-index views where they reuse the same source/destination/edge order.

Workstream mechanism: Name one semantic support-family identity and attach several independently target-specific physical views sharing the canonical edge spine. Compare per-operation specialists with generalized families under operation-mixture and transition costs; retain a Pareto frontier.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Validate every view against the same logical edge spine and operation-specific canonical referee.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-CROSSOP; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-X06 — Build the shared value and gradient identity spine

**Repository / subsystem / lane:** Cellerator · `compute/projection_family` · `CE-JBC-L-CROSSOP`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-CROSSOP; parallel with other provider/workstream lanes after JBC-G3-CELLERATOR-FRAGMENT.

### Why

**Purpose.** Build the shared value and gradient identity spine. Deliver this as one isolated, reviewable step in the Cross-operation Cellerator projection families workstream.

**Biological motivation.** The same biological support is repeatedly traversed by forward apply, transpose, support contraction, gating, normalization, moments, bundles, and gradients.

**Compiler-architectural reason.** The local compiler must expose specialized and operation-polymorphic physical-view families so CellShard can trade a small local loss for major cross-operation storage, preparation, and movement savings.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/compute/projection_family/
- [proposed] src/compute/projection_family/
- [proposed] tests/jbc/cross_operation/
- [proposed] bench/jbc/cross_operation/

**Existing code and permitted read scope:**

- include/Cellerator/geometry/relation_cover.hh
- include/Cellerator/execution/projection_value_plane/
- include/Cellerator/compute/operation/candidate_catalog_v3/
- include/Cellerator/compute/operation/relation_algebra_v2/

**Explicitly out of scope / forbidden shortcuts:**

- Do not force one physical projection to serve every operation.
- Do not merge physical mirrors into primary contribution ownership.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-X05

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.
- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Permit logical and projection-primary value/gradient planes to bind across forward, transpose, contraction, and update views without losing canonical edge identity.

Workstream mechanism: Name one semantic support-family identity and attach several independently target-specific physical views sharing the canonical edge spine. Compare per-operation specialists with generalized families under operation-mixture and transition costs; retain a Pareto frontier.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Validate every view against the same logical edge spine and operation-specific canonical referee.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-CROSSOP; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-X07 — Compare specialized and generalized view families

**Repository / subsystem / lane:** Cellerator · `compute/projection_family` · `CE-JBC-L-CROSSOP`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-CROSSOP; parallel with other provider/workstream lanes after JBC-G3-CELLERATOR-FRAGMENT.

### Why

**Purpose.** Compare specialized and generalized view families. Deliver this as one isolated, reviewable step in the Cross-operation Cellerator projection families workstream.

**Biological motivation.** The same biological support is repeatedly traversed by forward apply, transpose, support contraction, gating, normalization, moments, bundles, and gradients.

**Compiler-architectural reason.** The local compiler must expose specialized and operation-polymorphic physical-view families so CellShard can trade a small local loss for major cross-operation storage, preparation, and movement savings.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/compute/projection_family/
- [proposed] src/compute/projection_family/
- [proposed] tests/jbc/cross_operation/
- [proposed] bench/jbc/cross_operation/

**Existing code and permitted read scope:**

- include/Cellerator/geometry/relation_cover.hh
- include/Cellerator/execution/projection_value_plane/
- include/Cellerator/compute/operation/candidate_catalog_v3/
- include/Cellerator/compute/operation/relation_algebra_v2/

**Explicitly out of scope / forbidden shortcuts:**

- Do not force one physical projection to serve every operation.
- Do not merge physical mirrors into primary contribution ownership.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-X06

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.
- A machine-readable evidence record containing the baseline, candidate, complete-cost metrics, environment identity, and promotion disposition.

### Concrete mechanism

Construct per-operation specialists and one or more generalized families; quantify local kernel loss, avoided transforms, storage, preparation, and graph-family reuse.

Workstream mechanism: Name one semantic support-family identity and attach several independently target-specific physical views sharing the canonical edge spine. Compare per-operation specialists with generalized families under operation-mixture and transition costs; retain a Pareto frontier.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Validate every view against the same logical edge spine and operation-specific canonical referee.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-CROSSOP; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-X08 — Emit cross-operation Pareto frontier and promotion disposition

**Repository / subsystem / lane:** Cellerator · `compute/projection_family` · `CE-JBC-L-CROSSOP`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-CROSSOP; parallel with other provider/workstream lanes after JBC-G3-CELLERATOR-FRAGMENT.

### Why

**Purpose.** Emit cross-operation Pareto frontier and promotion disposition. Deliver this as one isolated, reviewable step in the Cross-operation Cellerator projection families workstream.

**Biological motivation.** The same biological support is repeatedly traversed by forward apply, transpose, support contraction, gating, normalization, moments, bundles, and gradients.

**Compiler-architectural reason.** The local compiler must expose specialized and operation-polymorphic physical-view families so CellShard can trade a small local loss for major cross-operation storage, preparation, and movement savings.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/compute/projection_family/
- [proposed] src/compute/projection_family/
- [proposed] tests/jbc/cross_operation/
- [proposed] bench/jbc/cross_operation/

**Existing code and permitted read scope:**

- include/Cellerator/geometry/relation_cover.hh
- include/Cellerator/execution/projection_value_plane/
- include/Cellerator/compute/operation/candidate_catalog_v3/
- include/Cellerator/compute/operation/relation_algebra_v2/

**Explicitly out of scope / forbidden shortcuts:**

- Do not force one physical projection to serve every operation.
- Do not merge physical mirrors into primary contribution ownership.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-X07

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.
- A machine-readable evidence record containing the baseline, candidate, complete-cost metrics, environment identity, and promotion disposition.

### Concrete mechanism

Retain nondominated families and mark generalized views promoted only for workload regions where complete graph cost wins; otherwise preserve specialists.

Workstream mechanism: Name one semantic support-family identity and attach several independently target-specific physical views sharing the canonical edge spine. Compare per-operation specialists with generalized families under operation-mixture and transition costs; retain a Pareto frontier.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Validate every view against the same logical edge spine and operation-specific canonical referee.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-CROSSOP; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


# Workstream CECOST: External global costs and bounded joint compiler exchange

**Repository:** Cellerator  
**Suggested lane:** `CE-JBC-L-EXTERNAL-COST`  
**Barrier:** `JBC-G3-CELLERATOR-FRAGMENT`

## Workstream design

Define an external cost vector, add it to local complete-cost and geometry objectives, implement a deterministic one-pass frontier, then a bounded request/response exchange. Column-generation pricing remains an experimental provider with duplicate and budget stops.

## Existing live source extended

- include/Cellerator/planner/
- include/Cellerator/geometry/compiler/v2/
- include/Cellerator/geometry/optimizer/portfolio_v1.hh
- include/Cellerator/compute/operation/candidate_catalog_v3/

## Likely source scope

- [proposed] include/Cellerator/planner/external_cost/
- [proposed] src/planner/external_cost/
- [proposed] tests/jbc/external_cost/
- [proposed] bench/jbc/compiler_exchange/

## Proposed Todos (6)

## CE-JBC-C01 — Define generic external cost vector v1

**Repository / subsystem / lane:** Cellerator · `planner/external_cost and geometry objective adapters` · `CE-JBC-L-EXTERNAL-COST`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-EXTERNAL-COST; parallel with other provider/workstream lanes after JBC-G3-CELLERATOR-FRAGMENT.

### Why

**Purpose.** Define generic external cost vector v1. Deliver this as one isolated, reviewable step in the External global costs and bounded joint compiler exchange workstream.

**Biological motivation.** A locally unusual order or projection may be globally superior when it preserves a reusable biological atom, avoids remote movement, or serves several operations.

**Compiler-architectural reason.** Cellerator needs a generic, CellShard-neutral way to accept global storage, movement, reduction, and reuse prices while retaining local correctness authority.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/planner/external_cost/
- [proposed] src/planner/external_cost/
- [proposed] tests/jbc/external_cost/
- [proposed] bench/jbc/compiler_exchange/

**Existing code and permitted read scope:**

- include/Cellerator/planner/
- include/Cellerator/geometry/compiler/v2/
- include/Cellerator/geometry/optimizer/portfolio_v1.hh
- include/Cellerator/compute/operation/candidate_catalog_v3/

**Explicitly out of scope / forbidden shortcuts:**

- Do not make Cellerator depend on a CellShard scheduler callback.
- Do not allow global prices to override correctness or numerical compatibility filters.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-F14

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.

### Concrete mechanism

Represent explicit prices/weights for storage, object build, replication, local/remote movement, canonicalization, layout transition, partial combine, graph launch, memory, invalidation risk, latency, and throughput.

Workstream mechanism: Define an external cost vector, add it to local complete-cost and geometry objectives, implement a deterministic one-pass frontier, then a bounded request/response exchange. Column-generation pricing remains an experimental provider with duplicate and budget stops.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Use planted cost surfaces where the globally preferred candidate differs from the local kernel winner.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-EXTERNAL-COST; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-C02 — Inject external costs into the local complete-cost planner

**Repository / subsystem / lane:** Cellerator · `planner/external_cost and geometry objective adapters` · `CE-JBC-L-EXTERNAL-COST`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-EXTERNAL-COST; parallel with other provider/workstream lanes after JBC-G3-CELLERATOR-FRAGMENT.

### Why

**Purpose.** Inject external costs into the local complete-cost planner. Deliver this as one isolated, reviewable step in the External global costs and bounded joint compiler exchange workstream.

**Biological motivation.** A locally unusual order or projection may be globally superior when it preserves a reusable biological atom, avoids remote movement, or serves several operations.

**Compiler-architectural reason.** Cellerator needs a generic, CellShard-neutral way to accept global storage, movement, reduction, and reuse prices while retaining local correctness authority.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/planner/external_cost/
- [proposed] src/planner/external_cost/
- [proposed] tests/jbc/external_cost/
- [proposed] bench/jbc/compiler_exchange/

**Existing code and permitted read scope:**

- include/Cellerator/planner/
- include/Cellerator/geometry/compiler/v2/
- include/Cellerator/geometry/optimizer/portfolio_v1.hh
- include/Cellerator/compute/operation/candidate_catalog_v3/

**Explicitly out of scope / forbidden shortcuts:**

- Do not make Cellerator depend on a CellShard scheduler callback.
- Do not allow global prices to override correctness or numerical compatibility filters.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-C01

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Add the supplied vector to candidate stage accounting without changing correctness filters or existing local cost components. Record external and local contributions separately.

Workstream mechanism: Define an external cost vector, add it to local complete-cost and geometry objectives, implement a deterministic one-pass frontier, then a bounded request/response exchange. Column-generation pricing remains an experimental provider with duplicate and budget stops.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Use planted cost surfaces where the globally preferred candidate differs from the local kernel winner.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-EXTERNAL-COST; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-C03 — Inject global reuse and movement prices into geometry objectives

**Repository / subsystem / lane:** Cellerator · `planner/external_cost and geometry objective adapters` · `CE-JBC-L-EXTERNAL-COST`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-EXTERNAL-COST; parallel with other provider/workstream lanes after JBC-G3-CELLERATOR-FRAGMENT.

### Why

**Purpose.** Inject global reuse and movement prices into geometry objectives. Deliver this as one isolated, reviewable step in the External global costs and bounded joint compiler exchange workstream.

**Biological motivation.** A locally unusual order or projection may be globally superior when it preserves a reusable biological atom, avoids remote movement, or serves several operations.

**Compiler-architectural reason.** Cellerator needs a generic, CellShard-neutral way to accept global storage, movement, reduction, and reuse prices while retaining local correctness authority.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/planner/external_cost/
- [proposed] src/planner/external_cost/
- [proposed] tests/jbc/external_cost/
- [proposed] bench/jbc/compiler_exchange/

**Existing code and permitted read scope:**

- include/Cellerator/planner/
- include/Cellerator/geometry/compiler/v2/
- include/Cellerator/geometry/optimizer/portfolio_v1.hh
- include/Cellerator/compute/operation/candidate_catalog_v3/

**Explicitly out of scope / forbidden shortcuts:**

- Do not make Cellerator depend on a CellShard scheduler callback.
- Do not allow global prices to override correctness or numerical compatibility filters.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-C02

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Allow semantic/target geometry strategies to score persistent atom boundaries, order continuity, remote cuts, and cross-operation reuse through generic external features.

Workstream mechanism: Define an external cost vector, add it to local complete-cost and geometry objectives, implement a deterministic one-pass frontier, then a bounded request/response exchange. Column-generation pricing remains an experimental provider with duplicate and budget stops.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Use planted cost surfaces where the globally preferred candidate differs from the local kernel winner.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-EXTERNAL-COST; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-C04 — Implement one-pass external-cost frontier

**Repository / subsystem / lane:** Cellerator · `planner/external_cost and geometry objective adapters` · `CE-JBC-L-EXTERNAL-COST`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-EXTERNAL-COST; parallel with other provider/workstream lanes after JBC-G3-CELLERATOR-FRAGMENT.

### Why

**Purpose.** Implement one-pass external-cost frontier. Deliver this as one isolated, reviewable step in the External global costs and bounded joint compiler exchange workstream.

**Biological motivation.** A locally unusual order or projection may be globally superior when it preserves a reusable biological atom, avoids remote movement, or serves several operations.

**Compiler-architectural reason.** Cellerator needs a generic, CellShard-neutral way to accept global storage, movement, reduction, and reuse prices while retaining local correctness authority.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/planner/external_cost/
- [proposed] src/planner/external_cost/
- [proposed] tests/jbc/external_cost/
- [proposed] bench/jbc/compiler_exchange/

**Existing code and permitted read scope:**

- include/Cellerator/planner/
- include/Cellerator/geometry/compiler/v2/
- include/Cellerator/geometry/optimizer/portfolio_v1.hh
- include/Cellerator/compute/operation/candidate_catalog_v3/

**Explicitly out of scope / forbidden shortcuts:**

- Do not make Cellerator depend on a CellShard scheduler callback.
- Do not allow global prices to override correctness or numerical compatibility filters.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-C03

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Given fixed external costs, run local geometry/candidate search once and emit a bounded deterministic Pareto frontier plus sensitivity diagnostics.

Workstream mechanism: Define an external cost vector, add it to local complete-cost and geometry objectives, implement a deterministic one-pass frontier, then a bounded request/response exchange. Column-generation pricing remains an experimental provider with duplicate and budget stops.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Use planted cost surfaces where the globally preferred candidate differs from the local kernel winner.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-EXTERNAL-COST; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-C05 — Implement bounded Cellerator–caller compiler exchange

**Repository / subsystem / lane:** Cellerator · `planner/external_cost and geometry objective adapters` · `CE-JBC-L-EXTERNAL-COST`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-EXTERNAL-COST; parallel with other provider/workstream lanes after JBC-G3-CELLERATOR-FRAGMENT.

### Why

**Purpose.** Implement bounded Cellerator–caller compiler exchange. Deliver this as one isolated, reviewable step in the External global costs and bounded joint compiler exchange workstream.

**Biological motivation.** A locally unusual order or projection may be globally superior when it preserves a reusable biological atom, avoids remote movement, or serves several operations.

**Compiler-architectural reason.** Cellerator needs a generic, CellShard-neutral way to accept global storage, movement, reduction, and reuse prices while retaining local correctness authority.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/planner/external_cost/
- [proposed] src/planner/external_cost/
- [proposed] tests/jbc/external_cost/
- [proposed] bench/jbc/compiler_exchange/

**Existing code and permitted read scope:**

- include/Cellerator/planner/
- include/Cellerator/geometry/compiler/v2/
- include/Cellerator/geometry/optimizer/portfolio_v1.hh
- include/Cellerator/compute/operation/candidate_catalog_v3/

**Explicitly out of scope / forbidden shortcuts:**

- Do not make Cellerator depend on a CellShard scheduler callback.
- Do not allow global prices to override correctness or numerical compatibility filters.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-C04

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Support a finite request/response loop where a caller updates prices or constraints and Cellerator emits only new/different candidates; enforce rounds, time, memory, and duplicate stops.

Workstream mechanism: Define an external cost vector, add it to local complete-cost and geometry objectives, implement a deterministic one-pass frontier, then a bounded request/response exchange. Column-generation pricing remains an experimental provider with duplicate and budget stops.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Use planted cost surfaces where the globally preferred candidate differs from the local kernel winner.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-EXTERNAL-COST; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-C06 — Implement experimental local pricing oracle for column generation

**Repository / subsystem / lane:** Cellerator · `planner/external_cost and geometry objective adapters` · `CE-JBC-L-EXTERNAL-COST`  
**Classification:** experimental candidate; baseline and negative result required  
**Parallelism:** Serial within CE-JBC-L-EXTERNAL-COST; parallel with other provider/workstream lanes after JBC-G3-CELLERATOR-FRAGMENT.

### Why

**Purpose.** Implement experimental local pricing oracle for column generation. Deliver this as one isolated, reviewable step in the External global costs and bounded joint compiler exchange workstream.

**Biological motivation.** A locally unusual order or projection may be globally superior when it preserves a reusable biological atom, avoids remote movement, or serves several operations.

**Compiler-architectural reason.** Cellerator needs a generic, CellShard-neutral way to accept global storage, movement, reduction, and reuse prices while retaining local correctness authority.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/Cellerator/planner/external_cost/
- [proposed] src/planner/external_cost/
- [proposed] tests/jbc/external_cost/
- [proposed] bench/jbc/compiler_exchange/

**Existing code and permitted read scope:**

- include/Cellerator/planner/
- include/Cellerator/geometry/compiler/v2/
- include/Cellerator/geometry/optimizer/portfolio_v1.hh
- include/Cellerator/compute/operation/candidate_catalog_v3/

**Explicitly out of scope / forbidden shortcuts:**

- Do not make Cellerator depend on a CellShard scheduler callback.
- Do not allow global prices to override correctness or numerical compatibility filters.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-C05

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Given dual/shadow prices from an external master problem, search for a fragment/projection with negative reduced global cost; return no-column as a valid result and never block the one-pass path.

Workstream mechanism: Define an external cost vector, add it to local complete-cost and geometry objectives, implement a deterministic one-pass frontier, then a bounded request/response exchange. Column-generation pricing remains an experimental provider with duplicate and budget stops.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Use planted cost surfaces where the globally preferred candidate differs from the local kernel winner.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-EXTERNAL-COST; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


# Workstream CEVAL: Cellerator exact verification, profiling, packaging, and integration

**Repository:** Cellerator  
**Suggested lane:** `CE-JBC-L-VERIFY-INTEGRATE`  
**Barrier:** `JBC-G5-PARTIAL-ARTIFACT`

## Workstream design

Use separate validators and canonical referees, stable mechanism IDs, complete phase accounting, standalone/embedded build matrices, and integration-only edits to root CMake, umbrella headers, registries, and component documentation.

## Existing live source extended

- all Cellerator JBC interfaces and implementations
- current standalone Cellerator targets
- components/CellShard

## Likely source scope

- [proposed] tests/jbc/
- [proposed] bench/jbc/
- [proposed] include/Cellerator/profiling/joint_compiler/
- CMakeLists.txt
- components/README.md

## Proposed Todos (6)

## CE-JBC-V01 — Implement an independent atom-fragment structural verifier

**Repository / subsystem / lane:** Cellerator · `tests/jbc, profiling, component bridge, integration` · `CE-JBC-L-VERIFY-INTEGRATE`  
**Classification:** validation/evidence  
**Parallelism:** Serial within CE-JBC-L-VERIFY-INTEGRATE; parallel with other provider/workstream lanes after JBC-G5-PARTIAL-ARTIFACT.

### Why

**Purpose.** Implement an independent atom-fragment structural verifier. Deliver this as one isolated, reviewable step in the Cellerator exact verification, profiling, packaging, and integration workstream.

**Biological motivation.** The integrated compiler is scientifically defensible only if exact atom-local execution and biological identity recovery remain independently testable while standalone Cellerator remains intact.

**Compiler-architectural reason.** This lane owns independent verification, profiler export, the privileged component bridge, central integration, and build/package gates. It does not invent provider mechanisms.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] tests/jbc/
- [proposed] bench/jbc/
- [proposed] include/Cellerator/profiling/joint_compiler/
- CMakeLists.txt
- components/README.md

**Existing code and permitted read scope:**

- all Cellerator JBC interfaces and implementations
- current standalone Cellerator targets
- components/CellShard

**Explicitly out of scope / forbidden shortcuts:**

- Do not weaken standalone Cellerator to make embedded tests pass.
- Do not auto-promote experimental providers during integration.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-R10
- CE-JBC-X08
- CE-JBC-C06
- CE-JBC-M10

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.
- Focused tests and adversarial fixtures demonstrating both success and explicit rejection paths.

### Concrete mechanism

Validate fragment identity, exact input/output coverage, local index maps, external decomposition, persistent order, projection, bindings, and output/partial affordance without reusing builder decisions as truth.

Workstream mechanism: Use separate validators and canonical referees, stable mechanism IDs, complete phase accounting, standalone/embedded build matrices, and integration-only edits to root CMake, umbrella headers, registries, and component documentation.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Run focused unit/property tests, sanitizers where supported, standalone build, embedded build, and package-consumer smoke.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-VERIFY-INTEGRATE; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-V02 — Implement scalable exact reconstruction and numerical verification

**Repository / subsystem / lane:** Cellerator · `tests/jbc, profiling, component bridge, integration` · `CE-JBC-L-VERIFY-INTEGRATE`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-VERIFY-INTEGRATE; parallel with other provider/workstream lanes after JBC-G5-PARTIAL-ARTIFACT.

### Why

**Purpose.** Implement scalable exact reconstruction and numerical verification. Deliver this as one isolated, reviewable step in the Cellerator exact verification, profiling, packaging, and integration workstream.

**Biological motivation.** The integrated compiler is scientifically defensible only if exact atom-local execution and biological identity recovery remain independently testable while standalone Cellerator remains intact.

**Compiler-architectural reason.** This lane owns independent verification, profiler export, the privileged component bridge, central integration, and build/package gates. It does not invent provider mechanisms.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] tests/jbc/
- [proposed] bench/jbc/
- [proposed] include/Cellerator/profiling/joint_compiler/
- CMakeLists.txt
- components/README.md

**Existing code and permitted read scope:**

- all Cellerator JBC interfaces and implementations
- current standalone Cellerator targets
- components/CellShard

**Explicitly out of scope / forbidden shortcuts:**

- Do not weaken standalone Cellerator to make embedded tests pass.
- Do not auto-promote experimental providers during integration.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-V01

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Reconstruct canonical results from disjoint or partial fragments using the declared algebra; compare with CPU high-precision and canonical GPU references while avoiding quadratic duplicate scans.

Workstream mechanism: Use separate validators and canonical referees, stable mechanism IDs, complete phase accounting, standalone/embedded build matrices, and integration-only edits to root CMake, umbrella headers, registries, and component documentation.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Run focused unit/property tests, sanitizers where supported, standalone build, embedded build, and package-consumer smoke.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-VERIFY-INTEGRATE; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-V03 — Export atom-aware profiler and mechanism manifests

**Repository / subsystem / lane:** Cellerator · `tests/jbc, profiling, component bridge, integration` · `CE-JBC-L-VERIFY-INTEGRATE`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-VERIFY-INTEGRATE; parallel with other provider/workstream lanes after JBC-G5-PARTIAL-ARTIFACT.

### Why

**Purpose.** Export atom-aware profiler and mechanism manifests. Deliver this as one isolated, reviewable step in the Cellerator exact verification, profiling, packaging, and integration workstream.

**Biological motivation.** The integrated compiler is scientifically defensible only if exact atom-local execution and biological identity recovery remain independently testable while standalone Cellerator remains intact.

**Compiler-architectural reason.** This lane owns independent verification, profiler export, the privileged component bridge, central integration, and build/package gates. It does not invent provider mechanisms.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] tests/jbc/
- [proposed] bench/jbc/
- [proposed] include/Cellerator/profiling/joint_compiler/
- CMakeLists.txt
- components/README.md

**Existing code and permitted read scope:**

- all Cellerator JBC interfaces and implementations
- current standalone Cellerator targets
- components/CellShard

**Explicitly out of scope / forbidden shortcuts:**

- Do not weaken standalone Cellerator to make embedded tests pass.
- Do not auto-promote experimental providers during integration.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-V02

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A bounded implementation unit, focused tests, mechanism statistics, and an integration receipt.

### Concrete mechanism

Extend stable profiling records with atom species/IDs, coverages, view family, phases bypassed, transforms, assembly, partials, persistent order, resource bytes, and empirical freshness.

Workstream mechanism: Use separate validators and canonical referees, stable mechanism IDs, complete phase accounting, standalone/embedded build matrices, and integration-only edits to root CMake, umbrella headers, registries, and component documentation.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Run focused unit/property tests, sanitizers where supported, standalone build, embedded build, and package-consumer smoke.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-VERIFY-INTEGRATE; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-V04 — Enforce standalone libCellerator build and ABI gate

**Repository / subsystem / lane:** Cellerator · `tests/jbc, profiling, component bridge, integration` · `CE-JBC-L-VERIFY-INTEGRATE`  
**Classification:** validation/evidence  
**Parallelism:** Serial within CE-JBC-L-VERIFY-INTEGRATE; parallel with other provider/workstream lanes after JBC-G5-PARTIAL-ARTIFACT.

### Why

**Purpose.** Enforce standalone libCellerator build and ABI gate. Deliver this as one isolated, reviewable step in the Cellerator exact verification, profiling, packaging, and integration workstream.

**Biological motivation.** The integrated compiler is scientifically defensible only if exact atom-local execution and biological identity recovery remain independently testable while standalone Cellerator remains intact.

**Compiler-architectural reason.** This lane owns independent verification, profiler export, the privileged component bridge, central integration, and build/package gates. It does not invent provider mechanisms.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] tests/jbc/
- [proposed] bench/jbc/
- [proposed] include/Cellerator/profiling/joint_compiler/
- CMakeLists.txt
- components/README.md

**Existing code and permitted read scope:**

- all Cellerator JBC interfaces and implementations
- current standalone Cellerator targets
- components/CellShard

**Explicitly out of scope / forbidden shortcuts:**

- Do not weaken standalone Cellerator to make embedded tests pass.
- Do not auto-promote experimental providers during integration.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-V03

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.
- A machine-readable evidence record containing the baseline, candidate, complete-cost metrics, environment identity, and promotion disposition.

### Concrete mechanism

Build/test with CellShard disabled and scan required Cellerator targets/headers for CellShard dependency. Verify public generic interfaces remain independently usable.

Workstream mechanism: Use separate validators and canonical referees, stable mechanism IDs, complete phase accounting, standalone/embedded build matrices, and integration-only edits to root CMake, umbrella headers, registries, and component documentation.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Run focused unit/property tests, sanitizers where supported, standalone build, embedded build, and package-consumer smoke.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-VERIFY-INTEGRATE; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-V05 — Enforce embedded privileged CellShard bridge gate

**Repository / subsystem / lane:** Cellerator · `tests/jbc, profiling, component bridge, integration` · `CE-JBC-L-VERIFY-INTEGRATE`  
**Classification:** validation/evidence  
**Parallelism:** Serial within CE-JBC-L-VERIFY-INTEGRATE; parallel with other provider/workstream lanes after JBC-G5-PARTIAL-ARTIFACT.

### Why

**Purpose.** Enforce embedded privileged CellShard bridge gate. Deliver this as one isolated, reviewable step in the Cellerator exact verification, profiling, packaging, and integration workstream.

**Biological motivation.** The integrated compiler is scientifically defensible only if exact atom-local execution and biological identity recovery remain independently testable while standalone Cellerator remains intact.

**Compiler-architectural reason.** This lane owns independent verification, profiler export, the privileged component bridge, central integration, and build/package gates. It does not invent provider mechanisms.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] tests/jbc/
- [proposed] bench/jbc/
- [proposed] include/Cellerator/profiling/joint_compiler/
- CMakeLists.txt
- components/README.md

**Existing code and permitted read scope:**

- all Cellerator JBC interfaces and implementations
- current standalone Cellerator targets
- components/CellShard

**Explicitly out of scope / forbidden shortcuts:**

- Do not weaken standalone Cellerator to make embedded tests pass.
- Do not auto-promote experimental providers during integration.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-V04

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A machine-readable evidence record containing the baseline, candidate, complete-cost metrics, environment identity, and promotion disposition.

### Concrete mechanism

Build with CellShard enabled, link the private bridge, run interface receipt and atom-fragment smoke, and prove no reverse required dependency enters canonical runtime targets.

Workstream mechanism: Use separate validators and canonical referees, stable mechanism IDs, complete phase accounting, standalone/embedded build matrices, and integration-only edits to root CMake, umbrella headers, registries, and component documentation.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Run focused unit/property tests, sanitizers where supported, standalone build, embedded build, and package-consumer smoke.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-VERIFY-INTEGRATE; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CE-JBC-V06 — Integrate and publish the Cellerator JBC handoff

**Repository / subsystem / lane:** Cellerator · `tests/jbc, profiling, component bridge, integration` · `CE-JBC-L-VERIFY-INTEGRATE`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CE-JBC-L-VERIFY-INTEGRATE; parallel with other provider/workstream lanes after JBC-G5-PARTIAL-ARTIFACT.

### Why

**Purpose.** Integrate and publish the Cellerator JBC handoff. Deliver this as one isolated, reviewable step in the Cellerator exact verification, profiling, packaging, and integration workstream.

**Biological motivation.** The integrated compiler is scientifically defensible only if exact atom-local execution and biological identity recovery remain independently testable while standalone Cellerator remains intact.

**Compiler-architectural reason.** This lane owns independent verification, profiler export, the privileged component bridge, central integration, and build/package gates. It does not invent provider mechanisms.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] tests/jbc/
- [proposed] bench/jbc/
- [proposed] include/Cellerator/profiling/joint_compiler/
- CMakeLists.txt
- components/README.md

**Existing code and permitted read scope:**

- all Cellerator JBC interfaces and implementations
- current standalone Cellerator targets
- components/CellShard

**Explicitly out of scope / forbidden shortcuts:**

- Do not weaken standalone Cellerator to make embedded tests pass.
- Do not auto-promote experimental providers during integration.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CE-JBC-V05

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Aggregate source-linked fragments, update integration-only headers/CMake/docs, run all focused gates, record exact interfaces/commits, and provide the CellShard consumer receipt without implementing CellShard policy.

Workstream mechanism: Use separate validators and canonical referees, stable mechanism IDs, complete phase accounting, standalone/embedded build matrices, and integration-only edits to root CMake, umbrella headers, registries, and component documentation.

### Data flow, ownership, and complexity

- Cold builders may own explicit temporary storage; public execution views remain non-owning pointer-plus-count records.
- Semantic identity and exact coverage are inputs or validated outputs; storage location and runtime pointers remain operational state.
- Central registries, umbrella headers, package exports, and root CMake are changed only by the designated integration task.

- Document asymptotic and peak-memory behavior.
- Reject unbounded all-pairs or unrestricted subgraph enumeration unless this Todo is an exact small-instance oracle.
- Use streaming, bounded top-L structures, sketches, sparse maps, count/scan/fill, radix/sort, or caller-owned marks as appropriate.

**Cold versus hot path.** The named mechanism is cold compilation unless explicitly described as runtime. Any steady-state path must perform no discovery, catalog parsing, hidden allocation, global sorting, or topology search.

### Invariants

- Canonical biological identity is explicit and recoverable; shape, ordinal position, or equal extent never establishes equivalence.
- Approximate evidence may propose work, but only independently certified exact coverage may enter execution.
- Proposal overlap, physical-representation overlap, and execution-contribution overlap remain distinct.
- Each logical contribution has one exact owner unless a versioned partial-result algebra proves reconstruction.
- Structure, mutable values, transient state, runtime residency, and preference/cost freshness have separate generations and invalidation.

### Failure cases and fallback

- Malformed or stale identity/generation data must be rejected before execution.
- Weak or unstable biological structure must produce a valid no-candidate or no-promotion outcome.
- Capacity overflow, duplicate identity, incomplete coverage, and candidate explosion must return explicit diagnostics.

### Validation

- Run focused unit/property tests, sanitizers where supported, standalone build, embedded build, and package-consumer smoke.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CE-JBC-L-VERIFY-INTEGRATE; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.
