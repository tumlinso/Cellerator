# Exact validation, biological evidence, ablation, vertical-slice, and integration Todo plan

This document collects the independent correctness and evidence tasks. The final claim must survive generic baselines and matched biological nulls; approximate proposal quality is irrelevant unless exact certified atoms produce measurable end-to-end benefit.

# Workstream CSCERT: Independent exact atom certification

**Repository:** CellShard  
**Suggested lane:** `CS-JBC-L-CERTIFICATION`  
**Barrier:** `JBC-G1-ATOM-THIN-WAIST`

## Workstream design

Validate domains, members, edge IDs, inverse maps, halos, replicas, contribution owners, residual coverage, multimodal and trajectory identity maps, partial algebras, generations, and dependency closure. Use global 64-bit identity, compact local widths, count/scan/fill or caller-owned marks, and a separate implementation from proposal builders.

## Existing live source extended

- CellShard atom core
- Cellerator relation_cover exact validation
- Cellerator hierarchical index spaces
- Cellerator partial-result algebra

## Likely source scope

- [proposed] include/CellShard/compiler/certification/
- [proposed] src/compiler/certification/
- [proposed] tests/jbc/certification/

## Proposed Todos (16)

## CS-JBC-C01 — Define atom-certification request and result

**Repository / subsystem / lane:** CellShard · `compiler/certification` · `CS-JBC-L-CERTIFICATION`  
**Classification:** validation/evidence  
**Parallelism:** Serial within CS-JBC-L-CERTIFICATION; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Define atom-certification request and result. Deliver this as one isolated, reviewable step in the Independent exact atom certification workstream.

**Biological motivation.** Approximate biological discovery is useful only if exact canonical identity, coverage, halos, replicas, contributions, and partial reconstruction are certified independently.

**Compiler-architectural reason.** Certification is a separate authority from every proposal provider and must scale to relations larger than one kernel-local index range.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/compiler/certification/
- [proposed] src/compiler/certification/
- [proposed] tests/jbc/certification/

**Existing code and permitted read scope:**

- CellShard atom core
- Cellerator relation_cover exact validation
- Cellerator hierarchical index spaces
- Cellerator partial-result algebra

**Explicitly out of scope / forbidden shortcuts:**

- Discovery providers may not self-certify by setting a flag.
- Do not use quadratic duplicate scans on potentially atlas-scale arrays.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-A20
- receipt:CE-JBC-I06

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.
- Focused tests and adversarial fixtures demonstrating both success and explicit rejection paths.

### Concrete mechanism

Accept one proposal, canonical biological sources, exact coverage ABI, identity mappings, and required affordances; return certified atom, residual, diagnostics, and exact certificate or explicit rejection.

Workstream mechanism: Validate domains, members, edge IDs, inverse maps, halos, replicas, contribution owners, residual coverage, multimodal and trajectory identity maps, partial algebras, generations, and dependency closure. Use global 64-bit identity, compact local widths, count/scan/fill or caller-owned marks, and a separate implementation from proposal builders.

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

- Maintain a corruption corpus, randomized property generator, exact small-instance oracle, and streaming large-instance validation.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-CERTIFICATION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-C02 — Validate canonical domain identities

**Repository / subsystem / lane:** CellShard · `compiler/certification` · `CS-JBC-L-CERTIFICATION`  
**Classification:** validation/evidence  
**Parallelism:** Serial within CS-JBC-L-CERTIFICATION; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Validate canonical domain identities. Deliver this as one isolated, reviewable step in the Independent exact atom certification workstream.

**Biological motivation.** Approximate biological discovery is useful only if exact canonical identity, coverage, halos, replicas, contributions, and partial reconstruction are certified independently.

**Compiler-architectural reason.** Certification is a separate authority from every proposal provider and must scale to relations larger than one kernel-local index range.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/compiler/certification/
- [proposed] src/compiler/certification/
- [proposed] tests/jbc/certification/

**Existing code and permitted read scope:**

- CellShard atom core
- Cellerator relation_cover exact validation
- Cellerator hierarchical index spaces
- Cellerator partial-result algebra

**Explicitly out of scope / forbidden shortcuts:**

- Discovery providers may not self-certify by setting a flag.
- Do not use quadratic duplicate scans on potentially atlas-scale arrays.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-C01

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- Focused tests and adversarial fixtures demonstrating both success and explicit rejection paths.

### Concrete mechanism

Resolve every atom port and coverage to exact typed domains/generations; reject shape-only, missing, duplicate, or cross-generation identity matches.

Workstream mechanism: Validate domains, members, edge IDs, inverse maps, halos, replicas, contribution owners, residual coverage, multimodal and trajectory identity maps, partial algebras, generations, and dependency closure. Use global 64-bit identity, compact local widths, count/scan/fill or caller-owned marks, and a separate implementation from proposal builders.

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

- Maintain a corruption corpus, randomized property generator, exact small-instance oracle, and streaming large-instance validation.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-CERTIFICATION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-C03 — Validate exact entity coverage

**Repository / subsystem / lane:** CellShard · `compiler/certification` · `CS-JBC-L-CERTIFICATION`  
**Classification:** validation/evidence  
**Parallelism:** Serial within CS-JBC-L-CERTIFICATION; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Validate exact entity coverage. Deliver this as one isolated, reviewable step in the Independent exact atom certification workstream.

**Biological motivation.** Approximate biological discovery is useful only if exact canonical identity, coverage, halos, replicas, contributions, and partial reconstruction are certified independently.

**Compiler-architectural reason.** Certification is a separate authority from every proposal provider and must scale to relations larger than one kernel-local index range.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/compiler/certification/
- [proposed] src/compiler/certification/
- [proposed] tests/jbc/certification/

**Existing code and permitted read scope:**

- CellShard atom core
- Cellerator relation_cover exact validation
- Cellerator hierarchical index spaces
- Cellerator partial-result algebra

**Explicitly out of scope / forbidden shortcuts:**

- Discovery providers may not self-certify by setting a flag.
- Do not use quadratic duplicate scans on potentially atlas-scale arrays.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-C02

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- Focused tests and adversarial fixtures demonstrating both success and explicit rejection paths.

### Concrete mechanism

Check sorted/unique canonical entities, interval bounds, union semantics, declared counts, and content digest using scalable marks/sort/index structures.

Workstream mechanism: Validate domains, members, edge IDs, inverse maps, halos, replicas, contribution owners, residual coverage, multimodal and trajectory identity maps, partial algebras, generations, and dependency closure. Use global 64-bit identity, compact local widths, count/scan/fill or caller-owned marks, and a separate implementation from proposal builders.

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

- Maintain a corruption corpus, randomized property generator, exact small-instance oracle, and streaming large-instance validation.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-CERTIFICATION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-C04 — Validate exact relation-edge coverage

**Repository / subsystem / lane:** CellShard · `compiler/certification` · `CS-JBC-L-CERTIFICATION`  
**Classification:** validation/evidence  
**Parallelism:** Serial within CS-JBC-L-CERTIFICATION; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Validate exact relation-edge coverage. Deliver this as one isolated, reviewable step in the Independent exact atom certification workstream.

**Biological motivation.** Approximate biological discovery is useful only if exact canonical identity, coverage, halos, replicas, contributions, and partial reconstruction are certified independently.

**Compiler-architectural reason.** Certification is a separate authority from every proposal provider and must scale to relations larger than one kernel-local index range.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/compiler/certification/
- [proposed] src/compiler/certification/
- [proposed] tests/jbc/certification/

**Existing code and permitted read scope:**

- CellShard atom core
- Cellerator relation_cover exact validation
- Cellerator hierarchical index spaces
- Cellerator partial-result algebra

**Explicitly out of scope / forbidden shortcuts:**

- Discovery providers may not self-certify by setting a flag.
- Do not use quadratic duplicate scans on potentially atlas-scale arrays.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-C03

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- Focused tests and adversarial fixtures demonstrating both success and explicit rejection paths.

### Concrete mechanism

Check logical edge IDs, source/destination endpoints, structure/epoch, duplicate-edge policy, and relation membership; support aggregate 64-bit edge counts.

Workstream mechanism: Validate domains, members, edge IDs, inverse maps, halos, replicas, contribution owners, residual coverage, multimodal and trajectory identity maps, partial algebras, generations, and dependency closure. Use global 64-bit identity, compact local widths, count/scan/fill or caller-owned marks, and a separate implementation from proposal builders.

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

- Maintain a corruption corpus, randomized property generator, exact small-instance oracle, and streaming large-instance validation.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-CERTIFICATION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-C05 — Detect duplicate members and edges scalably

**Repository / subsystem / lane:** CellShard · `compiler/certification` · `CS-JBC-L-CERTIFICATION`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CS-JBC-L-CERTIFICATION; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Detect duplicate members and edges scalably. Deliver this as one isolated, reviewable step in the Independent exact atom certification workstream.

**Biological motivation.** Approximate biological discovery is useful only if exact canonical identity, coverage, halos, replicas, contributions, and partial reconstruction are certified independently.

**Compiler-architectural reason.** Certification is a separate authority from every proposal provider and must scale to relations larger than one kernel-local index range.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/compiler/certification/
- [proposed] src/compiler/certification/
- [proposed] tests/jbc/certification/

**Existing code and permitted read scope:**

- CellShard atom core
- Cellerator relation_cover exact validation
- Cellerator hierarchical index spaces
- Cellerator partial-result algebra

**Explicitly out of scope / forbidden shortcuts:**

- Discovery providers may not self-certify by setting a flag.
- Do not use quadratic duplicate scans on potentially atlas-scale arrays.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-C04

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Use caller-owned mark arrays, inverse maps, radix/sort, or bounded hash tables; no O(n^2) duplicate scan on atlas-scale data.

Workstream mechanism: Validate domains, members, edge IDs, inverse maps, halos, replicas, contribution owners, residual coverage, multimodal and trajectory identity maps, partial algebras, generations, and dependency closure. Use global 64-bit identity, compact local widths, count/scan/fill or caller-owned marks, and a separate implementation from proposal builders.

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

- Maintain a corruption corpus, randomized property generator, exact small-instance oracle, and streaming large-instance validation.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-CERTIFICATION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-C06 — Build canonical-to-local and local-to-canonical maps

**Repository / subsystem / lane:** CellShard · `compiler/certification` · `CS-JBC-L-CERTIFICATION`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CS-JBC-L-CERTIFICATION; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Build canonical-to-local and local-to-canonical maps. Deliver this as one isolated, reviewable step in the Independent exact atom certification workstream.

**Biological motivation.** Approximate biological discovery is useful only if exact canonical identity, coverage, halos, replicas, contributions, and partial reconstruction are certified independently.

**Compiler-architectural reason.** Certification is a separate authority from every proposal provider and must scale to relations larger than one kernel-local index range.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/compiler/certification/
- [proposed] src/compiler/certification/
- [proposed] tests/jbc/certification/

**Existing code and permitted read scope:**

- CellShard atom core
- Cellerator relation_cover exact validation
- Cellerator hierarchical index spaces
- Cellerator partial-result algebra

**Explicitly out of scope / forbidden shortcuts:**

- Discovery providers may not self-certify by setting a flag.
- Do not use quadratic duplicate scans on potentially atlas-scale arrays.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-C05

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Construct deterministic bijections for owned membership, explicit sentinel handling for holes/halos, compact local index width, and inverse-map validation.

Workstream mechanism: Validate domains, members, edge IDs, inverse maps, halos, replicas, contribution owners, residual coverage, multimodal and trajectory identity maps, partial algebras, generations, and dependency closure. Use global 64-bit identity, compact local widths, count/scan/fill or caller-owned marks, and a separate implementation from proposal builders.

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

- Maintain a corruption corpus, randomized property generator, exact small-instance oracle, and streaming large-instance validation.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-CERTIFICATION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-C07 — Validate read-only halos

**Repository / subsystem / lane:** CellShard · `compiler/certification` · `CS-JBC-L-CERTIFICATION`  
**Classification:** validation/evidence  
**Parallelism:** Serial within CS-JBC-L-CERTIFICATION; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Validate read-only halos. Deliver this as one isolated, reviewable step in the Independent exact atom certification workstream.

**Biological motivation.** Approximate biological discovery is useful only if exact canonical identity, coverage, halos, replicas, contributions, and partial reconstruction are certified independently.

**Compiler-architectural reason.** Certification is a separate authority from every proposal provider and must scale to relations larger than one kernel-local index range.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/compiler/certification/
- [proposed] src/compiler/certification/
- [proposed] tests/jbc/certification/

**Existing code and permitted read scope:**

- CellShard atom core
- Cellerator relation_cover exact validation
- Cellerator hierarchical index spaces
- Cellerator partial-result algebra

**Explicitly out of scope / forbidden shortcuts:**

- Discovery providers may not self-certify by setting a flag.
- Do not use quadratic duplicate scans on potentially atlas-scale arrays.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-C06

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- Focused tests and adversarial fixtures demonstrating both success and explicit rejection paths.

### Concrete mechanism

Prove halo membership is outside or explicitly overlaps owned coverage, cannot receive contribution ownership, and is sufficient for provider-declared boundary reads.

Workstream mechanism: Validate domains, members, edge IDs, inverse maps, halos, replicas, contribution owners, residual coverage, multimodal and trajectory identity maps, partial algebras, generations, and dependency closure. Use global 64-bit identity, compact local widths, count/scan/fill or caller-owned marks, and a separate implementation from proposal builders.

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

- Maintain a corruption corpus, randomized property generator, exact small-instance oracle, and streaming large-instance validation.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-CERTIFICATION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-C08 — Validate physical replicas

**Repository / subsystem / lane:** CellShard · `compiler/certification` · `CS-JBC-L-CERTIFICATION`  
**Classification:** validation/evidence  
**Parallelism:** Serial within CS-JBC-L-CERTIFICATION; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Validate physical replicas. Deliver this as one isolated, reviewable step in the Independent exact atom certification workstream.

**Biological motivation.** Approximate biological discovery is useful only if exact canonical identity, coverage, halos, replicas, contributions, and partial reconstruction are certified independently.

**Compiler-architectural reason.** Certification is a separate authority from every proposal provider and must scale to relations larger than one kernel-local index range.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/compiler/certification/
- [proposed] src/compiler/certification/
- [proposed] tests/jbc/certification/

**Existing code and permitted read scope:**

- CellShard atom core
- Cellerator relation_cover exact validation
- Cellerator hierarchical index spaces
- Cellerator partial-result algebra

**Explicitly out of scope / forbidden shortcuts:**

- Discovery providers may not self-certify by setting a flag.
- Do not use quadratic duplicate scans on potentially atlas-scale arrays.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-C07

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- Focused tests and adversarial fixtures demonstrating both success and explicit rejection paths.

### Concrete mechanism

Prove each replica refers to the same semantic coverage/plane generation while permitting different encoding, location, and content representation; replicas never gain contribution ownership.

Workstream mechanism: Validate domains, members, edge IDs, inverse maps, halos, replicas, contribution owners, residual coverage, multimodal and trajectory identity maps, partial algebras, generations, and dependency closure. Use global 64-bit identity, compact local widths, count/scan/fill or caller-owned marks, and a separate implementation from proposal builders.

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

- Maintain a corruption corpus, randomized property generator, exact small-instance oracle, and streaming large-instance validation.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-CERTIFICATION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-C09 — Assign exact contribution owners

**Repository / subsystem / lane:** CellShard · `compiler/certification` · `CS-JBC-L-CERTIFICATION`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CS-JBC-L-CERTIFICATION; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Assign exact contribution owners. Deliver this as one isolated, reviewable step in the Independent exact atom certification workstream.

**Biological motivation.** Approximate biological discovery is useful only if exact canonical identity, coverage, halos, replicas, contributions, and partial reconstruction are certified independently.

**Compiler-architectural reason.** Certification is a separate authority from every proposal provider and must scale to relations larger than one kernel-local index range.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/compiler/certification/
- [proposed] src/compiler/certification/
- [proposed] tests/jbc/certification/

**Existing code and permitted read scope:**

- CellShard atom core
- Cellerator relation_cover exact validation
- Cellerator hierarchical index spaces
- Cellerator partial-result algebra

**Explicitly out of scope / forbidden shortcuts:**

- Discovery providers may not self-certify by setting a flag.
- Do not use quadratic duplicate scans on potentially atlas-scale arrays.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-C08

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Create one primary owner per logical contribution or a complete list of partial contributors tied to a valid algebra; reject ambiguous or missing ownership.

Workstream mechanism: Validate domains, members, edge IDs, inverse maps, halos, replicas, contribution owners, residual coverage, multimodal and trajectory identity maps, partial algebras, generations, and dependency closure. Use global 64-bit identity, compact local widths, count/scan/fill or caller-owned marks, and a separate implementation from proposal builders.

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

- Maintain a corruption corpus, randomized property generator, exact small-instance oracle, and streaming large-instance validation.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-CERTIFICATION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-C10 — Construct exact residual coverage

**Repository / subsystem / lane:** CellShard · `compiler/certification` · `CS-JBC-L-CERTIFICATION`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CS-JBC-L-CERTIFICATION; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Construct exact residual coverage. Deliver this as one isolated, reviewable step in the Independent exact atom certification workstream.

**Biological motivation.** Approximate biological discovery is useful only if exact canonical identity, coverage, halos, replicas, contributions, and partial reconstruction are certified independently.

**Compiler-architectural reason.** Certification is a separate authority from every proposal provider and must scale to relations larger than one kernel-local index range.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/compiler/certification/
- [proposed] src/compiler/certification/
- [proposed] tests/jbc/certification/

**Existing code and permitted read scope:**

- CellShard atom core
- Cellerator relation_cover exact validation
- Cellerator hierarchical index spaces
- Cellerator partial-result algebra

**Explicitly out of scope / forbidden shortcuts:**

- Discovery providers may not self-certify by setting a flag.
- Do not use quadratic duplicate scans on potentially atlas-scale arrays.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-C09

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Subtract accepted atom contribution coverage from the canonical relation/work scope and produce a certified residual; verify disjoint union or declared partial reconstruction.

Workstream mechanism: Validate domains, members, edge IDs, inverse maps, halos, replicas, contribution owners, residual coverage, multimodal and trajectory identity maps, partial algebras, generations, and dependency closure. Use global 64-bit identity, compact local widths, count/scan/fill or caller-owned marks, and a separate implementation from proposal builders.

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

- Maintain a corruption corpus, randomized property generator, exact small-instance oracle, and streaming large-instance validation.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-CERTIFICATION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-C11 — Validate multimodal identity mappings

**Repository / subsystem / lane:** CellShard · `compiler/certification` · `CS-JBC-L-CERTIFICATION`  
**Classification:** validation/evidence  
**Parallelism:** Serial within CS-JBC-L-CERTIFICATION; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Validate multimodal identity mappings. Deliver this as one isolated, reviewable step in the Independent exact atom certification workstream.

**Biological motivation.** Approximate biological discovery is useful only if exact canonical identity, coverage, halos, replicas, contributions, and partial reconstruction are certified independently.

**Compiler-architectural reason.** Certification is a separate authority from every proposal provider and must scale to relations larger than one kernel-local index range.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/compiler/certification/
- [proposed] src/compiler/certification/
- [proposed] tests/jbc/certification/

**Existing code and permitted read scope:**

- CellShard atom core
- Cellerator relation_cover exact validation
- Cellerator hierarchical index spaces
- Cellerator partial-result algebra

**Explicitly out of scope / forbidden shortcuts:**

- Discovery providers may not self-certify by setting a flag.
- Do not use quadratic duplicate scans on potentially atlas-scale arrays.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-C10

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.
- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.
- Focused tests and adversarial fixtures demonstrating both success and explicit rejection paths.

### Concrete mechanism

Check shared-spine IDs, per-modality local IDs, missing sentinels, duplicate mappings, domain generations, and cross-modal relation endpoints.

Workstream mechanism: Validate domains, members, edge IDs, inverse maps, halos, replicas, contribution owners, residual coverage, multimodal and trajectory identity maps, partial algebras, generations, and dependency closure. Use global 64-bit identity, compact local widths, count/scan/fill or caller-owned marks, and a separate implementation from proposal builders.

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

- Maintain a corruption corpus, randomized property generator, exact small-instance oracle, and streaming large-instance validation.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-CERTIFICATION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-C12 — Validate trajectory and lineage mappings

**Repository / subsystem / lane:** CellShard · `compiler/certification` · `CS-JBC-L-CERTIFICATION`  
**Classification:** validation/evidence  
**Parallelism:** Serial within CS-JBC-L-CERTIFICATION; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Validate trajectory and lineage mappings. Deliver this as one isolated, reviewable step in the Independent exact atom certification workstream.

**Biological motivation.** Approximate biological discovery is useful only if exact canonical identity, coverage, halos, replicas, contributions, and partial reconstruction are certified independently.

**Compiler-architectural reason.** Certification is a separate authority from every proposal provider and must scale to relations larger than one kernel-local index range.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/compiler/certification/
- [proposed] src/compiler/certification/
- [proposed] tests/jbc/certification/

**Existing code and permitted read scope:**

- CellShard atom core
- Cellerator relation_cover exact validation
- Cellerator hierarchical index spaces
- Cellerator partial-result algebra

**Explicitly out of scope / forbidden shortcuts:**

- Discovery providers may not self-certify by setting a flag.
- Do not use quadratic duplicate scans on potentially atlas-scale arrays.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-C11

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.
- Focused tests and adversarial fixtures demonstrating both success and explicit rejection paths.

### Concrete mechanism

Check state IDs, ancestry/transition edge identity, DAG/tree rules declared by the provider, branch membership, prefix membership, and delta parentage.

Workstream mechanism: Validate domains, members, edge IDs, inverse maps, halos, replicas, contribution owners, residual coverage, multimodal and trajectory identity maps, partial algebras, generations, and dependency closure. Use global 64-bit identity, compact local widths, count/scan/fill or caller-owned marks, and a separate implementation from proposal builders.

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

- Maintain a corruption corpus, randomized property generator, exact small-instance oracle, and streaming large-instance validation.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-CERTIFICATION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-C13 — Validate partial-result algebra compatibility

**Repository / subsystem / lane:** CellShard · `compiler/certification` · `CS-JBC-L-CERTIFICATION`  
**Classification:** validation/evidence  
**Parallelism:** Serial within CS-JBC-L-CERTIFICATION; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Validate partial-result algebra compatibility. Deliver this as one isolated, reviewable step in the Independent exact atom certification workstream.

**Biological motivation.** Approximate biological discovery is useful only if exact canonical identity, coverage, halos, replicas, contributions, and partial reconstruction are certified independently.

**Compiler-architectural reason.** Certification is a separate authority from every proposal provider and must scale to relations larger than one kernel-local index range.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/compiler/certification/
- [proposed] src/compiler/certification/
- [proposed] tests/jbc/certification/

**Existing code and permitted read scope:**

- CellShard atom core
- Cellerator relation_cover exact validation
- Cellerator hierarchical index spaces
- Cellerator partial-result algebra

**Explicitly out of scope / forbidden shortcuts:**

- Discovery providers may not self-certify by setting a flag.
- Do not use quadratic duplicate scans on potentially atlas-scale arrays.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-C12

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- Focused tests and adversarial fixtures demonstrating both success and explicit rejection paths.

### Concrete mechanism

Ensure every partial atom state layout, neutral/merge/finalize identity, numerical policy, contribution coverage, and deterministic-tree constraint match the operation provider.

Workstream mechanism: Validate domains, members, edge IDs, inverse maps, halos, replicas, contribution owners, residual coverage, multimodal and trajectory identity maps, partial algebras, generations, and dependency closure. Use global 64-bit identity, compact local widths, count/scan/fill or caller-owned marks, and a separate implementation from proposal builders.

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

- Maintain a corruption corpus, randomized property generator, exact small-instance oracle, and streaming large-instance validation.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-CERTIFICATION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-C14 — Validate generations and dependency closure

**Repository / subsystem / lane:** CellShard · `compiler/certification` · `CS-JBC-L-CERTIFICATION`  
**Classification:** validation/evidence  
**Parallelism:** Serial within CS-JBC-L-CERTIFICATION; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Validate generations and dependency closure. Deliver this as one isolated, reviewable step in the Independent exact atom certification workstream.

**Biological motivation.** Approximate biological discovery is useful only if exact canonical identity, coverage, halos, replicas, contributions, and partial reconstruction are certified independently.

**Compiler-architectural reason.** Certification is a separate authority from every proposal provider and must scale to relations larger than one kernel-local index range.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/compiler/certification/
- [proposed] src/compiler/certification/
- [proposed] tests/jbc/certification/

**Existing code and permitted read scope:**

- CellShard atom core
- Cellerator relation_cover exact validation
- Cellerator hierarchical index spaces
- Cellerator partial-result algebra

**Explicitly out of scope / forbidden shortcuts:**

- Discovery providers may not self-certify by setting a flag.
- Do not use quadratic duplicate scans on potentially atlas-scale arrays.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-C13

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- Focused tests and adversarial fixtures demonstrating both success and explicit rejection paths.

### Concrete mechanism

Traverse correctness dependencies, reject stale structure/value/state/graph/build inputs, and distinguish performance/cost staleness from invalidity.

Workstream mechanism: Validate domains, members, edge IDs, inverse maps, halos, replicas, contribution owners, residual coverage, multimodal and trajectory identity maps, partial algebras, generations, and dependency closure. Use global 64-bit identity, compact local widths, count/scan/fill or caller-owned marks, and a separate implementation from proposal builders.

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

- Maintain a corruption corpus, randomized property generator, exact small-instance oracle, and streaming large-instance validation.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-CERTIFICATION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-C15 — Emit the exact atom certificate

**Repository / subsystem / lane:** CellShard · `compiler/certification` · `CS-JBC-L-CERTIFICATION`  
**Classification:** validation/evidence  
**Parallelism:** Serial within CS-JBC-L-CERTIFICATION; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Emit the exact atom certificate. Deliver this as one isolated, reviewable step in the Independent exact atom certification workstream.

**Biological motivation.** Approximate biological discovery is useful only if exact canonical identity, coverage, halos, replicas, contributions, and partial reconstruction are certified independently.

**Compiler-architectural reason.** Certification is a separate authority from every proposal provider and must scale to relations larger than one kernel-local index range.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/compiler/certification/
- [proposed] src/compiler/certification/
- [proposed] tests/jbc/certification/

**Existing code and permitted read scope:**

- CellShard atom core
- Cellerator relation_cover exact validation
- Cellerator hierarchical index spaces
- Cellerator partial-result algebra

**Explicitly out of scope / forbidden shortcuts:**

- Discovery providers may not self-certify by setting a flag.
- Do not use quadratic duplicate scans on potentially atlas-scale arrays.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-C14

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.
- Focused tests and adversarial fixtures demonstrating both success and explicit rejection paths.

### Concrete mechanism

Serialize a compact certificate naming atom, canonical sources, coverage digests, owners/contributors, residual, maps, algebra, generations, verifier identity, and validation result.

Workstream mechanism: Validate domains, members, edge IDs, inverse maps, halos, replicas, contribution owners, residual coverage, multimodal and trajectory identity maps, partial algebras, generations, and dependency closure. Use global 64-bit identity, compact local widths, count/scan/fill or caller-owned marks, and a separate implementation from proposal builders.

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

- Maintain a corruption corpus, randomized property generator, exact small-instance oracle, and streaming large-instance validation.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-CERTIFICATION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-C16 — Build the independent streaming verifier, corruption corpus, and small oracle

**Repository / subsystem / lane:** CellShard · `compiler/certification` · `CS-JBC-L-CERTIFICATION`  
**Classification:** validation/evidence  
**Parallelism:** Serial within CS-JBC-L-CERTIFICATION; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Build the independent streaming verifier, corruption corpus, and small oracle. Deliver this as one isolated, reviewable step in the Independent exact atom certification workstream.

**Biological motivation.** Approximate biological discovery is useful only if exact canonical identity, coverage, halos, replicas, contributions, and partial reconstruction are certified independently.

**Compiler-architectural reason.** Certification is a separate authority from every proposal provider and must scale to relations larger than one kernel-local index range.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/compiler/certification/
- [proposed] src/compiler/certification/
- [proposed] tests/jbc/certification/

**Existing code and permitted read scope:**

- CellShard atom core
- Cellerator relation_cover exact validation
- Cellerator hierarchical index spaces
- Cellerator partial-result algebra

**Explicitly out of scope / forbidden shortcuts:**

- Discovery providers may not self-certify by setting a flag.
- Do not use quadratic duplicate scans on potentially atlas-scale arrays.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-C15

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.
- Focused tests and adversarial fixtures demonstrating both success and explicit rejection paths.

### Concrete mechanism

Implement a separate validation path, exact exhaustive small fixtures, randomized generators, malformed/corrupt records, 64-bit aggregate tests, and streaming validation with bounded memory.

Workstream mechanism: Validate domains, members, edge IDs, inverse maps, halos, replicas, contribution owners, residual coverage, multimodal and trajectory identity maps, partial algebras, generations, and dependency closure. Use global 64-bit identity, compact local widths, count/scan/fill or caller-owned marks, and a separate implementation from proposal builders.

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

- Maintain a corruption corpus, randomized property generator, exact small-instance oracle, and streaming large-instance validation.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-CERTIFICATION; final aggregation is integration-lane-only.

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


# Workstream CSVAL: Exact validation, biological evidence, vertical slices, and final integration

**Repository:** CellShard  
**Suggested lane:** `CS-JBC-L-VALIDATION-INTEGRATION`  
**Barrier:** `JBC-G6-RUNTIME-SUBSTRATE`

## Workstream design

Create synthetic and manifest-backed biological fixtures; implement byte/random/graph/hypergraph/materialized-view/JIT/trace-only baselines and biological-null transformations; record complete mechanism metrics; integrate required vertical slices; run standalone and embedded gates; advance the CellShard commit before the parent submodule pointer.

## Existing live source extended

- all CellShard JBC providers
- all Cellerator JBC interfaces
- legacy CSH5/CSPACK/sharded runtime baselines

## Likely source scope

- [proposed] tests/jbc/
- [proposed] bench/jbc/
- [proposed] docs/JBC/evidence/
- CMakeLists.txt
- include/CellShard/CellShard.hh

## Proposed Todos (20)

## CS-JBC-V01 — Build the shared biological and synthetic fixture corpus

**Repository / subsystem / lane:** CellShard · `tests/jbc, bench/jbc, integration` · `CS-JBC-L-VALIDATION-INTEGRATION`  
**Classification:** validation/evidence  
**Parallelism:** Serial within CS-JBC-L-VALIDATION-INTEGRATION; parallel with other provider/workstream lanes after JBC-G6-RUNTIME-SUBSTRATE.

### Why

**Purpose.** Build the shared biological and synthetic fixture corpus. Deliver this as one isolated, reviewable step in the Exact validation, biological evidence, vertical slices, and final integration workstream.

**Biological motivation.** The system earns its biological claim only if matched nulls and generic baselines isolate the benefit of biology-generated atom vocabulary, hierarchy, and reuse.

**Compiler-architectural reason.** This lane owns common fixtures, ablation harnesses, vertical slices, central integration, package/build matrices, and final acceptance. Provider lanes remain independently responsible for their focused tests.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] tests/jbc/
- [proposed] bench/jbc/
- [proposed] docs/JBC/evidence/
- CMakeLists.txt
- include/CellShard/CellShard.hh

**Existing code and permitted read scope:**

- all CellShard JBC providers
- all Cellerator JBC interfaces
- legacy CSH5/CSPACK/sharded runtime baselines

**Explicitly out of scope / forbidden shortcuts:**

- Do not claim biological novelty from kernel speed alone.
- Do not commit large private datasets; use manifests/checksums/extraction recipes or generated fixtures.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-RT22
- CS-JBC-ST29
- CS-JBC-BS18
- CS-JBC-IG10
- receipt:CE-JBC-V06

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.
- Focused tests and adversarial fixtures demonstrating both success and explicit rejection paths.

### Concrete mechanism

Create deterministic small/medium fixtures for scRNA-like support, scATAC-like support, regulatory relations, perturbations, trajectories, multimodal identity, segments, high-degree hubs, random nulls, stable structure/mutable values, and graph families; commit recipes/manifests, not private large data.

Workstream mechanism: Create synthetic and manifest-backed biological fixtures; implement byte/random/graph/hypergraph/materialized-view/JIT/trace-only baselines and biological-null transformations; record complete mechanism metrics; integrate required vertical slices; run standalone and embedded gates; advance the CellShard commit before the parent submodule pointer.

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

- Require independent exact certificates and canonical differential outputs for every vertical slice.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-VALIDATION-INTEGRATION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-V02 — Implement generic compiler baselines

**Repository / subsystem / lane:** CellShard · `tests/jbc, bench/jbc, integration` · `CS-JBC-L-VALIDATION-INTEGRATION`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CS-JBC-L-VALIDATION-INTEGRATION; parallel with other provider/workstream lanes after JBC-G6-RUNTIME-SUBSTRATE.

### Why

**Purpose.** Implement generic compiler baselines. Deliver this as one isolated, reviewable step in the Exact validation, biological evidence, vertical slices, and final integration workstream.

**Biological motivation.** The system earns its biological claim only if matched nulls and generic baselines isolate the benefit of biology-generated atom vocabulary, hierarchy, and reuse.

**Compiler-architectural reason.** This lane owns common fixtures, ablation harnesses, vertical slices, central integration, package/build matrices, and final acceptance. Provider lanes remain independently responsible for their focused tests.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] tests/jbc/
- [proposed] bench/jbc/
- [proposed] docs/JBC/evidence/
- CMakeLists.txt
- include/CellShard/CellShard.hh

**Existing code and permitted read scope:**

- all CellShard JBC providers
- all Cellerator JBC interfaces
- legacy CSH5/CSPACK/sharded runtime baselines

**Explicitly out of scope / forbidden shortcuts:**

- Do not claim biological novelty from kernel speed alone.
- Do not commit large private datasets; use manifests/checksums/extraction recipes or generated fixtures.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-V01

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Provide byte-balanced, random, generic graph, generic hypergraph, generic materialized-view, per-graph specialization, canonical-plus-JIT, Cellerator-only geometry, trace-only atoms, and biology-only atom candidates behind one comparison harness.

Workstream mechanism: Create synthetic and manifest-backed biological fixtures; implement byte/random/graph/hypergraph/materialized-view/JIT/trace-only baselines and biological-null transformations; record complete mechanism metrics; integrate required vertical slices; run standalone and embedded gates; advance the CellShard commit before the parent submodule pointer.

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

- Require independent exact certificates and canonical differential outputs for every vertical slice.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-VALIDATION-INTEGRATION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-V03 — Implement matched biological-null transformations

**Repository / subsystem / lane:** CellShard · `tests/jbc, bench/jbc, integration` · `CS-JBC-L-VALIDATION-INTEGRATION`  
**Classification:** validation/evidence  
**Parallelism:** Serial within CS-JBC-L-VALIDATION-INTEGRATION; parallel with other provider/workstream lanes after JBC-G6-RUNTIME-SUBSTRATE.

### Why

**Purpose.** Implement matched biological-null transformations. Deliver this as one isolated, reviewable step in the Exact validation, biological evidence, vertical slices, and final integration workstream.

**Biological motivation.** The system earns its biological claim only if matched nulls and generic baselines isolate the benefit of biology-generated atom vocabulary, hierarchy, and reuse.

**Compiler-architectural reason.** This lane owns common fixtures, ablation harnesses, vertical slices, central integration, package/build matrices, and final acceptance. Provider lanes remain independently responsible for their focused tests.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] tests/jbc/
- [proposed] bench/jbc/
- [proposed] docs/JBC/evidence/
- CMakeLists.txt
- include/CellShard/CellShard.hh

**Existing code and permitted read scope:**

- all CellShard JBC providers
- all Cellerator JBC interfaces
- legacy CSH5/CSPACK/sharded runtime baselines

**Explicitly out of scope / forbidden shortcuts:**

- Do not claim biological novelty from kernel speed alone.
- Do not commit large private datasets; use manifests/checksums/extraction recipes or generated fixtures.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-V02

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.
- Focused tests and adversarial fixtures demonstrating both success and explicit rejection paths.

### Concrete mechanism

Add degree-preserving rewiring, identity shuffle, stratum shuffle, trajectory disruption, modality-map shuffle, relation-type erasure, and operation-family reuse disruption while preserving relevant generic size/sparsity/degree properties.

Workstream mechanism: Create synthetic and manifest-backed biological fixtures; implement byte/random/graph/hypergraph/materialized-view/JIT/trace-only baselines and biological-null transformations; record complete mechanism metrics; integrate required vertical slices; run standalone and embedded gates; advance the CellShard commit before the parent submodule pointer.

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

- Require independent exact certificates and canonical differential outputs for every vertical slice.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-VALIDATION-INTEGRATION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-V04 — Freeze the biological mechanism and complete-cost metric schema

**Repository / subsystem / lane:** CellShard · `tests/jbc, bench/jbc, integration` · `CS-JBC-L-VALIDATION-INTEGRATION`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CS-JBC-L-VALIDATION-INTEGRATION; parallel with other provider/workstream lanes after JBC-G6-RUNTIME-SUBSTRATE.

### Why

**Purpose.** Freeze the biological mechanism and complete-cost metric schema. Deliver this as one isolated, reviewable step in the Exact validation, biological evidence, vertical slices, and final integration workstream.

**Biological motivation.** The system earns its biological claim only if matched nulls and generic baselines isolate the benefit of biology-generated atom vocabulary, hierarchy, and reuse.

**Compiler-architectural reason.** This lane owns common fixtures, ablation harnesses, vertical slices, central integration, package/build matrices, and final acceptance. Provider lanes remain independently responsible for their focused tests.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] tests/jbc/
- [proposed] bench/jbc/
- [proposed] docs/JBC/evidence/
- CMakeLists.txt
- include/CellShard/CellShard.hh

**Existing code and permitted read scope:**

- all CellShard JBC providers
- all Cellerator JBC interfaces
- legacy CSH5/CSPACK/sharded runtime baselines

**Explicitly out of scope / forbidden shortcuts:**

- Do not claim biological novelty from kernel speed alone.
- Do not commit large private datasets; use manifests/checksums/extraction recipes or generated fixtures.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-V03

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.

### Concrete mechanism

Record discovery, certification, atom, grammar, basis, materialization, I/O, movement, transforms, canonicalization, Cellerator phases, runtime, invalidation, and biological-stability metrics with stable IDs.

Workstream mechanism: Create synthetic and manifest-backed biological fixtures; implement byte/random/graph/hypergraph/materialized-view/JIT/trace-only baselines and biological-null transformations; record complete mechanism metrics; integrate required vertical slices; run standalone and embedded gates; advance the CellShard commit before the parent submodule pointer.

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

- Require independent exact certificates and canonical differential outputs for every vertical slice.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-VALIDATION-INTEGRATION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-V05 — Build the atom-discovery ablation harness

**Repository / subsystem / lane:** CellShard · `tests/jbc, bench/jbc, integration` · `CS-JBC-L-VALIDATION-INTEGRATION`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CS-JBC-L-VALIDATION-INTEGRATION; parallel with other provider/workstream lanes after JBC-G6-RUNTIME-SUBSTRATE.

### Why

**Purpose.** Build the atom-discovery ablation harness. Deliver this as one isolated, reviewable step in the Exact validation, biological evidence, vertical slices, and final integration workstream.

**Biological motivation.** The system earns its biological claim only if matched nulls and generic baselines isolate the benefit of biology-generated atom vocabulary, hierarchy, and reuse.

**Compiler-architectural reason.** This lane owns common fixtures, ablation harnesses, vertical slices, central integration, package/build matrices, and final acceptance. Provider lanes remain independently responsible for their focused tests.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] tests/jbc/
- [proposed] bench/jbc/
- [proposed] docs/JBC/evidence/
- CMakeLists.txt
- include/CellShard/CellShard.hh

**Existing code and permitted read scope:**

- all CellShard JBC providers
- all Cellerator JBC interfaces
- legacy CSH5/CSPACK/sharded runtime baselines

**Explicitly out of scope / forbidden shortcuts:**

- Do not claim biological novelty from kernel speed alone.
- Do not commit large private datasets; use manifests/checksums/extraction recipes or generated fixtures.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-V04

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Run support signature, co-support, bicluster, overlap, motif, factor, trajectory, multimodal, trace, combinations, no-discovery, and null variants through the same exact certification/cost path.

Workstream mechanism: Create synthetic and manifest-backed biological fixtures; implement byte/random/graph/hypergraph/materialized-view/JIT/trace-only baselines and biological-null transformations; record complete mechanism metrics; integrate required vertical slices; run standalone and embedded gates; advance the CellShard commit before the parent submodule pointer.

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

- Require independent exact certificates and canonical differential outputs for every vertical slice.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-VALIDATION-INTEGRATION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-V06 — Build the composition, grammar, basis, and superatom ablation harness

**Repository / subsystem / lane:** CellShard · `tests/jbc, bench/jbc, integration` · `CS-JBC-L-VALIDATION-INTEGRATION`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CS-JBC-L-VALIDATION-INTEGRATION; parallel with other provider/workstream lanes after JBC-G6-RUNTIME-SUBSTRATE.

### Why

**Purpose.** Build the composition, grammar, basis, and superatom ablation harness. Deliver this as one isolated, reviewable step in the Exact validation, biological evidence, vertical slices, and final integration workstream.

**Biological motivation.** The system earns its biological claim only if matched nulls and generic baselines isolate the benefit of biology-generated atom vocabulary, hierarchy, and reuse.

**Compiler-architectural reason.** This lane owns common fixtures, ablation harnesses, vertical slices, central integration, package/build matrices, and final acceptance. Provider lanes remain independently responsible for their focused tests.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] tests/jbc/
- [proposed] bench/jbc/
- [proposed] docs/JBC/evidence/
- CMakeLists.txt
- include/CellShard/CellShard.hh

**Existing code and permitted read scope:**

- all CellShard JBC providers
- all Cellerator JBC interfaces
- legacy CSH5/CSPACK/sharded runtime baselines

**Explicitly out of scope / forbidden shortcuts:**

- Do not claim biological novelty from kernel speed alone.
- Do not commit large private datasets; use manifests/checksums/extraction recipes or generated fixtures.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-V05

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Compare flat catalog, explicit DAG, induced grammar, one/multiple bases, disjoint/overlap, no superatoms, promotion/demotion, and no-basis fallback with complete costs.

Workstream mechanism: Create synthetic and manifest-backed biological fixtures; implement byte/random/graph/hypergraph/materialized-view/JIT/trace-only baselines and biological-null transformations; record complete mechanism metrics; integrate required vertical slices; run standalone and embedded gates; advance the CellShard commit before the parent submodule pointer.

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

- Require independent exact certificates and canonical differential outputs for every vertical slice.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-VALIDATION-INTEGRATION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-V07 — Build atom-store corruption, crash, consolidation, and GC tests

**Repository / subsystem / lane:** CellShard · `tests/jbc, bench/jbc, integration` · `CS-JBC-L-VALIDATION-INTEGRATION`  
**Classification:** validation/evidence  
**Parallelism:** Serial within CS-JBC-L-VALIDATION-INTEGRATION; parallel with other provider/workstream lanes after JBC-G6-RUNTIME-SUBSTRATE.

### Why

**Purpose.** Build atom-store corruption, crash, consolidation, and GC tests. Deliver this as one isolated, reviewable step in the Exact validation, biological evidence, vertical slices, and final integration workstream.

**Biological motivation.** The system earns its biological claim only if matched nulls and generic baselines isolate the benefit of biology-generated atom vocabulary, hierarchy, and reuse.

**Compiler-architectural reason.** This lane owns common fixtures, ablation harnesses, vertical slices, central integration, package/build matrices, and final acceptance. Provider lanes remain independently responsible for their focused tests.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] tests/jbc/
- [proposed] bench/jbc/
- [proposed] docs/JBC/evidence/
- CMakeLists.txt
- include/CellShard/CellShard.hh

**Existing code and permitted read scope:**

- all CellShard JBC providers
- all Cellerator JBC interfaces
- legacy CSH5/CSPACK/sharded runtime baselines

**Explicitly out of scope / forbidden shortcuts:**

- Do not claim biological novelty from kernel speed alone.
- Do not commit large private datasets; use manifests/checksums/extraction recipes or generated fixtures.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-V06

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.
- Focused tests and adversarial fixtures demonstrating both success and explicit rejection paths.

### Concrete mechanism

Inject malformed headers/offsets/digests, torn roots/frames, missing arenas, orphan generations, interrupted consolidation, stale indexes, pinned snapshots, and invalid action results.

Workstream mechanism: Create synthetic and manifest-backed biological fixtures; implement byte/random/graph/hypergraph/materialized-view/JIT/trace-only baselines and biological-null transformations; record complete mechanism metrics; integrate required vertical slices; run standalone and embedded gates; advance the CellShard commit before the parent submodule pointer.

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

- Require independent exact certificates and canonical differential outputs for every vertical slice.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-VALIDATION-INTEGRATION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-V08 — Integrate the support-signature basis vertical slice

**Repository / subsystem / lane:** CellShard · `tests/jbc, bench/jbc, integration` · `CS-JBC-L-VALIDATION-INTEGRATION`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CS-JBC-L-VALIDATION-INTEGRATION; parallel with other provider/workstream lanes after JBC-G6-RUNTIME-SUBSTRATE.

### Why

**Purpose.** Integrate the support-signature basis vertical slice. Deliver this as one isolated, reviewable step in the Exact validation, biological evidence, vertical slices, and final integration workstream.

**Biological motivation.** The system earns its biological claim only if matched nulls and generic baselines isolate the benefit of biology-generated atom vocabulary, hierarchy, and reuse.

**Compiler-architectural reason.** This lane owns common fixtures, ablation harnesses, vertical slices, central integration, package/build matrices, and final acceptance. Provider lanes remain independently responsible for their focused tests.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] tests/jbc/
- [proposed] bench/jbc/
- [proposed] docs/JBC/evidence/
- CMakeLists.txt
- include/CellShard/CellShard.hh

**Existing code and permitted read scope:**

- all CellShard JBC providers
- all Cellerator JBC interfaces
- legacy CSH5/CSPACK/sharded runtime baselines

**Explicitly out of scope / forbidden shortcuts:**

- Do not claim biological novelty from kernel speed alone.
- Do not commit large private datasets; use manifests/checksums/extraction recipes or generated fixtures.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-V07

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Execute canonical relation → support-signature evidence → exact certified atoms → greedy basis → atom-store materialization → CellShard residency → Cellerator relation apply; compare fixed row shards and canonical JIT.

Workstream mechanism: Create synthetic and manifest-backed biological fixtures; implement byte/random/graph/hypergraph/materialized-view/JIT/trace-only baselines and biological-null transformations; record complete mechanism metrics; integrate required vertical slices; run standalone and embedded gates; advance the CellShard commit before the parent submodule pointer.

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

- Require independent exact certificates and canonical differential outputs for every vertical slice.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-VALIDATION-INTEGRATION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-V09 — Integrate the cross-operation support-family vertical slice

**Repository / subsystem / lane:** CellShard · `tests/jbc, bench/jbc, integration` · `CS-JBC-L-VALIDATION-INTEGRATION`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CS-JBC-L-VALIDATION-INTEGRATION; parallel with other provider/workstream lanes after JBC-G6-RUNTIME-SUBSTRATE.

### Why

**Purpose.** Integrate the cross-operation support-family vertical slice. Deliver this as one isolated, reviewable step in the Exact validation, biological evidence, vertical slices, and final integration workstream.

**Biological motivation.** The system earns its biological claim only if matched nulls and generic baselines isolate the benefit of biology-generated atom vocabulary, hierarchy, and reuse.

**Compiler-architectural reason.** This lane owns common fixtures, ablation harnesses, vertical slices, central integration, package/build matrices, and final acceptance. Provider lanes remain independently responsible for their focused tests.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] tests/jbc/
- [proposed] bench/jbc/
- [proposed] docs/JBC/evidence/
- CMakeLists.txt
- include/CellShard/CellShard.hh

**Existing code and permitted read scope:**

- all CellShard JBC providers
- all Cellerator JBC interfaces
- legacy CSH5/CSPACK/sharded runtime baselines

**Explicitly out of scope / forbidden shortcuts:**

- Do not claim biological novelty from kernel speed alone.
- Do not commit large private datasets; use manifests/checksums/extraction recipes or generated fixtures.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-V08

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Reuse one exact support family across relation apply, transpose, support contraction, and one segment/gate operation; compare per-operation views with generalized family and explicit canonicalization.

Workstream mechanism: Create synthetic and manifest-backed biological fixtures; implement byte/random/graph/hypergraph/materialized-view/JIT/trace-only baselines and biological-null transformations; record complete mechanism metrics; integrate required vertical slices; run standalone and embedded gates; advance the CellShard commit before the parent submodule pointer.

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

- Require independent exact certificates and canonical differential outputs for every vertical slice.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-VALIDATION-INTEGRATION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-V10 — Integrate the stable-structure mutable-value vertical slice

**Repository / subsystem / lane:** CellShard · `tests/jbc, bench/jbc, integration` · `CS-JBC-L-VALIDATION-INTEGRATION`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CS-JBC-L-VALIDATION-INTEGRATION; parallel with other provider/workstream lanes after JBC-G6-RUNTIME-SUBSTRATE.

### Why

**Purpose.** Integrate the stable-structure mutable-value vertical slice. Deliver this as one isolated, reviewable step in the Exact validation, biological evidence, vertical slices, and final integration workstream.

**Biological motivation.** The system earns its biological claim only if matched nulls and generic baselines isolate the benefit of biology-generated atom vocabulary, hierarchy, and reuse.

**Compiler-architectural reason.** This lane owns common fixtures, ablation harnesses, vertical slices, central integration, package/build matrices, and final acceptance. Provider lanes remain independently responsible for their focused tests.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] tests/jbc/
- [proposed] bench/jbc/
- [proposed] docs/JBC/evidence/
- CMakeLists.txt
- include/CellShard/CellShard.hh

**Existing code and permitted read scope:**

- all CellShard JBC providers
- all Cellerator JBC interfaces
- legacy CSH5/CSPACK/sharded runtime baselines

**Explicitly out of scope / forbidden shortcuts:**

- Do not claim biological novelty from kernel speed alone.
- Do not commit large private datasets; use manifests/checksums/extraction recipes or generated fixtures.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-V09

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Persist stable support/projection, update projection-primary values and active masks across generations, compute direct gradients, publish generations, and prove no structure/geometry rebuild.

Workstream mechanism: Create synthetic and manifest-backed biological fixtures; implement byte/random/graph/hypergraph/materialized-view/JIT/trace-only baselines and biological-null transformations; record complete mechanism metrics; integrate required vertical slices; run standalone and embedded gates; advance the CellShard commit before the parent submodule pointer.

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

- Require independent exact certificates and canonical differential outputs for every vertical slice.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-VALIDATION-INTEGRATION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-V11 — Integrate the persistent moments and log-sum-exp partial slice

**Repository / subsystem / lane:** CellShard · `tests/jbc, bench/jbc, integration` · `CS-JBC-L-VALIDATION-INTEGRATION`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CS-JBC-L-VALIDATION-INTEGRATION; parallel with other provider/workstream lanes after JBC-G6-RUNTIME-SUBSTRATE.

### Why

**Purpose.** Integrate the persistent moments and log-sum-exp partial slice. Deliver this as one isolated, reviewable step in the Exact validation, biological evidence, vertical slices, and final integration workstream.

**Biological motivation.** The system earns its biological claim only if matched nulls and generic baselines isolate the benefit of biology-generated atom vocabulary, hierarchy, and reuse.

**Compiler-architectural reason.** This lane owns common fixtures, ablation harnesses, vertical slices, central integration, package/build matrices, and final acceptance. Provider lanes remain independently responsible for their focused tests.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] tests/jbc/
- [proposed] bench/jbc/
- [proposed] docs/JBC/evidence/
- CMakeLists.txt
- include/CellShard/CellShard.hh

**Existing code and permitted read scope:**

- all CellShard JBC providers
- all Cellerator JBC interfaces
- legacy CSH5/CSPACK/sharded runtime baselines

**Explicitly out of scope / forbidden shortcuts:**

- Do not claim biological novelty from kernel speed alone.
- Do not commit large private datasets; use manifests/checksums/extraction recipes or generated fixtures.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-V10

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Generate local moments and log-sum-exp partial atoms, persist/reload where profitable, combine hierarchically under exact algebra, and compare canonical single-device output.

Workstream mechanism: Create synthetic and manifest-backed biological fixtures; implement byte/random/graph/hypergraph/materialized-view/JIT/trace-only baselines and biological-null transformations; record complete mechanism metrics; integrate required vertical slices; run standalone and embedded gates; advance the CellShard commit before the parent submodule pointer.

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

- Require independent exact certificates and canonical differential outputs for every vertical slice.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-VALIDATION-INTEGRATION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-V12 — Integrate the trajectory prefix and branch vertical slice

**Repository / subsystem / lane:** CellShard · `tests/jbc, bench/jbc, integration` · `CS-JBC-L-VALIDATION-INTEGRATION`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CS-JBC-L-VALIDATION-INTEGRATION; parallel with other provider/workstream lanes after JBC-G6-RUNTIME-SUBSTRATE.

### Why

**Purpose.** Integrate the trajectory prefix and branch vertical slice. Deliver this as one isolated, reviewable step in the Exact validation, biological evidence, vertical slices, and final integration workstream.

**Biological motivation.** The system earns its biological claim only if matched nulls and generic baselines isolate the benefit of biology-generated atom vocabulary, hierarchy, and reuse.

**Compiler-architectural reason.** This lane owns common fixtures, ablation harnesses, vertical slices, central integration, package/build matrices, and final acceptance. Provider lanes remain independently responsible for their focused tests.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] tests/jbc/
- [proposed] bench/jbc/
- [proposed] docs/JBC/evidence/
- CMakeLists.txt
- include/CellShard/CellShard.hh

**Existing code and permitted read scope:**

- all CellShard JBC providers
- all Cellerator JBC interfaces
- legacy CSH5/CSPACK/sharded runtime baselines

**Explicitly out of scope / forbidden shortcuts:**

- Do not claim biological novelty from kernel speed alone.
- Do not commit large private datasets; use manifests/checksums/extraction recipes or generated fixtures.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-V11

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Use supplied lineage/transition fixtures to discover/certify prefix, branch, delta, temporal-window, and transition atoms; materialize only promoted candidates and run trajectory-null comparisons.

Workstream mechanism: Create synthetic and manifest-backed biological fixtures; implement byte/random/graph/hypergraph/materialized-view/JIT/trace-only baselines and biological-null transformations; record complete mechanism metrics; integrate required vertical slices; run standalone and embedded gates; advance the CellShard commit before the parent submodule pointer.

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

- Require independent exact certificates and canonical differential outputs for every vertical slice.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-VALIDATION-INTEGRATION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-V13 — Integrate the multimodal identity-spine vertical slice

**Repository / subsystem / lane:** CellShard · `tests/jbc, bench/jbc, integration` · `CS-JBC-L-VALIDATION-INTEGRATION`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CS-JBC-L-VALIDATION-INTEGRATION; parallel with other provider/workstream lanes after JBC-G6-RUNTIME-SUBSTRATE.

### Why

**Purpose.** Integrate the multimodal identity-spine vertical slice. Deliver this as one isolated, reviewable step in the Exact validation, biological evidence, vertical slices, and final integration workstream.

**Biological motivation.** The system earns its biological claim only if matched nulls and generic baselines isolate the benefit of biology-generated atom vocabulary, hierarchy, and reuse.

**Compiler-architectural reason.** This lane owns common fixtures, ablation harnesses, vertical slices, central integration, package/build matrices, and final acceptance. Provider lanes remain independently responsible for their focused tests.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] tests/jbc/
- [proposed] bench/jbc/
- [proposed] docs/JBC/evidence/
- CMakeLists.txt
- include/CellShard/CellShard.hh

**Existing code and permitted read scope:**

- all CellShard JBC providers
- all Cellerator JBC interfaces
- legacy CSH5/CSPACK/sharded runtime baselines

**Explicitly out of scope / forbidden shortcuts:**

- Do not claim biological novelty from kernel speed alone.
- Do not commit large private datasets; use manifests/checksums/extraction recipes or generated fixtures.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-V12

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.
- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Build a shared identity spine with RNA/ATAC-like overlays, missing entities, typed peak→gene relation, destination bundle, persistent order, and exact multimodal output.

Workstream mechanism: Create synthetic and manifest-backed biological fixtures; implement byte/random/graph/hypergraph/materialized-view/JIT/trace-only baselines and biological-null transformations; record complete mechanism metrics; integrate required vertical slices; run standalone and embedded gates; advance the CellShard commit before the parent submodule pointer.

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

- Require independent exact certificates and canonical differential outputs for every vertical slice.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-VALIDATION-INTEGRATION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-V14 — Integrate the external non-Cellerator provider vertical slice

**Repository / subsystem / lane:** CellShard · `tests/jbc, bench/jbc, integration` · `CS-JBC-L-VALIDATION-INTEGRATION`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CS-JBC-L-VALIDATION-INTEGRATION; parallel with other provider/workstream lanes after JBC-G6-RUNTIME-SUBSTRATE.

### Why

**Purpose.** Integrate the external non-Cellerator provider vertical slice. Deliver this as one isolated, reviewable step in the Exact validation, biological evidence, vertical slices, and final integration workstream.

**Biological motivation.** The system earns its biological claim only if matched nulls and generic baselines isolate the benefit of biology-generated atom vocabulary, hierarchy, and reuse.

**Compiler-architectural reason.** This lane owns common fixtures, ablation harnesses, vertical slices, central integration, package/build matrices, and final acceptance. Provider lanes remain independently responsible for their focused tests.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] tests/jbc/
- [proposed] bench/jbc/
- [proposed] docs/JBC/evidence/
- CMakeLists.txt
- include/CellShard/CellShard.hh

**Existing code and permitted read scope:**

- all CellShard JBC providers
- all Cellerator JBC interfaces
- legacy CSH5/CSPACK/sharded runtime baselines

**Explicitly out of scope / forbidden shortcuts:**

- Do not claim biological novelty from kernel speed alone.
- Do not commit large private datasets; use manifests/checksums/extraction recipes or generated fixtures.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-V13

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Schedule one mock sequence/halo or custom operation before/after a Cellerator fragment, pass atom identity/planes through CellShard, validate effects, and prove no Baseplane/model semantics entered core APIs.

Workstream mechanism: Create synthetic and manifest-backed biological fixtures; implement byte/random/graph/hypergraph/materialized-view/JIT/trace-only baselines and biological-null transformations; record complete mechanism metrics; integrate required vertical slices; run standalone and embedded gates; advance the CellShard commit before the parent submodule pointer.

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

- Require independent exact certificates and canonical differential outputs for every vertical slice.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-VALIDATION-INTEGRATION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-V15 — Integrate the two-logical-node /mnt/block vertical slice

**Repository / subsystem / lane:** CellShard · `tests/jbc, bench/jbc, integration` · `CS-JBC-L-VALIDATION-INTEGRATION`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CS-JBC-L-VALIDATION-INTEGRATION; parallel with other provider/workstream lanes after JBC-G6-RUNTIME-SUBSTRATE.

### Why

**Purpose.** Integrate the two-logical-node /mnt/block vertical slice. Deliver this as one isolated, reviewable step in the Exact validation, biological evidence, vertical slices, and final integration workstream.

**Biological motivation.** The system earns its biological claim only if matched nulls and generic baselines isolate the benefit of biology-generated atom vocabulary, hierarchy, and reuse.

**Compiler-architectural reason.** This lane owns common fixtures, ablation harnesses, vertical slices, central integration, package/build matrices, and final acceptance. Provider lanes remain independently responsible for their focused tests.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] tests/jbc/
- [proposed] bench/jbc/
- [proposed] docs/JBC/evidence/
- CMakeLists.txt
- include/CellShard/CellShard.hh

**Existing code and permitted read scope:**

- all CellShard JBC providers
- all Cellerator JBC interfaces
- legacy CSH5/CSPACK/sharded runtime baselines

**Explicitly out of scope / forbidden shortcuts:**

- Do not claim biological novelty from kernel speed alone.
- Do not commit large private datasets; use manifests/checksums/extraction recipes or generated fixtures.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-V14

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Read atom frames only through logical node 0 by test policy, schedule node-1 work through CellShard, compare route and compute-placement alternatives, and verify portable schedule relinking.

Workstream mechanism: Create synthetic and manifest-backed biological fixtures; implement byte/random/graph/hypergraph/materialized-view/JIT/trace-only baselines and biological-null transformations; record complete mechanism metrics; integrate required vertical slices; run standalone and embedded gates; advance the CellShard commit before the parent submodule pointer.

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

- Require independent exact certificates and canonical differential outputs for every vertical slice.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-VALIDATION-INTEGRATION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-V16 — Integrate the numaBraid route vertical slice

**Repository / subsystem / lane:** CellShard · `tests/jbc, bench/jbc, integration` · `CS-JBC-L-VALIDATION-INTEGRATION`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CS-JBC-L-VALIDATION-INTEGRATION; parallel with other provider/workstream lanes after JBC-G6-RUNTIME-SUBSTRATE.

### Why

**Purpose.** Integrate the numaBraid route vertical slice. Deliver this as one isolated, reviewable step in the Exact validation, biological evidence, vertical slices, and final integration workstream.

**Biological motivation.** The system earns its biological claim only if matched nulls and generic baselines isolate the benefit of biology-generated atom vocabulary, hierarchy, and reuse.

**Compiler-architectural reason.** This lane owns common fixtures, ablation harnesses, vertical slices, central integration, package/build matrices, and final acceptance. Provider lanes remain independently responsible for their focused tests.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] tests/jbc/
- [proposed] bench/jbc/
- [proposed] docs/JBC/evidence/
- CMakeLists.txt
- include/CellShard/CellShard.hh

**Existing code and permitted read scope:**

- all CellShard JBC providers
- all Cellerator JBC interfaces
- legacy CSH5/CSPACK/sharded runtime baselines

**Explicitly out of scope / forbidden shortcuts:**

- Do not claim biological novelty from kernel speed alone.
- Do not commit large private datasets; use manifests/checksums/extraction recipes or generated fixtures.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-V15

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Move selected GPU-resident atom planes through the numaBraid provider, compare direct P2P, host staging, and NCCL where legal, record host-memory savings, contention, and exact content.

Workstream mechanism: Create synthetic and manifest-backed biological fixtures; implement byte/random/graph/hypergraph/materialized-view/JIT/trace-only baselines and biological-null transformations; record complete mechanism metrics; integrate required vertical slices; run standalone and embedded gates; advance the CellShard commit before the parent submodule pointer.

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

- Require independent exact certificates and canonical differential outputs for every vertical slice.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-VALIDATION-INTEGRATION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-V17 — Integrate the crash and recovery vertical slice

**Repository / subsystem / lane:** CellShard · `tests/jbc, bench/jbc, integration` · `CS-JBC-L-VALIDATION-INTEGRATION`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CS-JBC-L-VALIDATION-INTEGRATION; parallel with other provider/workstream lanes after JBC-G6-RUNTIME-SUBSTRATE.

### Why

**Purpose.** Integrate the crash and recovery vertical slice. Deliver this as one isolated, reviewable step in the Exact validation, biological evidence, vertical slices, and final integration workstream.

**Biological motivation.** The system earns its biological claim only if matched nulls and generic baselines isolate the benefit of biology-generated atom vocabulary, hierarchy, and reuse.

**Compiler-architectural reason.** This lane owns common fixtures, ablation harnesses, vertical slices, central integration, package/build matrices, and final acceptance. Provider lanes remain independently responsible for their focused tests.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] tests/jbc/
- [proposed] bench/jbc/
- [proposed] docs/JBC/evidence/
- CMakeLists.txt
- include/CellShard/CellShard.hh

**Existing code and permitted read scope:**

- all CellShard JBC providers
- all Cellerator JBC interfaces
- legacy CSH5/CSPACK/sharded runtime baselines

**Explicitly out of scope / forbidden shortcuts:**

- Do not claim biological novelty from kernel speed alone.
- Do not commit large private datasets; use manifests/checksums/extraction recipes or generated fixtures.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-V16

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.
- Focused tests and adversarial fixtures demonstrating both success and explicit rejection paths.

### Concrete mechanism

Interrupt materialization, async read, transfer, partial combine, CUDA Graph execution, and generation publication; prove rollback/retry or explicit invalidation without duplicate contribution.

Workstream mechanism: Create synthetic and manifest-backed biological fixtures; implement byte/random/graph/hypergraph/materialized-view/JIT/trace-only baselines and biological-null transformations; record complete mechanism metrics; integrate required vertical slices; run standalone and embedded gates; advance the CellShard commit before the parent submodule pointer.

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

- Require independent exact certificates and canonical differential outputs for every vertical slice.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-VALIDATION-INTEGRATION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-V18 — Run the standalone, embedded, and package-consumer build matrix

**Repository / subsystem / lane:** CellShard · `tests/jbc, bench/jbc, integration` · `CS-JBC-L-VALIDATION-INTEGRATION`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CS-JBC-L-VALIDATION-INTEGRATION; parallel with other provider/workstream lanes after JBC-G6-RUNTIME-SUBSTRATE.

### Why

**Purpose.** Run the standalone, embedded, and package-consumer build matrix. Deliver this as one isolated, reviewable step in the Exact validation, biological evidence, vertical slices, and final integration workstream.

**Biological motivation.** The system earns its biological claim only if matched nulls and generic baselines isolate the benefit of biology-generated atom vocabulary, hierarchy, and reuse.

**Compiler-architectural reason.** This lane owns common fixtures, ablation harnesses, vertical slices, central integration, package/build matrices, and final acceptance. Provider lanes remain independently responsible for their focused tests.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] tests/jbc/
- [proposed] bench/jbc/
- [proposed] docs/JBC/evidence/
- CMakeLists.txt
- include/CellShard/CellShard.hh

**Existing code and permitted read scope:**

- all CellShard JBC providers
- all Cellerator JBC interfaces
- legacy CSH5/CSPACK/sharded runtime baselines

**Explicitly out of scope / forbidden shortcuts:**

- Do not claim biological novelty from kernel speed alone.
- Do not commit large private datasets; use manifests/checksums/extraction recipes or generated fixtures.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-V17

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Validate Cellerator without CellShard, embedded joint compiler, standalone CellShard against supplied Cellerator, optional CPU-only artifact inspection, provider controls, and no CE-AMP activation.

Workstream mechanism: Create synthetic and manifest-backed biological fixtures; implement byte/random/graph/hypergraph/materialized-view/JIT/trace-only baselines and biological-null transformations; record complete mechanism metrics; integrate required vertical slices; run standalone and embedded gates; advance the CellShard commit before the parent submodule pointer.

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

- Require independent exact certificates and canonical differential outputs for every vertical slice.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-VALIDATION-INTEGRATION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-V19 — Perform the biological-novelty readiness audit

**Repository / subsystem / lane:** CellShard · `tests/jbc, bench/jbc, integration` · `CS-JBC-L-VALIDATION-INTEGRATION`  
**Classification:** validation/evidence  
**Parallelism:** Serial within CS-JBC-L-VALIDATION-INTEGRATION; parallel with other provider/workstream lanes after JBC-G6-RUNTIME-SUBSTRATE.

### Why

**Purpose.** Perform the biological-novelty readiness audit. Deliver this as one isolated, reviewable step in the Exact validation, biological evidence, vertical slices, and final integration workstream.

**Biological motivation.** The system earns its biological claim only if matched nulls and generic baselines isolate the benefit of biology-generated atom vocabulary, hierarchy, and reuse.

**Compiler-architectural reason.** This lane owns common fixtures, ablation harnesses, vertical slices, central integration, package/build matrices, and final acceptance. Provider lanes remain independently responsible for their focused tests.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] tests/jbc/
- [proposed] bench/jbc/
- [proposed] docs/JBC/evidence/
- CMakeLists.txt
- include/CellShard/CellShard.hh

**Existing code and permitted read scope:**

- all CellShard JBC providers
- all Cellerator JBC interfaces
- legacy CSH5/CSPACK/sharded runtime baselines

**Explicitly out of scope / forbidden shortcuts:**

- Do not claim biological novelty from kernel speed alone.
- Do not commit large private datasets; use manifests/checksums/extraction recipes or generated fixtures.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-V18

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A machine-readable evidence record containing the baseline, candidate, complete-cost metrics, environment identity, and promotion disposition.

### Concrete mechanism

Verify that every biology-native mechanism exposes provenance, exact certification, generic baselines, null transforms, complete costs, valid negative outcomes, and no causal overclaim.

Workstream mechanism: Create synthetic and manifest-backed biological fixtures; implement byte/random/graph/hypergraph/materialized-view/JIT/trace-only baselines and biological-null transformations; record complete mechanism metrics; integrate required vertical slices; run standalone and embedded gates; advance the CellShard commit before the parent submodule pointer.

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

- Require independent exact certificates and canonical differential outputs for every vertical slice.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-VALIDATION-INTEGRATION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-V20 — Perform final dual-repository integration and acceptance

**Repository / subsystem / lane:** CellShard · `tests/jbc, bench/jbc, integration` · `CS-JBC-L-VALIDATION-INTEGRATION`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CS-JBC-L-VALIDATION-INTEGRATION; parallel with other provider/workstream lanes after JBC-G6-RUNTIME-SUBSTRATE.

### Why

**Purpose.** Perform final dual-repository integration and acceptance. Deliver this as one isolated, reviewable step in the Exact validation, biological evidence, vertical slices, and final integration workstream.

**Biological motivation.** The system earns its biological claim only if matched nulls and generic baselines isolate the benefit of biology-generated atom vocabulary, hierarchy, and reuse.

**Compiler-architectural reason.** This lane owns common fixtures, ablation harnesses, vertical slices, central integration, package/build matrices, and final acceptance. Provider lanes remain independently responsible for their focused tests.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] tests/jbc/
- [proposed] bench/jbc/
- [proposed] docs/JBC/evidence/
- CMakeLists.txt
- include/CellShard/CellShard.hh

**Existing code and permitted read scope:**

- all CellShard JBC providers
- all Cellerator JBC interfaces
- legacy CSH5/CSPACK/sharded runtime baselines

**Explicitly out of scope / forbidden shortcuts:**

- Do not claim biological novelty from kernel speed alone.
- Do not commit large private datasets; use manifests/checksums/extraction recipes or generated fixtures.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-V19

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A bounded implementation unit, focused tests, mechanism statistics, and an integration receipt.

### Concrete mechanism

Integrate CellShard commits first, push, update the parent submodule pointer in the Cellerator integration lane, run all acceptance gates, record exact commits/authorities/interfaces, and leave both trees clean without claiming CE-AMP.

Workstream mechanism: Create synthetic and manifest-backed biological fixtures; implement byte/random/graph/hypergraph/materialized-view/JIT/trace-only baselines and biological-null transformations; record complete mechanism metrics; integrate required vertical slices; run standalone and embedded gates; advance the CellShard commit before the parent submodule pointer.

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

- Require independent exact certificates and canonical differential outputs for every vertical slice.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-VALIDATION-INTEGRATION; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.
