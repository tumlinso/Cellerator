# Persistence, global graph, runtime, topology, I/O, transport, and residency Todo plan

This document covers downstream infrastructure required to materialize and execute the biology-generated hierarchy. Atom-store arenas, graph schedules, topology, async I/O, P2P, numaBraid, NCCL, residency, leases, CUDA Graphs, and recovery are mechanisms serving the atom compiler; they are not the novelty themselves.

# Workstream CSSTORE: Atom-native immutable persistence and lowering artifacts

**Repository:** CellShard  
**Suggested lane:** `CS-JBC-L-PERSISTENCE`  
**Barrier:** `JBC-G1-ATOM-THIN-WAIST`

## Workstream design

Freeze a new format family after collision audit. Separate semantic, content, materialization, replica, and action identity. Store root generation, atom dictionary, coverage/plane indexes, grammar, bases, superatoms, partials, lowering stages, physical views, lineage, actions, and large arena/frame payloads. Publish atomically; support recovery, consolidation, GC, CSH5/CSPACK import, codecs, and metadata-only inspection.

## Existing live source extended

- include/CellShard/artifact/image.hh
- include/CellShard/artifact/extent.hh
- include/CellShard/artifact/catalog.hh
- include/CellShard/artifact/snapshot.hh
- include/CellShard/io/pack/image_envelope.hh
- docs/SPEC_CSPACK_V1.md

## Likely source scope

- [proposed] include/CellShard/artifact/atom_store/
- [proposed] src/artifact/atom_store/
- [proposed] docs/SPEC_ATOM_STORE_V1.md
- [proposed] tests/jbc/atom_store/

## Proposed Todos (29)

## CS-JBC-ST01 — Freeze the atom-store v1 format charter and collision-free name

**Repository / subsystem / lane:** CellShard · `artifact/atom_store` · `CS-JBC-L-PERSISTENCE`  
**Classification:** generic infrastructure supporting the biology-native compiler  
**Parallelism:** Serial within CS-JBC-L-PERSISTENCE; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Freeze the atom-store v1 format charter and collision-free name. Deliver this as one isolated, reviewable step in the Atom-native immutable persistence and lowering artifacts workstream.

**Biological motivation.** The physical archive must persist an atomic reusable execution hierarchy, mutable planes, partials, grammar, bases, and progressively specialized Cellerator entry points—not predefined row shards.

**Compiler-architectural reason.** Use an adjacent immutable manifest-and-arena format rather than enlarging CSH5 or CSPACK. Large sequential arenas contain independently indexed atom frames; logical atom and storage extent boundaries remain separate.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/artifact/atom_store/
- [proposed] src/artifact/atom_store/
- [proposed] docs/SPEC_ATOM_STORE_V1.md
- [proposed] tests/jbc/atom_store/

**Existing code and permitted read scope:**

- include/CellShard/artifact/image.hh
- include/CellShard/artifact/extent.hh
- include/CellShard/artifact/catalog.hh
- include/CellShard/artifact/snapshot.hh
- include/CellShard/io/pack/image_envelope.hh
- docs/SPEC_CSPACK_V1.md

**Explicitly out of scope / forbidden shortcuts:**

- Do not create one file per atom by default.
- Do not use CSH5 or CSPACK as the new universal compiler database.
- Do not put source paths or placement epochs into immutable content identity.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-A20
- CS-JBC-O02

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A bounded implementation unit, focused tests, mechanism statistics, and an integration receipt.

### Concrete mechanism

Audit current CSH5, CSPACK, CPEXEC, CSHARD, and magic/version namespaces; choose a new format family and state that it persists atom hierarchy/planes/grammar/bases/partials rather than redefining canonical source science.

Workstream mechanism: Freeze a new format family after collision audit. Separate semantic, content, materialization, replica, and action identity. Store root generation, atom dictionary, coverage/plane indexes, grammar, bases, superatoms, partials, lowering stages, physical views, lineage, actions, and large arena/frame payloads. Publish atomically; support recovery, consolidation, GC, CSH5/CSPACK import, codecs, and metadata-only inspection.

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

- Use corruption/torn-write/orphan/partial-publication tests and verify decoded as well as encoded content identities.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Measure sequential I/O, selected-range I/O, frame count, read amplification, metadata memory, compression, assembly, and block-storage throughput.
- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-PERSISTENCE; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-ST02 — Add a strong algorithm-tagged content digest

**Repository / subsystem / lane:** CellShard · `artifact/atom_store` · `CS-JBC-L-PERSISTENCE`  
**Classification:** generic infrastructure supporting the biology-native compiler  
**Parallelism:** Serial within CS-JBC-L-PERSISTENCE; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Add a strong algorithm-tagged content digest. Deliver this as one isolated, reviewable step in the Atom-native immutable persistence and lowering artifacts workstream.

**Biological motivation.** The physical archive must persist an atomic reusable execution hierarchy, mutable planes, partials, grammar, bases, and progressively specialized Cellerator entry points—not predefined row shards.

**Compiler-architectural reason.** Use an adjacent immutable manifest-and-arena format rather than enlarging CSH5 or CSPACK. Large sequential arenas contain independently indexed atom frames; logical atom and storage extent boundaries remain separate.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/artifact/atom_store/
- [proposed] src/artifact/atom_store/
- [proposed] docs/SPEC_ATOM_STORE_V1.md
- [proposed] tests/jbc/atom_store/

**Existing code and permitted read scope:**

- include/CellShard/artifact/image.hh
- include/CellShard/artifact/extent.hh
- include/CellShard/artifact/catalog.hh
- include/CellShard/artifact/snapshot.hh
- include/CellShard/io/pack/image_envelope.hh
- docs/SPEC_CSPACK_V1.md

**Explicitly out of scope / forbidden shortcuts:**

- Do not create one file per atom by default.
- Do not use CSH5 or CSPACK as the new universal compiler database.
- Do not put source paths or placement epochs into immutable content identity.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-ST01

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A bounded implementation unit, focused tests, mechanism statistics, and an integration receipt.

### Concrete mechanism

Introduce a 256-bit or equivalent strong digest provider for decoded content and action identity, retain legacy FNV only for compatibility, and keep algorithm identity explicit.

Workstream mechanism: Freeze a new format family after collision audit. Separate semantic, content, materialization, replica, and action identity. Store root generation, atom dictionary, coverage/plane indexes, grammar, bases, superatoms, partials, lowering stages, physical views, lineage, actions, and large arena/frame payloads. Publish atomically; support recovery, consolidation, GC, CSH5/CSPACK import, codecs, and metadata-only inspection.

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

- Use corruption/torn-write/orphan/partial-publication tests and verify decoded as well as encoded content identities.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Measure sequential I/O, selected-range I/O, frame count, read amplification, metadata memory, compression, assembly, and block-storage throughput.
- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-PERSISTENCE; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-ST03 — Separate semantic, content, action, materialization, and replica identity

**Repository / subsystem / lane:** CellShard · `artifact/atom_store` · `CS-JBC-L-PERSISTENCE`  
**Classification:** generic infrastructure supporting the biology-native compiler  
**Parallelism:** Serial within CS-JBC-L-PERSISTENCE; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Separate semantic, content, action, materialization, and replica identity. Deliver this as one isolated, reviewable step in the Atom-native immutable persistence and lowering artifacts workstream.

**Biological motivation.** The physical archive must persist an atomic reusable execution hierarchy, mutable planes, partials, grammar, bases, and progressively specialized Cellerator entry points—not predefined row shards.

**Compiler-architectural reason.** Use an adjacent immutable manifest-and-arena format rather than enlarging CSH5 or CSPACK. Large sequential arenas contain independently indexed atom frames; logical atom and storage extent boundaries remain separate.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/artifact/atom_store/
- [proposed] src/artifact/atom_store/
- [proposed] docs/SPEC_ATOM_STORE_V1.md
- [proposed] tests/jbc/atom_store/

**Existing code and permitted read scope:**

- include/CellShard/artifact/image.hh
- include/CellShard/artifact/extent.hh
- include/CellShard/artifact/catalog.hh
- include/CellShard/artifact/snapshot.hh
- include/CellShard/io/pack/image_envelope.hh
- docs/SPEC_CSPACK_V1.md

**Explicitly out of scope / forbidden shortcuts:**

- Do not create one file per atom by default.
- Do not use CSH5 or CSPACK as the new universal compiler database.
- Do not put source paths or placement epochs into immutable content identity.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-ST02

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.

### Concrete mechanism

Define pointer-free durable records linking semantic atom family to decoded physical instance, compiler action, encoded replica, and archive generation without path-derived identity.

Workstream mechanism: Freeze a new format family after collision audit. Separate semantic, content, materialization, replica, and action identity. Store root generation, atom dictionary, coverage/plane indexes, grammar, bases, superatoms, partials, lowering stages, physical views, lineage, actions, and large arena/frame payloads. Publish atomically; support recovery, consolidation, GC, CSH5/CSPACK import, codecs, and metadata-only inspection.

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

- Use corruption/torn-write/orphan/partial-publication tests and verify decoded as well as encoded content identities.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Measure sequential I/O, selected-range I/O, frame count, read amplification, metadata memory, compression, assembly, and block-storage throughput.
- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-PERSISTENCE; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-ST04 — Define the immutable root-generation manifest

**Repository / subsystem / lane:** CellShard · `artifact/atom_store` · `CS-JBC-L-PERSISTENCE`  
**Classification:** generic infrastructure supporting the biology-native compiler  
**Parallelism:** Serial within CS-JBC-L-PERSISTENCE; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Define the immutable root-generation manifest. Deliver this as one isolated, reviewable step in the Atom-native immutable persistence and lowering artifacts workstream.

**Biological motivation.** The physical archive must persist an atomic reusable execution hierarchy, mutable planes, partials, grammar, bases, and progressively specialized Cellerator entry points—not predefined row shards.

**Compiler-architectural reason.** Use an adjacent immutable manifest-and-arena format rather than enlarging CSH5 or CSPACK. Large sequential arenas contain independently indexed atom frames; logical atom and storage extent boundaries remain separate.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/artifact/atom_store/
- [proposed] src/artifact/atom_store/
- [proposed] docs/SPEC_ATOM_STORE_V1.md
- [proposed] tests/jbc/atom_store/

**Existing code and permitted read scope:**

- include/CellShard/artifact/image.hh
- include/CellShard/artifact/extent.hh
- include/CellShard/artifact/catalog.hh
- include/CellShard/artifact/snapshot.hh
- include/CellShard/io/pack/image_envelope.hh
- docs/SPEC_CSPACK_V1.md

**Explicitly out of scope / forbidden shortcuts:**

- Do not create one file per atom by default.
- Do not use CSH5 or CSPACK as the new universal compiler database.
- Do not put source paths or placement epochs into immutable content identity.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-ST03

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.

### Concrete mechanism

Store dataset/canonical generation, parent generation, atom dictionary/index roots, grammar/basis/superatom/partial catalogs, arena list, action cache, lineage, schedules, feature flags, and root digest.

Workstream mechanism: Freeze a new format family after collision audit. Separate semantic, content, materialization, replica, and action identity. Store root generation, atom dictionary, coverage/plane indexes, grammar, bases, superatoms, partials, lowering stages, physical views, lineage, actions, and large arena/frame payloads. Publish atomically; support recovery, consolidation, GC, CSH5/CSPACK import, codecs, and metadata-only inspection.

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

- Use corruption/torn-write/orphan/partial-publication tests and verify decoded as well as encoded content identities.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Measure sequential I/O, selected-range I/O, frame count, read amplification, metadata memory, compression, assembly, and block-storage throughput.
- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-PERSISTENCE; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-ST05 — Define atom-dictionary records

**Repository / subsystem / lane:** CellShard · `artifact/atom_store` · `CS-JBC-L-PERSISTENCE`  
**Classification:** generic infrastructure supporting the biology-native compiler  
**Parallelism:** Serial within CS-JBC-L-PERSISTENCE; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Define atom-dictionary records. Deliver this as one isolated, reviewable step in the Atom-native immutable persistence and lowering artifacts workstream.

**Biological motivation.** The physical archive must persist an atomic reusable execution hierarchy, mutable planes, partials, grammar, bases, and progressively specialized Cellerator entry points—not predefined row shards.

**Compiler-architectural reason.** Use an adjacent immutable manifest-and-arena format rather than enlarging CSH5 or CSPACK. Large sequential arenas contain independently indexed atom frames; logical atom and storage extent boundaries remain separate.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/artifact/atom_store/
- [proposed] src/artifact/atom_store/
- [proposed] docs/SPEC_ATOM_STORE_V1.md
- [proposed] tests/jbc/atom_store/

**Existing code and permitted read scope:**

- include/CellShard/artifact/image.hh
- include/CellShard/artifact/extent.hh
- include/CellShard/artifact/catalog.hh
- include/CellShard/artifact/snapshot.hh
- include/CellShard/io/pack/image_envelope.hh
- docs/SPEC_CSPACK_V1.md

**Explicitly out of scope / forbidden shortcuts:**

- Do not create one file per atom by default.
- Do not use CSH5 or CSPACK as the new universal compiler database.
- Do not put source paths or placement epochs into immutable content identity.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-ST04

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.

### Concrete mechanism

Persist atom envelope identity, level/species, exact coverage references, ports, plane references, evidence, dependencies, and affordances using offsets/section IDs rather than native pointers.

Workstream mechanism: Freeze a new format family after collision audit. Separate semantic, content, materialization, replica, and action identity. Store root generation, atom dictionary, coverage/plane indexes, grammar, bases, superatoms, partials, lowering stages, physical views, lineage, actions, and large arena/frame payloads. Publish atomically; support recovery, consolidation, GC, CSH5/CSPACK import, codecs, and metadata-only inspection.

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

- Use corruption/torn-write/orphan/partial-publication tests and verify decoded as well as encoded content identities.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Measure sequential I/O, selected-range I/O, frame count, read amplification, metadata memory, compression, assembly, and block-storage throughput.
- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-PERSISTENCE; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-ST06 — Define the exact coverage index

**Repository / subsystem / lane:** CellShard · `artifact/atom_store` · `CS-JBC-L-PERSISTENCE`  
**Classification:** generic infrastructure supporting the biology-native compiler  
**Parallelism:** Serial within CS-JBC-L-PERSISTENCE; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Define the exact coverage index. Deliver this as one isolated, reviewable step in the Atom-native immutable persistence and lowering artifacts workstream.

**Biological motivation.** The physical archive must persist an atomic reusable execution hierarchy, mutable planes, partials, grammar, bases, and progressively specialized Cellerator entry points—not predefined row shards.

**Compiler-architectural reason.** Use an adjacent immutable manifest-and-arena format rather than enlarging CSH5 or CSPACK. Large sequential arenas contain independently indexed atom frames; logical atom and storage extent boundaries remain separate.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/artifact/atom_store/
- [proposed] src/artifact/atom_store/
- [proposed] docs/SPEC_ATOM_STORE_V1.md
- [proposed] tests/jbc/atom_store/

**Existing code and permitted read scope:**

- include/CellShard/artifact/image.hh
- include/CellShard/artifact/extent.hh
- include/CellShard/artifact/catalog.hh
- include/CellShard/artifact/snapshot.hh
- include/CellShard/io/pack/image_envelope.hh
- docs/SPEC_CSPACK_V1.md

**Explicitly out of scope / forbidden shortcuts:**

- Do not create one file per atom by default.
- Do not use CSH5 or CSPACK as the new universal compiler database.
- Do not put source paths or placement epochs into immutable content identity.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-ST05

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.

### Concrete mechanism

Index atom families/instances by domain, structure, coverage kind, canonical interval/list/edge/component identity, generation, and ownership role; support metadata-only lookup.

Workstream mechanism: Freeze a new format family after collision audit. Separate semantic, content, materialization, replica, and action identity. Store root generation, atom dictionary, coverage/plane indexes, grammar, bases, superatoms, partials, lowering stages, physical views, lineage, actions, and large arena/frame payloads. Publish atomically; support recovery, consolidation, GC, CSH5/CSPACK import, codecs, and metadata-only inspection.

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

- Use corruption/torn-write/orphan/partial-publication tests and verify decoded as well as encoded content identities.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Measure sequential I/O, selected-range I/O, frame count, read amplification, metadata memory, compression, assembly, and block-storage throughput.
- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-PERSISTENCE; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-ST07 — Define plane and physical-view indexes

**Repository / subsystem / lane:** CellShard · `artifact/atom_store` · `CS-JBC-L-PERSISTENCE`  
**Classification:** generic infrastructure supporting the biology-native compiler  
**Parallelism:** Serial within CS-JBC-L-PERSISTENCE; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Define plane and physical-view indexes. Deliver this as one isolated, reviewable step in the Atom-native immutable persistence and lowering artifacts workstream.

**Biological motivation.** The physical archive must persist an atomic reusable execution hierarchy, mutable planes, partials, grammar, bases, and progressively specialized Cellerator entry points—not predefined row shards.

**Compiler-architectural reason.** Use an adjacent immutable manifest-and-arena format rather than enlarging CSH5 or CSPACK. Large sequential arenas contain independently indexed atom frames; logical atom and storage extent boundaries remain separate.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/artifact/atom_store/
- [proposed] src/artifact/atom_store/
- [proposed] docs/SPEC_ATOM_STORE_V1.md
- [proposed] tests/jbc/atom_store/

**Existing code and permitted read scope:**

- include/CellShard/artifact/image.hh
- include/CellShard/artifact/extent.hh
- include/CellShard/artifact/catalog.hh
- include/CellShard/artifact/snapshot.hh
- include/CellShard/io/pack/image_envelope.hh
- docs/SPEC_CSPACK_V1.md

**Explicitly out of scope / forbidden shortcuts:**

- Do not create one file per atom by default.
- Do not use CSH5 or CSPACK as the new universal compiler database.
- Do not put source paths or placement epochs into immutable content identity.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-ST06

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.

### Concrete mechanism

Index structural/value/state/gradient/partial/physical/executable planes by atom, generation, order, target capability, projection ABI, encoding, and persistence status.

Workstream mechanism: Freeze a new format family after collision audit. Separate semantic, content, materialization, replica, and action identity. Store root generation, atom dictionary, coverage/plane indexes, grammar, bases, superatoms, partials, lowering stages, physical views, lineage, actions, and large arena/frame payloads. Publish atomically; support recovery, consolidation, GC, CSH5/CSPACK import, codecs, and metadata-only inspection.

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

- Use corruption/torn-write/orphan/partial-publication tests and verify decoded as well as encoded content identities.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Measure sequential I/O, selected-range I/O, frame count, read amplification, metadata memory, compression, assembly, and block-storage throughput.
- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-PERSISTENCE; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-ST08 — Define composition and grammar records

**Repository / subsystem / lane:** CellShard · `artifact/atom_store` · `CS-JBC-L-PERSISTENCE`  
**Classification:** generic infrastructure supporting the biology-native compiler  
**Parallelism:** Serial within CS-JBC-L-PERSISTENCE; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Define composition and grammar records. Deliver this as one isolated, reviewable step in the Atom-native immutable persistence and lowering artifacts workstream.

**Biological motivation.** The physical archive must persist an atomic reusable execution hierarchy, mutable planes, partials, grammar, bases, and progressively specialized Cellerator entry points—not predefined row shards.

**Compiler-architectural reason.** Use an adjacent immutable manifest-and-arena format rather than enlarging CSH5 or CSPACK. Large sequential arenas contain independently indexed atom frames; logical atom and storage extent boundaries remain separate.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/artifact/atom_store/
- [proposed] src/artifact/atom_store/
- [proposed] docs/SPEC_ATOM_STORE_V1.md
- [proposed] tests/jbc/atom_store/

**Existing code and permitted read scope:**

- include/CellShard/artifact/image.hh
- include/CellShard/artifact/extent.hh
- include/CellShard/artifact/catalog.hh
- include/CellShard/artifact/snapshot.hh
- include/CellShard/io/pack/image_envelope.hh
- docs/SPEC_CSPACK_V1.md

**Explicitly out of scope / forbidden shortcuts:**

- Do not create one file per atom by default.
- Do not use CSH5 or CSPACK as the new universal compiler database.
- Do not put source paths or placement epochs into immutable content identity.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-ST07

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.

### Concrete mechanism

Persist symbols, production IDs/versions, parameters, exact coverage-equation references, derivation parents, verifier receipts, and grammar roots.

Workstream mechanism: Freeze a new format family after collision audit. Separate semantic, content, materialization, replica, and action identity. Store root generation, atom dictionary, coverage/plane indexes, grammar, bases, superatoms, partials, lowering stages, physical views, lineage, actions, and large arena/frame payloads. Publish atomically; support recovery, consolidation, GC, CSH5/CSPACK import, codecs, and metadata-only inspection.

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

- Use corruption/torn-write/orphan/partial-publication tests and verify decoded as well as encoded content identities.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Measure sequential I/O, selected-range I/O, frame count, read amplification, metadata memory, compression, assembly, and block-storage throughput.
- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-PERSISTENCE; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-ST09 — Define basis and superatom records

**Repository / subsystem / lane:** CellShard · `artifact/atom_store` · `CS-JBC-L-PERSISTENCE`  
**Classification:** generic infrastructure supporting the biology-native compiler  
**Parallelism:** Serial within CS-JBC-L-PERSISTENCE; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Define basis and superatom records. Deliver this as one isolated, reviewable step in the Atom-native immutable persistence and lowering artifacts workstream.

**Biological motivation.** The physical archive must persist an atomic reusable execution hierarchy, mutable planes, partials, grammar, bases, and progressively specialized Cellerator entry points—not predefined row shards.

**Compiler-architectural reason.** Use an adjacent immutable manifest-and-arena format rather than enlarging CSH5 or CSPACK. Large sequential arenas contain independently indexed atom frames; logical atom and storage extent boundaries remain separate.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/artifact/atom_store/
- [proposed] src/artifact/atom_store/
- [proposed] docs/SPEC_ATOM_STORE_V1.md
- [proposed] tests/jbc/atom_store/

**Existing code and permitted read scope:**

- include/CellShard/artifact/image.hh
- include/CellShard/artifact/extent.hh
- include/CellShard/artifact/catalog.hh
- include/CellShard/artifact/snapshot.hh
- include/CellShard/io/pack/image_envelope.hh
- docs/SPEC_CSPACK_V1.md

**Explicitly out of scope / forbidden shortcuts:**

- Do not create one file per atom by default.
- Do not use CSH5 or CSPACK as the new universal compiler database.
- Do not put source paths or placement epochs into immutable content identity.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-ST08

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.

### Concrete mechanism

Persist workload-family identity, selected atoms/productions, objective/budgets, basis membership, superatom promotion lineage, validity, and freshness.

Workstream mechanism: Freeze a new format family after collision audit. Separate semantic, content, materialization, replica, and action identity. Store root generation, atom dictionary, coverage/plane indexes, grammar, bases, superatoms, partials, lowering stages, physical views, lineage, actions, and large arena/frame payloads. Publish atomically; support recovery, consolidation, GC, CSH5/CSPACK import, codecs, and metadata-only inspection.

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

- Use corruption/torn-write/orphan/partial-publication tests and verify decoded as well as encoded content identities.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Measure sequential I/O, selected-range I/O, frame count, read amplification, metadata memory, compression, assembly, and block-storage throughput.
- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-PERSISTENCE; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-ST10 — Define partial and lowering-resumption records

**Repository / subsystem / lane:** CellShard · `artifact/atom_store` · `CS-JBC-L-PERSISTENCE`  
**Classification:** generic infrastructure supporting the biology-native compiler  
**Parallelism:** Serial within CS-JBC-L-PERSISTENCE; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Define partial and lowering-resumption records. Deliver this as one isolated, reviewable step in the Atom-native immutable persistence and lowering artifacts workstream.

**Biological motivation.** The physical archive must persist an atomic reusable execution hierarchy, mutable planes, partials, grammar, bases, and progressively specialized Cellerator entry points—not predefined row shards.

**Compiler-architectural reason.** Use an adjacent immutable manifest-and-arena format rather than enlarging CSH5 or CSPACK. Large sequential arenas contain independently indexed atom frames; logical atom and storage extent boundaries remain separate.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/artifact/atom_store/
- [proposed] src/artifact/atom_store/
- [proposed] docs/SPEC_ATOM_STORE_V1.md
- [proposed] tests/jbc/atom_store/

**Existing code and permitted read scope:**

- include/CellShard/artifact/image.hh
- include/CellShard/artifact/extent.hh
- include/CellShard/artifact/catalog.hh
- include/CellShard/artifact/snapshot.hh
- include/CellShard/io/pack/image_envelope.hh
- docs/SPEC_CSPACK_V1.md

**Explicitly out of scope / forbidden shortcuts:**

- Do not create one file per atom by default.
- Do not use CSH5 or CSPACK as the new universal compiler database.
- Do not put source paths or placement epochs into immutable content identity.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-ST09

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.

### Concrete mechanism

Persist partial algebra/dependencies/state and Cellerator resumption stage, producer ABI, target restrictions, phases bypassed, and fallback parent.

Workstream mechanism: Freeze a new format family after collision audit. Separate semantic, content, materialization, replica, and action identity. Store root generation, atom dictionary, coverage/plane indexes, grammar, bases, superatoms, partials, lowering stages, physical views, lineage, actions, and large arena/frame payloads. Publish atomically; support recovery, consolidation, GC, CSH5/CSPACK import, codecs, and metadata-only inspection.

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

- Use corruption/torn-write/orphan/partial-publication tests and verify decoded as well as encoded content identities.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Measure sequential I/O, selected-range I/O, frame count, read amplification, metadata memory, compression, assembly, and block-storage throughput.
- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-PERSISTENCE; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-ST11 — Define the large arena header and directory

**Repository / subsystem / lane:** CellShard · `artifact/atom_store` · `CS-JBC-L-PERSISTENCE`  
**Classification:** generic infrastructure supporting the biology-native compiler  
**Parallelism:** Serial within CS-JBC-L-PERSISTENCE; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Define the large arena header and directory. Deliver this as one isolated, reviewable step in the Atom-native immutable persistence and lowering artifacts workstream.

**Biological motivation.** The physical archive must persist an atomic reusable execution hierarchy, mutable planes, partials, grammar, bases, and progressively specialized Cellerator entry points—not predefined row shards.

**Compiler-architectural reason.** Use an adjacent immutable manifest-and-arena format rather than enlarging CSH5 or CSPACK. Large sequential arenas contain independently indexed atom frames; logical atom and storage extent boundaries remain separate.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/artifact/atom_store/
- [proposed] src/artifact/atom_store/
- [proposed] docs/SPEC_ATOM_STORE_V1.md
- [proposed] tests/jbc/atom_store/

**Existing code and permitted read scope:**

- include/CellShard/artifact/image.hh
- include/CellShard/artifact/extent.hh
- include/CellShard/artifact/catalog.hh
- include/CellShard/artifact/snapshot.hh
- include/CellShard/io/pack/image_envelope.hh
- docs/SPEC_CSPACK_V1.md

**Explicitly out of scope / forbidden shortcuts:**

- Do not create one file per atom by default.
- Do not use CSH5 or CSPACK as the new universal compiler database.
- Do not put source paths or placement epochs into immutable content identity.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-ST10

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.

### Concrete mechanism

Create a pointer-free endian-defined arena with schema, archive generation, payload bytes, alignment, frame count, directory offsets, checksums, and no assumption that one arena equals one atom or shard.

Workstream mechanism: Freeze a new format family after collision audit. Separate semantic, content, materialization, replica, and action identity. Store root generation, atom dictionary, coverage/plane indexes, grammar, bases, superatoms, partials, lowering stages, physical views, lineage, actions, and large arena/frame payloads. Publish atomically; support recovery, consolidation, GC, CSH5/CSPACK import, codecs, and metadata-only inspection.

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

- Use corruption/torn-write/orphan/partial-publication tests and verify decoded as well as encoded content identities.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Measure sequential I/O, selected-range I/O, frame count, read amplification, metadata memory, compression, assembly, and block-storage throughput.
- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-PERSISTENCE; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-ST12 — Define independently addressable atom-frame headers

**Repository / subsystem / lane:** CellShard · `artifact/atom_store` · `CS-JBC-L-PERSISTENCE`  
**Classification:** generic infrastructure supporting the biology-native compiler  
**Parallelism:** Serial within CS-JBC-L-PERSISTENCE; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Define independently addressable atom-frame headers. Deliver this as one isolated, reviewable step in the Atom-native immutable persistence and lowering artifacts workstream.

**Biological motivation.** The physical archive must persist an atomic reusable execution hierarchy, mutable planes, partials, grammar, bases, and progressively specialized Cellerator entry points—not predefined row shards.

**Compiler-architectural reason.** Use an adjacent immutable manifest-and-arena format rather than enlarging CSH5 or CSPACK. Large sequential arenas contain independently indexed atom frames; logical atom and storage extent boundaries remain separate.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/artifact/atom_store/
- [proposed] src/artifact/atom_store/
- [proposed] docs/SPEC_ATOM_STORE_V1.md
- [proposed] tests/jbc/atom_store/

**Existing code and permitted read scope:**

- include/CellShard/artifact/image.hh
- include/CellShard/artifact/extent.hh
- include/CellShard/artifact/catalog.hh
- include/CellShard/artifact/snapshot.hh
- include/CellShard/io/pack/image_envelope.hh
- docs/SPEC_CSPACK_V1.md

**Explicitly out of scope / forbidden shortcuts:**

- Do not create one file per atom by default.
- Do not use CSH5 or CSPACK as the new universal compiler database.
- Do not put source paths or placement epochs into immutable content identity.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-ST11

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.

### Concrete mechanism

Each frame names physical instance/part, plane, coverage, codec pipeline, stored/decoded bytes, alignments, encoded and decoded digests, and payload offset.

Workstream mechanism: Freeze a new format family after collision audit. Separate semantic, content, materialization, replica, and action identity. Store root generation, atom dictionary, coverage/plane indexes, grammar, bases, superatoms, partials, lowering stages, physical views, lineage, actions, and large arena/frame payloads. Publish atomically; support recovery, consolidation, GC, CSH5/CSPACK import, codecs, and metadata-only inspection.

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

- Use corruption/torn-write/orphan/partial-publication tests and verify decoded as well as encoded content identities.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Measure sequential I/O, selected-range I/O, frame count, read amplification, metadata memory, compression, assembly, and block-storage throughput.
- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-PERSISTENCE; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-ST13 — Define multi-frame and multi-extent atom mappings

**Repository / subsystem / lane:** CellShard · `artifact/atom_store` · `CS-JBC-L-PERSISTENCE`  
**Classification:** generic infrastructure supporting the biology-native compiler  
**Parallelism:** Serial within CS-JBC-L-PERSISTENCE; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Define multi-frame and multi-extent atom mappings. Deliver this as one isolated, reviewable step in the Atom-native immutable persistence and lowering artifacts workstream.

**Biological motivation.** The physical archive must persist an atomic reusable execution hierarchy, mutable planes, partials, grammar, bases, and progressively specialized Cellerator entry points—not predefined row shards.

**Compiler-architectural reason.** Use an adjacent immutable manifest-and-arena format rather than enlarging CSH5 or CSPACK. Large sequential arenas contain independently indexed atom frames; logical atom and storage extent boundaries remain separate.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/artifact/atom_store/
- [proposed] src/artifact/atom_store/
- [proposed] docs/SPEC_ATOM_STORE_V1.md
- [proposed] tests/jbc/atom_store/

**Existing code and permitted read scope:**

- include/CellShard/artifact/image.hh
- include/CellShard/artifact/extent.hh
- include/CellShard/artifact/catalog.hh
- include/CellShard/artifact/snapshot.hh
- include/CellShard/io/pack/image_envelope.hh
- docs/SPEC_CSPACK_V1.md

**Explicitly out of scope / forbidden shortcuts:**

- Do not create one file per atom by default.
- Do not use CSH5 or CSPACK as the new universal compiler database.
- Do not put source paths or placement epochs into immutable content identity.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-ST12

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.
- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Allow one physical atom to span several frames/extents and several atoms to share one sequential arena; preserve ordered assembly and exact byte ranges.

Workstream mechanism: Freeze a new format family after collision audit. Separate semantic, content, materialization, replica, and action identity. Store root generation, atom dictionary, coverage/plane indexes, grammar, bases, superatoms, partials, lowering stages, physical views, lineage, actions, and large arena/frame payloads. Publish atomically; support recovery, consolidation, GC, CSH5/CSPACK import, codecs, and metadata-only inspection.

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

- Use corruption/torn-write/orphan/partial-publication tests and verify decoded as well as encoded content identities.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Measure sequential I/O, selected-range I/O, frame count, read amplification, metadata memory, compression, assembly, and block-storage throughput.
- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-PERSISTENCE; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-ST14 — Define encoded-replica descriptors

**Repository / subsystem / lane:** CellShard · `artifact/atom_store` · `CS-JBC-L-PERSISTENCE`  
**Classification:** generic infrastructure supporting the biology-native compiler  
**Parallelism:** Serial within CS-JBC-L-PERSISTENCE; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Define encoded-replica descriptors. Deliver this as one isolated, reviewable step in the Atom-native immutable persistence and lowering artifacts workstream.

**Biological motivation.** The physical archive must persist an atomic reusable execution hierarchy, mutable planes, partials, grammar, bases, and progressively specialized Cellerator entry points—not predefined row shards.

**Compiler-architectural reason.** Use an adjacent immutable manifest-and-arena format rather than enlarging CSH5 or CSPACK. Large sequential arenas contain independently indexed atom frames; logical atom and storage extent boundaries remain separate.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/artifact/atom_store/
- [proposed] src/artifact/atom_store/
- [proposed] docs/SPEC_ATOM_STORE_V1.md
- [proposed] tests/jbc/atom_store/

**Existing code and permitted read scope:**

- include/CellShard/artifact/image.hh
- include/CellShard/artifact/extent.hh
- include/CellShard/artifact/catalog.hh
- include/CellShard/artifact/snapshot.hh
- include/CellShard/io/pack/image_envelope.hh
- docs/SPEC_CSPACK_V1.md

**Explicitly out of scope / forbidden shortcuts:**

- Do not create one file per atom by default.
- Do not use CSH5 or CSPACK as the new universal compiler database.
- Do not put source paths or placement epochs into immutable content identity.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-ST13

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.

### Concrete mechanism

Represent codec, stored content digest, source/storage object/extents, location-independent replica identity, decoded physical instance, and operational source locations outside immutable identity.

Workstream mechanism: Freeze a new format family after collision audit. Separate semantic, content, materialization, replica, and action identity. Store root generation, atom dictionary, coverage/plane indexes, grammar, bases, superatoms, partials, lowering stages, physical views, lineage, actions, and large arena/frame payloads. Publish atomically; support recovery, consolidation, GC, CSH5/CSPACK import, codecs, and metadata-only inspection.

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

- Use corruption/torn-write/orphan/partial-publication tests and verify decoded as well as encoded content identities.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Measure sequential I/O, selected-range I/O, frame count, read amplification, metadata memory, compression, assembly, and block-storage throughput.
- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-PERSISTENCE; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-ST15 — Implement the positive action cache

**Repository / subsystem / lane:** CellShard · `artifact/atom_store` · `CS-JBC-L-PERSISTENCE`  
**Classification:** generic infrastructure supporting the biology-native compiler  
**Parallelism:** Serial within CS-JBC-L-PERSISTENCE; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Implement the positive action cache. Deliver this as one isolated, reviewable step in the Atom-native immutable persistence and lowering artifacts workstream.

**Biological motivation.** The physical archive must persist an atomic reusable execution hierarchy, mutable planes, partials, grammar, bases, and progressively specialized Cellerator entry points—not predefined row shards.

**Compiler-architectural reason.** Use an adjacent immutable manifest-and-arena format rather than enlarging CSH5 or CSPACK. Large sequential arenas contain independently indexed atom frames; logical atom and storage extent boundaries remain separate.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/artifact/atom_store/
- [proposed] src/artifact/atom_store/
- [proposed] docs/SPEC_ATOM_STORE_V1.md
- [proposed] tests/jbc/atom_store/

**Existing code and permitted read scope:**

- include/CellShard/artifact/image.hh
- include/CellShard/artifact/extent.hh
- include/CellShard/artifact/catalog.hh
- include/CellShard/artifact/snapshot.hh
- include/CellShard/io/pack/image_envelope.hh
- docs/SPEC_CSPACK_V1.md

**Explicitly out of scope / forbidden shortcuts:**

- Do not create one file per atom by default.
- Do not use CSH5 or CSPACK as the new universal compiler database.
- Do not put source paths or placement epochs into immutable content identity.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-ST14

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Map compiler action identity—inputs, production, compiler/provider versions, target/numeric/cost policy—to output atom/materialization IDs and measured build evidence.

Workstream mechanism: Freeze a new format family after collision audit. Separate semantic, content, materialization, replica, and action identity. Store root generation, atom dictionary, coverage/plane indexes, grammar, bases, superatoms, partials, lowering stages, physical views, lineage, actions, and large arena/frame payloads. Publish atomically; support recovery, consolidation, GC, CSH5/CSPACK import, codecs, and metadata-only inspection.

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

- Use corruption/torn-write/orphan/partial-publication tests and verify decoded as well as encoded content identities.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Measure sequential I/O, selected-range I/O, frame count, read amplification, metadata memory, compression, assembly, and block-storage throughput.
- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-PERSISTENCE; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-ST16 — Implement the negative action cache

**Repository / subsystem / lane:** CellShard · `artifact/atom_store` · `CS-JBC-L-PERSISTENCE`  
**Classification:** generic infrastructure supporting the biology-native compiler  
**Parallelism:** Serial within CS-JBC-L-PERSISTENCE; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Implement the negative action cache. Deliver this as one isolated, reviewable step in the Atom-native immutable persistence and lowering artifacts workstream.

**Biological motivation.** The physical archive must persist an atomic reusable execution hierarchy, mutable planes, partials, grammar, bases, and progressively specialized Cellerator entry points—not predefined row shards.

**Compiler-architectural reason.** Use an adjacent immutable manifest-and-arena format rather than enlarging CSH5 or CSPACK. Large sequential arenas contain independently indexed atom frames; logical atom and storage extent boundaries remain separate.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/artifact/atom_store/
- [proposed] src/artifact/atom_store/
- [proposed] docs/SPEC_ATOM_STORE_V1.md
- [proposed] tests/jbc/atom_store/

**Existing code and permitted read scope:**

- include/CellShard/artifact/image.hh
- include/CellShard/artifact/extent.hh
- include/CellShard/artifact/catalog.hh
- include/CellShard/artifact/snapshot.hh
- include/CellShard/io/pack/image_envelope.hh
- docs/SPEC_CSPACK_V1.md

**Explicitly out of scope / forbidden shortcuts:**

- Do not create one file per atom by default.
- Do not use CSH5 or CSPACK as the new universal compiler database.
- Do not put source paths or placement epochs into immutable content identity.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-ST15

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Persist rejected candidate/action, reason, exact evidence, break-even, relevant cost/reuse assumptions, and staleness predicate; allow reevaluation when assumptions change.

Workstream mechanism: Freeze a new format family after collision audit. Separate semantic, content, materialization, replica, and action identity. Store root generation, atom dictionary, coverage/plane indexes, grammar, bases, superatoms, partials, lowering stages, physical views, lineage, actions, and large arena/frame payloads. Publish atomically; support recovery, consolidation, GC, CSH5/CSPACK import, codecs, and metadata-only inspection.

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

- Use corruption/torn-write/orphan/partial-publication tests and verify decoded as well as encoded content identities.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Measure sequential I/O, selected-range I/O, frame count, read amplification, metadata memory, compression, assembly, and block-storage throughput.
- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-PERSISTENCE; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-ST17 — Implement composition lineage and provenance records

**Repository / subsystem / lane:** CellShard · `artifact/atom_store` · `CS-JBC-L-PERSISTENCE`  
**Classification:** generic infrastructure supporting the biology-native compiler  
**Parallelism:** Serial within CS-JBC-L-PERSISTENCE; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Implement composition lineage and provenance records. Deliver this as one isolated, reviewable step in the Atom-native immutable persistence and lowering artifacts workstream.

**Biological motivation.** The physical archive must persist an atomic reusable execution hierarchy, mutable planes, partials, grammar, bases, and progressively specialized Cellerator entry points—not predefined row shards.

**Compiler-architectural reason.** Use an adjacent immutable manifest-and-arena format rather than enlarging CSH5 or CSPACK. Large sequential arenas contain independently indexed atom frames; logical atom and storage extent boundaries remain separate.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/artifact/atom_store/
- [proposed] src/artifact/atom_store/
- [proposed] docs/SPEC_ATOM_STORE_V1.md
- [proposed] tests/jbc/atom_store/

**Existing code and permitted read scope:**

- include/CellShard/artifact/image.hh
- include/CellShard/artifact/extent.hh
- include/CellShard/artifact/catalog.hh
- include/CellShard/artifact/snapshot.hh
- include/CellShard/io/pack/image_envelope.hh
- docs/SPEC_CSPACK_V1.md

**Explicitly out of scope / forbidden shortcuts:**

- Do not create one file per atom by default.
- Do not use CSH5 or CSPACK as the new universal compiler database.
- Do not put source paths or placement epochs into immutable content identity.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-ST16

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.
- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Store parent atoms/actions/productions, exact coverage equation, parameterization, build environment, target assumptions, cost, verifier receipt, and output digests.

Workstream mechanism: Freeze a new format family after collision audit. Separate semantic, content, materialization, replica, and action identity. Store root generation, atom dictionary, coverage/plane indexes, grammar, bases, superatoms, partials, lowering stages, physical views, lineage, actions, and large arena/frame payloads. Publish atomically; support recovery, consolidation, GC, CSH5/CSPACK import, codecs, and metadata-only inspection.

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

- Use corruption/torn-write/orphan/partial-publication tests and verify decoded as well as encoded content identities.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Measure sequential I/O, selected-range I/O, frame count, read amplification, metadata memory, compression, assembly, and block-storage throughput.
- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-PERSISTENCE; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-ST18 — Implement two-pass atom-store writer requirements and fill

**Repository / subsystem / lane:** CellShard · `artifact/atom_store` · `CS-JBC-L-PERSISTENCE`  
**Classification:** generic infrastructure supporting the biology-native compiler  
**Parallelism:** Serial within CS-JBC-L-PERSISTENCE; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Implement two-pass atom-store writer requirements and fill. Deliver this as one isolated, reviewable step in the Atom-native immutable persistence and lowering artifacts workstream.

**Biological motivation.** The physical archive must persist an atomic reusable execution hierarchy, mutable planes, partials, grammar, bases, and progressively specialized Cellerator entry points—not predefined row shards.

**Compiler-architectural reason.** Use an adjacent immutable manifest-and-arena format rather than enlarging CSH5 or CSPACK. Large sequential arenas contain independently indexed atom frames; logical atom and storage extent boundaries remain separate.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/artifact/atom_store/
- [proposed] src/artifact/atom_store/
- [proposed] docs/SPEC_ATOM_STORE_V1.md
- [proposed] tests/jbc/atom_store/

**Existing code and permitted read scope:**

- include/CellShard/artifact/image.hh
- include/CellShard/artifact/extent.hh
- include/CellShard/artifact/catalog.hh
- include/CellShard/artifact/snapshot.hh
- include/CellShard/io/pack/image_envelope.hh
- docs/SPEC_CSPACK_V1.md

**Explicitly out of scope / forbidden shortcuts:**

- Do not create one file per atom by default.
- Do not use CSH5 or CSPACK as the new universal compiler database.
- Do not put source paths or placement epochs into immutable content identity.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-ST17

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Query all section/arena/index capacities, validate offsets and overflow, fill caller-owned or explicit writer-owned buffers deterministically, and expose no hidden whole-dataset copy.

Workstream mechanism: Freeze a new format family after collision audit. Separate semantic, content, materialization, replica, and action identity. Store root generation, atom dictionary, coverage/plane indexes, grammar, bases, superatoms, partials, lowering stages, physical views, lineage, actions, and large arena/frame payloads. Publish atomically; support recovery, consolidation, GC, CSH5/CSPACK import, codecs, and metadata-only inspection.

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

- Use corruption/torn-write/orphan/partial-publication tests and verify decoded as well as encoded content identities.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Measure sequential I/O, selected-range I/O, frame count, read amplification, metadata memory, compression, assembly, and block-storage throughput.
- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-PERSISTENCE; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-ST19 — Implement metadata-only reader and inspector

**Repository / subsystem / lane:** CellShard · `artifact/atom_store` · `CS-JBC-L-PERSISTENCE`  
**Classification:** generic infrastructure supporting the biology-native compiler  
**Parallelism:** Serial within CS-JBC-L-PERSISTENCE; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Implement metadata-only reader and inspector. Deliver this as one isolated, reviewable step in the Atom-native immutable persistence and lowering artifacts workstream.

**Biological motivation.** The physical archive must persist an atomic reusable execution hierarchy, mutable planes, partials, grammar, bases, and progressively specialized Cellerator entry points—not predefined row shards.

**Compiler-architectural reason.** Use an adjacent immutable manifest-and-arena format rather than enlarging CSH5 or CSPACK. Large sequential arenas contain independently indexed atom frames; logical atom and storage extent boundaries remain separate.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/artifact/atom_store/
- [proposed] src/artifact/atom_store/
- [proposed] docs/SPEC_ATOM_STORE_V1.md
- [proposed] tests/jbc/atom_store/

**Existing code and permitted read scope:**

- include/CellShard/artifact/image.hh
- include/CellShard/artifact/extent.hh
- include/CellShard/artifact/catalog.hh
- include/CellShard/artifact/snapshot.hh
- include/CellShard/io/pack/image_envelope.hh
- docs/SPEC_CSPACK_V1.md

**Explicitly out of scope / forbidden shortcuts:**

- Do not create one file per atom by default.
- Do not use CSH5 or CSPACK as the new universal compiler database.
- Do not put source paths or placement epochs into immutable content identity.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-ST18

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Open root/arena/index metadata, validate headers/digests/ranges, list atoms/planes/bases/actions without loading payload frames, and support selective lookup.

Workstream mechanism: Freeze a new format family after collision audit. Separate semantic, content, materialization, replica, and action identity. Store root generation, atom dictionary, coverage/plane indexes, grammar, bases, superatoms, partials, lowering stages, physical views, lineage, actions, and large arena/frame payloads. Publish atomically; support recovery, consolidation, GC, CSH5/CSPACK import, codecs, and metadata-only inspection.

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

- Use corruption/torn-write/orphan/partial-publication tests and verify decoded as well as encoded content identities.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Measure sequential I/O, selected-range I/O, frame count, read amplification, metadata memory, compression, assembly, and block-storage throughput.
- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-PERSISTENCE; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-ST20 — Implement exact multi-range atom source adapter

**Repository / subsystem / lane:** CellShard · `artifact/atom_store` · `CS-JBC-L-PERSISTENCE`  
**Classification:** generic infrastructure supporting the biology-native compiler  
**Parallelism:** Serial within CS-JBC-L-PERSISTENCE; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Implement exact multi-range atom source adapter. Deliver this as one isolated, reviewable step in the Atom-native immutable persistence and lowering artifacts workstream.

**Biological motivation.** The physical archive must persist an atomic reusable execution hierarchy, mutable planes, partials, grammar, bases, and progressively specialized Cellerator entry points—not predefined row shards.

**Compiler-architectural reason.** Use an adjacent immutable manifest-and-arena format rather than enlarging CSH5 or CSPACK. Large sequential arenas contain independently indexed atom frames; logical atom and storage extent boundaries remain separate.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/artifact/atom_store/
- [proposed] src/artifact/atom_store/
- [proposed] docs/SPEC_ATOM_STORE_V1.md
- [proposed] tests/jbc/atom_store/

**Existing code and permitted read scope:**

- include/CellShard/artifact/image.hh
- include/CellShard/artifact/extent.hh
- include/CellShard/artifact/catalog.hh
- include/CellShard/artifact/snapshot.hh
- include/CellShard/io/pack/image_envelope.hh
- docs/SPEC_CSPACK_V1.md

**Explicitly out of scope / forbidden shortcuts:**

- Do not create one file per atom by default.
- Do not use CSH5 or CSPACK as the new universal compiler database.
- Do not put source paths or placement epochs into immutable content identity.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-ST19

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Translate selected atom frames into ordered/coalescible exact byte-range requests through current payload_source_ref or the async source-v2 contract.

Workstream mechanism: Freeze a new format family after collision audit. Separate semantic, content, materialization, replica, and action identity. Store root generation, atom dictionary, coverage/plane indexes, grammar, bases, superatoms, partials, lowering stages, physical views, lineage, actions, and large arena/frame payloads. Publish atomically; support recovery, consolidation, GC, CSH5/CSPACK import, codecs, and metadata-only inspection.

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

- Use corruption/torn-write/orphan/partial-publication tests and verify decoded as well as encoded content identities.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Measure sequential I/O, selected-range I/O, frame count, read amplification, metadata memory, compression, assembly, and block-storage throughput.
- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-PERSISTENCE; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-ST21 — Implement atomic publication and durable root switch

**Repository / subsystem / lane:** CellShard · `artifact/atom_store` · `CS-JBC-L-PERSISTENCE`  
**Classification:** generic infrastructure supporting the biology-native compiler  
**Parallelism:** Serial within CS-JBC-L-PERSISTENCE; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Implement atomic publication and durable root switch. Deliver this as one isolated, reviewable step in the Atom-native immutable persistence and lowering artifacts workstream.

**Biological motivation.** The physical archive must persist an atomic reusable execution hierarchy, mutable planes, partials, grammar, bases, and progressively specialized Cellerator entry points—not predefined row shards.

**Compiler-architectural reason.** Use an adjacent immutable manifest-and-arena format rather than enlarging CSH5 or CSPACK. Large sequential arenas contain independently indexed atom frames; logical atom and storage extent boundaries remain separate.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/artifact/atom_store/
- [proposed] src/artifact/atom_store/
- [proposed] docs/SPEC_ATOM_STORE_V1.md
- [proposed] tests/jbc/atom_store/

**Existing code and permitted read scope:**

- include/CellShard/artifact/image.hh
- include/CellShard/artifact/extent.hh
- include/CellShard/artifact/catalog.hh
- include/CellShard/artifact/snapshot.hh
- include/CellShard/io/pack/image_envelope.hh
- docs/SPEC_CSPACK_V1.md

**Explicitly out of scope / forbidden shortcuts:**

- Do not create one file per atom by default.
- Do not use CSH5 or CSPACK as the new universal compiler database.
- Do not put source paths or placement epochs into immutable content identity.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-ST20

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Write temporary arenas/indexes/manifests, validate, sync required files, publish commit/root atomically, sync containing directory where relevant, and retain prior root.

Workstream mechanism: Freeze a new format family after collision audit. Separate semantic, content, materialization, replica, and action identity. Store root generation, atom dictionary, coverage/plane indexes, grammar, bases, superatoms, partials, lowering stages, physical views, lineage, actions, and large arena/frame payloads. Publish atomically; support recovery, consolidation, GC, CSH5/CSPACK import, codecs, and metadata-only inspection.

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

- Use corruption/torn-write/orphan/partial-publication tests and verify decoded as well as encoded content identities.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Measure sequential I/O, selected-range I/O, frame count, read amplification, metadata memory, compression, assembly, and block-storage throughput.
- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-PERSISTENCE; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-ST22 — Implement crash recovery and orphan detection

**Repository / subsystem / lane:** CellShard · `artifact/atom_store` · `CS-JBC-L-PERSISTENCE`  
**Classification:** generic infrastructure supporting the biology-native compiler  
**Parallelism:** Serial within CS-JBC-L-PERSISTENCE; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Implement crash recovery and orphan detection. Deliver this as one isolated, reviewable step in the Atom-native immutable persistence and lowering artifacts workstream.

**Biological motivation.** The physical archive must persist an atomic reusable execution hierarchy, mutable planes, partials, grammar, bases, and progressively specialized Cellerator entry points—not predefined row shards.

**Compiler-architectural reason.** Use an adjacent immutable manifest-and-arena format rather than enlarging CSH5 or CSPACK. Large sequential arenas contain independently indexed atom frames; logical atom and storage extent boundaries remain separate.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/artifact/atom_store/
- [proposed] src/artifact/atom_store/
- [proposed] docs/SPEC_ATOM_STORE_V1.md
- [proposed] tests/jbc/atom_store/

**Existing code and permitted read scope:**

- include/CellShard/artifact/image.hh
- include/CellShard/artifact/extent.hh
- include/CellShard/artifact/catalog.hh
- include/CellShard/artifact/snapshot.hh
- include/CellShard/io/pack/image_envelope.hh
- docs/SPEC_CSPACK_V1.md

**Explicitly out of scope / forbidden shortcuts:**

- Do not create one file per atom by default.
- Do not use CSH5 or CSPACK as the new universal compiler database.
- Do not put source paths or placement epochs into immutable content identity.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-ST21

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.
- Focused tests and adversarial fixtures demonstrating both success and explicit rejection paths.

### Concrete mechanism

Detect incomplete roots, missing/corrupt arenas, torn frames, orphaned temporary generations, and invalid action outputs; recover to last valid root without guessing.

Workstream mechanism: Freeze a new format family after collision audit. Separate semantic, content, materialization, replica, and action identity. Store root generation, atom dictionary, coverage/plane indexes, grammar, bases, superatoms, partials, lowering stages, physical views, lineage, actions, and large arena/frame payloads. Publish atomically; support recovery, consolidation, GC, CSH5/CSPACK import, codecs, and metadata-only inspection.

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

- Use corruption/torn-write/orphan/partial-publication tests and verify decoded as well as encoded content identities.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Measure sequential I/O, selected-range I/O, frame count, read amplification, metadata memory, compression, assembly, and block-storage throughput.
- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-PERSISTENCE; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-ST23 — Implement consolidation

**Repository / subsystem / lane:** CellShard · `artifact/atom_store` · `CS-JBC-L-PERSISTENCE`  
**Classification:** generic infrastructure supporting the biology-native compiler  
**Parallelism:** Serial within CS-JBC-L-PERSISTENCE; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Implement consolidation. Deliver this as one isolated, reviewable step in the Atom-native immutable persistence and lowering artifacts workstream.

**Biological motivation.** The physical archive must persist an atomic reusable execution hierarchy, mutable planes, partials, grammar, bases, and progressively specialized Cellerator entry points—not predefined row shards.

**Compiler-architectural reason.** Use an adjacent immutable manifest-and-arena format rather than enlarging CSH5 or CSPACK. Large sequential arenas contain independently indexed atom frames; logical atom and storage extent boundaries remain separate.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/artifact/atom_store/
- [proposed] src/artifact/atom_store/
- [proposed] docs/SPEC_ATOM_STORE_V1.md
- [proposed] tests/jbc/atom_store/

**Existing code and permitted read scope:**

- include/CellShard/artifact/image.hh
- include/CellShard/artifact/extent.hh
- include/CellShard/artifact/catalog.hh
- include/CellShard/artifact/snapshot.hh
- include/CellShard/io/pack/image_envelope.hh
- docs/SPEC_CSPACK_V1.md

**Explicitly out of scope / forbidden shortcuts:**

- Do not create one file per atom by default.
- Do not use CSH5 or CSPACK as the new universal compiler database.
- Do not put source paths or placement epochs into immutable content identity.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-ST22

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Merge small arenas/indexes, compact delta/value layers, materialize repeatedly assembled superatoms, or split overread-heavy frames while preserving semantic/content identity and snapshots.

Workstream mechanism: Freeze a new format family after collision audit. Separate semantic, content, materialization, replica, and action identity. Store root generation, atom dictionary, coverage/plane indexes, grammar, bases, superatoms, partials, lowering stages, physical views, lineage, actions, and large arena/frame payloads. Publish atomically; support recovery, consolidation, GC, CSH5/CSPACK import, codecs, and metadata-only inspection.

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

- Use corruption/torn-write/orphan/partial-publication tests and verify decoded as well as encoded content identities.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Measure sequential I/O, selected-range I/O, frame count, read amplification, metadata memory, compression, assembly, and block-storage throughput.
- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-PERSISTENCE; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-ST24 — Implement reachability garbage collection and snapshot pins

**Repository / subsystem / lane:** CellShard · `artifact/atom_store` · `CS-JBC-L-PERSISTENCE`  
**Classification:** generic infrastructure supporting the biology-native compiler  
**Parallelism:** Serial within CS-JBC-L-PERSISTENCE; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Implement reachability garbage collection and snapshot pins. Deliver this as one isolated, reviewable step in the Atom-native immutable persistence and lowering artifacts workstream.

**Biological motivation.** The physical archive must persist an atomic reusable execution hierarchy, mutable planes, partials, grammar, bases, and progressively specialized Cellerator entry points—not predefined row shards.

**Compiler-architectural reason.** Use an adjacent immutable manifest-and-arena format rather than enlarging CSH5 or CSPACK. Large sequential arenas contain independently indexed atom frames; logical atom and storage extent boundaries remain separate.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/artifact/atom_store/
- [proposed] src/artifact/atom_store/
- [proposed] docs/SPEC_ATOM_STORE_V1.md
- [proposed] tests/jbc/atom_store/

**Existing code and permitted read scope:**

- include/CellShard/artifact/image.hh
- include/CellShard/artifact/extent.hh
- include/CellShard/artifact/catalog.hh
- include/CellShard/artifact/snapshot.hh
- include/CellShard/io/pack/image_envelope.hh
- docs/SPEC_CSPACK_V1.md

**Explicitly out of scope / forbidden shortcuts:**

- Do not create one file per atom by default.
- Do not use CSH5 or CSPACK as the new universal compiler database.
- Do not put source paths or placement epochs into immutable content identity.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-ST23

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Trace retained roots, basis/schedule/action/lineage references and active pins; delete only unreachable replicas/arenas after policy and durability checks.

Workstream mechanism: Freeze a new format family after collision audit. Separate semantic, content, materialization, replica, and action identity. Store root generation, atom dictionary, coverage/plane indexes, grammar, bases, superatoms, partials, lowering stages, physical views, lineage, actions, and large arena/frame payloads. Publish atomically; support recovery, consolidation, GC, CSH5/CSPACK import, codecs, and metadata-only inspection.

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

- Use corruption/torn-write/orphan/partial-publication tests and verify decoded as well as encoded content identities.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Measure sequential I/O, selected-range I/O, frame count, read amplification, metadata memory, compression, assembly, and block-storage throughput.
- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-PERSISTENCE; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-ST25 — Implement CSH5, CSPACK, CPEXEC01/02 compatibility import

**Repository / subsystem / lane:** CellShard · `artifact/atom_store` · `CS-JBC-L-PERSISTENCE`  
**Classification:** compatibility/migration  
**Parallelism:** Serial within CS-JBC-L-PERSISTENCE; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Implement CSH5, CSPACK, CPEXEC01/02 compatibility import. Deliver this as one isolated, reviewable step in the Atom-native immutable persistence and lowering artifacts workstream.

**Biological motivation.** The physical archive must persist an atomic reusable execution hierarchy, mutable planes, partials, grammar, bases, and progressively specialized Cellerator entry points—not predefined row shards.

**Compiler-architectural reason.** Use an adjacent immutable manifest-and-arena format rather than enlarging CSH5 or CSPACK. Large sequential arenas contain independently indexed atom frames; logical atom and storage extent boundaries remain separate.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/artifact/atom_store/
- [proposed] src/artifact/atom_store/
- [proposed] docs/SPEC_ATOM_STORE_V1.md
- [proposed] tests/jbc/atom_store/

**Existing code and permitted read scope:**

- include/CellShard/artifact/image.hh
- include/CellShard/artifact/extent.hh
- include/CellShard/artifact/catalog.hh
- include/CellShard/artifact/snapshot.hh
- include/CellShard/io/pack/image_envelope.hh
- docs/SPEC_CSPACK_V1.md

**Explicitly out of scope / forbidden shortcuts:**

- Do not create one file per atom by default.
- Do not use CSH5 or CSPACK as the new universal compiler database.
- Do not put source paths or placement epochs into immutable content identity.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-ST24

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Read current canonical/archive/pack/image formats, map identities and payloads into atom-store canonical or physical records, preserve opaque CSG1/CPE2 bytes, and never reinterpret producer layouts.

Workstream mechanism: Freeze a new format family after collision audit. Separate semantic, content, materialization, replica, and action identity. Store root generation, atom dictionary, coverage/plane indexes, grammar, bases, superatoms, partials, lowering stages, physical views, lineage, actions, and large arena/frame payloads. Publish atomically; support recovery, consolidation, GC, CSH5/CSPACK import, codecs, and metadata-only inspection.

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

- Use corruption/torn-write/orphan/partial-publication tests and verify decoded as well as encoded content identities.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Measure sequential I/O, selected-range I/O, frame count, read amplification, metadata memory, compression, assembly, and block-storage throughput.
- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-PERSISTENCE; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-ST26 — Implement codec-provider registry and raw/index baselines

**Repository / subsystem / lane:** CellShard · `artifact/atom_store` · `CS-JBC-L-PERSISTENCE`  
**Classification:** generic infrastructure supporting the biology-native compiler  
**Parallelism:** Serial within CS-JBC-L-PERSISTENCE; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Implement codec-provider registry and raw/index baselines. Deliver this as one isolated, reviewable step in the Atom-native immutable persistence and lowering artifacts workstream.

**Biological motivation.** The physical archive must persist an atomic reusable execution hierarchy, mutable planes, partials, grammar, bases, and progressively specialized Cellerator entry points—not predefined row shards.

**Compiler-architectural reason.** Use an adjacent immutable manifest-and-arena format rather than enlarging CSH5 or CSPACK. Large sequential arenas contain independently indexed atom frames; logical atom and storage extent boundaries remain separate.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/artifact/atom_store/
- [proposed] src/artifact/atom_store/
- [proposed] docs/SPEC_ATOM_STORE_V1.md
- [proposed] tests/jbc/atom_store/

**Existing code and permitted read scope:**

- include/CellShard/artifact/image.hh
- include/CellShard/artifact/extent.hh
- include/CellShard/artifact/catalog.hh
- include/CellShard/artifact/snapshot.hh
- include/CellShard/io/pack/image_envelope.hh
- docs/SPEC_CSPACK_V1.md

**Explicitly out of scope / forbidden shortcuts:**

- Do not create one file per atom by default.
- Do not use CSH5 or CSPACK as the new universal compiler database.
- Do not put source paths or placement epochs into immutable content identity.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-ST25

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Provide source-linked codec descriptors, raw payload baseline, delta/bit-pack/RLE index pipelines, requirements, decode validation, and explicit lossy-policy rejection.

Workstream mechanism: Freeze a new format family after collision audit. Separate semantic, content, materialization, replica, and action identity. Store root generation, atom dictionary, coverage/plane indexes, grammar, bases, superatoms, partials, lowering stages, physical views, lineage, actions, and large arena/frame payloads. Publish atomically; support recovery, consolidation, GC, CSH5/CSPACK import, codecs, and metadata-only inspection.

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

- Use corruption/torn-write/orphan/partial-publication tests and verify decoded as well as encoded content identities.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Measure sequential I/O, selected-range I/O, frame count, read amplification, metadata memory, compression, assembly, and block-storage throughput.
- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-PERSISTENCE; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-ST27 — Implement CPU block-compression candidates

**Repository / subsystem / lane:** CellShard · `artifact/atom_store` · `CS-JBC-L-PERSISTENCE`  
**Classification:** generic infrastructure supporting the biology-native compiler  
**Parallelism:** Serial within CS-JBC-L-PERSISTENCE; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Implement CPU block-compression candidates. Deliver this as one isolated, reviewable step in the Atom-native immutable persistence and lowering artifacts workstream.

**Biological motivation.** The physical archive must persist an atomic reusable execution hierarchy, mutable planes, partials, grammar, bases, and progressively specialized Cellerator entry points—not predefined row shards.

**Compiler-architectural reason.** Use an adjacent immutable manifest-and-arena format rather than enlarging CSH5 or CSPACK. Large sequential arenas contain independently indexed atom frames; logical atom and storage extent boundaries remain separate.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/artifact/atom_store/
- [proposed] src/artifact/atom_store/
- [proposed] docs/SPEC_ATOM_STORE_V1.md
- [proposed] tests/jbc/atom_store/

**Existing code and permitted read scope:**

- include/CellShard/artifact/image.hh
- include/CellShard/artifact/extent.hh
- include/CellShard/artifact/catalog.hh
- include/CellShard/artifact/snapshot.hh
- include/CellShard/io/pack/image_envelope.hh
- docs/SPEC_CSPACK_V1.md

**Explicitly out of scope / forbidden shortcuts:**

- Do not create one file per atom by default.
- Do not use CSH5 or CSPACK as the new universal compiler database.
- Do not put source paths or placement epochs into immutable content identity.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-ST26

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Add one or more optional general-purpose block codecs behind the registry, preserve decoded digest and random frame access, and measure block-storage bytes/decode cost.

Workstream mechanism: Freeze a new format family after collision audit. Separate semantic, content, materialization, replica, and action identity. Store root generation, atom dictionary, coverage/plane indexes, grammar, bases, superatoms, partials, lowering stages, physical views, lineage, actions, and large arena/frame payloads. Publish atomically; support recovery, consolidation, GC, CSH5/CSPACK import, codecs, and metadata-only inspection.

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

- Use corruption/torn-write/orphan/partial-publication tests and verify decoded as well as encoded content identities.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Measure sequential I/O, selected-range I/O, frame count, read amplification, metadata memory, compression, assembly, and block-storage throughput.
- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-PERSISTENCE; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-ST28 — Implement GPU or atom-aware compression experiment

**Repository / subsystem / lane:** CellShard · `artifact/atom_store` · `CS-JBC-L-PERSISTENCE`  
**Classification:** generic infrastructure supporting the biology-native compiler  
**Parallelism:** Serial within CS-JBC-L-PERSISTENCE; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Implement GPU or atom-aware compression experiment. Deliver this as one isolated, reviewable step in the Atom-native immutable persistence and lowering artifacts workstream.

**Biological motivation.** The physical archive must persist an atomic reusable execution hierarchy, mutable planes, partials, grammar, bases, and progressively specialized Cellerator entry points—not predefined row shards.

**Compiler-architectural reason.** Use an adjacent immutable manifest-and-arena format rather than enlarging CSH5 or CSPACK. Large sequential arenas contain independently indexed atom frames; logical atom and storage extent boundaries remain separate.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/artifact/atom_store/
- [proposed] src/artifact/atom_store/
- [proposed] docs/SPEC_ATOM_STORE_V1.md
- [proposed] tests/jbc/atom_store/

**Existing code and permitted read scope:**

- include/CellShard/artifact/image.hh
- include/CellShard/artifact/extent.hh
- include/CellShard/artifact/catalog.hh
- include/CellShard/artifact/snapshot.hh
- include/CellShard/io/pack/image_envelope.hh
- docs/SPEC_CSPACK_V1.md

**Explicitly out of scope / forbidden shortcuts:**

- Do not create one file per atom by default.
- Do not use CSH5 or CSPACK as the new universal compiler database.
- Do not put source paths or placement epochs into immutable content identity.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-ST27

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Explore GPU decode or projection-aware compression for repeated masks, monotonic IDs, narrow indexes, and structured holes; retain raw/CPU fallback and require complete I/O-to-kernel evidence.

Workstream mechanism: Freeze a new format family after collision audit. Separate semantic, content, materialization, replica, and action identity. Store root generation, atom dictionary, coverage/plane indexes, grammar, bases, superatoms, partials, lowering stages, physical views, lineage, actions, and large arena/frame payloads. Publish atomically; support recovery, consolidation, GC, CSH5/CSPACK import, codecs, and metadata-only inspection.

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

- Use corruption/torn-write/orphan/partial-publication tests and verify decoded as well as encoded content identities.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Measure sequential I/O, selected-range I/O, frame count, read amplification, metadata memory, compression, assembly, and block-storage throughput.
- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-PERSISTENCE; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-ST29 — Implement experimental GPU-assisted atom linking

**Repository / subsystem / lane:** CellShard · `artifact/atom_store` · `CS-JBC-L-PERSISTENCE`  
**Classification:** experimental candidate; baseline and negative result required  
**Parallelism:** Serial within CS-JBC-L-PERSISTENCE; parallel with other provider/workstream lanes after JBC-G1-ATOM-THIN-WAIST.

### Why

**Purpose.** Implement experimental GPU-assisted atom linking. Deliver this as one isolated, reviewable step in the Atom-native immutable persistence and lowering artifacts workstream.

**Biological motivation.** The physical archive must persist an atomic reusable execution hierarchy, mutable planes, partials, grammar, bases, and progressively specialized Cellerator entry points—not predefined row shards.

**Compiler-architectural reason.** Use an adjacent immutable manifest-and-arena format rather than enlarging CSH5 or CSPACK. Large sequential arenas contain independently indexed atom frames; logical atom and storage extent boundaries remain separate.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/artifact/atom_store/
- [proposed] src/artifact/atom_store/
- [proposed] docs/SPEC_ATOM_STORE_V1.md
- [proposed] tests/jbc/atom_store/

**Existing code and permitted read scope:**

- include/CellShard/artifact/image.hh
- include/CellShard/artifact/extent.hh
- include/CellShard/artifact/catalog.hh
- include/CellShard/artifact/snapshot.hh
- include/CellShard/io/pack/image_envelope.hh
- docs/SPEC_CSPACK_V1.md

**Explicitly out of scope / forbidden shortcuts:**

- Do not create one file per atom by default.
- Do not use CSH5 or CSPACK as the new universal compiler database.
- Do not put source paths or placement epochs into immutable content identity.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-ST28

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Use Cellerator/CellShard gather, scatter, transpose, pack, checksum, or decode kernels to assemble new atoms when the result remains GPU-resident; compare with CPU linker and host staging.

Workstream mechanism: Freeze a new format family after collision audit. Separate semantic, content, materialization, replica, and action identity. Store root generation, atom dictionary, coverage/plane indexes, grammar, bases, superatoms, partials, lowering stages, physical views, lineage, actions, and large arena/frame payloads. Publish atomically; support recovery, consolidation, GC, CSH5/CSPACK import, codecs, and metadata-only inspection.

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

- Use corruption/torn-write/orphan/partial-publication tests and verify decoded as well as encoded content identities.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Measure sequential I/O, selected-range I/O, frame count, read amplification, metadata memory, compression, assembly, and block-storage throughput.
- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-PERSISTENCE; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


# Workstream CSGLOBAL: CellShard global operation graph and portable schedule compiler

**Repository:** CellShard  
**Suggested lane:** `CS-JBC-L-GLOBAL-IR`  
**Barrier:** `JBC-G3-CELLERATOR-FRAGMENT`

## Workstream design

Define provider-neutral operation nodes with typed atom ports and effects, graph-family and parameterized recipes, candidate physical graphs, portable schedules, replay strictness, legal rewrite descriptors, partial-result trees, exact certificate references, and serialization/profiling. Include a mock non-Cellerator provider.

## Existing live source extended

- CellShard atoms/bases/grammar
- Cellerator fragment export
- Cellerator partial algebra
- CellShard exact certificate

## Likely source scope

- [proposed] include/CellShard/compiler/graph/
- [proposed] include/CellShard/compiler/schedule/
- [proposed] src/compiler/graph/
- [proposed] src/compiler/schedule/
- [proposed] tests/jbc/global_ir/

## Proposed Todos (14)

## CS-JBC-Q01 — Define the source-linked global operation-provider descriptor

**Repository / subsystem / lane:** CellShard · `compiler/graph and compiler/schedule` · `CS-JBC-L-GLOBAL-IR`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CS-JBC-L-GLOBAL-IR; parallel with other provider/workstream lanes after JBC-G3-CELLERATOR-FRAGMENT.

### Why

**Purpose.** Define the source-linked global operation-provider descriptor. Deliver this as one isolated, reviewable step in the CellShard global operation graph and portable schedule compiler workstream.

**Biological motivation.** The global graph expresses biological atom flow, mutations, partials, and reusable graph families before topology or CUDA command selection.

**Compiler-architectural reason.** CellShard owns decomposition, materialization, placement intent, and schedule recipes while operation providers own local mathematics and effects.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/compiler/graph/
- [proposed] include/CellShard/compiler/schedule/
- [proposed] src/compiler/graph/
- [proposed] src/compiler/schedule/
- [proposed] tests/jbc/global_ir/

**Existing code and permitted read scope:**

- CellShard atoms/bases/grammar
- Cellerator fragment export
- Cellerator partial algebra
- CellShard exact certificate

**Explicitly out of scope / forbidden shortcuts:**

- Do not encode CUDA pointers, streams, NCCL communicators, file descriptors, or GPU ordinals in portable schedules.
- Do not allow graph rewriting without provider-declared effect/decomposition legality.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-BS04
- receipt:CE-JBC-F14

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.

### Concrete mechanism

Register provider identity/version, operation kinds, typed atom requirements/affordances, effects, decomposition, partial algebra, resource/cost query, preparation, command lowering, retry semantics, and profiler manifest.

Workstream mechanism: Define provider-neutral operation nodes with typed atom ports and effects, graph-family and parameterized recipes, candidate physical graphs, portable schedules, replay strictness, legal rewrite descriptors, partial-result trees, exact certificate references, and serialization/profiling. Include a mock non-Cellerator provider.

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

- Validate graph cycles, effects, atom generations, provider compatibility, replay/retarget modes, and exact certificate linkage.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-GLOBAL-IR; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-Q02 — Define global operation nodes and typed ports

**Repository / subsystem / lane:** CellShard · `compiler/graph and compiler/schedule` · `CS-JBC-L-GLOBAL-IR`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CS-JBC-L-GLOBAL-IR; parallel with other provider/workstream lanes after JBC-G3-CELLERATOR-FRAGMENT.

### Why

**Purpose.** Define global operation nodes and typed ports. Deliver this as one isolated, reviewable step in the CellShard global operation graph and portable schedule compiler workstream.

**Biological motivation.** The global graph expresses biological atom flow, mutations, partials, and reusable graph families before topology or CUDA command selection.

**Compiler-architectural reason.** CellShard owns decomposition, materialization, placement intent, and schedule recipes while operation providers own local mathematics and effects.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/compiler/graph/
- [proposed] include/CellShard/compiler/schedule/
- [proposed] src/compiler/graph/
- [proposed] src/compiler/schedule/
- [proposed] tests/jbc/global_ir/

**Existing code and permitted read scope:**

- CellShard atoms/bases/grammar
- Cellerator fragment export
- Cellerator partial algebra
- CellShard exact certificate

**Explicitly out of scope / forbidden shortcuts:**

- Do not encode CUDA pointers, streams, NCCL communicators, file descriptors, or GPU ordinals in portable schedules.
- Do not allow graph rewriting without provider-declared effect/decomposition legality.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-Q01

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.

### Concrete mechanism

Represent operation/provider identity, atom input/output ports, exact coverage, generation, alternative local incarnations, reuse/frequency, and control dependencies.

Workstream mechanism: Define provider-neutral operation nodes with typed atom ports and effects, graph-family and parameterized recipes, candidate physical graphs, portable schedules, replay strictness, legal rewrite descriptors, partial-result trees, exact certificate references, and serialization/profiling. Include a mock non-Cellerator provider.

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

- Validate graph cycles, effects, atom generations, provider compatibility, replay/retarget modes, and exact certificate linkage.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-GLOBAL-IR; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-Q03 — Define explicit access and effect semantics

**Repository / subsystem / lane:** CellShard · `compiler/graph and compiler/schedule` · `CS-JBC-L-GLOBAL-IR`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CS-JBC-L-GLOBAL-IR; parallel with other provider/workstream lanes after JBC-G3-CELLERATOR-FRAGMENT.

### Why

**Purpose.** Define explicit access and effect semantics. Deliver this as one isolated, reviewable step in the CellShard global operation graph and portable schedule compiler workstream.

**Biological motivation.** The global graph expresses biological atom flow, mutations, partials, and reusable graph families before topology or CUDA command selection.

**Compiler-architectural reason.** CellShard owns decomposition, materialization, placement intent, and schedule recipes while operation providers own local mathematics and effects.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/compiler/graph/
- [proposed] include/CellShard/compiler/schedule/
- [proposed] src/compiler/graph/
- [proposed] src/compiler/schedule/
- [proposed] tests/jbc/global_ir/

**Existing code and permitted read scope:**

- CellShard atoms/bases/grammar
- Cellerator fragment export
- Cellerator partial algebra
- CellShard exact certificate

**Explicitly out of scope / forbidden shortcuts:**

- Do not encode CUDA pointers, streams, NCCL communicators, file descriptors, or GPU ordinals in portable schedules.
- Do not allow graph rewriting without provider-declared effect/decomposition legality.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-Q02

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.

### Concrete mechanism

Support read, exclusive write, discard write, accumulate, reduce-by-algebra, mutate generation, publish generation, durable append, and commit; graph rewrites must preserve these effects.

Workstream mechanism: Define provider-neutral operation nodes with typed atom ports and effects, graph-family and parameterized recipes, candidate physical graphs, portable schedules, replay strictness, legal rewrite descriptors, partial-result trees, exact certificate references, and serialization/profiling. Include a mock non-Cellerator provider.

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

- Validate graph cycles, effects, atom generations, provider compatibility, replay/retarget modes, and exact certificate linkage.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-GLOBAL-IR; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-Q04 — Define atom dependency edges

**Repository / subsystem / lane:** CellShard · `compiler/graph and compiler/schedule` · `CS-JBC-L-GLOBAL-IR`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CS-JBC-L-GLOBAL-IR; parallel with other provider/workstream lanes after JBC-G3-CELLERATOR-FRAGMENT.

### Why

**Purpose.** Define atom dependency edges. Deliver this as one isolated, reviewable step in the CellShard global operation graph and portable schedule compiler workstream.

**Biological motivation.** The global graph expresses biological atom flow, mutations, partials, and reusable graph families before topology or CUDA command selection.

**Compiler-architectural reason.** CellShard owns decomposition, materialization, placement intent, and schedule recipes while operation providers own local mathematics and effects.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/compiler/graph/
- [proposed] include/CellShard/compiler/schedule/
- [proposed] src/compiler/graph/
- [proposed] src/compiler/schedule/
- [proposed] tests/jbc/global_ir/

**Existing code and permitted read scope:**

- CellShard atoms/bases/grammar
- Cellerator fragment export
- Cellerator partial algebra
- CellShard exact certificate

**Explicitly out of scope / forbidden shortcuts:**

- Do not encode CUDA pointers, streams, NCCL communicators, file descriptors, or GPU ordinals in portable schedules.
- Do not allow graph rewriting without provider-declared effect/decomposition legality.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-Q03

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.

### Concrete mechanism

Connect producer/consumer ports with exact atom/plane/generation/order requirements, optional transform/composition production, and ownership/partial roles.

Workstream mechanism: Define provider-neutral operation nodes with typed atom ports and effects, graph-family and parameterized recipes, candidate physical graphs, portable schedules, replay strictness, legal rewrite descriptors, partial-result trees, exact certificate references, and serialization/profiling. Include a mock non-Cellerator provider.

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

- Validate graph cycles, effects, atom generations, provider compatibility, replay/retarget modes, and exact certificate linkage.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-GLOBAL-IR; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-Q05 — Define graph-family identity and workload distribution

**Repository / subsystem / lane:** CellShard · `compiler/graph and compiler/schedule` · `CS-JBC-L-GLOBAL-IR`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CS-JBC-L-GLOBAL-IR; parallel with other provider/workstream lanes after JBC-G3-CELLERATOR-FRAGMENT.

### Why

**Purpose.** Define graph-family identity and workload distribution. Deliver this as one isolated, reviewable step in the CellShard global operation graph and portable schedule compiler workstream.

**Biological motivation.** The global graph expresses biological atom flow, mutations, partials, and reusable graph families before topology or CUDA command selection.

**Compiler-architectural reason.** CellShard owns decomposition, materialization, placement intent, and schedule recipes while operation providers own local mathematics and effects.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/compiler/graph/
- [proposed] include/CellShard/compiler/schedule/
- [proposed] src/compiler/graph/
- [proposed] src/compiler/schedule/
- [proposed] tests/jbc/global_ir/

**Existing code and permitted read scope:**

- CellShard atoms/bases/grammar
- Cellerator fragment export
- Cellerator partial algebra
- CellShard exact certificate

**Explicitly out of scope / forbidden shortcuts:**

- Do not encode CUDA pointers, streams, NCCL communicators, file descriptors, or GPU ordinals in portable schedules.
- Do not allow graph rewriting without provider-declared effect/decomposition legality.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-Q04

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.

### Concrete mechanism

Group related operation graphs under stable family identity, parameter domains, frequency/reuse estimates, uncertainty, and mutation profile; one graph remains distinguishable.

Workstream mechanism: Define provider-neutral operation nodes with typed atom ports and effects, graph-family and parameterized recipes, candidate physical graphs, portable schedules, replay strictness, legal rewrite descriptors, partial-result trees, exact certificate references, and serialization/profiling. Include a mock non-Cellerator provider.

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

- Validate graph cycles, effects, atom generations, provider compatibility, replay/retarget modes, and exact certificate linkage.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-GLOBAL-IR; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-Q06 — Implement compact parameterized graph recipes

**Repository / subsystem / lane:** CellShard · `compiler/graph and compiler/schedule` · `CS-JBC-L-GLOBAL-IR`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CS-JBC-L-GLOBAL-IR; parallel with other provider/workstream lanes after JBC-G3-CELLERATOR-FRAGMENT.

### Why

**Purpose.** Implement compact parameterized graph recipes. Deliver this as one isolated, reviewable step in the CellShard global operation graph and portable schedule compiler workstream.

**Biological motivation.** The global graph expresses biological atom flow, mutations, partials, and reusable graph families before topology or CUDA command selection.

**Compiler-architectural reason.** CellShard owns decomposition, materialization, placement intent, and schedule recipes while operation providers own local mathematics and effects.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/compiler/graph/
- [proposed] include/CellShard/compiler/schedule/
- [proposed] src/compiler/graph/
- [proposed] src/compiler/schedule/
- [proposed] tests/jbc/global_ir/

**Existing code and permitted read scope:**

- CellShard atoms/bases/grammar
- Cellerator fragment export
- Cellerator partial algebra
- CellShard exact certificate

**Explicitly out of scope / forbidden shortcuts:**

- Do not encode CUDA pointers, streams, NCCL communicators, file descriptors, or GPU ordinals in portable schedules.
- Do not allow graph rewriting without provider-declared effect/decomposition legality.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-Q05

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Represent repeated tasks over atom/partition/window/time/assay/panel/iteration/gradient parameters without eagerly materializing every command.

Workstream mechanism: Define provider-neutral operation nodes with typed atom ports and effects, graph-family and parameterized recipes, candidate physical graphs, portable schedules, replay strictness, legal rewrite descriptors, partial-result trees, exact certificate references, and serialization/profiling. Include a mock non-Cellerator provider.

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

- Validate graph cycles, effects, atom generations, provider compatibility, replay/retarget modes, and exact certificate linkage.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-GLOBAL-IR; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-Q07 — Define candidate physical-graph realizations

**Repository / subsystem / lane:** CellShard · `compiler/graph and compiler/schedule` · `CS-JBC-L-GLOBAL-IR`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CS-JBC-L-GLOBAL-IR; parallel with other provider/workstream lanes after JBC-G3-CELLERATOR-FRAGMENT.

### Why

**Purpose.** Define candidate physical-graph realizations. Deliver this as one isolated, reviewable step in the CellShard global operation graph and portable schedule compiler workstream.

**Biological motivation.** The global graph expresses biological atom flow, mutations, partials, and reusable graph families before topology or CUDA command selection.

**Compiler-architectural reason.** CellShard owns decomposition, materialization, placement intent, and schedule recipes while operation providers own local mathematics and effects.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/compiler/graph/
- [proposed] include/CellShard/compiler/schedule/
- [proposed] src/compiler/graph/
- [proposed] src/compiler/schedule/
- [proposed] tests/jbc/global_ir/

**Existing code and permitted read scope:**

- CellShard atoms/bases/grammar
- Cellerator fragment export
- Cellerator partial algebra
- CellShard exact certificate

**Explicitly out of scope / forbidden shortcuts:**

- Do not encode CUDA pointers, streams, NCCL communicators, file descriptors, or GPU ordinals in portable schedules.
- Do not allow graph rewriting without provider-declared effect/decomposition legality.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-Q06

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.

### Concrete mechanism

Attach chosen atoms, compositions, local Cellerator fragment candidates, transforms, partial trees, and materializations before topology binding.

Workstream mechanism: Define provider-neutral operation nodes with typed atom ports and effects, graph-family and parameterized recipes, candidate physical graphs, portable schedules, replay strictness, legal rewrite descriptors, partial-result trees, exact certificate references, and serialization/profiling. Include a mock non-Cellerator provider.

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

- Validate graph cycles, effects, atom generations, provider compatibility, replay/retarget modes, and exact certificate linkage.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-GLOBAL-IR; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-Q08 — Define the portable schedule artifact

**Repository / subsystem / lane:** CellShard · `compiler/graph and compiler/schedule` · `CS-JBC-L-GLOBAL-IR`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CS-JBC-L-GLOBAL-IR; parallel with other provider/workstream lanes after JBC-G3-CELLERATOR-FRAGMENT.

### Why

**Purpose.** Define the portable schedule artifact. Deliver this as one isolated, reviewable step in the CellShard global operation graph and portable schedule compiler workstream.

**Biological motivation.** The global graph expresses biological atom flow, mutations, partials, and reusable graph families before topology or CUDA command selection.

**Compiler-architectural reason.** CellShard owns decomposition, materialization, placement intent, and schedule recipes while operation providers own local mathematics and effects.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/compiler/graph/
- [proposed] include/CellShard/compiler/schedule/
- [proposed] src/compiler/graph/
- [proposed] src/compiler/schedule/
- [proposed] tests/jbc/global_ir/

**Existing code and permitted read scope:**

- CellShard atoms/bases/grammar
- Cellerator fragment export
- Cellerator partial algebra
- CellShard exact certificate

**Explicitly out of scope / forbidden shortcuts:**

- Do not encode CUDA pointers, streams, NCCL communicators, file descriptors, or GPU ordinals in portable schedules.
- Do not allow graph rewriting without provider-declared effect/decomposition legality.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-Q07

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.

### Concrete mechanism

Persist logical decomposition, atom/basis choices, contribution ownership, replication, partial-result trees, local candidate requirements, order continuity, and compatibility without machine IDs.

Workstream mechanism: Define provider-neutral operation nodes with typed atom ports and effects, graph-family and parameterized recipes, candidate physical graphs, portable schedules, replay strictness, legal rewrite descriptors, partial-result trees, exact certificate references, and serialization/profiling. Include a mock non-Cellerator provider.

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

- Validate graph cycles, effects, atom generations, provider compatibility, replay/retarget modes, and exact certificate linkage.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-GLOBAL-IR; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-Q09 — Implement exact, relink, retarget, and recompile replay modes

**Repository / subsystem / lane:** CellShard · `compiler/graph and compiler/schedule` · `CS-JBC-L-GLOBAL-IR`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CS-JBC-L-GLOBAL-IR; parallel with other provider/workstream lanes after JBC-G3-CELLERATOR-FRAGMENT.

### Why

**Purpose.** Implement exact, relink, retarget, and recompile replay modes. Deliver this as one isolated, reviewable step in the CellShard global operation graph and portable schedule compiler workstream.

**Biological motivation.** The global graph expresses biological atom flow, mutations, partials, and reusable graph families before topology or CUDA command selection.

**Compiler-architectural reason.** CellShard owns decomposition, materialization, placement intent, and schedule recipes while operation providers own local mathematics and effects.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/compiler/graph/
- [proposed] include/CellShard/compiler/schedule/
- [proposed] src/compiler/graph/
- [proposed] src/compiler/schedule/
- [proposed] tests/jbc/global_ir/

**Existing code and permitted read scope:**

- CellShard atoms/bases/grammar
- Cellerator fragment export
- Cellerator partial algebra
- CellShard exact certificate

**Explicitly out of scope / forbidden shortcuts:**

- Do not encode CUDA pointers, streams, NCCL communicators, file descriptors, or GPU ordinals in portable schedules.
- Do not allow graph rewriting without provider-declared effect/decomposition legality.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-Q08

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Exact requires matching objects/topology/candidates; relink preserves decomposition/objects; retarget permits new local projections; recompile reuses any compatible evidence/atoms.

Workstream mechanism: Define provider-neutral operation nodes with typed atom ports and effects, graph-family and parameterized recipes, candidate physical graphs, portable schedules, replay strictness, legal rewrite descriptors, partial-result trees, exact certificate references, and serialization/profiling. Include a mock non-Cellerator provider.

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

- Validate graph cycles, effects, atom generations, provider compatibility, replay/retarget modes, and exact certificate linkage.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-GLOBAL-IR; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-Q10 — Define legal graph-rewrite descriptors

**Repository / subsystem / lane:** CellShard · `compiler/graph and compiler/schedule` · `CS-JBC-L-GLOBAL-IR`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CS-JBC-L-GLOBAL-IR; parallel with other provider/workstream lanes after JBC-G3-CELLERATOR-FRAGMENT.

### Why

**Purpose.** Define legal graph-rewrite descriptors. Deliver this as one isolated, reviewable step in the CellShard global operation graph and portable schedule compiler workstream.

**Biological motivation.** The global graph expresses biological atom flow, mutations, partials, and reusable graph families before topology or CUDA command selection.

**Compiler-architectural reason.** CellShard owns decomposition, materialization, placement intent, and schedule recipes while operation providers own local mathematics and effects.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/compiler/graph/
- [proposed] include/CellShard/compiler/schedule/
- [proposed] src/compiler/graph/
- [proposed] src/compiler/schedule/
- [proposed] tests/jbc/global_ir/

**Existing code and permitted read scope:**

- CellShard atoms/bases/grammar
- Cellerator fragment export
- Cellerator partial algebra
- CellShard exact certificate

**Explicitly out of scope / forbidden shortcuts:**

- Do not encode CUDA pointers, streams, NCCL communicators, file descriptors, or GPU ordinals in portable schedules.
- Do not allow graph rewriting without provider-declared effect/decomposition legality.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-Q09

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.

### Concrete mechanism

Represent reorder, fuse, split, merge, replicate data, replicate compute, move compute to data, materialize/elide intermediates, change reduction tree, and preserve order only when provider effects/decomposition allow.

Workstream mechanism: Define provider-neutral operation nodes with typed atom ports and effects, graph-family and parameterized recipes, candidate physical graphs, portable schedules, replay strictness, legal rewrite descriptors, partial-result trees, exact certificate references, and serialization/profiling. Include a mock non-Cellerator provider.

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

- Validate graph cycles, effects, atom generations, provider compatibility, replay/retarget modes, and exact certificate linkage.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-GLOBAL-IR; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-Q11 — Compile global partial-result trees

**Repository / subsystem / lane:** CellShard · `compiler/graph and compiler/schedule` · `CS-JBC-L-GLOBAL-IR`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CS-JBC-L-GLOBAL-IR; parallel with other provider/workstream lanes after JBC-G3-CELLERATOR-FRAGMENT.

### Why

**Purpose.** Compile global partial-result trees. Deliver this as one isolated, reviewable step in the CellShard global operation graph and portable schedule compiler workstream.

**Biological motivation.** The global graph expresses biological atom flow, mutations, partials, and reusable graph families before topology or CUDA command selection.

**Compiler-architectural reason.** CellShard owns decomposition, materialization, placement intent, and schedule recipes while operation providers own local mathematics and effects.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/compiler/graph/
- [proposed] include/CellShard/compiler/schedule/
- [proposed] src/compiler/graph/
- [proposed] src/compiler/schedule/
- [proposed] tests/jbc/global_ir/

**Existing code and permitted read scope:**

- CellShard atoms/bases/grammar
- Cellerator fragment export
- Cellerator partial algebra
- CellShard exact certificate

**Explicitly out of scope / forbidden shortcuts:**

- Do not encode CUDA pointers, streams, NCCL communicators, file descriptors, or GPU ordinals in portable schedules.
- Do not allow graph rewriting without provider-declared effect/decomposition legality.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-Q10

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Choose contributor grouping, merge/finalize operations, owner, ordering/determinism, and topology-neutral tree shape from provider partial algebra.

Workstream mechanism: Define provider-neutral operation nodes with typed atom ports and effects, graph-family and parameterized recipes, candidate physical graphs, portable schedules, replay strictness, legal rewrite descriptors, partial-result trees, exact certificate references, and serialization/profiling. Include a mock non-Cellerator provider.

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

- Validate graph cycles, effects, atom generations, provider compatibility, replay/retarget modes, and exact certificate linkage.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-GLOBAL-IR; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-Q12 — Integrate the exact distributed certificate

**Repository / subsystem / lane:** CellShard · `compiler/graph and compiler/schedule` · `CS-JBC-L-GLOBAL-IR`  
**Classification:** validation/evidence  
**Parallelism:** Serial within CS-JBC-L-GLOBAL-IR; parallel with other provider/workstream lanes after JBC-G3-CELLERATOR-FRAGMENT.

### Why

**Purpose.** Integrate the exact distributed certificate. Deliver this as one isolated, reviewable step in the CellShard global operation graph and portable schedule compiler workstream.

**Biological motivation.** The global graph expresses biological atom flow, mutations, partials, and reusable graph families before topology or CUDA command selection.

**Compiler-architectural reason.** CellShard owns decomposition, materialization, placement intent, and schedule recipes while operation providers own local mathematics and effects.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/compiler/graph/
- [proposed] include/CellShard/compiler/schedule/
- [proposed] src/compiler/graph/
- [proposed] src/compiler/schedule/
- [proposed] tests/jbc/global_ir/

**Existing code and permitted read scope:**

- CellShard atoms/bases/grammar
- Cellerator fragment export
- Cellerator partial algebra
- CellShard exact certificate

**Explicitly out of scope / forbidden shortcuts:**

- Do not encode CUDA pointers, streams, NCCL communicators, file descriptors, or GPU ordinals in portable schedules.
- Do not allow graph rewriting without provider-declared effect/decomposition legality.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-Q11

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.
- Focused tests and adversarial fixtures demonstrating both success and explicit rejection paths.

### Concrete mechanism

Bind schedule nodes/edges/partials/owners/halos/generations to the independently verified certificate; no schedule may execute without a valid certificate.

Workstream mechanism: Define provider-neutral operation nodes with typed atom ports and effects, graph-family and parameterized recipes, candidate physical graphs, portable schedules, replay strictness, legal rewrite descriptors, partial-result trees, exact certificate references, and serialization/profiling. Include a mock non-Cellerator provider.

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

- Validate graph cycles, effects, atom generations, provider compatibility, replay/retarget modes, and exact certificate linkage.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-GLOBAL-IR; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-Q13 — Serialize global IR and emit profiler identities

**Repository / subsystem / lane:** CellShard · `compiler/graph and compiler/schedule` · `CS-JBC-L-GLOBAL-IR`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CS-JBC-L-GLOBAL-IR; parallel with other provider/workstream lanes after JBC-G3-CELLERATOR-FRAGMENT.

### Why

**Purpose.** Serialize global IR and emit profiler identities. Deliver this as one isolated, reviewable step in the CellShard global operation graph and portable schedule compiler workstream.

**Biological motivation.** The global graph expresses biological atom flow, mutations, partials, and reusable graph families before topology or CUDA command selection.

**Compiler-architectural reason.** CellShard owns decomposition, materialization, placement intent, and schedule recipes while operation providers own local mathematics and effects.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/compiler/graph/
- [proposed] include/CellShard/compiler/schedule/
- [proposed] src/compiler/graph/
- [proposed] src/compiler/schedule/
- [proposed] tests/jbc/global_ir/

**Existing code and permitted read scope:**

- CellShard atoms/bases/grammar
- Cellerator fragment export
- Cellerator partial algebra
- CellShard exact certificate

**Explicitly out of scope / forbidden shortcuts:**

- Do not encode CUDA pointers, streams, NCCL communicators, file descriptors, or GPU ordinals in portable schedules.
- Do not allow graph rewriting without provider-declared effect/decomposition legality.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-Q12

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Persist graph/schedule recipes and stable stage/atom/rewrite IDs, predicted complete costs, decisions, and compatibility while excluding live runtime handles.

Workstream mechanism: Define provider-neutral operation nodes with typed atom ports and effects, graph-family and parameterized recipes, candidate physical graphs, portable schedules, replay strictness, legal rewrite descriptors, partial-result trees, exact certificate references, and serialization/profiling. Include a mock non-Cellerator provider.

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

- Validate graph cycles, effects, atom generations, provider compatibility, replay/retarget modes, and exact certificate linkage.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-GLOBAL-IR; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-Q14 — Implement a mock non-Cellerator operation provider

**Repository / subsystem / lane:** CellShard · `compiler/graph and compiler/schedule` · `CS-JBC-L-GLOBAL-IR`  
**Classification:** settled required mechanism  
**Parallelism:** Serial within CS-JBC-L-GLOBAL-IR; parallel with other provider/workstream lanes after JBC-G3-CELLERATOR-FRAGMENT.

### Why

**Purpose.** Implement a mock non-Cellerator operation provider. Deliver this as one isolated, reviewable step in the CellShard global operation graph and portable schedule compiler workstream.

**Biological motivation.** The global graph expresses biological atom flow, mutations, partials, and reusable graph families before topology or CUDA command selection.

**Compiler-architectural reason.** CellShard owns decomposition, materialization, placement intent, and schedule recipes while operation providers own local mathematics and effects.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/compiler/graph/
- [proposed] include/CellShard/compiler/schedule/
- [proposed] src/compiler/graph/
- [proposed] src/compiler/schedule/
- [proposed] tests/jbc/global_ir/

**Existing code and permitted read scope:**

- CellShard atoms/bases/grammar
- Cellerator fragment export
- Cellerator partial algebra
- CellShard exact certificate

**Explicitly out of scope / forbidden shortcuts:**

- Do not encode CUDA pointers, streams, NCCL communicators, file descriptors, or GPU ordinals in portable schedules.
- Do not allow graph rewriting without provider-declared effect/decomposition legality.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-Q13

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Provide provider-defined coverage, one read/transform/write operation, optional halo, partial algebra, and command lowering to prove the graph is extensible without Baseplane or model semantics.

Workstream mechanism: Define provider-neutral operation nodes with typed atom ports and effects, graph-family and parameterized recipes, candidate physical graphs, portable schedules, replay strictness, legal rewrite descriptors, partial-result trees, exact certificate references, and serialization/profiling. Include a mock non-Cellerator provider.

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

- Validate graph cycles, effects, atom generations, provider compatibility, replay/retarget modes, and exact certificate linkage.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-GLOBAL-IR; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


# Workstream CSRUNTIME: CellShard topology, I/O, transport, residency, and runtime lowering

**Repository:** CellShard  
**Suggested lane:** `CS-JBC-L-RUNTIME`  
**Barrier:** `JBC-G5-PARTIAL-ARTIFACT`

## Workstream design

Build stable topology and measured calibration layers, async multi-range atom sources, read coalescing and pinned staging, source-linked transport providers, numaBraid/NCCL/P2P/host routes, plane-aware residency and leases, reconstruction-aware eviction, command IR, scheduler, CUDA Graph lowering, transactional recovery, and the dual-NUMA logical-node test.

## Existing live source extended

- include/CellShard/runtime/source/payload_source.hh
- include/CellShard/runtime/source/local_file_source.hh
- include/CellShard/runtime/residency/host.hh
- include/CellShard/runtime/residency/device.cuh
- CellShard global schedule IR
- Cellerator execution session
- numaBraid

## Likely source scope

- [proposed] include/CellShard/runtime/v2/
- [proposed] src/runtime/v2/
- [proposed] tests/jbc/runtime/
- [proposed] bench/jbc/runtime/

## Proposed Todos (22)

## CS-JBC-RT01 — Define the immutable topology profile

**Repository / subsystem / lane:** CellShard · `runtime/v2` · `CS-JBC-L-RUNTIME`  
**Classification:** generic infrastructure supporting the biology-native compiler  
**Parallelism:** Serial within CS-JBC-L-RUNTIME; parallel with other provider/workstream lanes after JBC-G5-PARTIAL-ARTIFACT.

### Why

**Purpose.** Define the immutable topology profile. Deliver this as one isolated, reviewable step in the CellShard topology, I/O, transport, residency, and runtime lowering workstream.

**Biological motivation.** Biologically compiled atoms must be placed, prefetched, combined, and retained according to their reuse, reconstruction cost, mutation half-life, and operation graph—not merely bytes.

**Compiler-architectural reason.** Runtime machinery is downstream infrastructure. It must execute the biology-generated schedule without becoming the source ontology.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/runtime/v2/
- [proposed] src/runtime/v2/
- [proposed] tests/jbc/runtime/
- [proposed] bench/jbc/runtime/

**Existing code and permitted read scope:**

- include/CellShard/runtime/source/payload_source.hh
- include/CellShard/runtime/source/local_file_source.hh
- include/CellShard/runtime/residency/host.hh
- include/CellShard/runtime/residency/device.cuh
- CellShard global schedule IR
- Cellerator execution session
- numaBraid

**Explicitly out of scope / forbidden shortcuts:**

- Do not hardcode the local two-node test policy as universal reachability.
- Do not expose numaBraid relay/BDF/chunk internals to portable scheduling.
- Do not perform topology discovery or allocation inside sealed replay.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-Q14
- CS-JBC-ST20

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.

### Concrete mechanism

Represent nodes, NUMA domains, CPUs, GPUs, storage endpoints, memory tiers, links, capacities, reachability, and provider capabilities separately from measured performance and temporary load.

Workstream mechanism: Build stable topology and measured calibration layers, async multi-range atom sources, read coalescing and pinned staging, source-linked transport providers, numaBraid/NCCL/P2P/host routes, plane-aware residency and leases, reconstruction-aware eviction, command IR, scheduler, CUDA Graph lowering, transactional recovery, and the dual-NUMA logical-node test.

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

- Use unavailable-route, capacity, cancellation, stale generation, concurrent stream, and logical two-node tests.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Measure source I/O, staging, route setup, transfer, overlap, residency hits, reconstruction, graph replay, host RAM, and complete graph latency.
- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-RUNTIME; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-RT02 — Define logical nodes and NUMA ownership

**Repository / subsystem / lane:** CellShard · `runtime/v2` · `CS-JBC-L-RUNTIME`  
**Classification:** generic infrastructure supporting the biology-native compiler  
**Parallelism:** Serial within CS-JBC-L-RUNTIME; parallel with other provider/workstream lanes after JBC-G5-PARTIAL-ARTIFACT.

### Why

**Purpose.** Define logical nodes and NUMA ownership. Deliver this as one isolated, reviewable step in the CellShard topology, I/O, transport, residency, and runtime lowering workstream.

**Biological motivation.** Biologically compiled atoms must be placed, prefetched, combined, and retained according to their reuse, reconstruction cost, mutation half-life, and operation graph—not merely bytes.

**Compiler-architectural reason.** Runtime machinery is downstream infrastructure. It must execute the biology-generated schedule without becoming the source ontology.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/runtime/v2/
- [proposed] src/runtime/v2/
- [proposed] tests/jbc/runtime/
- [proposed] bench/jbc/runtime/

**Existing code and permitted read scope:**

- include/CellShard/runtime/source/payload_source.hh
- include/CellShard/runtime/source/local_file_source.hh
- include/CellShard/runtime/residency/host.hh
- include/CellShard/runtime/residency/device.cuh
- CellShard global schedule IR
- Cellerator execution session
- numaBraid

**Explicitly out of scope / forbidden shortcuts:**

- Do not hardcode the local two-node test policy as universal reachability.
- Do not expose numaBraid relay/BDF/chunk internals to portable scheduling.
- Do not perform topology discovery or allocation inside sealed replay.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-RT01

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.

### Concrete mechanism

Map physical sockets/NUMA domains and GPUs into policy-defined logical nodes, retain physical identity, and permit the local machine to emulate two nodes without changing portable schedules.

Workstream mechanism: Build stable topology and measured calibration layers, async multi-range atom sources, read coalescing and pinned staging, source-linked transport providers, numaBraid/NCCL/P2P/host routes, plane-aware residency and leases, reconstruction-aware eviction, command IR, scheduler, CUDA Graph lowering, transactional recovery, and the dual-NUMA logical-node test.

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

- Use unavailable-route, capacity, cancellation, stale generation, concurrent stream, and logical two-node tests.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Measure source I/O, staging, route setup, transfer, overlap, residency hits, reconstruction, graph replay, host RAM, and complete graph latency.
- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-RUNTIME; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-RT03 — Define storage endpoints and access policy

**Repository / subsystem / lane:** CellShard · `runtime/v2` · `CS-JBC-L-RUNTIME`  
**Classification:** generic infrastructure supporting the biology-native compiler  
**Parallelism:** Serial within CS-JBC-L-RUNTIME; parallel with other provider/workstream lanes after JBC-G5-PARTIAL-ARTIFACT.

### Why

**Purpose.** Define storage endpoints and access policy. Deliver this as one isolated, reviewable step in the CellShard topology, I/O, transport, residency, and runtime lowering workstream.

**Biological motivation.** Biologically compiled atoms must be placed, prefetched, combined, and retained according to their reuse, reconstruction cost, mutation half-life, and operation graph—not merely bytes.

**Compiler-architectural reason.** Runtime machinery is downstream infrastructure. It must execute the biology-generated schedule without becoming the source ontology.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/runtime/v2/
- [proposed] src/runtime/v2/
- [proposed] tests/jbc/runtime/
- [proposed] bench/jbc/runtime/

**Existing code and permitted read scope:**

- include/CellShard/runtime/source/payload_source.hh
- include/CellShard/runtime/source/local_file_source.hh
- include/CellShard/runtime/residency/host.hh
- include/CellShard/runtime/residency/device.cuh
- CellShard global schedule IR
- Cellerator execution session
- numaBraid

**Explicitly out of scope / forbidden shortcuts:**

- Do not hardcode the local two-node test policy as universal reachability.
- Do not expose numaBraid relay/BDF/chunk internals to portable scheduling.
- Do not perform topology discovery or allocation inside sealed replay.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-RT02

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.

### Concrete mechanism

Represent filesystem/device identity, storage-local node/NUMA/root complex, capabilities, exact-read alignment, and policy reachability; encode /mnt/block node-0-only access only in the test profile.

Workstream mechanism: Build stable topology and measured calibration layers, async multi-range atom sources, read coalescing and pinned staging, source-linked transport providers, numaBraid/NCCL/P2P/host routes, plane-aware residency and leases, reconstruction-aware eviction, command IR, scheduler, CUDA Graph lowering, transactional recovery, and the dual-NUMA logical-node test.

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

- Use unavailable-route, capacity, cancellation, stale generation, concurrent stream, and logical two-node tests.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Measure source I/O, staging, route setup, transfer, overlap, residency hits, reconstruction, graph replay, host RAM, and complete graph latency.
- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-RUNTIME; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-RT04 — Implement topology and route cost calibration records

**Repository / subsystem / lane:** CellShard · `runtime/v2` · `CS-JBC-L-RUNTIME`  
**Classification:** generic infrastructure supporting the biology-native compiler  
**Parallelism:** Serial within CS-JBC-L-RUNTIME; parallel with other provider/workstream lanes after JBC-G5-PARTIAL-ARTIFACT.

### Why

**Purpose.** Implement topology and route cost calibration records. Deliver this as one isolated, reviewable step in the CellShard topology, I/O, transport, residency, and runtime lowering workstream.

**Biological motivation.** Biologically compiled atoms must be placed, prefetched, combined, and retained according to their reuse, reconstruction cost, mutation half-life, and operation graph—not merely bytes.

**Compiler-architectural reason.** Runtime machinery is downstream infrastructure. It must execute the biology-generated schedule without becoming the source ontology.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/runtime/v2/
- [proposed] src/runtime/v2/
- [proposed] tests/jbc/runtime/
- [proposed] bench/jbc/runtime/

**Existing code and permitted read scope:**

- include/CellShard/runtime/source/payload_source.hh
- include/CellShard/runtime/source/local_file_source.hh
- include/CellShard/runtime/residency/host.hh
- include/CellShard/runtime/residency/device.cuh
- CellShard global schedule IR
- Cellerator execution session
- numaBraid

**Explicitly out of scope / forbidden shortcuts:**

- Do not hardcode the local two-node test policy as universal reachability.
- Do not expose numaBraid relay/BDF/chunk internals to portable scheduling.
- Do not perform topology discovery or allocation inside sealed replay.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-RT03

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.
- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Measure or import bandwidth, latency, setup, concurrency, host-memory, and contention by provider/link/size; separate calibration freshness from topology validity.

Workstream mechanism: Build stable topology and measured calibration layers, async multi-range atom sources, read coalescing and pinned staging, source-linked transport providers, numaBraid/NCCL/P2P/host routes, plane-aware residency and leases, reconstruction-aware eviction, command IR, scheduler, CUDA Graph lowering, transactional recovery, and the dual-NUMA logical-node test.

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

- Use unavailable-route, capacity, cancellation, stale generation, concurrent stream, and logical two-node tests.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Measure source I/O, staging, route setup, transfer, overlap, residency hits, reconstruction, graph replay, host RAM, and complete graph latency.
- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-RUNTIME; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-RT05 — Define asynchronous multi-range atom-source ABI

**Repository / subsystem / lane:** CellShard · `runtime/v2` · `CS-JBC-L-RUNTIME`  
**Classification:** generic infrastructure supporting the biology-native compiler  
**Parallelism:** Serial within CS-JBC-L-RUNTIME; parallel with other provider/workstream lanes after JBC-G5-PARTIAL-ARTIFACT.

### Why

**Purpose.** Define asynchronous multi-range atom-source ABI. Deliver this as one isolated, reviewable step in the CellShard topology, I/O, transport, residency, and runtime lowering workstream.

**Biological motivation.** Biologically compiled atoms must be placed, prefetched, combined, and retained according to their reuse, reconstruction cost, mutation half-life, and operation graph—not merely bytes.

**Compiler-architectural reason.** Runtime machinery is downstream infrastructure. It must execute the biology-generated schedule without becoming the source ontology.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/runtime/v2/
- [proposed] src/runtime/v2/
- [proposed] tests/jbc/runtime/
- [proposed] bench/jbc/runtime/

**Existing code and permitted read scope:**

- include/CellShard/runtime/source/payload_source.hh
- include/CellShard/runtime/source/local_file_source.hh
- include/CellShard/runtime/residency/host.hh
- include/CellShard/runtime/residency/device.cuh
- CellShard global schedule IR
- Cellerator execution session
- numaBraid

**Explicitly out of scope / forbidden shortcuts:**

- Do not hardcode the local two-node test policy as universal reachability.
- Do not expose numaBraid relay/BDF/chunk internals to portable scheduling.
- Do not perform topology discovery or allocation inside sealed replay.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-RT04

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.

### Concrete mechanism

Add query/submit/cancel/completion operations over ordered byte ranges, caller-owned buffers, source identity, backpressure, and stable request IDs; preserve synchronous payload_source_ref adapter.

Workstream mechanism: Build stable topology and measured calibration layers, async multi-range atom sources, read coalescing and pinned staging, source-linked transport providers, numaBraid/NCCL/P2P/host routes, plane-aware residency and leases, reconstruction-aware eviction, command IR, scheduler, CUDA Graph lowering, transactional recovery, and the dual-NUMA logical-node test.

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

- Use unavailable-route, capacity, cancellation, stale generation, concurrent stream, and logical two-node tests.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Measure source I/O, staging, route setup, transfer, overlap, residency hits, reconstruction, graph replay, host RAM, and complete graph latency.
- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-RUNTIME; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-RT06 — Implement the synchronous exact-read baseline

**Repository / subsystem / lane:** CellShard · `runtime/v2` · `CS-JBC-L-RUNTIME`  
**Classification:** generic infrastructure supporting the biology-native compiler  
**Parallelism:** Serial within CS-JBC-L-RUNTIME; parallel with other provider/workstream lanes after JBC-G5-PARTIAL-ARTIFACT.

### Why

**Purpose.** Implement the synchronous exact-read baseline. Deliver this as one isolated, reviewable step in the CellShard topology, I/O, transport, residency, and runtime lowering workstream.

**Biological motivation.** Biologically compiled atoms must be placed, prefetched, combined, and retained according to their reuse, reconstruction cost, mutation half-life, and operation graph—not merely bytes.

**Compiler-architectural reason.** Runtime machinery is downstream infrastructure. It must execute the biology-generated schedule without becoming the source ontology.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/runtime/v2/
- [proposed] src/runtime/v2/
- [proposed] tests/jbc/runtime/
- [proposed] bench/jbc/runtime/

**Existing code and permitted read scope:**

- include/CellShard/runtime/source/payload_source.hh
- include/CellShard/runtime/source/local_file_source.hh
- include/CellShard/runtime/residency/host.hh
- include/CellShard/runtime/residency/device.cuh
- CellShard global schedule IR
- Cellerator execution session
- numaBraid

**Explicitly out of scope / forbidden shortcuts:**

- Do not hardcode the local two-node test policy as universal reachability.
- Do not expose numaBraid relay/BDF/chunk internals to portable scheduling.
- Do not perform topology discovery or allocation inside sealed replay.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-RT05

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Use current local-file/payload-source semantics to read exact atom frames deterministically with explicit errors and no speculative asynchronous behavior.

Workstream mechanism: Build stable topology and measured calibration layers, async multi-range atom sources, read coalescing and pinned staging, source-linked transport providers, numaBraid/NCCL/P2P/host routes, plane-aware residency and leases, reconstruction-aware eviction, command IR, scheduler, CUDA Graph lowering, transactional recovery, and the dual-NUMA logical-node test.

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

- Use unavailable-route, capacity, cancellation, stale generation, concurrent stream, and logical two-node tests.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Measure source I/O, staging, route setup, transfer, overlap, residency hits, reconstruction, graph replay, host RAM, and complete graph latency.
- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-RUNTIME; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-RT07 — Implement an asynchronous multi-range file provider

**Repository / subsystem / lane:** CellShard · `runtime/v2` · `CS-JBC-L-RUNTIME`  
**Classification:** generic infrastructure supporting the biology-native compiler  
**Parallelism:** Serial within CS-JBC-L-RUNTIME; parallel with other provider/workstream lanes after JBC-G5-PARTIAL-ARTIFACT.

### Why

**Purpose.** Implement an asynchronous multi-range file provider. Deliver this as one isolated, reviewable step in the CellShard topology, I/O, transport, residency, and runtime lowering workstream.

**Biological motivation.** Biologically compiled atoms must be placed, prefetched, combined, and retained according to their reuse, reconstruction cost, mutation half-life, and operation graph—not merely bytes.

**Compiler-architectural reason.** Runtime machinery is downstream infrastructure. It must execute the biology-generated schedule without becoming the source ontology.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/runtime/v2/
- [proposed] src/runtime/v2/
- [proposed] tests/jbc/runtime/
- [proposed] bench/jbc/runtime/

**Existing code and permitted read scope:**

- include/CellShard/runtime/source/payload_source.hh
- include/CellShard/runtime/source/local_file_source.hh
- include/CellShard/runtime/residency/host.hh
- include/CellShard/runtime/residency/device.cuh
- CellShard global schedule IR
- Cellerator execution session
- numaBraid

**Explicitly out of scope / forbidden shortcuts:**

- Do not hardcode the local two-node test policy as universal reachability.
- Do not expose numaBraid relay/BDF/chunk internals to portable scheduling.
- Do not perform topology discovery or allocation inside sealed replay.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-RT06

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Use the supported Linux async mechanism, preferably io_uring when available, with registered/fixed buffers where beneficial, bounded queue depth, cancellation, and fallback.

Workstream mechanism: Build stable topology and measured calibration layers, async multi-range atom sources, read coalescing and pinned staging, source-linked transport providers, numaBraid/NCCL/P2P/host routes, plane-aware residency and leases, reconstruction-aware eviction, command IR, scheduler, CUDA Graph lowering, transactional recovery, and the dual-NUMA logical-node test.

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

- Use unavailable-route, capacity, cancellation, stale generation, concurrent stream, and logical two-node tests.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Measure source I/O, staging, route setup, transfer, overlap, residency hits, reconstruction, graph replay, host RAM, and complete graph latency.
- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-RUNTIME; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-RT08 — Implement range coalescing and read planning

**Repository / subsystem / lane:** CellShard · `runtime/v2` · `CS-JBC-L-RUNTIME`  
**Classification:** generic infrastructure supporting the biology-native compiler  
**Parallelism:** Serial within CS-JBC-L-RUNTIME; parallel with other provider/workstream lanes after JBC-G5-PARTIAL-ARTIFACT.

### Why

**Purpose.** Implement range coalescing and read planning. Deliver this as one isolated, reviewable step in the CellShard topology, I/O, transport, residency, and runtime lowering workstream.

**Biological motivation.** Biologically compiled atoms must be placed, prefetched, combined, and retained according to their reuse, reconstruction cost, mutation half-life, and operation graph—not merely bytes.

**Compiler-architectural reason.** Runtime machinery is downstream infrastructure. It must execute the biology-generated schedule without becoming the source ontology.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/runtime/v2/
- [proposed] src/runtime/v2/
- [proposed] tests/jbc/runtime/
- [proposed] bench/jbc/runtime/

**Existing code and permitted read scope:**

- include/CellShard/runtime/source/payload_source.hh
- include/CellShard/runtime/source/local_file_source.hh
- include/CellShard/runtime/residency/host.hh
- include/CellShard/runtime/residency/device.cuh
- CellShard global schedule IR
- Cellerator execution session
- numaBraid

**Explicitly out of scope / forbidden shortcuts:**

- Do not hardcode the local two-node test policy as universal reachability.
- Do not expose numaBraid relay/BDF/chunk internals to portable scheduling.
- Do not perform topology discovery or allocation inside sealed replay.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-RT07

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Sort requested frames by storage extent, merge nearby ranges under overread budget, preserve frame boundaries/digests, and emit read amplification and seek/range counts.

Workstream mechanism: Build stable topology and measured calibration layers, async multi-range atom sources, read coalescing and pinned staging, source-linked transport providers, numaBraid/NCCL/P2P/host routes, plane-aware residency and leases, reconstruction-aware eviction, command IR, scheduler, CUDA Graph lowering, transactional recovery, and the dual-NUMA logical-node test.

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

- Use unavailable-route, capacity, cancellation, stale generation, concurrent stream, and logical two-node tests.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Measure source I/O, staging, route setup, transfer, overlap, residency hits, reconstruction, graph replay, host RAM, and complete graph latency.
- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-RUNTIME; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-RT09 — Implement NUMA-local pinned staging pools

**Repository / subsystem / lane:** CellShard · `runtime/v2` · `CS-JBC-L-RUNTIME`  
**Classification:** generic infrastructure supporting the biology-native compiler  
**Parallelism:** Serial within CS-JBC-L-RUNTIME; parallel with other provider/workstream lanes after JBC-G5-PARTIAL-ARTIFACT.

### Why

**Purpose.** Implement NUMA-local pinned staging pools. Deliver this as one isolated, reviewable step in the CellShard topology, I/O, transport, residency, and runtime lowering workstream.

**Biological motivation.** Biologically compiled atoms must be placed, prefetched, combined, and retained according to their reuse, reconstruction cost, mutation half-life, and operation graph—not merely bytes.

**Compiler-architectural reason.** Runtime machinery is downstream infrastructure. It must execute the biology-generated schedule without becoming the source ontology.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/runtime/v2/
- [proposed] src/runtime/v2/
- [proposed] tests/jbc/runtime/
- [proposed] bench/jbc/runtime/

**Existing code and permitted read scope:**

- include/CellShard/runtime/source/payload_source.hh
- include/CellShard/runtime/source/local_file_source.hh
- include/CellShard/runtime/residency/host.hh
- include/CellShard/runtime/residency/device.cuh
- CellShard global schedule IR
- Cellerator execution session
- numaBraid

**Explicitly out of scope / forbidden shortcuts:**

- Do not hardcode the local two-node test policy as universal reachability.
- Do not expose numaBraid relay/BDF/chunk internals to portable scheduling.
- Do not perform topology discovery or allocation inside sealed replay.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-RT08

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Allocate explicit bounded pinned buffers on the storage-local NUMA node, pipeline reads/decode/H2D, expose capacity/high-water marks, and avoid hidden pageable copies.

Workstream mechanism: Build stable topology and measured calibration layers, async multi-range atom sources, read coalescing and pinned staging, source-linked transport providers, numaBraid/NCCL/P2P/host routes, plane-aware residency and leases, reconstruction-aware eviction, command IR, scheduler, CUDA Graph lowering, transactional recovery, and the dual-NUMA logical-node test.

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

- Use unavailable-route, capacity, cancellation, stale generation, concurrent stream, and logical two-node tests.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Measure source I/O, staging, route setup, transfer, overlap, residency hits, reconstruction, graph replay, host RAM, and complete graph latency.
- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-RUNTIME; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-RT10 — Define the source-linked transport-provider registry

**Repository / subsystem / lane:** CellShard · `runtime/v2` · `CS-JBC-L-RUNTIME`  
**Classification:** generic infrastructure supporting the biology-native compiler  
**Parallelism:** Serial within CS-JBC-L-RUNTIME; parallel with other provider/workstream lanes after JBC-G5-PARTIAL-ARTIFACT.

### Why

**Purpose.** Define the source-linked transport-provider registry. Deliver this as one isolated, reviewable step in the CellShard topology, I/O, transport, residency, and runtime lowering workstream.

**Biological motivation.** Biologically compiled atoms must be placed, prefetched, combined, and retained according to their reuse, reconstruction cost, mutation half-life, and operation graph—not merely bytes.

**Compiler-architectural reason.** Runtime machinery is downstream infrastructure. It must execute the biology-generated schedule without becoming the source ontology.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/runtime/v2/
- [proposed] src/runtime/v2/
- [proposed] tests/jbc/runtime/
- [proposed] bench/jbc/runtime/

**Existing code and permitted read scope:**

- include/CellShard/runtime/source/payload_source.hh
- include/CellShard/runtime/source/local_file_source.hh
- include/CellShard/runtime/residency/host.hh
- include/CellShard/runtime/residency/device.cuh
- CellShard global schedule IR
- Cellerator execution session
- numaBraid

**Explicitly out of scope / forbidden shortcuts:**

- Do not hardcode the local two-node test policy as universal reachability.
- Do not expose numaBraid relay/BDF/chunk internals to portable scheduling.
- Do not perform topology discovery or allocation inside sealed replay.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-RT09

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.

### Concrete mechanism

Register same-device, host copy, CUDA P2P, numaBraid, NCCL/collective, loopback, and future network providers with query/plan/prepare/launch/completion/cancel/release.

Workstream mechanism: Build stable topology and measured calibration layers, async multi-range atom sources, read coalescing and pinned staging, source-linked transport providers, numaBraid/NCCL/P2P/host routes, plane-aware residency and leases, reconstruction-aware eviction, command IR, scheduler, CUDA Graph lowering, transactional recovery, and the dual-NUMA logical-node test.

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

- Use unavailable-route, capacity, cancellation, stale generation, concurrent stream, and logical two-node tests.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Measure source I/O, staging, route setup, transfer, overlap, residency hits, reconstruction, graph replay, host RAM, and complete graph latency.
- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-RUNTIME; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-RT11 — Implement same-device alias and CUDA P2P providers

**Repository / subsystem / lane:** CellShard · `runtime/v2` · `CS-JBC-L-RUNTIME`  
**Classification:** generic infrastructure supporting the biology-native compiler  
**Parallelism:** Serial within CS-JBC-L-RUNTIME; parallel with other provider/workstream lanes after JBC-G5-PARTIAL-ARTIFACT.

### Why

**Purpose.** Implement same-device alias and CUDA P2P providers. Deliver this as one isolated, reviewable step in the CellShard topology, I/O, transport, residency, and runtime lowering workstream.

**Biological motivation.** Biologically compiled atoms must be placed, prefetched, combined, and retained according to their reuse, reconstruction cost, mutation half-life, and operation graph—not merely bytes.

**Compiler-architectural reason.** Runtime machinery is downstream infrastructure. It must execute the biology-generated schedule without becoming the source ontology.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/runtime/v2/
- [proposed] src/runtime/v2/
- [proposed] tests/jbc/runtime/
- [proposed] bench/jbc/runtime/

**Existing code and permitted read scope:**

- include/CellShard/runtime/source/payload_source.hh
- include/CellShard/runtime/source/local_file_source.hh
- include/CellShard/runtime/residency/host.hh
- include/CellShard/runtime/residency/device.cuh
- CellShard global schedule IR
- Cellerator execution session
- numaBraid

**Explicitly out of scope / forbidden shortcuts:**

- Do not hardcode the local two-node test policy as universal reachability.
- Do not expose numaBraid relay/BDF/chunk internals to portable scheduling.
- Do not perform topology discovery or allocation inside sealed replay.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-RT10

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Provide no-copy alias where legal and explicit peer-copy plans with capability, size, stream/event, and fallback checks.

Workstream mechanism: Build stable topology and measured calibration layers, async multi-range atom sources, read coalescing and pinned staging, source-linked transport providers, numaBraid/NCCL/P2P/host routes, plane-aware residency and leases, reconstruction-aware eviction, command IR, scheduler, CUDA Graph lowering, transactional recovery, and the dual-NUMA logical-node test.

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

- Use unavailable-route, capacity, cancellation, stale generation, concurrent stream, and logical two-node tests.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Measure source I/O, staging, route setup, transfer, overlap, residency hits, reconstruction, graph replay, host RAM, and complete graph latency.
- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-RUNTIME; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-RT12 — Implement host-staged and QPI/NUMA transfer provider

**Repository / subsystem / lane:** CellShard · `runtime/v2` · `CS-JBC-L-RUNTIME`  
**Classification:** generic infrastructure supporting the biology-native compiler  
**Parallelism:** Serial within CS-JBC-L-RUNTIME; parallel with other provider/workstream lanes after JBC-G5-PARTIAL-ARTIFACT.

### Why

**Purpose.** Implement host-staged and QPI/NUMA transfer provider. Deliver this as one isolated, reviewable step in the CellShard topology, I/O, transport, residency, and runtime lowering workstream.

**Biological motivation.** Biologically compiled atoms must be placed, prefetched, combined, and retained according to their reuse, reconstruction cost, mutation half-life, and operation graph—not merely bytes.

**Compiler-architectural reason.** Runtime machinery is downstream infrastructure. It must execute the biology-generated schedule without becoming the source ontology.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/runtime/v2/
- [proposed] src/runtime/v2/
- [proposed] tests/jbc/runtime/
- [proposed] bench/jbc/runtime/

**Existing code and permitted read scope:**

- include/CellShard/runtime/source/payload_source.hh
- include/CellShard/runtime/source/local_file_source.hh
- include/CellShard/runtime/residency/host.hh
- include/CellShard/runtime/residency/device.cuh
- CellShard global schedule IR
- Cellerator execution session
- numaBraid

**Explicitly out of scope / forbidden shortcuts:**

- Do not hardcode the local two-node test policy as universal reachability.
- Do not expose numaBraid relay/BDF/chunk internals to portable scheduling.
- Do not perform topology discovery or allocation inside sealed replay.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-RT11

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Move through explicit storage-local and destination-local pinned buffers, account cross-socket bytes, host RAM, copies, and concurrency; retain as universal local fallback.

Workstream mechanism: Build stable topology and measured calibration layers, async multi-range atom sources, read coalescing and pinned staging, source-linked transport providers, numaBraid/NCCL/P2P/host routes, plane-aware residency and leases, reconstruction-aware eviction, command IR, scheduler, CUDA Graph lowering, transactional recovery, and the dual-NUMA logical-node test.

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

- Use unavailable-route, capacity, cancellation, stale generation, concurrent stream, and logical two-node tests.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Measure source I/O, staging, route setup, transfer, overlap, residency hits, reconstruction, graph replay, host RAM, and complete graph latency.
- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-RUNTIME; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-RT13 — Integrate numaBraid build and capability discovery

**Repository / subsystem / lane:** CellShard · `runtime/v2` · `CS-JBC-L-RUNTIME`  
**Classification:** generic infrastructure supporting the biology-native compiler  
**Parallelism:** Serial within CS-JBC-L-RUNTIME; parallel with other provider/workstream lanes after JBC-G5-PARTIAL-ARTIFACT.

### Why

**Purpose.** Integrate numaBraid build and capability discovery. Deliver this as one isolated, reviewable step in the CellShard topology, I/O, transport, residency, and runtime lowering workstream.

**Biological motivation.** Biologically compiled atoms must be placed, prefetched, combined, and retained according to their reuse, reconstruction cost, mutation half-life, and operation graph—not merely bytes.

**Compiler-architectural reason.** Runtime machinery is downstream infrastructure. It must execute the biology-generated schedule without becoming the source ontology.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/runtime/v2/
- [proposed] src/runtime/v2/
- [proposed] tests/jbc/runtime/
- [proposed] bench/jbc/runtime/

**Existing code and permitted read scope:**

- include/CellShard/runtime/source/payload_source.hh
- include/CellShard/runtime/source/local_file_source.hh
- include/CellShard/runtime/residency/host.hh
- include/CellShard/runtime/residency/device.cuh
- CellShard global schedule IR
- Cellerator execution session
- numaBraid

**Explicitly out of scope / forbidden shortcuts:**

- Do not hardcode the local two-node test policy as universal reachability.
- Do not expose numaBraid relay/BDF/chunk internals to portable scheduling.
- Do not perform topology discovery or allocation inside sealed replay.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-RT12

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Resolve the canonical numaBraid target/package, make it a dependency of the full CellShard transport/runtime target, and import topology/capability without exposing relay internals.

Workstream mechanism: Build stable topology and measured calibration layers, async multi-range atom sources, read coalescing and pinned staging, source-linked transport providers, numaBraid/NCCL/P2P/host routes, plane-aware residency and leases, reconstruction-aware eviction, command IR, scheduler, CUDA Graph lowering, transactional recovery, and the dual-NUMA logical-node test.

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

- Use unavailable-route, capacity, cancellation, stale generation, concurrent stream, and logical two-node tests.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Measure source I/O, staging, route setup, transfer, overlap, residency hits, reconstruction, graph replay, host RAM, and complete graph latency.
- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-RUNTIME; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-RT14 — Implement numaBraid transport planning and launch

**Repository / subsystem / lane:** CellShard · `runtime/v2` · `CS-JBC-L-RUNTIME`  
**Classification:** generic infrastructure supporting the biology-native compiler  
**Parallelism:** Serial within CS-JBC-L-RUNTIME; parallel with other provider/workstream lanes after JBC-G5-PARTIAL-ARTIFACT.

### Why

**Purpose.** Implement numaBraid transport planning and launch. Deliver this as one isolated, reviewable step in the CellShard topology, I/O, transport, residency, and runtime lowering workstream.

**Biological motivation.** Biologically compiled atoms must be placed, prefetched, combined, and retained according to their reuse, reconstruction cost, mutation half-life, and operation graph—not merely bytes.

**Compiler-architectural reason.** Runtime machinery is downstream infrastructure. It must execute the biology-generated schedule without becoming the source ontology.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/runtime/v2/
- [proposed] src/runtime/v2/
- [proposed] tests/jbc/runtime/
- [proposed] bench/jbc/runtime/

**Existing code and permitted read scope:**

- include/CellShard/runtime/source/payload_source.hh
- include/CellShard/runtime/source/local_file_source.hh
- include/CellShard/runtime/residency/host.hh
- include/CellShard/runtime/residency/device.cuh
- CellShard global schedule IR
- Cellerator execution session
- numaBraid

**Explicitly out of scope / forbidden shortcuts:**

- Do not hardcode the local two-node test policy as universal reachability.
- Do not expose numaBraid relay/BDF/chunk internals to portable scheduling.
- Do not perform topology discovery or allocation inside sealed replay.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-RT13

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Given source/destination GPU, buffer ranges, size, stream/event, and policy, request an eligible numaBraid plan, launch asynchronously, report completion/error, batch transfers, and preserve fallback.

Workstream mechanism: Build stable topology and measured calibration layers, async multi-range atom sources, read coalescing and pinned staging, source-linked transport providers, numaBraid/NCCL/P2P/host routes, plane-aware residency and leases, reconstruction-aware eviction, command IR, scheduler, CUDA Graph lowering, transactional recovery, and the dual-NUMA logical-node test.

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

- Use unavailable-route, capacity, cancellation, stale generation, concurrent stream, and logical two-node tests.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Measure source I/O, staging, route setup, transfer, overlap, residency hits, reconstruction, graph replay, host RAM, and complete graph latency.
- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-RUNTIME; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-RT15 — Implement the NCCL collective provider

**Repository / subsystem / lane:** CellShard · `runtime/v2` · `CS-JBC-L-RUNTIME`  
**Classification:** generic infrastructure supporting the biology-native compiler  
**Parallelism:** Serial within CS-JBC-L-RUNTIME; parallel with other provider/workstream lanes after JBC-G5-PARTIAL-ARTIFACT.

### Why

**Purpose.** Implement the NCCL collective provider. Deliver this as one isolated, reviewable step in the CellShard topology, I/O, transport, residency, and runtime lowering workstream.

**Biological motivation.** Biologically compiled atoms must be placed, prefetched, combined, and retained according to their reuse, reconstruction cost, mutation half-life, and operation graph—not merely bytes.

**Compiler-architectural reason.** Runtime machinery is downstream infrastructure. It must execute the biology-generated schedule without becoming the source ontology.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/runtime/v2/
- [proposed] src/runtime/v2/
- [proposed] tests/jbc/runtime/
- [proposed] bench/jbc/runtime/

**Existing code and permitted read scope:**

- include/CellShard/runtime/source/payload_source.hh
- include/CellShard/runtime/source/local_file_source.hh
- include/CellShard/runtime/residency/host.hh
- include/CellShard/runtime/residency/device.cuh
- CellShard global schedule IR
- Cellerator execution session
- numaBraid

**Explicitly out of scope / forbidden shortcuts:**

- Do not hardcode the local two-node test policy as universal reachability.
- Do not expose numaBraid relay/BDF/chunk internals to portable scheduling.
- Do not perform topology discovery or allocation inside sealed replay.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-RT14

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Use NCCL for supported collectives/point-to-point where selected, create communicators cold, expose graph-capture constraints, and keep CellShard in charge of participant set and reduction plan.

Workstream mechanism: Build stable topology and measured calibration layers, async multi-range atom sources, read coalescing and pinned staging, source-linked transport providers, numaBraid/NCCL/P2P/host routes, plane-aware residency and leases, reconstruction-aware eviction, command IR, scheduler, CUDA Graph lowering, transactional recovery, and the dual-NUMA logical-node test.

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

- Use unavailable-route, capacity, cancellation, stale generation, concurrent stream, and logical two-node tests.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Measure source I/O, staging, route setup, transfer, overlap, residency hits, reconstruction, graph replay, host RAM, and complete graph latency.
- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-RUNTIME; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-RT16 — Define atom-plane resident instances

**Repository / subsystem / lane:** CellShard · `runtime/v2` · `CS-JBC-L-RUNTIME`  
**Classification:** generic infrastructure supporting the biology-native compiler  
**Parallelism:** Serial within CS-JBC-L-RUNTIME; parallel with other provider/workstream lanes after JBC-G5-PARTIAL-ARTIFACT.

### Why

**Purpose.** Define atom-plane resident instances. Deliver this as one isolated, reviewable step in the CellShard topology, I/O, transport, residency, and runtime lowering workstream.

**Biological motivation.** Biologically compiled atoms must be placed, prefetched, combined, and retained according to their reuse, reconstruction cost, mutation half-life, and operation graph—not merely bytes.

**Compiler-architectural reason.** Runtime machinery is downstream infrastructure. It must execute the biology-generated schedule without becoming the source ontology.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/runtime/v2/
- [proposed] src/runtime/v2/
- [proposed] tests/jbc/runtime/
- [proposed] bench/jbc/runtime/

**Existing code and permitted read scope:**

- include/CellShard/runtime/source/payload_source.hh
- include/CellShard/runtime/source/local_file_source.hh
- include/CellShard/runtime/residency/host.hh
- include/CellShard/runtime/residency/device.cuh
- CellShard global schedule IR
- Cellerator execution session
- numaBraid

**Explicitly out of scope / forbidden shortcuts:**

- Do not hardcode the local two-node test policy as universal reachability.
- Do not expose numaBraid relay/BDF/chunk internals to portable scheduling.
- Do not perform topology discovery or allocation inside sealed replay.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-RT15

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.

### Concrete mechanism

Represent atom/plane/materialization/replica, tier, node/device, pointer/range, order, generation, ready event, reconstruction action, consumers, and pin/lease counts.

Workstream mechanism: Build stable topology and measured calibration layers, async multi-range atom sources, read coalescing and pinned staging, source-linked transport providers, numaBraid/NCCL/P2P/host routes, plane-aware residency and leases, reconstruction-aware eviction, command IR, scheduler, CUDA Graph lowering, transactional recovery, and the dual-NUMA logical-node test.

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

- Use unavailable-route, capacity, cancellation, stale generation, concurrent stream, and logical two-node tests.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Measure source I/O, staging, route setup, transfer, overlap, residency hits, reconstruction, graph replay, host RAM, and complete graph latency.
- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-RUNTIME; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-RT17 — Implement leases and pins

**Repository / subsystem / lane:** CellShard · `runtime/v2` · `CS-JBC-L-RUNTIME`  
**Classification:** generic infrastructure supporting the biology-native compiler  
**Parallelism:** Serial within CS-JBC-L-RUNTIME; parallel with other provider/workstream lanes after JBC-G5-PARTIAL-ARTIFACT.

### Why

**Purpose.** Implement leases and pins. Deliver this as one isolated, reviewable step in the CellShard topology, I/O, transport, residency, and runtime lowering workstream.

**Biological motivation.** Biologically compiled atoms must be placed, prefetched, combined, and retained according to their reuse, reconstruction cost, mutation half-life, and operation graph—not merely bytes.

**Compiler-architectural reason.** Runtime machinery is downstream infrastructure. It must execute the biology-generated schedule without becoming the source ontology.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/runtime/v2/
- [proposed] src/runtime/v2/
- [proposed] tests/jbc/runtime/
- [proposed] bench/jbc/runtime/

**Existing code and permitted read scope:**

- include/CellShard/runtime/source/payload_source.hh
- include/CellShard/runtime/source/local_file_source.hh
- include/CellShard/runtime/residency/host.hh
- include/CellShard/runtime/residency/device.cuh
- CellShard global schedule IR
- Cellerator execution session
- numaBraid

**Explicitly out of scope / forbidden shortcuts:**

- Do not hardcode the local two-node test policy as universal reachability.
- Do not expose numaBraid relay/BDF/chunk internals to portable scheduling.
- Do not perform topology discovery or allocation inside sealed replay.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-RT16

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Guarantee identity, generation, residency, and graph-stable address for a prepared consumer; release explicitly, reject stale use, and propagate failed readiness.

Workstream mechanism: Build stable topology and measured calibration layers, async multi-range atom sources, read coalescing and pinned staging, source-linked transport providers, numaBraid/NCCL/P2P/host routes, plane-aware residency and leases, reconstruction-aware eviction, command IR, scheduler, CUDA Graph lowering, transactional recovery, and the dual-NUMA logical-node test.

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

- Use unavailable-route, capacity, cancellation, stale generation, concurrent stream, and logical two-node tests.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Measure source I/O, staging, route setup, transfer, overlap, residency hits, reconstruction, graph replay, host RAM, and complete graph latency.
- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-RUNTIME; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-RT18 — Implement reconstruction-aware admission and eviction

**Repository / subsystem / lane:** CellShard · `runtime/v2` · `CS-JBC-L-RUNTIME`  
**Classification:** generic infrastructure supporting the biology-native compiler  
**Parallelism:** Serial within CS-JBC-L-RUNTIME; parallel with other provider/workstream lanes after JBC-G5-PARTIAL-ARTIFACT.

### Why

**Purpose.** Implement reconstruction-aware admission and eviction. Deliver this as one isolated, reviewable step in the CellShard topology, I/O, transport, residency, and runtime lowering workstream.

**Biological motivation.** Biologically compiled atoms must be placed, prefetched, combined, and retained according to their reuse, reconstruction cost, mutation half-life, and operation graph—not merely bytes.

**Compiler-architectural reason.** Runtime machinery is downstream infrastructure. It must execute the biology-generated schedule without becoming the source ontology.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/runtime/v2/
- [proposed] src/runtime/v2/
- [proposed] tests/jbc/runtime/
- [proposed] bench/jbc/runtime/

**Existing code and permitted read scope:**

- include/CellShard/runtime/source/payload_source.hh
- include/CellShard/runtime/source/local_file_source.hh
- include/CellShard/runtime/residency/host.hh
- include/CellShard/runtime/residency/device.cuh
- CellShard global schedule IR
- Cellerator execution session
- numaBraid

**Explicitly out of scope / forbidden shortcuts:**

- Do not hardcode the local two-node test policy as universal reachability.
- Do not expose numaBraid relay/BDF/chunk internals to portable scheduling.
- Do not perform topology discovery or allocation inside sealed replay.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-RT17

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Rank resident atoms by predicted reuse, fetch/decode/assembly/recompile cost, persistent-order value, downstream fan-out, mutability, and capacity; keep a size/LRU baseline.

Workstream mechanism: Build stable topology and measured calibration layers, async multi-range atom sources, read coalescing and pinned staging, source-linked transport providers, numaBraid/NCCL/P2P/host routes, plane-aware residency and leases, reconstruction-aware eviction, command IR, scheduler, CUDA Graph lowering, transactional recovery, and the dual-NUMA logical-node test.

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

- Use unavailable-route, capacity, cancellation, stale generation, concurrent stream, and logical two-node tests.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Measure source I/O, staging, route setup, transfer, overlap, residency hits, reconstruction, graph replay, host RAM, and complete graph latency.
- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-RUNTIME; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-RT19 — Define runtime command IR and scheduler

**Repository / subsystem / lane:** CellShard · `runtime/v2` · `CS-JBC-L-RUNTIME`  
**Classification:** generic infrastructure supporting the biology-native compiler  
**Parallelism:** Serial within CS-JBC-L-RUNTIME; parallel with other provider/workstream lanes after JBC-G5-PARTIAL-ARTIFACT.

### Why

**Purpose.** Define runtime command IR and scheduler. Deliver this as one isolated, reviewable step in the CellShard topology, I/O, transport, residency, and runtime lowering workstream.

**Biological motivation.** Biologically compiled atoms must be placed, prefetched, combined, and retained according to their reuse, reconstruction cost, mutation half-life, and operation graph—not merely bytes.

**Compiler-architectural reason.** Runtime machinery is downstream infrastructure. It must execute the biology-generated schedule without becoming the source ontology.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/runtime/v2/
- [proposed] src/runtime/v2/
- [proposed] tests/jbc/runtime/
- [proposed] bench/jbc/runtime/

**Existing code and permitted read scope:**

- include/CellShard/runtime/source/payload_source.hh
- include/CellShard/runtime/source/local_file_source.hh
- include/CellShard/runtime/residency/host.hh
- include/CellShard/runtime/residency/device.cuh
- CellShard global schedule IR
- Cellerator execution session
- numaBraid

**Explicitly out of scope / forbidden shortcuts:**

- Do not hardcode the local two-node test policy as universal reachability.
- Do not expose numaBraid relay/BDF/chunk internals to portable scheduling.
- Do not perform topology discovery or allocation inside sealed replay.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-RT18

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A versioned pointer-first contract or cold owning-builder contract with static layout checks and an independent validator.

### Concrete mechanism

Lower storage read, decode, assemble, transfer, Cellerator fragment, external operation, partial combine, canonicalize, publish, persist, and evict commands with dependencies/resources/retry/profiler IDs.

Workstream mechanism: Build stable topology and measured calibration layers, async multi-range atom sources, read coalescing and pinned staging, source-linked transport providers, numaBraid/NCCL/P2P/host routes, plane-aware residency and leases, reconstruction-aware eviction, command IR, scheduler, CUDA Graph lowering, transactional recovery, and the dual-NUMA logical-node test.

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

- Use unavailable-route, capacity, cancellation, stale generation, concurrent stream, and logical two-node tests.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Measure source I/O, staging, route setup, transfer, overlap, residency hits, reconstruction, graph replay, host RAM, and complete graph latency.
- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-RUNTIME; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-RT20 — Implement worker-local CUDA Graph lowering

**Repository / subsystem / lane:** CellShard · `runtime/v2` · `CS-JBC-L-RUNTIME`  
**Classification:** generic infrastructure supporting the biology-native compiler  
**Parallelism:** Serial within CS-JBC-L-RUNTIME; parallel with other provider/workstream lanes after JBC-G5-PARTIAL-ARTIFACT.

### Why

**Purpose.** Implement worker-local CUDA Graph lowering. Deliver this as one isolated, reviewable step in the CellShard topology, I/O, transport, residency, and runtime lowering workstream.

**Biological motivation.** Biologically compiled atoms must be placed, prefetched, combined, and retained according to their reuse, reconstruction cost, mutation half-life, and operation graph—not merely bytes.

**Compiler-architectural reason.** Runtime machinery is downstream infrastructure. It must execute the biology-generated schedule without becoming the source ontology.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/runtime/v2/
- [proposed] src/runtime/v2/
- [proposed] tests/jbc/runtime/
- [proposed] bench/jbc/runtime/

**Existing code and permitted read scope:**

- include/CellShard/runtime/source/payload_source.hh
- include/CellShard/runtime/source/local_file_source.hh
- include/CellShard/runtime/residency/host.hh
- include/CellShard/runtime/residency/device.cuh
- CellShard global schedule IR
- Cellerator execution session
- numaBraid

**Explicitly out of scope / forbidden shortcuts:**

- Do not hardcode the local two-node test policy as universal reachability.
- Do not expose numaBraid relay/BDF/chunk internals to portable scheduling.
- Do not perform topology discovery or allocation inside sealed replay.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-RT19

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.

### Concrete mechanism

Capture stable local command subgraphs after addresses are leased, support legal parameter/pointer updates, invalidate/recapture on topology or structural changes, and retain ordinary launch fallback.

Workstream mechanism: Build stable topology and measured calibration layers, async multi-range atom sources, read coalescing and pinned staging, source-linked transport providers, numaBraid/NCCL/P2P/host routes, plane-aware residency and leases, reconstruction-aware eviction, command IR, scheduler, CUDA Graph lowering, transactional recovery, and the dual-NUMA logical-node test.

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

- Use unavailable-route, capacity, cancellation, stale generation, concurrent stream, and logical two-node tests.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Measure source I/O, staging, route setup, transfer, overlap, residency hits, reconstruction, graph replay, host RAM, and complete graph latency.
- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-RUNTIME; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-RT21 — Implement transactional runtime recovery

**Repository / subsystem / lane:** CellShard · `runtime/v2` · `CS-JBC-L-RUNTIME`  
**Classification:** generic infrastructure supporting the biology-native compiler  
**Parallelism:** Serial within CS-JBC-L-RUNTIME; parallel with other provider/workstream lanes after JBC-G5-PARTIAL-ARTIFACT.

### Why

**Purpose.** Implement transactional runtime recovery. Deliver this as one isolated, reviewable step in the CellShard topology, I/O, transport, residency, and runtime lowering workstream.

**Biological motivation.** Biologically compiled atoms must be placed, prefetched, combined, and retained according to their reuse, reconstruction cost, mutation half-life, and operation graph—not merely bytes.

**Compiler-architectural reason.** Runtime machinery is downstream infrastructure. It must execute the biology-generated schedule without becoming the source ontology.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/runtime/v2/
- [proposed] src/runtime/v2/
- [proposed] tests/jbc/runtime/
- [proposed] bench/jbc/runtime/

**Existing code and permitted read scope:**

- include/CellShard/runtime/source/payload_source.hh
- include/CellShard/runtime/source/local_file_source.hh
- include/CellShard/runtime/residency/host.hh
- include/CellShard/runtime/residency/device.cuh
- CellShard global schedule IR
- Cellerator execution session
- numaBraid

**Explicitly out of scope / forbidden shortcuts:**

- Do not hardcode the local two-node test policy as universal reachability.
- Do not expose numaBraid relay/BDF/chunk internals to portable scheduling.
- Do not perform topology discovery or allocation inside sealed replay.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-RT20

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A source implementation that consumes explicit caller-owned storage or declares its allocations and produces deterministic diagnostics.
- Focused tests and adversarial fixtures demonstrating both success and explicit rejection paths.

### Concrete mechanism

Classify retryable/idempotent commands, cancel dependent work, reject incomplete partials, recover resident/materialized state, and publish generations/objects only after all required commands succeed.

Workstream mechanism: Build stable topology and measured calibration layers, async multi-range atom sources, read coalescing and pinned staging, source-linked transport providers, numaBraid/NCCL/P2P/host routes, plane-aware residency and leases, reconstruction-aware eviction, command IR, scheduler, CUDA Graph lowering, transactional recovery, and the dual-NUMA logical-node test.

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

- Use unavailable-route, capacity, cancellation, stale generation, concurrent stream, and logical two-node tests.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Measure source I/O, staging, route setup, transfer, overlap, residency hits, reconstruction, graph replay, host RAM, and complete graph latency.
- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-RUNTIME; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.


## CS-JBC-RT22 — Run the dual-NUMA logical-node and process-model campaign

**Repository / subsystem / lane:** CellShard · `runtime/v2` · `CS-JBC-L-RUNTIME`  
**Classification:** generic infrastructure supporting the biology-native compiler  
**Parallelism:** Serial within CS-JBC-L-RUNTIME; parallel with other provider/workstream lanes after JBC-G5-PARTIAL-ARTIFACT.

### Why

**Purpose.** Run the dual-NUMA logical-node and process-model campaign. Deliver this as one isolated, reviewable step in the CellShard topology, I/O, transport, residency, and runtime lowering workstream.

**Biological motivation.** Biologically compiled atoms must be placed, prefetched, combined, and retained according to their reuse, reconstruction cost, mutation half-life, and operation graph—not merely bytes.

**Compiler-architectural reason.** Runtime machinery is downstream infrastructure. It must execute the biology-generated schedule without becoming the source ontology.

### Scope

**Likely write scope** — paths marked `[proposed]` do not yet exist:

- [proposed] include/CellShard/runtime/v2/
- [proposed] src/runtime/v2/
- [proposed] tests/jbc/runtime/
- [proposed] bench/jbc/runtime/

**Existing code and permitted read scope:**

- include/CellShard/runtime/source/payload_source.hh
- include/CellShard/runtime/source/local_file_source.hh
- include/CellShard/runtime/residency/host.hh
- include/CellShard/runtime/residency/device.cuh
- CellShard global schedule IR
- Cellerator execution session
- numaBraid

**Explicitly out of scope / forbidden shortcuts:**

- Do not hardcode the local two-node test policy as universal reachability.
- Do not expose numaBraid relay/BDF/chunk internals to portable scheduling.
- Do not perform topology discovery or allocation inside sealed replay.
- Do not reinterpret a statistical cluster as a causal biological module.
- Do not hide allocation, canonicalization, packing, assembly, transfer, or synchronization.
- Do not encode V100 tile dimensions, GPU ordinals, topology routes, or runtime pointers in portable biological identity.
- Do not mutate frozen CSG1, CPE2, CPK1, CSPACK, or CSH5 wire semantics without an adjacent version and compatibility route.
- Do not move Cellerator mathematical semantics into CellShard or CellShard global placement/storage semantics into standalone Cellerator.

### Inputs and dependencies

- CS-JBC-RT21

Expected inputs:

- The frozen shared atom/coverage/identity contracts applicable to this workstream.
- Exact live source identities revalidated before implementation.
- Caller-owned capacities, cost/reuse policy, and biological-stratum metadata where relevant.

### Outputs

- A bounded implementation unit, focused tests, mechanism statistics, and an integration receipt.

### Concrete mechanism

Treat the host as two logical nodes, assign /mnt/block to node 0, compare host/QPI, local GPU+numaBraid, and compute-near-storage routes; compare in-process, per-NUMA-node, and where feasible per-GPU models without hard policy.

Workstream mechanism: Build stable topology and measured calibration layers, async multi-range atom sources, read coalescing and pinned staging, source-linked transport providers, numaBraid/NCCL/P2P/host routes, plane-aware residency and leases, reconstruction-aware eviction, command IR, scheduler, CUDA Graph lowering, transactional recovery, and the dual-NUMA logical-node test.

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

- Use unavailable-route, capacity, cancellation, stale generation, concurrent stream, and logical two-node tests.
- Add deterministic unit tests, malformed-input tests, and randomized property tests.
- Where an approximate proposal is used, compare it with an exact rescan and independently validate the certified output.
- Keep at least one canonical/reference fallback and differential-test against it.

### Performance evidence and promotion

- Measure source I/O, staging, route setup, transfer, overlap, residency hits, reconstruction, graph replay, host RAM, and complete graph latency.
- Record cold build time, peak memory, candidate count, and work avoided or added.
- Do not promote from microkernel speed alone; report complete cost and reuse break-even.
- A measured non-promotion is an acceptable terminal result for experimental mechanisms.

### Integration and completion

**Integration point.** Source-linked fragment or frozen interface owned by lane CS-JBC-L-RUNTIME; final aggregation is integration-lane-only.

**Downstream consumers:**

- None.

**Completion condition.** The contract/implementation is source-complete, independently validated, profiler-visible, and consumable through fixtures without central-file edits.
