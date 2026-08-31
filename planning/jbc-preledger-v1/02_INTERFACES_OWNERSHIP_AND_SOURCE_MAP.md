# Cross-project interfaces, ownership, and live source map

The interfaces below are proposed pre-ledger contracts. Existing paths are exact; `[proposed]` paths must be created additively. Function pointers belong only in source-linked runtime/provider registries. Persistent artifacts store stable identities and schemas.

| ID | Interface | Owner | Consumers | Proposed path |
| --- | --- | --- | --- | --- |
| JBC-I01 | Namespace-qualified persistent identity | Cellerator | CellShard atom core, all operation providers | [proposed] include/Cellerator/execution/joint_compiler/persistent_identity_v1.hh |
| JBC-I02 | Exact logical coverage and role | Cellerator | CellShard atoms, Cellerator fragments, external providers | [proposed] include/Cellerator/execution/joint_compiler/logical_coverage_v1.hh |
| JBC-I03 | Atom requirement descriptor | Cellerator | CellShard materializer and linker | [proposed] include/Cellerator/execution/joint_compiler/atom_requirement_v1.hh |
| JBC-I04 | Atom affordance descriptor | Cellerator | CellShard atom store, graph compiler, persistence planner | [proposed] include/Cellerator/execution/joint_compiler/atom_affordance_v1.hh |
| JBC-I05 | Partial-result algebra | Cellerator/operation provider | CellShard global graph, partial persistence, collection | [proposed] include/Cellerator/compute/decomposition/partial_result_algebra_v1.hh |
| JBC-I06 | Operation decomposition alternatives | Cellerator/operation provider | CellShard global compiler | [proposed] include/Cellerator/compute/decomposition/decomposition_v1.hh |
| JBC-I07 | Atom-fragment request and candidate frontier | Cellerator | Embedded CellShard bridge, standalone callers | [proposed] include/Cellerator/execution/atom_fragment/fragment_v1.hh |
| JBC-I08 | Multi-extent external binding | Cellerator | CellShard residency/runtime | [proposed] include/Cellerator/execution/object_binding/external_binding_v1.hh |
| JBC-I09 | Lowering-resumption contract | Cellerator | CellShard artifact store and linker | [proposed] include/Cellerator/execution/lowering_resumption/resumption_v1.hh |
| JBC-I10 | External global cost vector | Cellerator | CellShard global optimizer, other standalone global planners | [proposed] include/Cellerator/planner/external_cost/external_cost_v1.hh |
| JBC-I11 | Atom-aware execution export v2 | Cellerator | CellShard graph/basis/topology compilers | [proposed] include/Cellerator/profiling/joint_compiler/execution_export_v2.hh |
| JBC-I12 | Common atom envelope | CellShard | CellShard compiler/runtime, embedded Cellerator bridge | [proposed] include/CellShard/compiler/atom/atom_v1.hh |
| JBC-I13 | Atom evidence atlas | CellShard | Discovery portfolio, basis/grammar planners | [proposed] include/CellShard/compiler/evidence/atlas_v1.hh |
| JBC-I14 | Composition production and derivation | CellShard | Grammar, basis, superatom, artifact linker | [proposed] include/CellShard/compiler/composition/production_v1.hh |
| JBC-I15 | Biological execution basis manifest | CellShard | Materializer, graph-family linker, persistence | [proposed] include/CellShard/compiler/basis/basis_v1.hh |
| JBC-I16 | Persistent partial atom | CellShard with Cellerator algebra | Global graph, store, collection | [proposed] include/CellShard/compiler/partial/partial_atom_v1.hh |
| JBC-I17 | Atom-store generation and physical instance | CellShard | CellShard linker/runtime, Cellerator external payload/binding adapters | [proposed] include/CellShard/artifact/atom_store/ |
| JBC-I18 | Global operation provider and portable schedule | CellShard | Cellerator bridge, future Baseplane/model providers | [proposed] include/CellShard/compiler/graph/provider_v1.hh and schedule_v1.hh |
| JBC-I19 | Exact distributed certificate | CellShard | Scheduler, runtime, recovery, validation | [proposed] include/CellShard/compiler/certification/distributed_certificate_v1.hh |
| JBC-I20 | Topology realization, transport and residency lease | CellShard | Runtime executors, Cellerator external bindings | [proposed] include/CellShard/runtime/v2/ |

## JBC-I01 — Namespace-qualified persistent identity

**Owner:** Cellerator

**Existing substrate:**

- include/Cellerator/compute/operation/operation_core_v2/schema.hh
- include/Cellerator/execution/identity.hh
- include/CellShard/identity/strong_id.hh

**Proposed location:** `[proposed] include/Cellerator/execution/joint_compiler/persistent_identity_v1.hh`

**Semantics:** Explicit producer namespace plus local 64-bit identity; adapters preserve legacy IDs without hashing pointers or conflating content identity.

**Freeze rule:** Freeze the smallest pointer-first semantic contract needed for provider fan-out. Add adjacent versions rather than mutating frozen wire formats. Consumers record exact version/hash/source receipts.

## JBC-I02 — Exact logical coverage and role

**Owner:** Cellerator

**Existing substrate:**

- include/Cellerator/geometry/relation_cover.hh
- include/Cellerator/execution/index_space/hierarchical_index_space_v1.hh

**Proposed location:** `[proposed] include/Cellerator/execution/joint_compiler/logical_coverage_v1.hh`

**Semantics:** Typed axes, structure/epoch, exact member or edge set, role, canonical maps, global 64-bit identity, compact local indexes; proposal/halo/replica/contribution roles are distinct.

**Freeze rule:** Freeze the smallest pointer-first semantic contract needed for provider fan-out. Add adjacent versions rather than mutating frozen wire formats. Consumers record exact version/hash/source receipts.

## JBC-I03 — Atom requirement descriptor

**Owner:** Cellerator

**Existing substrate:**

- include/Cellerator/compute/operation/candidate_catalog_v3/
- include/Cellerator/execution/geometry_acquisition_v2/schema.hh

**Proposed location:** `[proposed] include/Cellerator/execution/joint_compiler/atom_requirement_v1.hh`

**Semantics:** Coverage, planes, order, numeric/index types, extent and alignment rules, generation, target, graph-stability, and accepted transform routes.

**Freeze rule:** Freeze the smallest pointer-first semantic contract needed for provider fan-out. Add adjacent versions rather than mutating frozen wire formats. Consumers record exact version/hash/source receipts.

## JBC-I04 — Atom affordance descriptor

**Owner:** Cellerator

**Existing substrate:**

- include/Cellerator/profiling/partition_export.h
- include/Cellerator/execution/projection_value_plane/

**Proposed location:** `[proposed] include/Cellerator/execution/joint_compiler/atom_affordance_v1.hh`

**Semantics:** What a physical or output atom can directly satisfy, including operation families, planes, orders, partial forms, target ABI, direct gradient/output, and persistence eligibility.

**Freeze rule:** Freeze the smallest pointer-first semantic contract needed for provider fan-out. Add adjacent versions rather than mutating frozen wire formats. Consumers record exact version/hash/source receipts.

## JBC-I05 — Partial-result algebra

**Owner:** Cellerator/operation provider

**Existing substrate:**

- include/Cellerator/compute/operation/relation_algebra_v2/
- include/Cellerator/compute/operation/operation_core_v2/schema.hh

**Proposed location:** `[proposed] include/Cellerator/compute/decomposition/partial_result_algebra_v1.hh`

**Semantics:** State schema, neutral element, merge, finalize, algebraic laws, order/determinism constraints, numeric policy, and stable implementation identities.

**Freeze rule:** Freeze the smallest pointer-first semantic contract needed for provider fan-out. Add adjacent versions rather than mutating frozen wire formats. Consumers record exact version/hash/source receipts.

## JBC-I06 — Operation decomposition alternatives

**Owner:** Cellerator/operation provider

**Existing substrate:**

- include/Cellerator/compute/operation/operation_core_v2/
- include/Cellerator/geometry/relation_cover.hh

**Proposed location:** `[proposed] include/Cellerator/compute/decomposition/decomposition_v1.hh`

**Semantics:** Legal split axes, exact input/output/partial coverage, replication and halos, order constraints, partial algebra, numeric consequences, and complete unsplit fallback.

**Freeze rule:** Freeze the smallest pointer-first semantic contract needed for provider fan-out. Add adjacent versions rather than mutating frozen wire formats. Consumers record exact version/hash/source receipts.

## JBC-I07 — Atom-fragment request and candidate frontier

**Owner:** Cellerator

**Existing substrate:**

- include/Cellerator/compute/operation/candidate_catalog_v3/
- include/Cellerator/execution/program/program_v2.h
- include/Cellerator/geometry/compiler/v2/

**Proposed location:** `[proposed] include/Cellerator/execution/atom_fragment/fragment_v1.hh`

**Semantics:** Atom-bound operation request plus bounded Pareto alternatives, exact cover, requirements/affordances, program recipe, resources, local complete cost, and verifier receipt.

**Freeze rule:** Freeze the smallest pointer-first semantic contract needed for provider fan-out. Add adjacent versions rather than mutating frozen wire formats. Consumers record exact version/hash/source receipts.

## JBC-I08 — Multi-extent external binding

**Owner:** Cellerator

**Existing substrate:**

- include/Cellerator/execution/geometry_acquisition_v2/external_payload.hh
- include/CellShard/runtime/residency/

**Proposed location:** `[proposed] include/Cellerator/execution/object_binding/external_binding_v1.hh`

**Semantics:** Ordered extents with address space, offsets, alignment, order, generation, readiness and lease tokens; raw runtime handles are not durable.

**Freeze rule:** Freeze the smallest pointer-first semantic contract needed for provider fan-out. Add adjacent versions rather than mutating frozen wire formats. Consumers record exact version/hash/source receipts.

## JBC-I09 — Lowering-resumption contract

**Owner:** Cellerator

**Existing substrate:**

- include/Cellerator/execution/geometry_acquisition_v2/
- include/Cellerator/geometry/persistence/semantic_geometry_image_v1.hh
- include/Cellerator/geometry/persistence/execution_image_v2.hh

**Proposed location:** `[proposed] include/Cellerator/execution/lowering_resumption/resumption_v1.hh`

**Semantics:** Canonical, evidence, semantic atom/basis, target cover, projection, packed operand, executable recipe, topology-linked and resident stages with validation, fallback and phases-bypassed.

**Freeze rule:** Freeze the smallest pointer-first semantic contract needed for provider fan-out. Add adjacent versions rather than mutating frozen wire formats. Consumers record exact version/hash/source receipts.

## JBC-I10 — External global cost vector

**Owner:** Cellerator

**Existing substrate:**

- include/Cellerator/planner/
- include/Cellerator/geometry/compiler/v2/workload_profile.hh

**Proposed location:** `[proposed] include/Cellerator/planner/external_cost/external_cost_v1.hh`

**Semantics:** Generic storage, build, memory, movement, canonicalization, combine, replication, invalidation, latency and throughput prices; never overrides correctness.

**Freeze rule:** Freeze the smallest pointer-first semantic contract needed for provider fan-out. Add adjacent versions rather than mutating frozen wire formats. Consumers record exact version/hash/source receipts.

## JBC-I11 — Atom-aware execution export v2

**Owner:** Cellerator

**Existing substrate:**

- include/Cellerator/profiling/partition_export.h

**Proposed location:** `[proposed] include/Cellerator/profiling/joint_compiler/execution_export_v2.hh`

**Semantics:** Exact coverage, decomposition, requirements/affordances, partial algebra, persistent orders, stage graph, candidate frontier, complete local cost, compatibility and freshness.

**Freeze rule:** Freeze the smallest pointer-first semantic contract needed for provider fan-out. Add adjacent versions rather than mutating frozen wire formats. Consumers record exact version/hash/source receipts.

## JBC-I12 — Common atom envelope

**Owner:** CellShard

**Existing substrate:**

- include/CellShard/artifact/image.hh
- include/CellShard/artifact/catalog.hh
- include/CellShard/artifact/snapshot.hh

**Proposed location:** `[proposed] include/CellShard/compiler/atom/atom_v1.hh`

**Semantics:** Level-relative atom identity, species, exact coverage, typed ports, planes, dependencies, evidence, affordances and lineage; candidate/certified/basis/physical/replica/partial/super/resident states remain distinct.

**Freeze rule:** Freeze the smallest pointer-first semantic contract needed for provider fan-out. Add adjacent versions rather than mutating frozen wire formats. Consumers record exact version/hash/source receipts.

## JBC-I13 — Atom evidence atlas

**Owner:** CellShard

**Existing substrate:**

- include/Cellerator/geometry/support_atlas.hh

**Proposed location:** `[proposed] include/CellShard/compiler/evidence/atlas_v1.hh`

**Semantics:** Overlapping approximate proposal evidence, provenance, strata, confidence/stability, negative evidence and exact-rescan status; never execution ownership.

**Freeze rule:** Freeze the smallest pointer-first semantic contract needed for provider fan-out. Add adjacent versions rather than mutating frozen wire formats. Consumers record exact version/hash/source receipts.

## JBC-I14 — Composition production and derivation

**Owner:** CellShard

**Existing substrate:**

- include/Cellerator/compute/operation/relation_algebra_v2/composition.hh
- include/CellShard/artifact/image.hh

**Proposed location:** `[proposed] include/CellShard/compiler/composition/production_v1.hh`

**Semantics:** Typed inputs/output, parameters, exact coverage equation, identity/order/generation rules, contribution semantics, cost, invalidation, verifier and multi-parent derivation DAG.

**Freeze rule:** Freeze the smallest pointer-first semantic contract needed for provider fan-out. Add adjacent versions rather than mutating frozen wire formats. Consumers record exact version/hash/source receipts.

## JBC-I15 — Biological execution basis manifest

**Owner:** CellShard

**Existing substrate:**

- include/CellShard/artifact/snapshot.hh
- include/Cellerator/geometry/compiler/v2/workload_profile.hh

**Proposed location:** `[proposed] include/CellShard/compiler/basis/basis_v1.hh`

**Semantics:** Selected atoms/productions for workload-family distribution, budgets, objective vector, memberships, no-basis fallback, dependencies, validity and cost freshness.

**Freeze rule:** Freeze the smallest pointer-first semantic contract needed for provider fan-out. Add adjacent versions rather than mutating frozen wire formats. Consumers record exact version/hash/source receipts.

## JBC-I16 — Persistent partial atom

**Owner:** CellShard with Cellerator algebra

**Existing substrate:**

- include/Cellerator/compute/operation/relation_algebra_v2/
- include/Cellerator/execution/projection_value_plane/

**Proposed location:** `[proposed] include/CellShard/compiler/partial/partial_atom_v1.hh`

**Semantics:** Exact dependency closure, contribution coverage, merge/finalize algebra, numeric policy, generation, persistence legality and profitability.

**Freeze rule:** Freeze the smallest pointer-first semantic contract needed for provider fan-out. Add adjacent versions rather than mutating frozen wire formats. Consumers record exact version/hash/source receipts.

## JBC-I17 — Atom-store generation and physical instance

**Owner:** CellShard

**Existing substrate:**

- include/CellShard/artifact/extent.hh
- include/CellShard/artifact/catalog.hh
- include/CellShard/artifact/snapshot.hh
- include/CellShard/io/pack/image_envelope.hh

**Proposed location:** `[proposed] include/CellShard/artifact/atom_store/`

**Semantics:** Immutable root, atom dictionary, indexes, grammar/bases/partials, actions/lineage, large arenas and frames, encoded replicas, atomic publication, recovery, consolidation and GC.

**Freeze rule:** Freeze the smallest pointer-first semantic contract needed for provider fan-out. Add adjacent versions rather than mutating frozen wire formats. Consumers record exact version/hash/source receipts.

## JBC-I18 — Global operation provider and portable schedule

**Owner:** CellShard

**Existing substrate:**

- include/CellShard/runtime/
- include/Cellerator/execution/program/program_v2.h

**Proposed location:** `[proposed] include/CellShard/compiler/graph/provider_v1.hh and schedule_v1.hh`

**Semantics:** Provider-neutral typed operations/effects, atom flow, graph families, rewrites, local fragment alternatives, partial trees, replay modes and machine-independent schedule.

**Freeze rule:** Freeze the smallest pointer-first semantic contract needed for provider fan-out. Add adjacent versions rather than mutating frozen wire formats. Consumers record exact version/hash/source receipts.

## JBC-I19 — Exact distributed certificate

**Owner:** CellShard

**Existing substrate:**

- include/Cellerator/geometry/relation_cover.hh
- include/Cellerator/execution/projection_value_plane/composite_plane_v1.hh

**Proposed location:** `[proposed] include/CellShard/compiler/certification/distributed_certificate_v1.hh`

**Semantics:** Exact inputs, owners, contributors, halos, replicas, partial algebra/tree, generations, effects, canonical recovery and duplicate/omission proof.

**Freeze rule:** Freeze the smallest pointer-first semantic contract needed for provider fan-out. Add adjacent versions rather than mutating frozen wire formats. Consumers record exact version/hash/source receipts.

## JBC-I20 — Topology realization, transport and residency lease

**Owner:** CellShard

**Existing substrate:**

- include/CellShard/runtime/source/
- include/CellShard/runtime/residency/
- include/Cellerator/runtime/

**Proposed location:** `[proposed] include/CellShard/runtime/v2/`

**Semantics:** Topology-linked placement, sources, routes, resident atom planes, ready events, lease/pin lifetime, reconstruction, runtime command graph and no durable raw handles.

**Freeze rule:** Freeze the smallest pointer-first semantic contract needed for provider fan-out. Add adjacent versions rather than mutating frozen wire formats. Consumers record exact version/hash/source receipts.

# Ownership map

## Cellerator

Owns typed biological axes and relations, canonical structure/edge identity, exact local coverage and decomposition, partial-result algebra, numerical policy, local projections/packing, local transforms, atom-aware fragment compilation, local candidates/resources, projection-primary values/gradients, lowering validation, and prepared execution.

## CellShard

Owns atom evidence across operations/workloads/datasets, atom envelope and species, exact global certification, composition grammar, bases, superatoms, partial persistence, physical materialization, global graph/schedule, storage, topology, residency, transport, collection, recovery, and cross-run lineage.

## Explicit nonownership

CellShard does not own Cellerator mathematics. Cellerator does not own CellShard global storage/placement. Neither owns model objectives, optimizer policy, pseudotime inference, or Baseplane science in this program.

# Cross-authority interface protocol

The later Todo Orchestrator plan should create an owner task in the owning authority, a freeze checkpoint, and a mirrored consumer receipt task. The receipt records interface ID, version, content hash, source commit, Todo revision, and paths. A matching interface name is not sufficient. The parent submodule pointer is advanced only after the CellShard commit is integrated and pushed.

# Source preservation and migration map

## Cellerator preserve/elevate

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

## Cellerator compatibility-only or migration

- include/Cellerator/interop/cellshard/access.cuh
- include/Cellerator/runtime/multi_gpu/fleet.cuh (standalone local support remains; global authority moves downstream)
- CPK1/CP-BP v1 semantics and wire bytes
- Cellerator examples/models directly consuming legacy CellShard sharded<T>

## CellShard preserve/elevate

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

## CellShard compatibility-only or replacement after migration

- include/CellShard/runtime/layout/sharded.cuh
- include/CellShard/runtime/distributed/distributed.cuh
- include/CellShard/runtime/device/sharded_device.cuh
- include/CellShard/runtime/storage/shard_storage.cuh
- include/CellShard/io/csh5/api.cuh
- include/CellShard/io/pack/execution_payload.cuh
- CSH5/CSPACK/CPEXEC01 direct hot execution paths

# Integration-only files and contention

Likely integration-only Cellerator paths: root `CMakeLists.txt`, `components/README.md`, umbrella headers, central provider/candidate registries, package exports, and top-level test/bench CMake. Likely integration-only CellShard paths: root `CMakeLists.txt`, `include/CellShard/CellShard.hh`, package exports, central discovery/codec/transport registries, final format registration, and top-level test CMake.

All provider lanes should emit source-linked fragments in isolated directories. They must not edit central registries directly.
