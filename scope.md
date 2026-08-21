# Cellerator Scope Boundary

Last updated: 2026-08-21

## Purpose

Cellerator is CellStack's performance-critical biological execution core. It
exploits repeated, modular, hierarchical, and correlated biological structure
when that structure reduces execution time, memory movement, synchronization,
preparation, communication, or whole-module work. It is not a biological
wrapper around generic sparse matrix multiplication.

The central model is a biological program over typed domains and relations:
sequence predicates, regulatory elements, genes and transcripts, cells,
modules, pathways, learned state, and downstream behavior. SpMM remains one
operation family, not the system ontology. Conventional CSR, SELL, BSR,
Blocked-ELL, dense, and vendor-library paths remain first-class candidates and
must win whenever measured end-to-end cost is lower.

## Core Contracts

Cellerator owns these versioned foundations:

- biological domain, order, semantic geometry, partition, structure,
  structure epoch, value generation, and projection identity;
- heterogeneous dense, bit-plane, event, segment, sparse-relation, and small
  parameter operand views;
- immutable relation structures separated from mutable value planes and
  per-launch bindings;
- execution-order contracts that preserve compatible internal order and make
  canonicalization an explicit graph operation;
- one execution session with structure-persistent, plan-persistent, and
  stream-ordered launch-transient storage;
- an operation registry shared by native kernels, composed paths, and vendor
  libraries;
- a projection catalog and planner that charge all material preparation,
  conversion, execution, order, synchronization, and communication cost;
- pointer-free relocatable execution images with versioned projection
  directories.

Semantic geometry describes stable execution-relevant organization. A physical
projection describes concrete bytes and scheduling for an operation, numeric
policy, device class, and reuse regime. One geometry may have many projections;
no universal layout is presumed.

## Repository Ownership

- Cellerator owns computational meaning, biological identity, numerical
  interpretation, projection policy, operation preparation, execution-order
  policy, and end-to-end planning.
- CellPack owns Cellerator's packing and execution-image construction. CP-BP v1
  plans, row order, tiles, canonical recovery, and CPK1 remain validated
  compatibility artifacts behind adapters.
- Baseplane owns packed sequence, validity-aware bit logic, exact motif and
  grammar predicates, masks, events, segments, and low-level sequence-program
  preparation. Baseplane consumes a small Cellerator-owned ABI and does not own
  a separate numerical runtime. Cellerator owns materialize-versus-fuse
  decisions and the first genuinely numerical sequence-to-state interaction.
- CellShard owns storage, pack publication, fetch, delivery, staging, and
  distributed placement. Its CSPACK01/CPEXEC01 envelope treats the inner
  Cellerator image as opaque. CellShard does not choose or interpret physical
  execution projections.
- CelleraTorch owns Torch/libtorch adapters. Framework wrappers call stable
  native operations and do not define Cellerator's hot execution model.

## In Scope

- Domain-aware relation execution across cells, genes, regulatory elements,
  sequence coordinates, modules, pathways, graphs, and learned state.
- Structure-aware packing, semantic geometry, projection construction, value
  remapping, and explicit order transforms.
- Native and library-backed sparse/dense operators, reductions, transforms,
  graph operations, quantized operations, and sequence-state fusion.
- Forward and transpose projection contracts, sparse learned values, mixed
  precision, and quantization mechanisms when activated by concrete work.
- Real-data and adversarial evidence that can select a conventional fallback.
- Device-fleet and nested partition identities needed by future multi-GPU and
  multi-node work, without burdening every hot record with unused metadata.
- Preprocessing and search kernels when they are reusable layout-aware compute
  primitives. Human-facing workflow policy remains thin and separate.

## Out Of Scope

- Durable storage, publication, fetch policy, or interpretation of CellShard's
  outer envelopes.
- A generic sparse matrix package with no biological execution advantage.
- A second CP-Math runtime, owned prepared-operation stream, hidden allocation,
  or generic-SpMM-only planner.
- Mandatory canonical order between compatible internal operations.
- Mandatory dense, CSR, event-table, or matrix materialization at the Baseplane
  boundary.
- Expanding Torch-linked code as Cellerator core.
- Placeholder implementations for hypothetical architectures, distributed
  execution, backward kernels, or precision modes without a current consumer.

## Compatibility Rules

- Old persisted meanings are immutable. New objectives, identities,
  projections, or execution images require new versions.
- Preserve validated v1 objects through read-only adapters before considering
  rewrites or removal.
- CPK1 remains loadable; CPEXEC01 remains the CellShard-owned opaque envelope.
- Pointer addresses never define semantic identity.
- A prepared operation may freeze semantics, structure, projection, algorithm,
  descriptors, and declared workspace requirements. Inputs, outputs, mutable
  values, scalars, stream, and transient workspace remain launch bindings.
- Hot run paths perform no allocation, discovery, hashing, device selection,
  descriptor construction, or synchronization.

## Agent Rules

- Read this file, `optimization.md`, and `planning_strategy.md` before changing
  execution ABI, runtime, packing, projection, planner, or Baseplane seams.
- Preserve execution order unless an external or incompatible consumer
  requires an explicit transform.
- Treat the configured sparse-layout default as legacy compatibility policy,
  not planner authority.
- Record data movement, persistent bytes, transient workspace, preparation
  break-even, and order-transform cost for performance claims.
- Do not edit CellShard to accommodate an unproven Cellerator layout change;
  stop at an external interface decision if CPEXEC01 cannot remain opaque.
- Use `out_of_scope_inventory.md` for unrelated ownership migration rather than
  normalizing scope drift.
