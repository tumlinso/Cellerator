# Cellerator Planning Strategy

Last updated: 2026-08-21

The biological execution architecture recovery is complete. New work starts
from the frozen identity, lifetime, runtime, projection, planner, and Baseplane
contracts rather than reopening CP-BP v1 or reviving experimental CP-Math.

## Planning Principle

Plan complete biological operations, not isolated formats or kernels. A task
must identify its domains and orders, immutable structure, mutable values,
candidate projections, launch bindings, correctness oracle, full data movement,
workspace, reuse horizon, and measured fallback.

Performance is the governing decision. Biological organization matters when it
creates reusable supports, modules, bounded coherence, locality, scheduling,
caching, communication reduction, fusion, or work skipping. A conventional
layout is the correct result when those advantages do not pay for themselves.

## Stable Foundations

New work consumes these existing seams:

1. The biological ABI distinguishes domain, exact order, semantic geometry,
   partition, immutable structure and epoch, mutable value generation, and
   physical projection. Runtime handles are generation-checked aliases for
   persistent identities, never durable identity themselves.
2. The execution-order contract makes transforms graph-visible and permits
   compatible producers and consumers to stay packed.
3. The execution session owns device facts, streams, library handles, and
   persistent/transient allocation. Persistent objects never alias or move;
   sealed launch binding allocates, discovers, and synchronizes nothing. No
   second math runtime is permitted.
4. CellPack's semantic geometry and execution image v2 support a projection
   catalog while preserving CP-BP v1 and CPK1 through adapters.
5. The operation core represents native, vendor, and composed candidates with
   bounded multi-structure dependencies, explicit output effects, direct
   prepared dispatch, and launch-time bindings.
6. The end-to-end planner separates semantic, structure, projection, device,
   build, and policy keys and can select measured conventional fallbacks.
7. The Baseplane seam supports native sequence operands and explicit
   materialized or fused sequence-to-state execution.
8. CellShard continues to wrap opaque Cellerator bytes in CPEXEC01.
9. Execution Image v2 is validated cold on host and prebound hot from validated
   offsets plus the current host or device image base.

## Activation Workflow

### 1. State the biological operation

Name the source and destination domains, input and output orders, geometry,
structure reuse, value-change rate, numeric policy, and whether forward,
transpose, or backward behavior is required. Do not disguise a new operation as
SpMM merely to reuse an existing backend interface.

### 2. Register capabilities, not a preferred format

Describe correctness limits, projection requirements, persistent preprocessing,
transient workspace, determinism, graph compatibility, output order, and
architecture class for each candidate. Reuse current projections before adding
new bytes. New projections require an identity, schema, validation, ownership,
and construction-cost contract.

### 3. Preserve structure/value/binding lifetimes

Immutable relation structure and semantic geometry outlive mutable value
planes. Value updates do not rebuild structure. Per-launch pointers, values,
scalars, streams, and transient workspace do not enter semantic plan identity.
Stale structure epochs, value generations, orders, and devices fail explicitly.

### 4. Establish correctness before performance

Use independent scalar or logical referees, adversarial dimensions, invalid
identity/order/generation cases, guard regions, and CUDA sanitizer checks where
memory safety is at issue. Preserve accepted CP-BP and CPK1 reconstruction
evidence when an adapter is involved.

### 5. Measure the complete workflow

Separate host preparation, semantic packing, projection construction, H2D,
dynamic packing, kernel, epilogue, order transform, synchronization,
communication, D2H, persistent bytes, and transient workspace. Distinguish
one-shot, bounded reuse, and persistent reuse. Use the CUDA background
controller for serialized benchmarks and deep profiles.

### 6. Promote only measured winners

The planner rejects illegal candidates, analytically ranks the remainder,
empirically measures a bounded shortlist when reuse warrants it, records the
winner and confidence, and rejects stale evidence. Native Cellerator formats
receive no unmeasured preference over CSR, SELL, BSR, valid Blocked-ELL,
cuSPARSE, or dense cuBLAS.

## Baseplane Work

Baseplane-local validity, coordinate, ambiguity, strand, predicate, event, and
segment work remains Baseplane-owned. Cross-library work uses the frozen common
ABI and stays lazy:

- materialize masks/events/segments when static sequence results will be reused;
- fuse predicates into regulatory or gene-state operations when avoiding an
  intermediate wins end to end;
- remain bitwise/integer until affinity, activity, occupancy, enhancer strength,
  learned weight, or another quantitative interaction requires floating point.

Do not introduce a host event-table requirement, dense motif tensor, mandatory
CSR relation, arbitrary persisted device function pointer, or Baseplane-owned
numerical runtime.

## Future Compatibility

Transpose/backward projections, sparse-value gradients, mixed precision,
module quantization, CUDA Graphs, persistent CTAs, work queues, nested GPU/node
partitions, and architecture-specific kernels are activation-gated. They may
use optional image sections and candidate capabilities already reserved by the
foundations. They do not justify placeholder kernels or metadata in every hot
record.

## Pickup Checklist

Before claiming implementation work, answer:

- What biological operation exists when the task is done?
- Which identities and orders are consumed and produced?
- Which structure, value, prepared, and launch lifetimes change?
- Which existing projection and compatibility adapters are reused?
- What conventional fallback is legal?
- What adversarial case should make the native approach lose?
- What are the persistent, transient, transfer, and order-transform costs?
- What evidence and checkpoint allow the next task to proceed?
- Does the work preserve standalone Baseplane and opaque CellShard CPEXEC01?

If these answers are absent, the task is not ready for implementation.
