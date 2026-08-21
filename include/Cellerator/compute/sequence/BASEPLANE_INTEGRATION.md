# Cellerator and Baseplane sequence integration v1

This interface consumes frozen Baseplane sequence-predicate v1 and the
Cellerator biological ABI. Baseplane remains the owner of packed sequence,
validity, normalized motifs, portable predicate programs, and scalar/CUDA
references. Cellerator owns axis identity, relation semantics, numerical edge
values, strategy selection, execution order, stream binding, and the
sequence-to-regulatory-to-gene operation.

The first precompiled primitive accepts one verified forward exact-motif mask
program. Unsupported programs fail preparation rather than entering a general
device interpreter. A prepared operation freezes the Baseplane semantic hash,
motif, immutable interval/element-to-gene projection, structure epoch, and
output execution order. Input/output/value pointers, value generation, stream,
and transient workspace remain launch bindings.

Two strategies share one operation contract:

- `materialize_mask` writes a caller-owned one-bit predicate mask, then joins
  sorted non-overlapping regulatory intervals and accumulates weighted
  element-to-gene edges. The mask can be retained by compatible consumers.
- `fuse_predicate` evaluates validity and the motif while joining and
  accumulating, avoiding predicate-mask traffic.

Automatic selection uses fusion for one-shot execution and materialization
when predicate reuse is declared. This is an explicit bounded policy, not the
future empirical planner. Both paths operate on device-resident Baseplane
planes, explicit validity, typed biological axes, and caller-owned streams.
They allocate nothing, synchronize nothing, perform no host round trip, create
no dense motif tensor, and build no CSR intermediate. Floating point begins at
the element-to-gene value plane.

The current interval projection is sorted and non-overlapping. Overlapping
regulatory targets, allowed motifs, predicate DAGs, stable event composition,
segments, and broader density strategies require separately registered
precompiled primitives and evidence; they do not silently widen v1.

Imported interfaces:

- `cellerator-biological-abi-v1`:
  `708c359577f347cc2f6540aab378c152d2f67386a563ad47e0a3ed901f2eb272`
- `baseplane-sequence-event-v1`:
  `c9ec0c0210e4b41eee2748ad8d79479779db6164870393919089aecbcb8ad2bd`
- `baseplane-sequence-predicate-v1`:
  `4c422b44c2f73aa0e9c0c4ef93c90b9a30977a9968500cf191b85d58b602a685`

The runtime dispatch pointer is process-local prepared state and is never
persisted. `DeviceMathContext`, arbitrary persisted function pointers,
CellShard image interpretation, and Baseplane-owned numerical policy are not
part of this boundary.
