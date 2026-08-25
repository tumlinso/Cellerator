# Cellerator and Baseplane sequence integration v3

This interface consumes frozen Baseplane sequence-predicate v1 and the
Cellerator biological ABI. Baseplane remains the owner of packed sequence,
validity, normalized motifs, portable predicate programs, and scalar/CUDA
references. Cellerator owns axis identity, relation semantics, numerical edge
values, strategy selection, execution order, stream binding, and the
sequence-to-regulatory-to-gene operation.

The first precompiled primitive accepts one verified forward exact-motif mask
program at the exact `baseplane::seq::sequence_predicate_abi_version` required
by Cellerator. Unsupported programs fail preparation rather than entering a
general device interpreter. A prepared operation freezes the Baseplane semantic
hash, motif, coordinate-to-regulatory relation, regulatory-to-gene relation,
both structure epochs, and output contracts. Input/output/value pointers, value
generation, stream, and transient workspace remain launch bindings.

Three strategies share one operation contract:

- `materialize_mask` writes a caller-owned one-bit predicate mask, then joins
  sorted non-overlapping regulatory intervals and accumulates weighted
  element-to-gene edges. The mask can be retained by compatible consumers.
- `fuse_predicate` evaluates validity and the motif while joining and
  accumulating, avoiding predicate-mask traffic.
- `materialize_relation` writes a device-resident regulatory-element local id
  (or an invalid sentinel) for every local coordinate. This is the direct
  relation-builder output: it carries coordinate domain/order in its output
  axis and removes the interval join from later dynamic cell-state uses.

Automatic selection requires comparable evidence and uses the complete reuse
horizon: fused cost is charged for every cell state, while direct relation
materialization charges its first build/consume cost once and its cached
consumer cost for later states. Missing, stale, noisy, or build/device-mismatched
evidence requests empirical measurement instead of guessing. All paths operate on device-resident Baseplane
planes, explicit validity, typed biological axes, and caller-owned streams.
They allocate nothing, synchronize nothing, perform no host round trip, create
no dense motif tensor, and build no CSR intermediate. Floating point begins at
the element-to-gene value plane.

The biological chain is represented honestly as coordinate axis to regulatory
axis to gene axis. Mutable weights belong only to the regulatory-to-gene
relation. Preparation validates both relation identities, axes, epochs, logical
counts, and the weight binding; the fused physical projection may contain both
indexes without inventing a coordinate-to-gene relation or duplicating values.

Validity is authoritative at this seam. Canonical A/C/G/T/U symbols encode with
validity present; every other symbol is invalid. A raw two-bit payload cannot
represent ambiguity, absent validity is rejected, and an invalid payload value
of zero must never become biological A evidence. Predicate semantic hashes are
validated at preparation and stale hashes fail.

Gene-state output declares `accumulate` because the fused consumer adds into
caller-initialized numerical state. Materialized sequence outputs declare
`overwrite`; no strategy silently zeroes gene state. Materialization versus
fusion remains a planner decision, and the legacy mask remains available for
consumers that need a predicate bit plane.

The direct relation cache is mutable session state, never prepared state. A new
regulatory value generation can reuse the same static sequence relation; a new
sequence generation, coordinate order, coordinate structure, or regulatory
projection cannot. Its readiness event permits same-device stream reuse without
a host synchronization boundary. Pointer rebinding requires a fresh cache
entry, but pointer identity never determines semantic reuse.

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
