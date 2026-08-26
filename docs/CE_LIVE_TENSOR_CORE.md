# CE-LIVE bounded V100 Tensor Core candidate contract

`CELLERATOR_TENSOR_CORE_CONTRACT_READY` defines one optional candidate lane. It
does not implement a kernel, register a candidate, alter `operation_candidate`,
add a hot semantic-geometry field, or claim a performance win.

The lane is a Volta `sm_70` dense-fragment projection for sparse-dense multiply.
It preserves the CE-LIVE-11 logical relation—feature source to row or module
destination—and shares structure identity, epoch, and logical edge indices with
CPK1, FMP1, CSR, and CTP1 projections. The machine-readable design is
`bench/ce_live/tensor_core/contract/v100_dense_fragment_candidate_v1.json`.

## Why the lane is bounded

The current CE-ARCH-92 V100 corpus has 36 correct measurements but no Tensor
Core candidate. At eight-use amortization, CSR wins the full real/adversarial
N=1 regimes, row-masked wins one high-sharing N=1 regime, feature-major wins
N=16, and feature-major CTA wins N=32. Those results establish strong baselines
and projection plurality; they do not imply that density alone will make WMMA
win. Full PBMC3K and GSE147520 block-width-32 scalar occupancies are only about
5.1% and 7.8%, respectively, so whole-structure densification is specifically
rejected.

The dominant expected risks are padded FLOPs and generation packing at low
density, HBM traffic for fragment and dense-input panels, and backward/residual
accumulation. Tensor throughput is useful only after those costs are paid.

## Candidate geometry and numeric contract

One qualified physical tile represents a 16-destination by 16-source relation
fragment. A WMMA operation combines that relation tile with a 16-source by
16-dense-column panel and accumulates a 16-destination by 16-column FP32 tile.
Absent logical edges are explicit FP16 zeros in projection-local storage.

The only v1 numeric tuple is:

- FP16 relation values and FP16 dense input;
- FP16 multiply with FP32 accumulation and FP32 output;
- nearest-even conversion, no saturation, bias, or quantization;
- explicit f32-to-f16 input packing when the producer supplies f32.

The projection and packed dense panels use at least 32-byte base alignment.
Relation and dense-panel leading dimensions are multiples of 16 FP16 elements;
the accumulator/output leading dimension is a multiple of 16 FP32 elements.
These constraints are stricter than the minimum useful Volta alignment and
make each row and 16-wide tile naturally 32-byte aligned.

This is architecture-specific physical metadata, never a stable semantic ABI.
Ampere or later paths require their own runtime-selected candidate and may not
leak TF32, BF16, sparse-MMA, or asynchronous-copy assumptions into this lane.

## Density classification and qualification

Classification uses exact `nnz / 256` for each 16 by 16 candidate tile. Audit
buckets are empty, low `(0, .25)`, medium `[.25, .5)`, high `[.5, .75)`, and
near-dense `[.75, 1]`. Buckets support evidence stratification; none is an
automatic activation threshold.

CellPack currently counts `capacity >= 16 && occupancy >= 0.5` as a
`dense_fragment_candidate`. That remains a cheap shortlist statistic, not a
kernel-selection rule. The activation threshold must be learned from correct,
serialized measurements for the exact device-performance identity, build,
dense width, tail distribution, and reuse horizon. A fragment qualifies only
when complete amortized cost beats the best legal candidate outside the
planner's practical tolerance.

Detailed occupancy distributions, maps, and break-even evidence remain cold
planner/projection sidecars. No mandatory dense-fragment flag is added to every
hot CellPack record.

## Packing, logical edges, and generations

Immutable projection construction produces:

1. fragment coordinates in existing execution row/feature orders;
2. `logical_edge_to_fragment_slot` and an inverse slot map whose zero-fill
   entries use an invalid sentinel;
3. an explicit transpose schedule with exact logical-to-position and
   position-to-logical maps;
4. fragment density, padding, residual, metadata, and expected-reuse facts.

The forward and transpose schedules share the logical relation and edge IDs.
They do not reverse source/destination semantics. A transpose schedule may load
the same 16 by 16 packed value tile with the opposite matrix layout, but any
tile-order or reduction metadata is explicit and independently validated.

Numerical values remain a mutable value plane. Packing a value generation into
FP16 fragment slots is generation-scoped runtime work and must wait on the
runtime readiness contract without placing streams or events in persistent
identity. A changed value pointer or generation does not rebuild topology, but
its packing cost and readiness dependency remain visible. Reusable packed
generation buffers belong to the runtime/session, not CellPack or a prepared
semantic plan.

## Tails and residual execution

WMMA owns only complete 16 by 16 by 16 work. Dense-column padding to 16 is legal
only when the padded FLOPs, input conversion, output slicing, and reuse are
measured. Row, feature, or dense-column residuals must be assigned to an
explicit legal sibling candidate: row-masked, feature-major warp/CTA, CSR, or a
future measured scalar residual. The residual candidate has its own projection,
order/effect contract, launch count, and planner cost.

A composed WMMA-plus-residual plan must prove disjoint output ownership or an
explicit accumulation effect. It may not silently fall back inside a hot kernel
or require canonicalization between the two paths. Missing residual coverage,
overlapping writes, non-invertible maps, and capacity overflow reject the
candidate before launch.

## Complete planner cost and rejection

The planner charges density classification, projection construction, forward
and transpose maps, persistent metadata, per-generation value packing,
f32-to-f16 dense input packing, transfers, WMMA and residual kernels, epilogue,
output order work, synchronization, communication, padding FLOPs, memory, and
structure/projection/value reuse. Kernel-only time cannot promote the lane.

Reject before measurement or selection when:

- the device is not `sm_70`, the numeric tuple/alignment is unsupported, or
  fragment dimensions are incompatible;
- domain/order/geometry/partition, structure epoch, projection identity, or
  forward/transpose edge maps disagree;
- the required value generation is not visible on the launch stream;
- tails lack a legal explicit residual, capacities overflow, or output effects
  overlap ambiguously;
- empirical break-even evidence is absent or stale;
- independent correctness, generation-rebind, adversarial-tail, canary, or
  Compute Sanitizer validation fails;
- complete amortized cost does not beat the best correct baseline outside the
  declared practical tolerance.

## Future evidence campaign

An implementation task may add a candidate only after an independent CPU
referee and projection-map tests pass. Its serialized V100 campaign should use
the checksum-pinned CE-ARCH-92 real/high-sharing/adversarial traces plus
synthetic tiles spanning every density bucket, empty fragments, 15/16/17 row,
feature, and N tails, and at least two mutable value generations.

Measure N=16, 32, and 64; one-shot and reuse horizons including eight; and
compare row-masked, feature-major warp/CTA, custom CSR, a maintained dense
library path, and the WMMA-plus-residual plan. Record all planner phases,
transfer and synchronization inclusion, bytes, padded/useful interaction
ratio, launch count, registers, shared memory, occupancy, Tensor Core activity,
DRAM behavior, numerical tolerance, sample count, MAD/spread, resource lease,
hardware/toolchain/build identities, and correctness digest. Use correctness
before timing and report inconclusive evidence rather than forcing a winner.

Until that campaign succeeds, `dense_fragment` remains a reserved projection
kind and this candidate remains unimplemented and unregistered.
