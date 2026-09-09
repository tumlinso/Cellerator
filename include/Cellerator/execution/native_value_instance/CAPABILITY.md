# Native shared relation values, NF1 V07

The compiled owner remains `prepared_relation.cu` and its retained relation,
readiness, sparse projection and gradient providers. `Cellerator::native_value_instance`
links these owners and the existing atom validators. This document describes the
bounded executable capability qualified by `ce_nf1_v07`; it is not a general-width
accelerator or a public stable ABI.

- Device route: CUDA sm70, one owning device and caller stream per instance.
  Forward and transpose accept dense widths 1 and 16, f32 state/result arithmetic,
  and the existing FMA/reassociation/nonfinite contract. Other widths or numeric
  policies receive explicit status. No graph replay contract is qualified.
- Structure: synchronously consumed host CSR is converted once into counted,
  immutable shared projections/maps. New instances retain that actual owner,
  allocate independent values/readiness, and inherit no publication. Checked
  source close leaves siblings usable. Reports expose actual candidate names,
  shared structure IDs/bytes, private value bytes and generations.
- Values: f16 authority preserves the retained arithmetic. Explicit f32 authority
  retains changes smaller than half precision. Optional `derive_f16` allocates a
  separate RNE half projection; explicit derived execution observes that projection.
  Float-only instances allocate no half value buffer. Publication and physical
  updates refresh the requested derived generation in stream order.
- Updates: f16 uses `RNE_half(f32(w)+delta)` or
  `RNE_half(fma(-alpha,g,f32(w)))`. F32 stores the f32 sum or fused result directly;
  an explicitly requested half projection receives RNE of that result. N16 edge
  gradients use retained full-f32 or explicitly half-rounded operand policies.
  Gradient stamps bind instance incarnation, topology, order, input versions and
  generation. Sibling mutation does not invalidate another instance's response.
- Lifetime: the retained external read lease exposes a const physical **f16** plane
  only. Typed f32 borrowing is unsupported. Returned reader events order mutation;
  live borrows prevent close/replacement. Generations are not historical snapshots.
  Epoch replacement prepares first, preserves the old owner on failure, then drains
  and retires that instance while unaffected siblings retain their own epoch.
- Atom association: the current composite API accepts validated compact logical
  f16 value planes and f32 logical gradient outputs, with explicit native/atom axis
  association and retained map validation. Physical-primary and typed f32 atom
  publication are unsupported; ordinary f32 publication is available separately.

`ce_nf1_v07` executes independent formulas, real address-reuse stale-ticket rejection,
source-CSR destruction, sibling survival, delayed reader ordering, pending close,
FP32 precision, both update storage policies and optional projection branches.
Wrong stream/device and captured mutation/preparation preserve output, generation,
counters and an empty capture graph. The launcher runs Compute Sanitizer and all
23 retained `ru1_` tests using their existing source targets, inside the same native
lease and sealed runner lock. It hashes the actual binaries and scripts in the
external child evidence, which is cryptographically linked by the adapter receipt.
No timing-based performance promotion, multi-device execution, broad widths,
physical-primary atom route or full integrated consumer acceptance is asserted here.
