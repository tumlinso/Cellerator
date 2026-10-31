# V100 CUDA framework qualification, 2026-10-29

## Result

All eleven new CUDA smoke families and the nine-test Python binding suite have
passed their independent device checks on Tesla V100-SXM2-16GB (`sm_70`). Each
path also passed Compute Sanitizer memcheck, racecheck, initcheck and synccheck
with zero reported errors or hazards after the fixture corrections below.
This is bounded numerical and resource-lifetime qualification, not biological
validation, a complete training-system qualification, or a universal speedup.

The base publication is `a5190c168b25e2764ccc4e09e406472ab845cae6`.
Qualification includes the subsequent working-tree fixes and expanded tests;
the accepted evidence manifest binds their source and binary identities.

## Coverage

| Family | Independent checks |
| --- | --- |
| Runtime | Device views, offsets, pinned H2D/D2H, D2D subviews, producer/consumer streams and events. |
| Dispatch | FP16/FP32 gather, weighted combine and both backwards; repeated and idle origins, signed/zero weights, feature tails, empty routes and zero-feature weight gradients. |
| Quantization | 1/2/4/8-bit FP16/FP32 encoding/decoding, reordered features, partial words, exact override prefixes, adjacent packed updates and invalid-input status. |
| Event memory | FP16/FP32 decay and event recurrence at 1/33/257 channels, repeated impulses, lazy reads and duplicate reads. |
| Changed frontier | Exact host set-union oracle, duplicate/overlapping postings, empty work, skewed rows, capacity and malformed-input status. |
| Attention | FP16/FP32 staged and fused forward/VJP; CSR degrees 0/1/2/31/32/33/65; value widths 1/5/31/32/33/65; a 65,536-row block-stride case. Host FP64 finite differences supplement the device comparisons. |
| Grouped linear | Ragged and empty experts, FP16/FP32 forward/dX/dW, overwrite and seeded accumulation. References consume represented input values, accumulate in FP32 and respect output rounding. |
| Packed embedding | All four code widths, valid ragged codebooks, feature remapping, exact interpolation, repeated weighted occurrences, empty bags, FP16/FP32 and analytic gradients. |
| Reaction events | Exact int64 counts, statuses and initialized changed-source log prefixes against an independent sequential interpreter; negative counts, overflow and capacity cases. This applies supplied events, not GPU SSA. |
| Regulatory gates | Occurrence and bitplane paths against scalar predicates, cross-word features, 67-cell tails, duplicates, invalid inputs and capacities. |
| Parameter transfer | Byte identity, exact revision pins, publication/retirement while in use, downstream stream lifetime and zero-byte revisions. |
| Python binding | Six host contract checks plus three real-device integration checks for cross-stream dispatch, grouped-linear gradients and pinned transfers. |

Grouped-linear acceptance uses per-output FP32 accumulation bounds based on dot
length and the sum of absolute products, plus FP16 output rounding. Numerical
tolerances and exact fixtures remain in the source tests; they are not
whole-library precision guarantees.

## Corrections established by real-device testing

- Fixed fused attention's full-warp shuffle inside a dimension-dependent branch.
  Only output-owning lanes entered that branch, violating the shuffle mask.
  Shared probability/key staging and synchronization now preserve participation.
- Fixed zero-feature dispatch weight derivatives: their empty dot products must
  be assigned zero rather than leaving caller output storage untouched.
- Corrected test references before accepting results: an event-memory impulse
  at width one, dispatch directional derivatives and empty-route inputs, grouped
  accumulation seeds and represented FP16 inputs, packed table offsets and
  1-bit codebook limits, and 32-bit regulatory bitplane word counts.
- Corrected quantization and reaction fixtures to download only initialized
  output-log prefixes. The kernels do not promise initialized spare capacity.
- Corrected the experimental cuBLAS grouped-batched comparison's matrix-pointer
  arrays to reside on the device. This was a prototype harness error.

## Execution and reproduction

All GPU work runs through UUID-selected exclusive Device Scheduler grants.
The existing Cellerator-scoped client handles finite correctness/sanitizer
jobs; registered measurements use Autobench source snapshots and quiescence.
No existing claims, unrelated jobs or submodule references were modified.
One device was intermittently unavailable to admission despite idle telemetry;
its queued task-owned jobs were cancelled and rerouted after verified cleanup.

Builds target only SM70 with CUDA 12.9.86 and GCC 12.4. Assertions are retained
with `-UNDEBUG` in Release smoke/benchmark executables. Sanitizers use the
installed CUDA 12.9 executable directly:
`/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/compute-sanitizer/compute-sanitizer`
(version 2025.2.1). The compiler-directory wrapper incorrectly selected an
uninstalled CUDA 13.1 tool and is not accepted as sanitizer evidence.

The [plan](PLAN.md), per-family drafts, registered Autobench definition and
`scripts/validation/` runners retain the contracts and reproduction commands.
Project Control returned `workflow_task_already_active`; existing claims were
preserved rather than replaced.

## Optimization boundary

Work is paused at the user's request. The [continuation checkpoint](PAUSED.md)
records the preserved state and remaining comparisons.

Matched resident and preparation/transfer-aware comparisons remain unfinished.
The packed-embedding warp and frontier reservation alternatives remain
experimental until correctness and measured promotion requirements are met.
The existing default paths remain available. Performance results and retained
choices will be added here after evidence review.
