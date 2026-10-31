# Python CUDA boundary validation

## Goal

Exercise the optional Python CUDA adapter against real device memory while
using independent NumPy references. Validate cross-stream ordering, pinned
transfer tickets, sparse dispatch outputs and gradients, and ragged grouped
linear forward/backward. The host-only tests remain usable without a GPU.

## Baseline and risk

The installed `_modeling_cuda` host contract suite passes six tests. It does not
initialize CUDA, transfer bytes, execute operations, or test event ordering.
The principal integration risks are lost owners before asynchronous completion,
incorrect stream handoff, device/shape/dtype mismatches, and wrapper arithmetic
disagreeing with its native kernel contracts.

## Contract and references

- Compare dispatch gather, weighted combine, and both backward projections
  against direct NumPy indexing and reductions, including repeated origins and
  signed weights.
- Compare grouped linear forward, input gradients, and every expert's weight
  gradient with NumPy matmul, including an empty expert; exercise FP32 and FP16
  storage with separate tolerances.
- Copy pinned host buffers asynchronously in both directions and wait on each
  returned ticket before checking data.
- Start input arrays on a different stream from grouped execution to exercise
  the wrapper's event-based cross-stream dependency.
- Use the test only under an admitted Device Scheduler job and gate device cases
  behind `CELLERATOR_TEST_CUDA=1`; the test never schedules work itself.

## Performance boundary

This is an integration correctness test, not a benchmark. Root's per-kernel
owners handle resident-kernel and complete-operation optimization evidence.

## Validation and acceptance

Run the existing six host-only tests first. For device acceptance, build/install
the extension for SM70, then execute this file inside an admitted scheduler job
with `CELLERATOR_TEST_CUDA=1` and the scheduler-assigned device ordinal. Require
all NumPy comparisons to pass and all asynchronous tickets to complete. Root
retains scheduler admission and final acceptance.
