# CE-LIVE concurrency and hot-path acceptance

The CE-LIVE N=16 training program passes the adversarial runtime acceptance
suite on one Tesla V100 (`sm_70`). The suite uses the sole Cellerator execution
session with two caller-created nonblocking streams and one immutable prepared
FMP1/CTP1 topology.

## Accepted behavior

- Generation 1 to 2 is enqueued on a producer stream.
- Generation 2 to 3 is enqueued on a different consumer stream without a host
  wait. The established readiness event supplies the cross-stream dependency.
- The same prepared program accepts relocated dense input and output pointers;
  its preparation count remains one across all accepted generations.
- Stale generation readiness, a stale structure epoch, and insufficient
  workspace are rejected before useful work is accepted.
- The explicit caller stream is returned in result metadata for each enqueue.
- Session allocation, device-query, library-handle preparation, and
  synchronization accounting do not change across the hot-path launches.
- The captured fixed-binding step contains all five native training kernels.
  Its readiness event is recorded as part of the captured stream work, the
  graph instantiates, and one graph launch publishes generation 4.

CUDA Graph capture describes one fixed binding and one explicit generation
transition. Blindly replaying that graph as though host-side generation
metadata had advanced again is not supported. A caller must establish a legal
generation/binding contract before capturing another transition. This keeps
runtime readiness explicit instead of pretending graph replay updates
persistent biological identity.

## Hot-path audit

`run_training_program` performs only readiness ordering, validation, dispatch
to the already-prepared native slice, result description, and counters. The
training program and underlying native run contain no hot-path `cudaMalloc`,
`cudaFree`, `cudaSetDevice`, `cudaDeviceSynchronize`, descriptor creation,
structural hash, transfer, or topology construction. Test-side stream
synchronization is used only to inspect completed results.

The concurrency test includes the complete CE-LIVE-33 contract test, then adds
two-stream reuse, stale identity cases, session-accounting assertions, and CUDA
Graph capture/launch. It is reproducible with:

```bash
bash bench/ce_live/concurrency/run_concurrency_test.sh
```

The one-GPU controller evidence is recorded in
`bench/ce_live/concurrency/acceptance_evidence.json`. Compute Sanitizer memcheck
passes the same executable.

## Boundaries

- Structure and value generation remain independent identities.
- Readiness events and streams remain runtime-only state.
- Caller streams are honored; no internal stream redirection is introduced.
- Workspace remains caller-owned and capacity-checked.
- FMP1 and CTP1 remain non-owning typed views over pointer-free projections.
- No Tensor Core, general-N training, Torch, storage, or planner policy is
  introduced by this acceptance task.
