# V100 CUDA runtime and parameter-transfer validation

## Contract and baseline

Validate `DeviceGuard`, streams/events, device/pinned buffers, and
`ParameterTransfer`/`ParameterExecutionLease` on Tesla V100 (`sm_70`). The
runtime supplies typed device ownership and async copies. Parameter upload
copies one exact host-backed immutable parameter revision byte-for-byte to a
device staging buffer; a consumer lease inserts an upload dependency and pins
the transfer and source until downstream stream work completes.

Current baseline is the registered CUDA 12.9 smoke executables and source-level
receipts in `planning/framework-push-2026-10-29/cuda/VALIDATION.md`. Those
receipts establish compilation and host checks, not device execution. The
existing GPU smoke covers an exact-byte upload/readback, revision replacement
while pinned, a consumer stream copy, explicit completion and implicit
completion at lease destruction.

## Risks and unknowns

- No admitted GPU descriptor or Device Scheduler grant is available to this
  owner. Do not run a CUDA executable locally or infer permission from device
  visibility. Root coordinates GPU execution through Device Scheduler.
- The ownership contract depends on event synchronization completing before
  pinned staging, source lease, and device storage are released.
- Device change/restoration, borrowed versus owned stream lifetime, empty
  transfers, repeated/moved tickets, and exceptional enqueue paths need device
  execution evidence. CUDA runtime APIs may surface deferred errors at sync.
- Transfer optimization is relevant only for a material repeated-upload use
  case and must separate resident byte copy, staging/setup, and complete upload
  cost. No kernel optimization is currently justified.

## Validation sequence

1. Build only the runtime/parameter-transfer libraries and their two smoke
   executables for CUDA 12.9, C++17, and `CMAKE_CUDA_ARCHITECTURES=70`.
2. Run both existing host-only modes for argument and ownership preconditions.
3. With a root-provided Device Scheduler grant, run both device modes on the
   exact granted UUID. Verify bytes independently on the host; repeat with
   empty and varied-size payloads, nondefault streams, device guards, event
   ordering, ticket drops, exact revision replacement, and destruction without
   explicit `finish()`.
4. Run Compute Sanitizer memcheck and initcheck on the admitted device smoke.
   Add racecheck/synccheck only where the exercised CUDA mechanisms apply.
5. If the admitted smoke establishes a repeated transfer workload, compare
   warmed pinned async H2D with the existing ticket path and a plain CUDA
   baseline using CUDA events. Report upload-only and complete-path latency and
   throughput, along with payload sizes and repetition count.

## Promotion criteria

All device bytes must exactly equal the source payload; source and device owners
must remain live through the recorded completion; no sanitizer finding may
occur for covered code. Any optimization must preserve those checks and improve
matched complete-path cost reproducibly. If the shared scheduler is unavailable,
retain host/source evidence and report V100 device correctness as unvalidated.
