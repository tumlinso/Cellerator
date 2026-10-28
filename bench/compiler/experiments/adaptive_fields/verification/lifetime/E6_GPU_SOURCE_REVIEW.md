# E6 GPU lifetime source review

Date: 2026-10-28
Scope: static source review only; no GPU command was run by this verifier.
Source revision: `c63f591d04ef74b7566937d161c80936dcc4cf05`.

## Verdict before runtime evidence

**Partial support, synthetic fixture.** The source implements a plausible host-side ownership protocol around actual CUDA driver modules and functions. On its normal path, explicit event ordering keeps the old executable leased through the old forward and backward kernels, and a generic fallback is selected after the structure epoch changes. This is not evidence about prepared relation projections, their buffers, or the Cellerator runtime.

Acceptance of the blocked-callback/module-load overlap and sanitizer safety remains **unresolved** until the root-owned admitted run finishes. The replacement is loaded with `cuModuleLoadData` while stream B's host callback is intentionally blocked. Source inspection cannot establish whether this driver/toolchain combination completes that load without waiting for stream progress.

## Ownership and ordering represented in the code

- `executable` owns a `CUmodule` and its forward/backward `CUfunction`s. `plan`, `plan_store`, local launch leases, and a forward tape use `shared_ptr` ownership.
- A callback in stream B sets `entered` and waits on a condition variable. An event recorded after that callback gates stream A. The old forward is queued on A, replacement PTX is compiled on a host worker, and replacement module loading/publication occurs on main while the old launch is pending.
- After the gate releases, A executes old forward then old backward; B executes the replacement. Recorded completion events are synchronized before the old tape/launch leases are reset. This is the key module-retirement proof intended by the fixture.
- CUDA APIs are not called from the host callback. NVRTC runs on a host worker; driver module load and kernel launch run on the main thread with the checked current context.
- A failed compile and a pre-NVRTC cancellation leave the active plan pointer unchanged. A stale structure epoch is rejected by `plan_store::publish`, and `select()` chooses the epoch-generic fallback.

The source order supports these claims, but the independent verifier has not seen the E6 GPU result or sanitizer logs yet.

## Runtime evidence received

The first root-owned run, scheduler job `81de9963-f128-4915-a956-4282acf9ff19` (attempt `a424972d-be33-48f0-8dc3-7f7d0feaaaae`), completed the E7 compiler runner's memcheck/racecheck/synccheck with zero reported errors/hazards, then stalled in E6 without a result or stage log. The root canceled it; the terminal record reports `state=cancelled` and `cleanup_verified=true` in `evidence/scheduler/runtime_sanitizers_v2_terminal.json`. This attempt provides no E6 runtime pass.

The diagnostic v3 scheduler job `ca84c045-076d-441c-87d8-51acc24f3311` / attempt `6f6c1605-5868-4194-9151-aa49315743a7` completed with verified cleanup. It used runner SHA-256 `8d2e90be851c7e87b8292f8716a34e6ed961a3a8e83e6772999d22b8df9c7806` and source SHA-256 `02305c68cc4e245f7e5c1787cc0dce7bcfa10e04ea0c7b908b3dff2871b14888`. Its E6 lifetime memcheck, racecheck, and synccheck all completed with zero errors/hazards/warnings; the results are under `evidence/runtime_lifetime_v3/`. However, every sanitizer log shows the same order: callback entered at about 24 ms, replacement `module_load_begin` at 38–39 ms, callback watchdog released the gate at 10,025 ms, and only then `module_load_end` appeared at about 10,025 ms (`elapsed_ms` about 9,987). The result JSON reports `callback_watchdog_timed_out=true` and its trace explicitly says the module-load overlap may serialize.

This does **not** pass the requested successful-replacement-while-old-work-is-in-flight scenario. The staged evidence shows module loading did not complete during the blocked callback; release of the gate unblocked stream A before publication proceeded. No event query or marker proves whether the tiny old kernel had finished by the publication point, so the strongest supported statement is that overlap was not demonstrated and the path is consistent with module-load serialization. The result JSON's unconditional `successful_replacement_with_pending_old_launch=true` and trace text overstate the scenario and must not be used as acceptance evidence. The zero-error sanitizer reports apply only to the watchdog-released, serialized path.

An unsanitized run of the same v3 binary confirms the delay is not introduced by Compute Sanitizer. Scheduler job `e32938ad-64b2-4e50-bd31-e14636b5c77b` / attempt `9cb3046e-9e75-4700-8f05-5e69e8cd21ca` succeeded with verified cleanup. Its markers show replacement `module_load_begin` at 37 ms, watchdog release at 10 seconds, and `module_load_end` at 10,026 ms (`elapsed_ms=9,988`). This reproduces the same gated load delay without instrumentation. It supports a claim about this binary, driver, context, and blocked-callback setup; it does not establish universal CUDA driver serialization behavior.

The v3 checks do support the narrower synthetic claims that the old module stayed retained through the later tape backward and was released after its completion event, and that the tiny kernels produced their exact host-reference values. Those assertions do not establish real training-tape or prepared-projection ownership.

## Material limits and untested paths

1. **Potential callback/load deadlock and callback cleanup.** The replacement path loads a module before releasing the host callback. A load failure or exception before `release_gate(stream_gate)` unwinds the `shared_ptr<compile_gate>` while the CUDA callback still holds its raw `get()` pointer and is waiting. The catch returns from `main`, but does not release that gate or synchronize/destroy the streams. This failure path can hang or leave the callback using destroyed host memory. The injected NVRTC failure happens later, after the stream gate is released, so it does not cover this overlap failure.

2. **Duplicate assertion is weak by itself.** The check compares only whether the two `shared_future`s are valid; two independent futures would both pass. The source does look up and return the existing `compile_request` for a duplicate key, and increments `duplicate_requests_`, so source control flow supports coalescing. A runtime compile count and identical future result would be stronger independent evidence.

3. **Failure/cancellation keys stay cached.** `requests_` retains every key and shared future. A retry after a failed or canceled request returns the same exceptional future, rather than scheduling a new compile. A successful request may intentionally serve as a cache entry, but failure/cancel retry policy and bounded cache lifetime are not exercised here.

4. **Structure identity is synthetic metadata.** `FIELD_STRUCTURE_EPOCH` is supplied as an NVRTC define but is not referenced by `lifetime_kernels.cu`; kernels only multiply runtime input by runtime scale. Structure staleness is tested by plan metadata and a manually supplied string key. It does not demonstrate rejection of an executable specialized for an actual changed relation, projection, or physical order.

5. **No production tape or buffer integration.** The tape is a local `shared_ptr<executable>`, not a training autograd tape. Device buffers are scalar-kernel fixture buffers; there are no prepared projections, worksets, actual relation values, or production execution session handles.

6. **Retirement counter is not an error-checked unload receipt.** `executable::~executable()` ignores the return value from `cuModuleUnload` and increments `g_unloaded_modules` regardless. The host event/lease assertions are more meaningful than that count; sanitizer/runtime outcomes are still needed.

7. **Numerical coverage is deliberately narrow.** Eight-element FP32 multiply kernels have exact simple host references. This says nothing about recurrent training arithmetic or biological workload semantics.

## Sanitizer runner review

`runtime_sanitizers.py` checks the runner, generated device artifact, source, and lifetime runner/kernel hashes before execution. It runs memcheck, racecheck, and synccheck serially and preserves command/stdout/stderr records. It stops at the first failed check. Scheduler admission is carried by the external scheduler receipt; the marker, correlation key, and UUID route are accidental-run/correlation guards only. The previous first attempt stopped at its admission guard before any GPU command, and is not sanitizer evidence.

The sanitizer script invokes the lifetime binary directly rather than through `run.py`, but the binary itself checks the admission marker, one UUID route, and the JobSpec correlation key. The `run.py` wrapper additionally validates the build/source hashes and records correlation details for a normal non-sanitizer run. Neither guard establishes scheduler authority on its own.

## Frozen source identities

```text
v2 lifetime_runner.cu
5c597670c62fd2e6e4fef659f4af1395859ff2ee3efa1654a7e780277127f46f

v3 diagnostic lifetime_runner.cu
02305c68cc4e245f7e5c1787cc0dce7bcfa10e04ea0c7b908b3dff2871b14888

lifetime_kernels.cu
004372efeb1158eacb763975f9ef94aa40d99edd5d5f7e7fd97f321316f7956a

runtime_sanitizers.py
c9823a8decb1ca9b4a103e4595c279a19cea540309ec871c6fe5203264997b1b
```
