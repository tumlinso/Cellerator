# E6 CPU executable lifetime probe

**Decision: supported for this experiment-local CPU publication protocol.**

The harness compiled and linked real generated shared objects through Cellerator's existing `compile_generated_cpp_object_v1`, then invoked the loaded symbol through `dlopen`/`dlsym`. All scheduling barriers use mutexes and condition variables; there are no sleeps in the tested interleavings.

The old module's exported function calls a host callback and blocks while another generated module is compiled and published. A `shared_ptr` lease captured by the executing call keeps the original module loaded after replacement. Once the call is released and returns, its weak handle expires and the RAII owner calls `dlclose`. This establishes a concrete CPU ownership rule: each invocation and tape-like consumer must retain an executable lease until its work completes.

The same run exercised one compiler worker and one pending replacement slot. A duplicate in-flight request coalesced to the existing request ID; a distinct request occupied the pending slot; a third request was rejected. The pending request was cancelled and never compiled. A replacement remained compatible with a value-generation update because its generated function receives numerical input at call time. A completed compilation for stale structure epoch 7 was rejected after the current epoch advanced to 8, and the host generic fallback remained callable. An injected malformed C++ translation unit failed compilation; the fallback remained active. Completion consumption removed the request key, and explicit retries after successful compilation, cancellation, and compilation failure each received a new request ID and followed their respective path again.

Observed trace:

```text
old_call_result=3 old_code_retired_after_completion=1
replacement_published=1 stale_rejected=1 failed_rejected=1 value_generation_now=5 structural_epoch_now=8 duplicate_coalesced=1 pending_capacity_rejected=1 pending_cancelled=2 retry_after_success_cancel_failure=pass generic_fallback_callable=1 module_close_count=3
```

`module_close_count=3` is measured before process teardown: the old module, stale unpublished replacement, and successful but unused compilation result were retired after their final references were released. The currently published replacement remains leased; other successful retry results retained by local handles are retired at process teardown. Intentionally failed compiles produce no loaded module.

## Reproduction

```sh
g++ -std=c++20 -O2 -Wall -Wextra -Iinclude \
  bench/compiler/experiments/adaptive_fields/cpu/runtime/lifetime_probe.cc \
  src/compiler/backend/compile_generated_c_into_ordinary_objects.cc \
  -ldl -pthread \
  -o bench/compiler/experiments/adaptive_fields/cpu/runtime/artifacts/lifetime_probe

bench/compiler/experiments/adaptive_fields/cpu/runtime/artifacts/lifetime_probe \
  bench/compiler/experiments/adaptive_fields/cpu/runtime/artifacts/modules
```

The two intentional failed-compilation attempts print compiler diagnostics for the deliberately invalid generated source; the harness consumes both failures and exits successfully.

## Boundaries

- This is a CPU-only adapter prototype. It does not implement CUDA stream events, device buffer ownership, cancellation of an already-running compiler process, or a production executable registry.
- The generated module's `float -> float` body stands in for a compiled region. The host generic fallback is a local callable, not a general execution-plan fallback API.
- The harness validates a single consumer lease and one worker with one pending slot. Multi-consumer tapes, cache eviction policy, process crashes, and resource quotas remain untested.
- This demonstrates the ownership/publication contract independently from E4's artifact-key and cache tests; it does not establish a public ABI.
