# Independent E4 artifact and E6 lifetime verification

Date: 2026-10-28
Repository revision: `c63f591d04ef74b7566937d161c80936dcc4cf05`
Environment: Linux x86_64, g++ 13.3.0, clang 18.1.3; CPU-only. No CUDA device work was run.

## Verdicts

| Slice | Verdict | Evidence boundary |
|---|---|---|
| E4 CPU runtime compile and artifact reuse | **Supported, bounded** | The generated CPU region compiled through Cellerator's existing object compiler, linked and loaded as a shared object, and executed. Cold and cross-process warm cache behavior, value rebinding, and key changes were observed. The prototype is dependency-closed over generated source and standard headers; per-header content invalidation is not established. |
| E6 CPU module lifetime and publication | **Supported, bounded** | A real loaded shared object remained callable while an old invocation was blocked, a replacement was published, then retired after the invocation released its lease. Stale, failed, canceled, duplicate, and retry cases were exercised with deterministic host synchronization. |
| E6 CUDA stream/tape lifetime | **Unresolved** | This verifier did not run the two-stream GPU path or memory/race/synchronization sanitizers. CPU ownership evidence does not establish CUDA module or buffer retirement safety. |

## E4 checks

Built the probe from source and the repository's existing object-compilation implementation:

```sh
g++ -std=c++20 -O2 -Iinclude \
  bench/compiler/experiments/adaptive_fields/cpu/jit_probe.cc \
  src/compiler/backend/compile_generated_c_into_ordinary_objects.cc \
  -ldl -o bench/compiler/experiments/adaptive_fields/verification/lifetime/jit_probe_verify
```

Invoked it in separate processes against isolated verifier-owned cache directories. For `g++`, the cold invocation reported one compile and one link; a second process reported `warm_hit=1`, zero compile/link operations, and loaded the same artifact. The same-process second `prepare()` call is also checked by the probe. A ctypes call to the emitted `.so` independently matched an FP64 CSR-plus-segmented-softmax reference with maximum absolute error `1.89346924e-08`; the generated function also observed a numerical value rebinding without recompilation.

Changing compiler command from g++ 13.3.0 to clang++ 18.1.3 generated a different key and artifact. Changing the numerical contract from ordered FP32 to fast FP32 and changing the topology tag from `csr4x3-seg2-v1` to `...-v2` also generated distinct artifacts. Tampering with the manifest artifact hash and, separately, the dependency-file text caused a cold rebuild. The source's manifest identity includes source, compiler command/version, flags, numerical contract, topology tag, dependency-file hash, and produced shared-object hash.

The topology tag is included in generated source/cache identity, but it does not specialize the generic CSR loops; topology arrays remain runtime inputs. This is valid as a generic executable probe and is not evidence that topology-specific physical code is beneficial. The depfile contains the generated `.cc` only because compilation uses `-MMD`; standard headers are not individually content-hashed. Thus include-content invalidation beyond the current generated-source/standard-library boundary remains uncovered. Cache measurements are single probe observations, not a performance campaign.

Independent FP64 oracle: [verify_loaded_artifact.py](verify_loaded_artifact.py). Relevant generated code and manifests are under `cache_gpp/`, `cache_clang/`, `cache_contract/`, and `cache_topology/`.

## E6 CPU checks

Built and ran the experiment-local lifetime harness:

```sh
g++ -std=c++20 -O2 -Iinclude \
  bench/compiler/experiments/adaptive_fields/cpu/runtime/lifetime_probe.cc \
  src/compiler/backend/compile_generated_c_into_ordinary_objects.cc \
  -ldl -pthread \
  -o bench/compiler/experiments/adaptive_fields/verification/lifetime/lifetime_probe_verify

bench/compiler/experiments/adaptive_fields/verification/lifetime/lifetime_probe_verify \
  bench/compiler/experiments/adaptive_fields/verification/lifetime/lifetime_modules_retry
```

The harness compiles, links, and `dlopen`s actual generated shared objects. Its callback blocks an old executable call at a condition variable while replacement compilation and publication proceed. Assertions verified the replacement result, stale structure-epoch rejection, pending-cancel and compile-failure handling, bounded pending capacity, duplicate coalescing while work is outstanding, and old-module lease retention until the blocked call completes. The printed trace ended with `retry_after_success_cancel_failure=pass`; it also reported `old_code_retired_after_completion=1` and `generic_fallback_callable=1`.

An independent driver submits the same request after consuming its completion. Before the owner fixed request-key cleanup, this exposed that a consumed request ID could be returned as a duplicate with no completion available. The latest owner version clears the coalescing key on completion consumption; the driver now receives a fresh ID for the retry. The owner's harness additionally checks retries after successful compilation, cancellation, and failure. The driver is [verify_duplicate_retry.cc](verify_duplicate_retry.cc).

The tested ownership contract is a host-side `shared_ptr<Module>` lease around a synchronous loaded-function call. It is not Cellerator production runtime code. It does not establish asynchronous CUDA stream/event ownership, forward-tape retention through backward, cancellation of a compiler already executing, sanitizer cleanliness, or publication against real prepared projections and device buffers. A printed close count is taken before all end-of-scope owners are destroyed; the old-call weak-reference assertion is the direct retirement evidence.

## Exact source identities

```text
cpu/jit_probe.cc
f01f00315a10cc13e0752733cf077453186b8f0b4d888337b1cf0ae25100c15b

cpu/runtime/lifetime_probe.cc
efae5f015a3755fff2ac63835ee6ff596abe8e7bacaf5ea68f077dac9619fef8
```
