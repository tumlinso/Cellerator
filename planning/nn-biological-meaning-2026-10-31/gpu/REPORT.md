# Biological-meaning NN V100 integration qualification

## Result

The corrected admitted V100 smoke passed **6/6 tests**, with zero failures, errors, or skips. The tested artifact matched the pre-submission source manifest, and source hashes stayed unchanged during execution. Device Scheduler verified cleanup for the successful run.

This is focused CUDA correctness evidence for the exercised compositions. It makes no performance, scaling, exhaustive correctness, or biological validity claim.

## Workload and coverage

The six admitted-only tests covered:

- Named `FeatureBinding`/`FeatureEncoder` alignment with noncontiguous model indices and repeated occurrences; selected readout outputs preserve requested identity order and match gathering from the full readout.
- Prepared packed relations with duplicate edges and an empty neighborhood: FP32 and FP16 forward outputs, input VJPs, and parameter VJPs matched both raw CSR and an independent per-target Torch softmax reference.
- Dense Boolean support and deduplicated CSR support produced the same relation-attention outputs as raw CSR. Explicit host-to-V100 relation preparation, same-device reuse, value-only changes, stale occurrence-order rejection, and stale structural-generation rejection were checked.
- Queryable RNA/ATAC-style readouts reused one base evaluation for queries sharing the same conditioning object, ran separate evaluations for distinct conditioning objects, changed output under registered conditioning, and propagated gradients through the conditioner. Frozen-base mode propagated input/readout gradients without updating base parameters; joint mode propagated base gradients. Conflicting packed history-window offsets were rejected.
- Overlapping shared interaction programs produced zero for empty support and backpropagated through feature, context, and shared program parameters. The reverse prepared incidence structure was materialized before forward execution and reused.
- Bound encoder output composed with the existing native grouped-linear operation. Native FP32 outputs and gradients matched its Torch reference; no native numerical kernel was changed.

The tests used deterministic inputs and disabled dropout. Feature encoding, readout, and interaction programs used FP32 parameters/activations/outputs. Prepared attention was compared in FP32 and FP16; its ordinary Torch arithmetic has no separately promised accumulator type. Grouped-linear used FP32 inputs, weights, outputs, and its existing FP32 accumulator contract. Tolerances were `rtol=3e-4, atol=3e-5` for FP32 and `rtol=8e-3, atol=6e-3` for FP16.

## Execution evidence

Successful run:

- Device Scheduler job: `5f310ca8-9a45-4bae-9bb3-833e81c293dd`
- Attempt: `8a211721-a6af-43c8-afe9-d98a0fe2f80f`
- Device: Tesla V100-SXM2-16GB, SM70, UUID `GPU-d745da9c-7649-8334-41f1-483afa6f3206`
- Runtime: Python 3.13.11, PyTorch 2.7.0+cu126, CUDA runtime 12.6
- `cleanup_verified=true`; job exit code 0
- Native extension: `/tmp/cellerator-nn-grouped-linear-sm70/_modeling_cuda.cpython-313-x86_64-linux-gnu.so`, SHA-256 `d5ffa9284ae5a7dd96ce1e6f8adcbd347b86b58064d0a00496d04f1a8b6ec537`
- Artifact manifest: [`artifact_manifest-attempt-2.json`](evidence/artifact_manifest-attempt-2.json), SHA-256 `a45c8194aff17dea33c4c664824822592e69ae8281c7e878264f547b39340b5c`
- Test result and source/runtime hashes: [`cuda-run-attempt-2.json`](evidence/cuda-run-attempt-2.json)
- Scheduler submission/status/logs: [`submission-attempt-2.json`](evidence/submission-attempt-2.json), [`job-get-attempt-2.json`](evidence/job-get-attempt-2.json), [`job-logs-attempt-2.json`](evidence/job-logs-attempt-2.json)

The first bounded run (job `7f092dfe-4d78-4bb4-9928-92e4620c480d`, attempt `df890ece-4013-4f89-ba7d-f72ae5ffeb91`) completed with verified cleanup but reported one assertion failure and one test error caused by fixture assumptions: it expected repeated host-to-device preparation to return the same object, and it moved a wrapper to CUDA before registering its readout modules. Those fixture assumptions were corrected. The first run's test output and scheduler receipts are retained as `cuda-run.json`, `job-get-attempt-1.json`, and `job-logs-attempt-1.json`; it is not counted as passing evidence.

## CPU preflight and limits

- `py_compile` passed for the test file and scheduler runner.
- Ordinary test discovery with `CELLERATOR_TEST_CUDA` unset collected the six tests and skipped all six without initializing CUDA.
- Five non-native test methods also passed when directly exercised on CPU using the same deterministic fixtures; the native grouped-linear case was qualified only in the admitted V100 run.
- The prepared-relation no-scalar-read guard ran around CUDA forward and VJP. Queryable readout selection also ran under a `Tensor.item()` guard.
- No profiler, sanitizer, benchmark, driver/toolchain repair, or kernel rebuild was performed. The only native binary used was the pre-existing isolated extension identified above.

The qualification covers only these shapes, dtypes, and paths on one V100/PyTorch environment. It does not cover every supported model composition or large-scale execution.
