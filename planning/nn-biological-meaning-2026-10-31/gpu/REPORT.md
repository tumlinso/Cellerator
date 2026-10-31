# Biological-meaning NN V100 integration qualification

## Status

The isolated admitted-only test entrypoint and Device Scheduler runner are prepared. GPU execution has **not yet been submitted**. The implementation owners are completing a final source-level correction in query coordinate validation; source hashes and job submission are intentionally deferred until they confirm the affected files are frozen.

## CPU preflight completed

- Python 3.13 `py_compile` passed for `tests/framework/test_biological_gpu.py` and `scripts/qualify.py`.
- Ordinary unittest discovery, with `CELLERATOR_TEST_CUDA` unset and CUDA hidden, collected six GPU tests and skipped all six without initializing a CUDA device.
- Five non-native test methods were directly exercised on CPU using the same deterministic fixtures and ordinary Torch references; all passed. This is harness/reference-path evidence only, not V100 qualification.
- The optional prebuilt grouped-linear CUDA extension exists at `/tmp/cellerator-nn-grouped-linear-sm70/_modeling_cuda.cpython-313-x86_64-linux-gnu.so`; it has not been imported or executed in this pass.

## Pending admitted run

The job will request one exclusive, UUID-selected V100 with a finite timeout and the Cellerator Device Scheduler client. It will run only `test_biological_gpu.py`, capture the test/source/tool/native-extension hashes, require SM70, and report cleanup through the scheduler receipt. No benchmark or profiler is requested. The numerical kernel sources remain unchanged.

## Evidence boundaries

No V100 output, VJP, or cleanup result is claimed in this draft report. Passing CPU tests and skip behavior do not substitute for the pending admitted GPU run. A successful smoke will cover only its compact deterministic cases and will not establish performance, scaling, exhaustive numerical correctness, or biological validity.
