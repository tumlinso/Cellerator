# Prepared condition root-map production validation

The production CUDA workset test was built by the root build in
`build-sparse-support/tests/sparse_support/cellerator_sparse_support_cuda_worksets`.
The exact path resolved to that executable, and its SHA-256 was
`2c1c2b9b4fbdfe9e9ad0dd864aeea1076855271a9d9ba049d29ee6631b32d489`. A
read-only snapshot with the same hash was used for the GPU run. The source,
test, static-library, build-flags, linker-command, and build-log hashes are in
`sha256.txt`.

## Validation

- Root reports host CTest: 3/3 passed. The target build and its log are retained
  here as `build.log`.
- CUDA correctness ran on the frozen snapshot. The test covers prepared root
  map construction and caller capacity handling, then checks prepared full and
  dirty bitmap paths against expected results from the retained fallback.
- Scheduler job `1974439f-c87c-475e-bafe-6bcd47a2eade` succeeded on
  `gpu:testlsserver:GPU-d745da9c-7649-8334-41f1-483afa6f3206`; cleanup was
  verified. The GPU quiescence gate remained at 60 seconds, three consecutive
  samples, and zero utilization. The successful observation was 3%, 3%, then
  0%, 0%, 0%; admission used the final three idle samples.
- Compute Sanitizer 2025.2.1.0, from the installed CUDA 12.9 path, reported
  memcheck 0 errors, racecheck 0 hazards / 0 errors / 0 warnings, and synccheck
  0 errors. Full scheduler receipt and logs are retained as
  `scheduler_result_02.json` and `job_logs_02.json`.

The first admitted attempt, `3ffe8e8a-0388-481c-9b71-1355c52b10d1`, also ran
the CUDA correctness test successfully and cleaned up. Its sanitizer stage
failed before instrumentation because the generic `compute-sanitizer` launcher
redirected to unavailable CUDA 13.1. That attempt's receipt and logs are kept as
`scheduler_result_01.json` and `job_logs_01.json`; the corrected second attempt
used the installed CUDA 12.9 sanitizer executable directly.

No performance claim is made by this validation. Performance evidence remains
the isolated complete-path experiment in the parent directory.
