# Sparse support lifecycle verification

Date: 2026-10-27

## Scope

Independent checks for cross-stream parameter publication, support epoch and
parameter generation separation, live tape fencing, abandoned graph cleanup,
and optimizer/native owner continuation across support promotion and revocation.
The CUDA package snapshot is
`build-sparse-support/evidence/torch-v3/package`, identified by its source
inventory SHA256
`9e3cfcfa3a84eaa267ba0c17552f5835a024b9e9d606f95430c4c58adb6d9523`.

## Evidence

- Root's v3 full-suite job `674942d2-1a9f-4878-8b56-00bc3eda4599` ran 22 tests:
  21 passed and one errored. The error was in the added cross-stream test because
  it constructed SGD while `guarded_step` explicitly requires Adam. The test
  has been corrected to Adam in the source and in this report directory's
  frozen test copy; the corrected test passed in the subsequent GPU1 run below.
- Independent lifecycle job `3e4de619-90a7-42d4-90e5-ca608020f1a6` ran the
  original three-case test copy on GPU3. The live-tape publication gate and
  promotion/revocation checkpoint continuation passed. The cross-stream case
  errored at optimizer validation for the same SGD/Adam mismatch. Scheduler
  reports `cleanup_verified=true`.
- The final combined job `9e2f5743-2c43-4597-b6e9-d47ea72e4f26` pins this
  corrected test (SHA256
  `5ee6bd30bf18d300fe18ccd997be918753df0cf7e5c663c86d3429b941d9b51d`) and the
  existing mechanism regression file (SHA256
  `8e51bbae5089e9acb1c50ecca23a7a343dc7642d358ac99519697841f25a44c9`). It did
  not start: the scheduler's `gpu.quiescence` precondition was blocked because
  quiescence/utilization requirements were not observed before the deadline.
  Scheduler reports `cleanup_verified=true`; no test command ran.
- The GPU2 retry `8a01d44c-2e5e-429a-a453-2188f19e9639` passed scheduler
  admission but failed before collection because `python -m pytest` could not
  import pytest in the supervised process environment. Cleanup is verified.
  The corrected console-script runner `2ddca49c-3bf7-4d6e-bd7e-d020ce7b006d`
  was then blocked by the same strict quiescence precondition: utilization was
  observed at 3% before the deadline, despite a later idle sample. Cleanup is
  verified and no tests ran. The precondition was not relaxed.
- Final GPU1 job `78cfba09-82c8-467c-8e76-c3296529e05b` ran the corrected
  lifecycle test and the existing mechanism regression file against the frozen
  v3 package. Scheduler quiescence was verified with three 0%-utilization
  samples, and cleanup is verified. All 29 tests passed in 4.53s. The retained
  output SHA256 is
  `16bc8d5fc9e72a11eb2d61bc1d9e619225313430b4b13701cc4287908423c865`.
- Python compilation passed for both frozen files. Local discovery of the
  CUDA-gated lifecycle test reported three skipped tests because the scheduler
  admission variable was unset. The CPU native owner protocol suite passed 2/2
  tests with the v3 package. Pytest 9.1.1 is installed in the selected runtime.

## Coverage and limits

The successful GPU1 run covers cross-stream parameter visibility after an Adam
publication, refusal to publish support while a training tape is live, unchanged
parameter generation and leaf identity across a support revision, and
checkpoint restoration plus deterministic Adam continuation after promotion
and revocation. The existing mechanism regressions passed for shared module
ownership, checkpoint continuation, nondefault streams, live tapes, abandoned
reader cleanup, and poisoning recovery. These results apply to the frozen v3
package and listed test inputs; they do not establish broader training or
scientific correctness.
