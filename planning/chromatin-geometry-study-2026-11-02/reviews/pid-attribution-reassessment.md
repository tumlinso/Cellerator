# Independent PID attribution assessment

**Assessment:** PID `1047837` is strongly consistent with a study native child,
but its identity at the compute-process observation is not established by the
retained timestamps. The observation is unresolved, not proof of a foreign
process. The retained evidence alone does not justify clearing the guard or
qualifying the packet's timings. A separate reassessment could change that
conclusion if it obtained decisive process-identity evidence and satisfied the
other study gates; it must preserve the original failed worker result and raw
sample.

## Evidence

All packet paths below refer to
`bench/chromatin_geometry/runs/packets/45bab455-239f-4e32-8ebd-fd33ae03e979/`.

* `native-child-lifetimes.jsonl` lines 161–162 record invocation
  `a4b0e7cf-86e2-425f-a98e-d9092e7c845e`, PID `1047837`, target device
  `GPU-cf22c41f-5b58-77b1-3535-8fadd1ca6505`, a direct teacher/linear FP32
  campaign command and successful return. The output is
  `kda/supplemental/round-05-direct.json`, whose independently computed
  SHA-256 is `ddf3b771c5608b3c3ce2974168b407e2c56c890bcccf8bc1ae4ff605dc3a2e27`,
  matching the end event. Its recorded case, backend, workload and device
  match the command. This is substantial corroboration, but the output does
  not identify the process observed by the later telemetry query.
* The lifetime file hashes to
  `20e696361929df8cb1c5ecbe12cf104bcf08d83f4c2886f5bdd454649257b88e`,
  matching `serial-telemetry.json`. It contains 180 paired starts/ends with
  180 distinct recorded PIDs, all successful, all recording the frozen
  binary and target UUID. I verified all 180 original output paths exist and
  hash to their corresponding end records. This excludes another *recorded
  study child* with the same PID; it cannot exclude reuse by an unrecorded
  process after reaping.
* `serial-telemetry.json` lines 3806–3841 contains the only sample with an
  issue among 263 samples. It records that PID and GPU, `owned: false`, and
  `foreign GPU compute process observed`. The separate utilization query
  reports zero on every GPU for this sample. The wrapper performs utilization
  and process queries sequentially; those are not an atomic snapshot. Zero
  utilization does not establish which process the subsequent query saw.
* The child records `spawn_upper_ns = 1793659178411575576` and
  `reap_upper_ns = 1793659179195143585`. The sample records
  `gpu_query_start_ns = 1793659179105514431` and
  `gpu_query_end_ns = 1793659179144586457`. Thus the utilization query is
  bracketed by those *outer* lifetime bounds; its end is about 50.6 ms before
  the recorded reap upper bound. But `runtime-wrapper.py` lines 34–42 launches
  the **compute-process query after** `gpu_query_end_ns` and does not retain
  that query's start, end, or observation time. The archived `kda_driver.py`
  lines 99–109 records `reap_upper_ns` only **after** `communicate()` returns.
  Reaping can precede this upper bound, and the recording process can be
  descheduled. Therefore neither the recorded GPU query nor the reap upper
  bound proves that the PID was still held by the child at the compute-process
  observation. A different process could in principle reuse it, or the
  process snapshot could occur after reaping. The timings make a disappearing
  descendant plausible; they do not establish it.
* The runtime wrapper's `is_descendant` reads the live `/proc` parent chain
  after `nvidia-smi` returns and fails closed on missing or unreadable status.
  Consequently `owned: false` means that this later live check did not prove
  ancestry; it does not prove the observed process was foreign. The wrapper
  leaves this issue in place even when it separately attributes sampled target
  utilization to bounded child lifetimes. Its `clean` result requires every
  sample to have no issues, and the recorded worker `result.json` is failed
  with return code 1. The inner campaign returned 0, which does not override
  the guard.

## Source identity and admission

The retained registered command uses immutable snapshot
`fc149dc6edc4c37caee5ed5dec71f211f00cb98d80b967d1a3ca3c3dab9bd008`.
I independently hashed its actual closure archive as
`f131948000849270e22c392459264fa599566f4a1d51492a09e72d9e1d8f72c7`,
its frozen binary as
`b0292b309e12751655b8a476c4d34e23d1c877549e175d33c5dcdca12aa25ddf`,
and packet `runtime-wrapper.py` as
`ac6a8d225c049abc9bc4703fe09f4e287b283a607d2fa71ce3431da02c58c6e7`;
these agree with `runtime-wrapper-identity.json` and the registered command.
The archive's driver hashes to
`f760a9f14bbf33069510bfda647ac3288adf705797a6606451ea9e1042d7544c`,
identical to the inspected local driver. The packet's and approved v3 freeze's
SHA-256 are both
`4033eea4a8168f5e898212873252c8782943d12f5325e5a31ca45487f9751f06`.

The Scheduler receipt at `bench/chromatin_geometry/runs/receipts/45bab455-239f-4e32-8ebd-fd33ae03e979.json`
records an exclusive target-device allocation, two clean precondition samples,
attempt `f09caad3-2485-41d6-8187-b04166334cb0`, verified cleanup and released
lease. Its systemd attempt reports success, while the separate worker result
reports the guard failure. Admission and cleanup do not prove the identity of
the process in this particular query or exclude unmanaged activity. Retained
sampled telemetry also cannot exclude activity between samples.

## Condition for reassessment

A fail-closed retrospective attribution needs independently retained evidence
tying the **same process instance**, not just the same numeric PID, to the
reported target GPU at the process snapshot. For example, a reliable bound for
the actual process observation entirely after that child's spawn and before a
known lower bound on its reap would preclude PID reuse in that interval;
equivalent process-instance evidence could also suffice. The evidence must be
bound to this packet, invocation, device and frozen inputs, with no conflicting
process or observation evidence. The present sample lacks such a process-query
bound and the child lacks a reap lower bound, so it fails that condition. Do
not replace the existing raw sample or worker/guard status. Even if future
evidence resolves identity, a new timing assessment must separately establish
the protocol's comparison, correctness, isolation and other applicable gates;
this review did not establish them or authorize promotion.
