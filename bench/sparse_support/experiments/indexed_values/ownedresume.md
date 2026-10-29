# Resume note

## Objective and boundary

This isolated probe evaluates `U = unique(P union D)` shared edge-value
computation, preserving the canonical Torch parameter owner and native sparse
relation ABI. It owns only this experiment directory. No production source was
edited and no indexed-values GPU job has been submitted.

## Probe and KDA state

- `probe.py` computes either the existing full-E shared endpoint formula or
  scores only U, then differentiably scatters into an E-length FP32 vector.
  It checks outputs and the full shared-parameter VJP before timing complete
  native SIMT training steps. U is rebuilt each invocation; values are
  recomputed after every guarded Adam publication.
- `test_probe.py` validates P∪D coverage, zeros outside U, and embedding/scorer
  VJP against a full Torch reference. It passes.
- `task.json`, `kda_adapter.py`, candidate manifests/selectors, `run_kda.sh`,
  and `gpu3_job.json` are refreshed for the persisted Torch v12 validator.
- KDA workspace `kda_workspace_v12` has `full_scan` and `indexed_union`
  registered and both CPU builds prepared. The old v11 `kda_workspace` and its
  failed attempts are preserved.
- `gpu3_job.json` is only a proposal; it is not submitted. The latest resource
  selector is the prior GPU3 UUID and must be rechecked after handoff.

Frozen package identities:

- identity file: `build-sparse-support/evidence/torch-v12-validator/identity.json`,
  SHA256 `99da528db6f4ba14bec983b6dfe31f12a065dbdc8b39c0dfb224ed0e969fca07`;
- extension: `_torch.cpython-313-x86_64-linux-gnu.so`, SHA256
  `6a85eebb37d26bf7ddc94a74026a67723d5592470a30b3a55349f2fa95b6071c`;
- frozen `sparse_support.py` SHA256
  `b81d5dcd555e850cd76e68f35a574f420db3e2c6516c7c9cc3e38fdcb1f9026f`.

Fixture manifests and host posting counts revalidated on resume:

- events manifest SHA256
  `910a4c742b795a2a827cb21e62c510a3bc5039e6878ae75d4826fc0d33234be8`,
  E=1,280,000, 8 cells, 20,000 entities; U=70,466 (5.51% of E), 82,959
  total cell-level derivative candidates;
- ranked-512 manifest SHA256
  `674d5d4660befcb6ddbcd618b16b136203d3478ddc577d09a037e116bffd6e32`,
  E=262,144, 32 cells, 4,096 entities, 512 ranks/cell; U=205,281 (78.31% of
  E), 1,039,948 total cell-level derivative candidates;
- small control uses the benchmark's deterministic synthetic 64-edge support.

These are host-derived workload sizes, not performance results.

## Checks already run

- `python -m unittest bench.sparse_support.experiments.indexed_values.test_probe -v`:
  1 test passed under Torch 2.7.0.
- Python compilation of `probe.py`, `kda_adapter.py`, and `test_probe.py` passed.
- `python -m json.tool task.json` passed.
- KDA v12 `register` and `prepare` succeeded for both candidates.
- Fixture loader verified the selected large fixture arrays against their
  manifest hashes; package and extension hashes match persisted v12 files.

## GPU attempts and next steps

The initial GPU3 handoff was received after index-support jobs
`e6531eec-0397-47e7-a228-ac53832805fb`, `cf4b66ab-948c-412b-aeea-933fc9e34f20`,
`9b288b3c-c592-4106-ac44-0039a376bee0`, and
`dfc428d3-abe0-4595-be6b-ee4bf6b93f5b` were terminal with cleanup verified.

The canonical scheduler invocation for this host is:

```text
/home/tumlinson/device-scheduler/.venv/bin/device-scheduler \
  --config /home/tumlinson/.config/device-scheduler/cellerator-client.toml ...
```

GPU3 is `gpu:testlsserver:GPU-6c1cac7f-a360-0aef-ba98-2828bfd1db1a`. Always
query current availability and use the scheduler's `gpu.quiescence` precondition
before submitting; an earlier inventory or handoff is not a current grant.

Attempts retained:

- Job `19fe940f-66aa-44cd-aa72-1d1b7bf3ebf2` stopped in KDA preflight because
  the adapter omitted required `driver_version`; its evidence is under
  `evidence/attempt-preflight-v1/`.
- Job `b6ca53b5-7efa-476f-bd0f-95b92d722bb6` stopped in KDA validation. The
  failure was a probe accounting bug: untimed warmup calls were counted before
  the generation baseline. Its evidence is under
  `evidence/attempt-adapter-probe-v2/` and KDA run
  `82bfb950-ee57-4069-bdee-63341e0077f5`.
- Job `01744e2a-046f-4339-afcc-875c1695510d` passed quiescence, VJP/training
  checks, then stopped because a Torch `_CUuuid` object was not JSON serializable.
  Its receipt and output are under `evidence/attempt-counter-fix-v3/`; KDA run
  `c9c9c20b-48a3-4d87-a822-9ad75cae2cd4`.
- Job `bb9a6ca2-92f6-475a-91b1-829af210b3dd` was canceled at root's request
  after about nine minutes in large `full_scan` validation. Scheduler verified
  cleanup. The partial KDA run is `63df7cca-9311-4a9b-a466-bec563793ca1`, and
  the scheduler receipt is at `evidence/attempt-json-fix-v4/scheduler_job.json`.
  Root identified the cause in frozen v12 native support-owner construction:
  `training.cu` checks P scalar parameter IDs for uniqueness with nested
  pairwise loops, O(P^2). The 1.28M-edge event fixture makes this setup step
  impractical. The native worker is fixing the exact uniqueness check; wait for
  root's frozen v13 package before another GPU run. Do not hand GPU3 to another
  experiment before root confirms the indexed-values rerun is complete or
  releases it.

Probe fixes applied locally:

- `probe.py` resets refresh-call counters after the untimed warmup so counted
  refreshes and parameter-generation deltas cover the same interval.
- Device UUID is converted to `str` before JSON serialization.
- The v12 `gpu3_job.json` is restored to its original v2 selector; v13 uses
  the separate `gpu3_job_v13.json` and `run_kda_v13.sh`.

## v13 retry preparation

Root froze the owner-validation package at
`build-sparse-support/evidence/torch-v13-owner-validation/package`:

- identity SHA256: `4d3b6f0e1a63ea20785975597849bfad0b0f081221c47d9b858d6581b487afd4`;
- extension SHA256: `51b0838ad3466f4c03f693d620aa6b800e8af9ed87a8a3fceb3bd432a05a4f24`;
- `sparse_support.py` SHA256 remains
  `b81d5dcd555e850cd76e68f35a574f420db3e2c6516c7c9cc3e38fdcb1f9026f`.

Prepared separately from v12:

- `task_v13.json` references the v13 identity/module/extension and has a new
  task ID;
- `candidates_v13/{full_scan,indexed_union}` have v13 dependency manifests;
- `kda_workspace_v13` registered both candidates and CPU `prepare` passed;
- `run_kda_v13.sh` uses the frozen v13 package and workspace;
- `gpu3_job_v13.json` has a fresh idempotency key/tag and preserves the UUID,
  exclusive request, and quiescence precondition.

Local CPU unittest, Python compilation, JSON parsing, and `git diff --check`
passed after the v13 preparation. At the last update from root, its GPU3
regression job `98343670-57dc-4bfe-963f-acdf4b476213` was queued. Wait for its
explicit pass, cleanup verification, and GPU3 handoff before submitting the
v13 KDA evaluation. The owner fix is an exact O(P log P) uniqueness validation
that preserves scalar ID order and validation precedence; it does not change
the numerical kernel or this probe's operation.

V13 GPU evaluation completed successfully as scheduler job
`5c3a61c7-a5e9-45bf-a7dd-04f203909f23` (cleanup verified), KDA run
`d919fe94-0e4c-4437-b086-3e63539dabc9`. Both candidates validated all three
cases. KDA selected indexed values with composite 1.395x speedup (95% interval
1.375–1.414): large-events 2.935x, small 0.929x, ranked512 0.989x. Detailed
timings, gradient errors, memory, setup cost, source hashes, and reproduction
instructions are in `evidence/qualification/v13/REPORT.md`. GPU3 was handed to
`counted_condition_investigation` after cleanup, as root requested.

The result remains an isolated SIMT probe. Packed CE-GEO and autograd-graph
reuse are outside this measurement.
