# Resume note

## Objective and scope

This is an isolated probe of deriving shared edge values only for
`U = unique(P union D)`, where P is the primal workset and D is the derivative
candidate workset. It preserves the Torch shared parameter owner and native
relation ABI. The assigned boundary is this directory only; there are no
production edits and no GPU job has been submitted.

## Current artifacts

- `probe.py`: computes full-reference values with the existing shared endpoint
  formula, or computes U and uses differentiable `index_copy` into an E-sized
  FP32 vector. It compares output and full shared-parameter VJP before timing
  complete native SIMT training steps. U is recomputed per invocation, and
  edge values are recomputed after every Adam publication.
- `test_probe.py`: CPU Torch oracle covers P∪D inclusion, zero values outside
  U, and the shared embedding/scorer VJP. It passes with `unittest`.
- `task.json`, `kda_adapter.py`, candidate manifests/selectors, and
  `run_kda.sh`: KDA experiment setup. Candidate registration and CPU build
  preparation succeeded; GPU execution has not started.
- `gpu3_job.json`: proposed exclusive GPU3 job payload. It is not submitted.
- `README.md`: method and boundaries.

The fixture manifests validated locally are:

- events: SHA256 `910a4c742b795a2a827cb21e62c510a3bc5039e6878ae75d4826fc0d33234be8`,
  E=1,280,000, 8 cells, 20,000 entities;
- ranked: SHA256 `674d5d4660befcb6ddbcd618b16b136203d3478ddc577d09a037e116bffd6e32`,
  E=262,144, 32 cells, 4,096 entities, 512 ranks per cell.

Using the benchmark's host posting builder on the frozen query sets gives
events U=70,466 and 82,959 total per-cell candidate items; ranked-512 U=205,281
and 1,039,948 per-cell candidate items. These are host-computed workload
counts, not GPU performance claims.

Frozen Torch v11 identity is `887f8a27b2e8c1e858690291401919f7cec6898a401798377c31294326a07a5b`;
the extension SHA256 is `430646858b79724b2eee8129e5a7ec57df6fa9632d89d798fd03d80b67324c5a`.

## Checks already run

- `python -m unittest bench.sparse_support.experiments.indexed_values.test_probe -v`:
  1 test passed under the Cellerator Torch 2.7 environment.
- Python compilation of `probe.py`, `kda_adapter.py`, and `test_probe.py` passed.
- `python -m json.tool task.json` passed.
- KDA `register` succeeded for `full_scan` and `indexed_union`; `prepare`
  succeeded for both. Two earlier failed `prepare` attempts remain preserved in
  the KDA workspace; the later successful attempt used the corrected adapter.

## Resume steps

1. Verify current source files and KDA task hashes after restart; root may have
   checkpointed the workspace meanwhile.
2. Confirm the index-support agent's GPU3 sanitizer/fullscan job is terminal
   with cleanup verified; it had not yet sent final release when the pause
   arrived. Do not submit if GPU3 is still occupied or fails quiescence.
3. Run the finite KDA evaluation via the scheduler using the exact resource ID
   and current client config, not direct CUDA. The drafted payload targets
   `gpu:testlsserver:GPU-6c1cac7f-a360-0aef-ba98-2828bfd1db1a`; recheck this
   resource and the current scheduler admission state before use.
4. If native VJP or output comparison fails, fix only this experiment probe,
   rerun the CPU oracle, and preserve failed evidence. If the run completes,
   retain scheduler receipts, KDA attempt files, raw stdout/stderr, source
   hashes, and report full-E versus indexed-U costs, per-case U/E, VJP error,
   and peak temporary memory. Do not promote the probe into production.

## Open limits

The GPU comparison is not yet run. The current candidate uses native SIMT only;
it does not measure CE-GEO packed forward or reuse an autograd value graph across
a frozen multi-event window. The full-step comparison includes per-step union
construction, value computation, native relation, backward, and guarded Adam;
the edge-value-only refresh timings are diagnostic. No speed or memory benefit
has been established.
