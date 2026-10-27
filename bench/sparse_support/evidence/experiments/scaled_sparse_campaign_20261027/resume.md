# Sparse-support performance campaign checkpoint

Checkpoint requested before the computer restart. No further GPU jobs or
profilers should be started until the user resumes. All owned scheduler jobs
are terminal and cleanup-verified; no local execution session remains active.

## Completed evidence

- Native v3 rescan/postings/gating baseline: `native_baseline/REPORT.md` and
  `native_baseline/run_001/`. Exact fixture hashes and the frozen pre-fastpath
  binary identity are in `frozen_baseline/manifest.json`. Results remain
  negative for the then-current serial k-way merge: q24 full-rescan discovery
  was 3.172 ms vs postings 26.211 ms on large-ranked, and 3.425 ms vs 21.816 ms
  on large-events. Complete rescan paths were 9.586 and 27.860 ms; complete
  posting paths were 32.100 and 46.338 ms. These do not reject the later
  ordered-disjoint fastpath.
- Host exact-set / gate oracle outputs are in `host_oracles/`.
- NCU profile artifacts are in
  `profiles/serial_postings_q24_ranked_v3_ncu3_20261027/`. Both query kernels
  collected with Nsight Compute 2025.2.1.0, 81 passes each, on the explicit
  GPU2 UUID. The `.ncu-rep` files and scheduler receipt are present. NCU emitted
  “No source files were imported” despite `--import-source yes`, `--source-folders`
  pointing to the frozen exact snapshot, and CUDA debug-line sections in the
  binary. Helper analysis and correct source mapping are not complete; make no
  source-line/stall claims. This profile is diagnostic only.
- A first NCU attempt is preserved under
  `profiles/serial_postings_q24_ranked_v3_20261027/`. It exited before profile
  collection because the benchmark requires at least three samples. The
  corrected profile used three ordinary benchmark samples while `ncu -c 1`
  collected one matching launch.

## Scheduler jobs owned here

| Job | Attempt | Final state | Cleanup |
|---|---|---|---|
| `3f12203b-6104-43a6-aa42-9afcaf590f80` | `7c7e83db-760a-49a3-8660-8c9480f2151e` | succeeded native baseline | verified |
| `d9b87ab9-94d6-4c06-8e69-6b90e931d740` | `8a8e1479-03c8-41b5-a1df-61534f588e34` | failed NCU sample guard | verified |
| `21971184-010c-43de-88bd-17f8708f7c88` | `dc946c6c-f015-4098-abc5-ad026bfafb90` | succeeded NCU collection | verified |

No KDA Torch comparison job has been submitted. There are no queued or running
jobs from this owner.

## Staged large/small Torch comparison

Root supplied the fixed large-image validator package at
`build-sparse-support/evidence/torch-v12-validator/package/`, identity file
SHA-256 `99da528db6f4ba14bec983b6dfe31f12a065dbdc8b39c0dfb224ed0e969fca07`,
extension SHA-256
`6a85eebb37d26bf7ddc94a74026a67723d5592470a30b3a55349f2fa95b6071c`.
Root reports the change is validator-only and the quadratic physical-cover
validator has been fixed; native image-builder and differential validation
passed in the geometry workstream.

The KDA v12 task and workspace have been registered/prepared, but not
evaluated:

- Task: `kda_task_v12.json`, SHA-256
  `32daea0096be0b32b3c7d709767187081797c52076a238490cfbcf75c5abc60b`.
- Workspace: `kda_workspace_v12/`; both SIMT and packed selector candidates
  built successfully through the KDA experiment engine.
- Snapshot SHA-256: SIMT
  `d989f10def6a7652078e7583934b1dc55673729e6a84767f1d90f825d1359205`; packed
  `9619c740c4d6618218074adbea795236c37d7470db536e28cbaecbac7edc6666`.
- The four task cases include small event-local and ranked-batch anchors plus
  large-ranked (E=262,144, N=4,096, B=32, 512 selected identities) and
  large-events (E=1,280,000, N=20,000, B=8, 256-observation count/time window).
  The task measures full inference and training costs including global all-edge
  value refresh, derivative-only gate gradient work, Adam, setup, and memory.
- Inputs use fixture generator SHA-256
  `887437fc3418e06c25e515487891a3224f8dd56493b689bf41bc70e66ae5a76b`, with
  immutable v3 bundle manifests pinned in the task.
- To resume, submit `run_packed_training_kda.py` as a finite exclusive,
  quiescence-checked job on the approved GPU2 UUID, setting
  `CELLERATOR_SPARSE_SUPPORT_BENCH_ADMITTED=1`,
  `CELLERATOR_SPARSE_PACKAGE_ROOT=/home/tumlinson/Cellerator/build-sparse-support/evidence/torch-v12-validator/package`,
  and `PYTHONPATH=/home/tumlinson/kda:/home/tumlinson/Cellerator/build-sparse-support/evidence/torch-v12-validator/package`.
  Keep the existing task's 21 samples, 5 warmups, paired rounds, and setup cost.

## Remaining bounded work

1. Finish KDA v12 SIMT vs CE-GEO full-cost qualification on the staged small and
   large cases; preserve regressions and numerical checks.
2. Analyze the two retained NCU reports with KDA helpers. If source import is
   still material, inspect the binary's recorded source paths and retry only
   after restart/resumption, in a new immutable profile directory.
3. When the production ordered-disjoint posting fastpath is ready, compare it
   against the frozen old posting binary and rescan using byte-identical v3
   fixtures. Keep setup, worklist/gating costs, and full-pipeline results
   visible; do not claim a universal winner.

## Source / artifact identities

- Native baseline report SHA-256:
  `1c4b5e22ef153476dec67ac3da1dbb9800b0c3ff2f108bae01e696bf82c1a3fd`.
- Frozen baseline binary SHA-256:
  `a628a22579a576d51f942f91d3c0f279ec7c371fdec36f0b454a956e7a44ae7f`.
- Frozen posting CUDA source SHA-256:
  `a3aecefb1ed359342bbc6ffb6843bda079f8a28082030f07510a0e78275e3853`.
- Successful profile report hashes: counts
  `4a6b53f0bfc462e234171749b99cf5bb16b241038e9d98de864f38626d634841`, scatter
  `1865a1431daa7256c3960b557697f3f22e79285885c06bb9283860fcefa19118`.
- NCU profile collection manifest SHA-256:
  `0acf05cb2dddf209d7c1f830894bade10ec473866161bfae95887023e0c7c29e`.
- Current benchmark driver SHA-256 at checkpoint:
  `a1d29ecabb79591556ee1b0bc7dc1381d52e11ab5561357733aa8dd60c0d547e`.
- NCU offline scratch extraction (optional): `/tmp/cellerator_profile_elf/`;
  imported source inspection output: `/tmp/ncu_import_source.txt`.

Root owns integration and final acceptance/commit. This checkpoint does not
assert completion of the overall performance objective.
