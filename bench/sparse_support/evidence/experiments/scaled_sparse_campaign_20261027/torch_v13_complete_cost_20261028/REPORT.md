# v13 packed-forward versus SIMT complete-cost comparison

This is a source-bound comparison of native SIMT sparse relation forward/VJP
against CE-GEO N64 packed forward with the same native SIMT VJP. It covers the
four frozen synthetic slices in the task. The result describes this backend
and these fixtures; it does not eliminate CE-GEO, packing, or any other sparse
mechanism family, and it does not select a Cellerator default for real data.

## Method and validation

The KDA experiment engine ran on a Tesla V100-SXM2-16GB (SM70), resource
`GPU-d745da9c-7649-8334-41f1-483afa6f3206`, driver 580.178.04, with Torch
2.7.0+cu126. Each candidate/case contributed 84 synchronized-wall samples
(four KDA rounds × 21 samples, 3 iterations/sample, 5 warmups). KDA's primary
decision uses its two fresh paired confirmation rounds; those confidence
intervals are shown separately from the observed p10–p90 ranges of all 84
samples. Module setup was primed, then measured in five balanced AB/BA pairs.

Both paths passed validation and every benchmark's numerical checks for all
four cases: relation output, loss, parameter gradients, released read tickets,
and finite nonzero STE gradients for derivative-only gate proposals (8,
447, 519,961, and 41,478 such candidates, respectively). The
measured full training step includes zero-grad, shared gate scoring and STE,
all-edge value refresh, sparse relation forward, loss, backward including
derivative-only proposals, and guarded Adam. The packed candidate uses the
existing N64 CE-GEO forward and the same SIMT VJP; this is not a packed
backward result.

## Complete operation timing

Times are milliseconds per step. Each entry is median [observed p10, p90] over
84 samples. KDA score is the paired SIMT/packed speed ratio: values above 1
favor packed; values below 1 favor SIMT. Its interval is a 95% bootstrap
interval from the confirmation stage.

| Slice | Synthetic workload | SIMT inference | Packed inference | SIMT training | Packed training | KDA SIMT/packed score [95% CI] |
|---|---|---:|---:|---:|---:|---:|
| event-local | B=4, N=32, 1 event/cell, E=64, 16 candidates | 0.963 [0.942, 0.990] | 1.055 [1.015, 1.076] | 7.392 [6.738, 7.807] | 7.652 [6.849, 7.836] | 1.004 [0.915, 1.103] |
| ranked batch | B=32, N=32, 8 selected genes/cell, E=64, 896 candidates | 0.946 [0.900, 0.973] | 1.049 [1.021, 1.094] | 6.665 [6.565, 7.648] | 7.339 [5.945, 7.886] | 0.916 [0.913, 0.918] |
| large ranked | B=32, N=4,096, 512 selected identities/cell, E=262,144, 1,039,948 candidates | 5.491 [5.481, 5.522] | 7.994 [7.985, 8.024] | 13.117 [13.098, 13.267] | 15.593 [15.568, 15.681] | 0.841 [0.841, 0.841] |
| large event window | B=8, N=20,000, 512 occurrences/cell, count/time intersection 256 occurrences/cell, E=1,280,000, 82,959 candidates | 5.545 [5.533, 5.562] | 6.262 [6.252, 6.281] | 14.098 [14.052, 14.177] | 14.812 [14.770, 15.096] | 0.944 [0.934, 0.954] |

The aggregate KDA score is 0.924 [0.901, 0.948], equivalent to packed being
about 1.08x slower (reciprocal interval 1.055–1.110x), so this specific packed
candidate did not meet the configured requirement for positive improvement;
KDA kept SIMT as the preferred candidate in this experiment workspace.
The large ranked case shows the clearest difference: packed forward was about
45.6% slower for inference and 18.8% slower for training by the paired KDA
score. The event-window case's training score corresponds to about 5.9% slower
packed execution. The event-local score is close to parity and its interval
includes both directions. These are synthetic-fixture observations, not a
reason to remove the corresponding algorithms before testing real data.

## Refresh phase, preparation, and memory

The separately measured `global_edge_value_refresh_fwd_plus_ones_vjp` phase
was similar across backends: large-ranked 2.386 ms SIMT vs 2.383 ms packed;
large-events 11.309 vs 11.310 ms. This is diagnostic phase attribution; it is
not added to the complete step a second time. It confirms that this packed
forward probe did not make the full-edge value refresh cheaper.

| Slice | SIMT module setup median (ms) | Packed module setup median (ms) | Extra packed cold setup incl. geometry compile (ms) | Max measured training allocation delta, either path |
|---|---:|---:|---:|---:|
| event-local | 40.92 | 40.97 | 0.07 | 0.50 MB |
| ranked batch | 40.69 | 40.66 | 0.03 | 1.71 MB |
| large ranked | 532.09 | 662.26 | 442.61 | 1.18 GB |
| large event window | 2,445.91 | 3,171.34 | 2,688.97 | 1.80 GB |

The extra cold setup includes the host geometry compilation cost: about 310 ms
on large-ranked and 1,962 ms on the event-window fixture. Large-event setup
also had about 2.61 seconds of fixture-owner preparation. These are reported
separately from the steady step and explain why a faster steady-state sample,
if one occurs in another regime, must still repay preparation. Here the
complete step already favors SIMT on the two large cases.

Memory values above are measured peak allocated deltas during the full
training phase. The task also recorded a preflight estimate, but its old
benchmark-source version undercounted large-case memory because it read an
estimate from the wrong nesting level. In particular, it reported about 45 MB
for large-events while the recorded shape components sum to about 1.40 GB
before CE-GEO active tiles; the measured training peak was about 1.80 GB. A
post-run host-only fix recomputes estimates from B, N, E, width, candidate
count, workset bytes, and active tile count; the corrected event-window
expected peak is 1.444 GB. `test_packed_training_preflight.py` passes this exact
large-E regression. The fix changes no candidate kernels or performance
outputs, so the already-completed GPU run was not repeated. No candidate was
truncated and no OOM occurred; the reported performance timings used the
unchanged frozen driver captured in KDA's trusted-input manifest.

## Interpretation boundary

- Inputs are deterministic, pair-unique synthetic supports with controlled
  shared identity, gate states, and event windows. They are not biological
  datasets and do not validate biological semantics.
- The event slice preserves repeated occurrences in its event context and
  intersects count and time bounds before candidate construction. The ranked
  slice uses 512 identities/cell with cohort overlap and independent residual
  ordering. Neither synthetic generator replaces evaluation on real
  Geneformer-style or Karyoplast inputs.
- KDA's local preferred reference remains SIMT because packed had no supported
  improvement. This task-local result is not a global mechanism selection,
  production dispatch rule, or rejection of the CE-GEO representation.
- The v12 attempt cancelled during its quadratic owner-ID uniqueness setup;
  it is retained separately. This report uses the v13 package after the exact
  O(P log P) host uniqueness validation fix.
- No profiler was used to explain the packed slowdown here. The measurements
  establish whole-operation behavior, not a source-line bottleneck.

## Provenance

- KDA task: `task.json`, SHA-256
  `b05f70900612a1c30cbfae26c2b2ab8ee275fd3076b112c1a3de30ea656ecdca`.
- KDA attempt: `c70c526c-3452-40fd-9fca-a4c6066d7f68`; complete raw driver
  outputs and all per-case records are under `workspace/runs/kda-experiments/`
  and summarized in `analysis.json`.
- Scheduler job `b0a983c2-1bcf-4c44-8977-6f272170db97`, attempt
  `1a0376ff-cc41-48b1-bd51-e1b67de727a9`; succeeded, lease released, and
  `cleanup_verified=true`. Receipt and final logs are `scheduler_job.json` and
  `scheduler_logs.json`.
- v13 package identity SHA-256
  `4d3b6f0e1a63ea20785975597849bfad0b0f081221c47d9b858d6581b487afd4`; Torch
  extension SHA-256
  `51b0838ad3466f4c03f693d620aa6b800e8af9ed87a8a3fceb3bd432a05a4f24`.
- Fixture generator SHA-256
  `887437fc3418e06c25e515487891a3224f8dd56493b689bf41bc70e66ae5a76b`; exact
  fixture manifests and per-array identities are pinned in `task.json` and
  `workspace/runs/kda-experiments/attempts/c70c526c-3452-40fd-9fca-a4c6066d7f68/trusted-identities.json`.
- The run used benchmark-driver SHA-256
  `a1d29ecabb79591556ee1b0bc7dc1381d52e11ab5561357733aa8dd60c0d547e` and
  KDA adapter SHA-256
  `e901341ce5d28198ebb5e912490b9778d735371b38ab3f935cf9746ed216c991`; see
  KDA's trusted identity list for the complete source closure.
- Post-run host memory-estimate fix source SHA-256
  `14f5d763eb04f3a0095d623762e1beb383e2dec75fd2dca876459379ed4eb0de` and host
  regression SHA-256
  `022a7ec40447f468e18f63e3c3f0853cc40c121f0b63b7f1c975fa1c86e08fd7`. These
  are separate from the frozen source identity used for GPU timing.
