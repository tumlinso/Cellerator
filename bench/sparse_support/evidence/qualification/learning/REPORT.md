# Sparse support synthetic learning run (v8, invalidated for pruning)

> **Validity notice:** A subsequent binding review found that, when the derivative workset is a subset of the full gate-score array, the custom backward scattered derivative values into the wrong score slots. This affects gradients after pruning in this run. The 10-epoch all-candidate soft warmup and soft-only controls are unaffected; pruned-training metrics below are retained as raw diagnostics only and do not qualify the declared estimator or support a pruning-quality conclusion. See the v9 follow-up evidence when available.

## Run identity and reproduction

The frozen `torch-v8` source/package snapshot was used without modification. Its SHA-256 manifest is in `source_package_hashes.json`; the runtime extension hashes are recorded separately in `runtime_binary_hashes.json`. No binary files are copied into this evidence directory. All 10 live Python package files present in the snapshot match their frozen SHA-256 values; the training script and selected tests are also pinned in the source manifest. Detailed comparisons are in `live_source_hash_verification.json`.

Scheduler job `0f816b06-6b0e-4139-920e-e23803b10751` succeeded with exit code 0 and verified cleanup. The requested Tesla V100-SXM2-16GB was admitted exclusively after the configured three-sample quiescence check. Exact scheduler receipt, spec, and logs are retained alongside this report.

Scheduler-executed argv and environment (the captured job spec is `job_spec_and_result.json`; submit through Device Scheduler, not as a direct GPU command):

```sh
PYTHONPATH=/home/tumlinson/Cellerator/build-sparse-support/evidence/torch-v8/package:/home/tumlinson/Cellerator/build-sparse-support/evidence/torch-v8 \
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/home/tumlinson/Software/venvs/cellerator-ml2-py313-cu126/bin/python \
/home/tumlinson/Cellerator/build-sparse-support/evidence/torch-v8/examples/sparse_support/train_synthetic.py \
  --device 0 --epochs 50 --soft-epochs 10 --adapt-epoch 20 \
  --support-growth-budget 16 --seed 41027
```

The run trained six models: learned-support, soft-only, and shuffled-support controls for each slice. All used the same fixture seed and update count. Training used 10 soft epochs followed by 40 epochs with thresholded gates at `0.50`; support adaptation was requested at epoch 20.

## Held-out results

Held-out evaluation used 64 cells per fixture. Task chance references are 0.125 for the eight-class ranked target and 0.25 for the event target.

| Slice / model | Soft loss | Soft accuracy | Pruned loss | Pruned accuracy | Pruned admission |
|---|---:|---:|---:|---:|---:|
| Ranked learned support | 1.094883 | 0.6406 | 2.596051 | 0.2812 | 0.5191 |
| Ranked soft-only | 0.602330 | 0.7812 | 4.184646 | 0.1719 | 0.4263 |
| Ranked shuffled support | 1.934409 | 0.4531 | 2.554960 | 0.2500 | 0.3326 |
| Events learned support | 2.318265 | 0.2812 | 2.572302 | 0.2812 | 0.7207 |
| Events soft-only | 2.081066 | 0.2500 | 2.702804 | 0.2500 | 0.6047 |
| Events shuffled support | 2.914433 | 0.2344 | 3.044789 | 0.1406 | 0.6123 |

The all-candidate soft warmup and soft-only controls are valid soft-training evidence. Every result from models trained through the post-warmup pruned phase is invalid for gradient-based learning qualification because of the binding scatter defect above. Their forward numbers remain reproducible diagnostics, but cannot support conclusions about pruning, learned sparse supports, or estimator quality. The 64-cell evaluation is small and has no confidence interval.

## Discovery and lifecycle diagnostics

The gradients-dependent proposal scores, route choices, promotions, and trained post-warmup state also derive from the affected training run; retain them as run diagnostics rather than valid adaptive-learning evidence. Forward-only checkpoint and stream-continuation checks still record what the submitted code reported, but do not repair the training-gradient defect.

- The proposal call used `proposal_budget=64` and `attempt_budget=2048`; the native result reported 72 attempts in each slice against a candidate-space upper bound of 2,304. The result marked the missed-candidate lower bound as exact at 2,048 and sampled coverage lower bound as 0.027778.
- The fixture starts with 96 support edges (`pair_support`: 16 sources × 6 targets each). The proposal log reported four ranked and six event candidates and one selected novel promotion per slice (ranked source 5 to destination 20, score 0.103465; events source 3 to destination 23, score 0.047233). Each promotion raised that slice’s support from 96 to 97 edges. Because post-warmup scoring followed the invalidated gradient phase, these are topology-change diagnostics, not valid adaptive-learning evidence.
- On held-out data, the rotating dormant derivative sample selected 256 of 1,300 ranked dormant candidates and 247 of 535 event dormant candidates. Sampling policy is `biased_cyclic_v1`; these are sampled surrogate gradients, not exact gradients for the full dormant pool.
- Parameter/support checkpoint continuation passed for both slices. Event-stream prefix/suffix continuation passed for a 6-event prefix and 6-event suffix.

## Costs recorded by the application

- Geometry compile: 0.001483 s.
- Slice preparation: 3.080577 s.
- Route-cache refresh: 0.250930 s aggregate.
- Support publication: 0.002130 s aggregate.
- Peak allocated CUDA memory: 26,975,744 bytes.
- Epoch wall time after the first epoch was approximately 0.33–0.54 s in this run; all per-epoch values are retained in `raw_stdout.txt`.

These are application-run measurements. They do not compare complete training-step cost against a dense or alternate sparse backend.
