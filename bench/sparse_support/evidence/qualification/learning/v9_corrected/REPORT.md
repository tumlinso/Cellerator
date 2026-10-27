# Corrected sparse support synthetic learning qualification (v9)

## Run and validity

This is the corrected rerun of the invalidated v8 diagnostic. The v9 binding fix scatters the native derivative-workset vector back to full gate-score order by the derivative gate indices. The frozen v9 build passed the independent GPU regression suite (26/26, cleanup verified), including complete Torch VJP comparison and repeated candidate gate indices across two cells. The regression receipt is retained at [`v9-gradient-scatter`](../../lifecycle/v9-gradient-scatter/).

The training job `2b7608f5-ac63-419d-94a3-435d4d72c5d0` succeeded with exit code 0 and verified cleanup. It used the exclusive Tesla V100-SXM2-16GB after the configured three-sample quiescence check. Frozen identity SHA-256: `be20f601882e918cdd6c74ace0843234a1b4a5c1fd74e2036cfb476222f86fa2`; `_torch` extension SHA-256: `93dd66d4559e455d2fda7be4bd95b79d2ca9d460e7cd309d2978706a4ab958d2`. The source-only manifest is in `source_package_hashes.json`; no binary artifacts are included. `sparse_support.py` has a post-freeze whitespace-only EOF cleanup in the live worktree, so the frozen snapshot hash is authoritative for this run.

Scheduler-executed argv and environment (the captured job spec is `job_spec_and_result.json`; submit through Device Scheduler, not as a direct GPU command):

```sh
PYTHONPATH=/home/tumlinson/Cellerator/build-sparse-support/evidence/torch-v9/package:/home/tumlinson/Cellerator/build-sparse-support/evidence/torch-v9 \
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/home/tumlinson/Software/venvs/cellerator-ml2-py313-cu126/bin/python -u \
/home/tumlinson/Cellerator/build-sparse-support/evidence/torch-v9/examples/sparse_support/train_synthetic.py \
  --device 0 --epochs 50 --soft-epochs 10 --adapt-epoch 20 \
  --support-growth-budget 16 --seed 41027
```

The six models were learned-support, soft-only, and shuffled-support controls for each slice. They shared fixture seed and update count. The run used 10 soft epochs, then 40 thresholded epochs at threshold `0.50`, with adaptation requested at epoch 20.

## Held-out results

Held-out evaluation used 64 cells per slice. Chance references are 0.125 for the eight-class ranked target and 0.25 for the event target.

| Slice / model | Soft loss | Soft accuracy | Pruned loss | Pruned accuracy | Pruned admission |
|---|---:|---:|---:|---:|---:|
| Ranked learned support | 1.253947 | 0.6875 | 1.674484 | 0.5625 | 0.4624 |
| Ranked soft-only | 0.602328 | 0.7812 | 4.183590 | 0.1719 | 0.4249 |
| Ranked shuffled support | 1.792296 | 0.5156 | 2.429595 | 0.3438 | 0.3601 |
| Events learned support | 2.349621 | 0.2812 | 2.556622 | 0.2812 | 0.5979 |
| Events soft-only | 2.083489 | 0.2500 | 2.705913 | 0.2500 | 0.6038 |
| Events shuffled support | 2.923618 | 0.2188 | 2.975315 | 0.1562 | 0.7302 |

Ranked-gene learning is above chance in both soft and pruned evaluation, and the pruned result exceeds the shuffled-support control. The learned pruned model remains below the learned soft model. Event accuracy is weak and only modestly above its 0.25 reference; the event pruned loss is better than the shuffled control, but the 64-cell sample does not establish robust mechanism identification. These fixture results support the corrected trainable path and a limited held-out learning signal; they do not establish generalization to biological data or a universal pruning quality benefit. The held-out sample is small and no confidence interval was computed.

## Discovery, gates, and lifecycle

- The proposal call used `proposal_budget=64` and `attempt_budget=2048`; the native result reported 72 attempts per slice against a candidate-space upper bound of 2,304. The missed-candidate lower bound was marked exact at 2,048; sampled coverage lower bound was 0.027778.
- The fixture begins with 96 support edges (`pair_support`: 16 sources × 6 targets each). Logs reported four ranked and six event proposals and one novel route promoted per slice: ranked source 5 to destination 20 (score 0.106770), events source 3 to destination 23 (score 0.030045). Each promotion raised that slice’s support from 96 to 97 edges, demonstrating one net structural addition per slice.
- Held-out gate admission was 0.4624 ranked and 0.5979 events. Rotating derivative samples selected 256 of 1,453 ranked dormant candidates and 252 of 784 event dormant candidates. The sampler policy is biased cyclic; these remain sampled surrogate gradients, not exact gradients for the whole dormant pool.
- Checkpoint continuation passed for both slices. Event stream continuation passed for a six-event prefix and six-event suffix.

## Recorded costs

- Geometry compile: 0.001357 s.
- Slice preparation: 3.032750 s.
- Route-cache refresh: 0.204705 s aggregate.
- Support publication: 0.002031 s aggregate.
- Peak allocated CUDA memory: 26,968,064 bytes.
- Epoch wall time after initialization was about 0.34–0.47 s; individual timings are in `raw_stdout.txt`.

These are application-run measurements, not a full-cost comparison against dense or alternate sparse implementations.
