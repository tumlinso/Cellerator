# Event-task route/readout diagnostic

## Verdict

The corrected v9 event task has a real route/readout support ceiling. At each
event step, `KaryoplastEventSlice` passes only the **current event entity** to
the sparse relation. History and latent state enter gate scoring, but they do
not enter the relation input that produces target rows. The held-out fixture's
last event is a distractor in IDs 8–15. Its seeded support has only 25 of the
32 possible routes from those inputs to the four label destinations 16–19.

On the exact v9 held-out seed and 64-cell fixture, only **47/64** true labels
have an edge from the current last-event source. The reported learned soft and
pruned accuracy is **18/64**. Because a target with no route has an exactly
zero output row, and the model has no readout bias, its target logit is zero.
Output IDs 0–15 also have no incoming support and therefore have zero logits;
`argmax` selects an earlier zero output instead of an unsupported target at
16–19. Thus at least 17 held-out errors are structurally forced by the current
input support. This explains part of the weak result, but it does not explain
all of it: the other 29 errors occur among the 47 examples with a target route.

The qualifying run is [the v9 corrected report](../../evidence/qualification/learning/v9_corrected/REPORT.md),
with frozen source identity
`be20f601882e918cdd6c74ace0843234a1b4a5c1fd74e2036cfb476222f86fa2`. Its
`source_package_hashes.json` identifies the exact training driver and Torch
module used below.

## CPU coverage and expressivity check

`route_coverage.py` imports the frozen v9 fixture generator with the run's
training seed 41027 and held-out seed 241027. It uses the existing
`pair_support(seed)` policy (16 sources, six of eight destinations per source),
the observed event/timestamp arrays, and the public count/time windows. It
reads only the fixture's `events`, `targets`, `timestamps`, `count_windows`, and
`time_windows` fields. It does not read `validator_support` or
`validator_mechanism`; it does not edit the generator, labels, support, or
validation rules.

The exact coverage result is saved in [`route_coverage.json`](route_coverage.json):

- Current last-event-only input: 47/64 examples have a true-target route;
  17/64 (26.6%) do not.
- The post-training promotion reported for v9 is source 3 to destination 23.
  It changes none of those paths: last-event inputs are 8–15 and task labels
  are 16–19.
- The same allowed count/time-window history contains five event occurrences
  per held-out cell. If the relation input used that observed history instead
  of only the current event, at least one support route to the true target is
  present for **64/64** examples. This is a route-coverage upper bound, not a
  trained accuracy estimate; other history routes and numerical interference
  can still affect the readout.

The implementation evidence is in the frozen v9 package named by the
qualification record:

- `train_synthetic.py::event_provider` returns the window-eligible occurrence
  sources for gate context; the same list is aggregated into `history_input`.
- `KaryoplastEventSlice._forward_pinned` constructs `inputs` from only the
  current event, builds gate context from `history_input + latent_state`, then
  calls the sparse relation with `inputs`.
- `next_event_logits` emits one logit for each of the 24 entity destinations,
  and `training_loss` passes all 24 logits to cross-entropy. The report's 25%
  chance reference assumes only the four valid targets are choices; the
  implemented readout is not masked to those four. Uniform random choice over
  all 24 IDs would score 1/24 (4.17%), although the model's actual zero-tie and route
  structure is not uniform.
- `tests/sparse_support/test_cuda_training.py::test_relation_forward_and_ste_vjp_match_torch_reference`
  builds a sparse relation output from zero and confirms that only the
  destination with an explicit edge receives a value.

The local Project Control evidence store had no matching observation; the
qualification report and frozen v9 source/hash manifest are the evidence used
here. No GPU work was run.

## Controlled readout comparison

The CPU route gap motivated one matched comparison. A paired history-input
model receives the same count/time-window occurrence sum already used in gate
context, while the baseline relation receives only the last event. Fixtures,
targets, support, gate context, threshold, optimizer and schedule are matched;
paired parameter vectors were verified bit-identical. Training uses the
existing unmasked 24-logit cross-entropy. No validator-only field enters either
model.

## Reproduction

```sh
PYTHONPATH=/home/tumlinson/Cellerator/build-sparse-support/evidence/torch-v9/package:/home/tumlinson/Cellerator/build-sparse-support/evidence/torch-v9 \
  /home/tumlinson/Software/venvs/cellerator-ml2-py313-cu126/bin/python \
  bench/sparse_support/experiments/event_learning_diagnostic/route_coverage.py
```

Hashes for the diagnostic source/result and frozen v9 inputs are in
`evidence/sha256.txt`.

## Fixed 512-cell follow-up

The bounded CPU prespecification uses the frozen v12-validator package
(`identity.json` SHA-256
`99da528db6f4ba14bec983b6dfe31f12a065dbdc8b39c0dfb224ed0e969fca07`) and the
same training seed 41027 / held-out seed 241027. On the fixed 512-cell fixture,
the learned support gives a current-last-event route to the observed target for
392/512 cells, while one of the five observed count/time-window occurrences
provides a route for 512/512. The degree-preserving shuffled support gives
274/512 and 489/512, respectively. This is a route-availability bound, not
trained accuracy.

The held-out target counts are [127, 123, 136, 126] for IDs 16–19; the empirical
constant-label prior is 26.56%. Because the last distractor source is repeated
by cell-index phase, its held-out-only conditional-majority oracle is 32.03%.
That oracle uses the held-out labels only for diagnosis and is never exposed to
training. These finite-fixture priors make the historical 25% reference
insufficient on its own.

The prespecification is in [`trial_spec.json`](trial_spec.json) and the driver
is [`history_readout_trial.py`](history_readout_trial.py). CPU coverage and
prior diagnostics are [`cpu_coverage_512.py`](cpu_coverage_512.py) and
[`cpu_coverage_512_v13.json`](cpu_coverage_512_v13.json).

The same CPU check was repeated against the frozen v13 owner-validation
snapshot (identity SHA-256
`4d3b6f0e1a63ea20785975597849bfad0b0f081221c47d9b858d6581b487afd4`) and
produced identical support coverage and label-prior counts; see
[`cpu_coverage_512_v13.json`](cpu_coverage_512_v13.json). The GPU driver
requires the selected immutable snapshot path and identity digest via
environment variables and verifies the runner, module, and extension against
that snapshot before training. Its [v13 qualification](../../evidence/qualification/v13-owner-validation/REPORT.md)
completed 63 tests before this trial.

The fixed run used 50 epochs (10 soft warmup, then 40 pruned epochs for the
learned and shuffled controls), LR 0.02, 96 support edges, no support growth,
32 train cells per epoch, and the held-out seed 241027 with 512 cells. The
single successful job `b83b39e2-bfe3-4633-abd0-8ee05e6e6f7a` ran on GPU3
`GPU-6c1cac7f-a360-0aef-ba98-2828bfd1db1a`; admission passed three consecutive
idle samples. It exited 0 and cleanup was verified. The preceding attempt
`8e1b662b-19d0-4f62-9dcb-4395e9492c74` stopped before completing epoch 0 due
to an FP16/FP32 prototype-buffer mismatch; it also has a terminal cleanup
receipt. The correction changed only the temporary history accumulation
buffer dtype. The trial config and learning schedule were unchanged on retry.

Held-out accuracy and cross-entropy use all 24 logits. Each cell below gives
`top-1 accuracy / CE`; the soft-only control was trained with gates unpruned
for all 50 epochs, while its pruned result is evaluation-only.

| Support/training control | Current input soft | Current input pruned | History input soft | History input pruned |
|---|---:|---:|---:|---:|
| Learned support, pruned training | 0.238 / 2.378 | 0.238 / 2.649 | 0.955 / 0.055 | 0.994 / 0.047 |
| Learned support, soft-only training | 0.246 / 2.065 | 0.246 / 2.861 | 1.000 / 0.000014 | 0.752 / 1.578 |
| Shuffled support, pruned training | 0.221 / 2.799 | 0.133 / 2.948 | 0.656 / 1.146 | 0.635 / 1.314 |

The learned/pruned history-input model scored 509/512 in the full 24-way
readout, compared with 122/512 for current-only. Its pruned gate admission
rates were nearly identical (0.7098 vs 0.7085), so the gain is not explained
by admitting more edges. The history/pruned model predicted ID 0 for three
examples and the correct target IDs for the other 509. By contrast, current
only predicted IDs 16–18 and never selected target 19. The shuffled history
model improved over its current-only counterpart but remained well below the
learned-support model.

The result supports a relation-input bottleneck on this task: gate context
already contains the window history, but the current-only relation lacks
routes from many observed targets' last distractor sources. Current-only
coverage was 392/512, versus 512/512 from a window occurrence. Coverage is not
the whole story: 392 routes were available but the learned current-only model
correctly predicted only 122 examples. The held-out global constant-label
prior is 26.56%, and a held-out-only current-source-conditioned oracle is
32.03%; the current-only model is below both. These baselines are diagnostics,
not training inputs.

This is one synthetic seed and one 512-cell fixture, without a confidence
interval. It establishes the bottleneck for this fixture family, not for other
tasks or biological data. Full predictions and training summaries are in
[`gpu3_trial_result.json`](evidence/gpu3_trial_result.json); exact scheduler
receipts/logs and post-cleanup resource availability are retained under
[`evidence/`](evidence/).
