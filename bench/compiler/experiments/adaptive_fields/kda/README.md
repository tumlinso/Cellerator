# Trusted portfolio collection through KDA

The single registered `portfolio_collector` is a trusted measurement harness,
not a competing numerical implementation. Its benchmark request executes the
six bounded indexer/numerical combinations paired against `scan_simt_full`.
The KDA metric contains baseline samples for protocol validation; its
single-candidate preferred state is **not** evidence of a numerical winner.
Candidate comparisons come from the six actual paired rows retained alongside
each request/result record.

KDA supplies two exploration and two confirmation rounds. Each collector
request runs one round of five warmups and 21 paired samples per combination.
This produces 84 measured pairs for each combination/fixture. Validation
requests must first pass all six implementations on each required fixture.
Workset IDs and masks are exact; arithmetic tolerances remain operation-specific.

CPU-only setup from the repository root:

```sh
python bench/compiler/experiments/adaptive_fields/kda/create_task.py
PYTHONPATH=/home/tumlinson/kda python -m kda_experiments \
  --task bench/compiler/experiments/adaptive_fields/kda/task.json \
  --workspace bench/compiler/experiments/adaptive_fields/kda/workspace \
  register portfolio_collector \
  bench/compiler/experiments/adaptive_fields/kda/collector \
  --manifest bench/compiler/experiments/adaptive_fields/kda/collector/manifest.json \
  --hypothesis 'Measure the bounded six-way complete relation portfolio' \
  --change 'Isolated paired measurement collector' --baseline
```

Run `evaluate --candidates portfolio_collector --incumbent portfolio_collector`
only inside the root-owned finite Device Scheduler job. Retain its actual
submission, attempt, logs, and verified cleanup receipt. The task's direct
execution setting is subordinate to that external admission; it never grants
GPU access. A numerical failure stops measurement rather than shrinking or
silently skipping a fixture.

Measured reuse horizons and the adaptive trace are separately retained probes;
they are not inferred from multiplying a steady-state median.
