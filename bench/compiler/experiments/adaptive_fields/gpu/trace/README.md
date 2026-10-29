# E7 adaptive support trace prototype

This CPU-only prototype exercises a low → high → low workset-density trace and
the real host-owned `adaptive_support_manager_v1` publication lifecycle. It
pins three 16-step windows, stages promotion of stable logical edge ID 100 while
the first window is pinned, verifies that publication is rejected with a live
tape, retires that tape, and publishes structure epoch 2. The 48 resulting
worksets are checked against the frozen E7 host-scan oracle.

Run the CPU trace with:

```sh
python3 bench/compiler/experiments/adaptive_fields/gpu/trace/run_trace.py
```

This compiles and runs only the native host support manager and emits the raw
trace, correctness report, and source/toolchain manifest under `artifacts/`.
It does not import Torch, initialize CUDA, call the experiment indexer, or
produce GPU timings.

## Scheduler-owned GPU follow-up

`run_gpu_trace.py` is the CPU-authored runner for the admitted follow-up. It
imports the current frozen `e7_relation_runner.py` helpers only after the
measured-cost table, source identities, and parent scheduler environment pass
preflight. Without `--cost-table`, it writes an explicit unresolved preflight
record and exits before importing Torch or initializing CUDA. With a table,
`--preflight-only` checks the fixture, frozen package, indexer library, E7
runner, and measurement identities without device calls.

The required table schema is
`cellerator.adaptive_fields.e7_trace_measured_cost_table.v1`. It must set
`evidence_kind` to `measured_e7`, identify the hashed source report, and bind
the exact E7 runner, indexer library, trace runner, and small fixture manifest hashes. It
must declare `complete_step_metric` as
`synchronized_training_step_wall_ms`, `evaluation_interval_steps` as `16`, a
measured `policy_evaluation_ms`, an `initial_candidate`, and measured low/high
step, candidate preparation, and candidate switch costs for exactly
`scan_simt_full` and `ordered_postings_simt_full`. Scenario and modeled tables
are rejected before GPU setup.

This is an intentionally bounded choice between two indexers using SIMT with
full-support values on the actual E7 `small` fixture. Any fixed-candidate
comparison or regret claim applies only to these two choices; it says nothing
about the broader six-way E7 portfolio. The CPU support trace's 64-edge fixture
is not a source for this table or for GPU timing costs.

The GPU runner executes a fixed candidate trace and a separate adaptive trace
from the same seed. Each uses three pinned 16-step windows, stages a real
logical support promotion after the first window, publishes it only after
retiring native readers, constructs an epoch-2 immutable C ABI indexer, and
replaces the epoch-1 handle. Every step records exact IDs, canonical ordinals,
state generation, structure epoch, density, and training-step wall time; host
oracle mismatches or stale epochs stop the run. The policy observes the prior
window, evaluates every 16 steps, and compares measured step savings with
preparation already needed and counted in the adaptive trace, measured or
previously observed switch cost, actual window-observation time, and the
measured policy-evaluation allowance. Candidate preparation happens before the
choice so the candidate is ready; its real cost is included once in complete
trace cost, not charged again as an incremental switch cost. Shared support
publication and epoch-indexer rebuild costs are included in complete trace
cost and decision records. They cancel from the switching inequality because
both fixed and adaptive traces must publish the same support and rebuild the
same epoch-bound indexer.

The root task owner supplies scheduler admission. The calibration companion
measures five warmup steps and four rounds of 21 paired full training steps
for each exact low/high query regime. It resets the canonical native owner and
Adam state before each candidate measurement, alternates scan-first and
ordered-first, checks the resulting workset hashes, and separately times each
candidate's indexer preparation, candidate-switch parsing, and the actual
policy function. The generated report is hashed into the measured cost table;
the table is bound to the actual E7 small fixture manifest, frozen E7 runner,
indexer library, and trace runner. These are 84 paired samples per density
regime, not costs copied from the mixed-density E7 run or the CPU 64-edge trace.

Calibration and the 48-step trace must run serially in the same finite
root-admitted GPU job so the table is produced from the exact current inputs:

```sh
python3 bench/compiler/experiments/adaptive_fields/gpu/trace/calibrate_trace_costs.py \
  --report-output bench/compiler/experiments/adaptive_fields/gpu/trace/artifacts/trace_cost_calibration.json \
  --cost-table-output bench/compiler/experiments/adaptive_fields/gpu/trace/artifacts/trace_cost_table.json
python3 bench/compiler/experiments/adaptive_fields/gpu/trace/run_gpu_trace.py \
  --cost-table bench/compiler/experiments/adaptive_fields/gpu/trace/artifacts/trace_cost_table.json \
  --output bench/compiler/experiments/adaptive_fields/gpu/trace/artifacts/gpu_trace_result.json
```

No GPU invocation or timing result is claimed by this preflight implementation.

`test_policy.py` belongs to the CPU support-trace prototype above. Its optional
cost replay uses a separate schema and checks policy arithmetic only; it is
not accepted by `run_gpu_trace.py` and does not provide measured E7 input.

Run the policy arithmetic checks with:

```sh
python3 bench/compiler/experiments/adaptive_fields/gpu/trace/test_policy.py
```

## Scope and unresolved GPU integration

`adaptive_support_manager_v1` provides real staged promotion, stable identity,
structure epochs, pin/retire protection, and publication; the trace executes
those native host APIs. Its C ABI does not produce numeric workset kernels.
The separate E7 experiment indexer is immutable: `af_index_create_v1` prepares
one support epoch and exposes only run/prepare/destroy operations, with no
publication or update call. Consequently this prototype validates support
publication and exact worksets on CPU. The isolated GPU trace runner now
constructs a new epoch-2 indexer from the published support and retires the
epoch-1 handle at the boundary, but this path remains unvalidated until a
root-admitted run supplies the measured candidate table and completes the GPU
trace. No production API was changed.
