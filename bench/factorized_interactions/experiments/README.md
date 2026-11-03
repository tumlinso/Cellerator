# Factorized interaction KDA adapter

This directory contains task-owned glue for evaluating the factorized
interaction candidates with the installed `kda_experiments` engine. The task
uses KDA's direct mode only when launched by `in_job_kda.py` inside the already
admitted Autobench/Device Scheduler job; it does not submit a nested job.

## Evidence boundary

The KDA workspace used for this task is **provisional and experiment-only**.
KDA's preferred candidate is a ranking produced by its engine, not authority to
change Cellerator's production route. `selection_guard.py` writes a conditional
selection record when KDA's direct engine receipt and the fixed project gates
pass. It never accepts an Autobench/Device Scheduler allocation or cleanup
receipt; Oracle must match the final outer receipt, job, attempt, UUID, and
cleanup before root acceptance. Failed guards are retained as rejected evidence.

The root policy is encoded in `protocol.json`:

* 7 paired exploration blocks and 7 paired confirmation blocks per case;
  variants run baseline/challenger, then challenger/baseline, alternating
  within each case across both phases.
* The candidate needs at least 1.10x point median speedup on every case marked
  `candidate_win_required`, with a paired 95% bootstrap lower bound above 1.
* Every declared supported case must have a per-case lower bound of at least
  `1 / 1.05` in benefit units. This is the exact conversion of the 5% maximum
  time regression: KDA stores `max_regression = 1 - 1 / 1.05`.
* The primary metric is the host wall time per call for a batch of 30
  uninstrumented warm forward/VJP calls followed by one stream synchronization.
  This includes caller dispatch, kernel launches, tape work, in-kernel output
  and gradient reductions, and amortized synchronization. CUDA-event forward,
  backward, and forward/VJP vectors are secondary diagnostics; they do not
  decide selection. Preparation, layout upload, conversions, allocation,
  parameter refresh, and setup-plus-warm estimates are retained separately.
  Parameter-refresh samples measure the runner's host wall time for staging and
  completing three parameter uploads. The runner reuploads the same parameter
  values and has no optimizer update, so these are forward/VJP comparisons,
  not optimizer steps.

KDA itself ranks provisionally using `min_improvement = 0`; the trusted
post-selection guard applies the separate 1.10x point-median requirement and
checks every confirmation block, backend identity, order, and per-case loss
bound. Its conditional record remains pending outer receipt/cleanup matching;
do not use the KDA preferred pointer as the accepted route.

## Candidate integration

`prepare_task.py` consumes an explicit candidate-variant map, scheduler site
policy, and win-case list. Its map must contain exactly one incumbent and one
challenger. It creates a CUDA `sm_70` task in KDA direct mode and binds the
expected GPU UUID plus trusted in-job entrypoint. `in_job_kda.py` verifies the
inherited Scheduler job/attempt, CUDA visibility, and exact attempt cgroup
before invoking KDA. `register_candidate.py` refuses candidate snapshots that
are not complete Cellerator source trees with a checked KDA manifest. The
source tree and manifest must be finalized before registration.

The source-bound candidate runner accepts four explicit modes:
`native_direct`, `minimal_direct`, `direct_unique_output`, and `packet`. The
incumbent is mapped per generated workload: the exact record/backend predicate
chooses `direct_unique_output` when outputs are one-to-one and the direct
backend is applicable; otherwise it chooses `minimal_direct`. The challenger
uses its fixed packet mode. The runner reports requested mode and actual
implementation/layout, including stable fallbacks. The adapter checks those
facts against task-bound workload records before retaining measurements.

The build command invokes the task-trusted candidate helper only after checking
its bytes against the helper in KDA's immutable candidate snapshot. It reports
the native build manifest, runner, and linked CUDA library as KDA build
artifacts. The validator always delegates to the existing
`validate_runner.py` and independent reference. The benchmark adapter runs both
candidate binaries for one case and one KDA round, alternates their order,
stores both full metric records in the attempt directory, and returns the
requested candidate's one raw block sample to KDA. The KDA engine then performs
its normal paired ranking/bootstrap decision.

All correctness and timing invocations pass the validator's optional
`--reference-cache-dir` argument using the same
`factorized-reference-cache/` directory under that KDA attempt's monitored
output tree. The cache stores only verified independent-reference arrays; the
validator still runs every native candidate and checks its outputs and all
VJPs on every call. The cache helper is itself included as a trusted task
input, alongside the reference, workloads and wire protocol. It does not write
to candidate source or build directories.

During admitted validation, each candidate is checked on every fixed
correctness-only workload (including cancellation, saturation, repeated-role,
empty-shape, and high-fan-in cases). Packet validation uses packet mode on all
12 workloads. Incumbent validation applies the generated record's strongest
direct mode per workload. These are not timed as route-selection cases. Each
paired timing block also measures `native_direct` for descriptive comparison;
its position rotates across blocks and it is excluded from KDA's paired score.
The selection manifest lists every untimed performance case and every
correctness-only case as a direct fallback.

## Setup and use

Use a dedicated provisional KDA workspace under the outer Autobench packet's
managed `{output}` directory, outside any production preference or routing
state. This keeps each attempt's shared reference cache in the monitored output
tree. Build the candidate source snapshot from a finalized,
source-bound Cellerator closure, then create its KDA manifest from that closure
with the helper described by the task owner. Example flow (the site configuration
and candidate map are installation-specific):

```sh
python3 bench/factorized_interactions/experiments/prepare_task.py \
  --output /path/to/provisional/task.json \
  --variant-map /path/to/provisional/variants.json \
  --site-policy /path/to/autobench-scheduler-policy.json \
  --nvcc /opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/bin/nvcc \
  --expected-uuid GPU-00000000-0000-0000-0000-000000000000 \
  --candidate-win-case packet_a4_m32_u16 \
  --task-id factorized-interaction-kda \
  --max-seconds 14400

python3 bench/factorized_interactions/experiments/candidate_manifest.py \
  --source /path/to/frozen/source \
  --closure-manifest /path/to/source-closure.json \
  --output /path/to/provisional/direct.manifest.json

python3 bench/factorized_interactions/experiments/register_candidate.py \
  --task /path/to/provisional/task.json \
  --workspace /path/to/provisional/workspace \
  --candidate-id direct-control \
  --source /path/to/frozen/source \
  --manifest /path/to/provisional/direct.manifest.json \
  --variant-map /path/to/provisional/variants.json \
  --hypothesis 'qualified strongest direct control' \
  --change 'bind the exact source and native runner' \
  --baseline --kda-root /home/tumlinson/kda
```

Register the challenger using the same wrapper. Before outer GPU admission,
prepare both candidates with KDA's CPU-only build command:

```sh
PYTHONPATH=/home/tumlinson/kda python3 -m kda_experiments \
  --task /path/to/provisional/task.json \
  --workspace /path/to/provisional/workspace \
  prepare direct-control packet-candidate --jobs 1
```

Then configure the Autobench packet environment to include
`PYTHONPATH=/home/tumlinson/kda` (prepend it to any required existing value),
and run this command in the admitted worker with the exact UUID bound in the
task:

```sh
python3 bench/factorized_interactions/experiments/in_job_kda.py \
  --task /path/to/provisional/task.json \
  --workspace /path/to/provisional/workspace \
  --expected-uuid GPU-00000000-0000-0000-0000-000000000000 \
  --candidates packet-candidate \
  --incumbent direct-control
```

The entrypoint records the observed job, attempt, UUID, cgroup, task hash, and
entrypoint hash before `exec`ing KDA. That context is not a Scheduler receipt.
KDA's direct probe, validators, and benchmark then run in the same admitted
job and inherit its environment. Do not invoke the KDA `evaluate` command or
task probe/validate/benchmark commands directly outside this entrypoint.

The paired block's one KDA sample is the runner's uninstrumented host-wall
batch mean; it is not fabricated from the event vectors or from their p50.
Each KDA round is one independent paired process block, with seven exploration
and seven confirmation blocks. `selection_guard.py` recomputes each case's
point speedup from those confirmation-block values, checks the engine's
per-case bootstrap result and raw source/binary identities, and preserves the
secondary sample vectors from all blocks. Event-based setup-plus-warm p50
estimates are secondary and are not used as host-wall reuse horizons.

After an engine run completes, call `selection_guard.py` with that run ID and
an output path outside the provisional KDA workspace. It reads the engine's
retained summary and raw response artifacts; it does not recompute or replace
KDA's ranking. On rejection it writes a rejected-gate record. When numerical
and performance gates pass, it writes a **conditional** selection record with
outer receipt/cleanup verification pending. Oracle and root must match the
completed outer receipt to the recorded job, attempt, and UUID before accepting
any route. The record includes case routes, metric summaries, backend IDs,
source/binary hashes, task and run hashes, the KDA direct receipt, and observed
outer context.

Accepted routes are scoped to the exact benchmark case IDs in the task.
Unmeasured performance and correctness-only cases remain direct fallbacks.
This case-name map is evidence for a later root-owned routing decision, not a
general runtime predicate or production selection.

CPU-only adapter contract tests live in `tests/`. They use synthetic requests
and measurements and do not launch CUDA, contact Autobench, or schedule jobs.
