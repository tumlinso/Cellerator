# Packed embedding KDA comparison

This workspace compares the existing FP32 occurrence-major backward kernel
with its compile-time warp-per-occurrence alternative. Candidate snapshots,
task contract, common-source hash manifest, focused CMake definition, and
trusted adapter are separate inputs. The packed kernel/header candidate files are frozen at
registration; the build adapter rejects source drift and records the actual
binary hashes. The duplicate smoke file in the candidate manifest is only a
source identity check; candidate builds compile the separately declared
trusted smoke harness.

The numerical gate runs the task-hashed, trusted host-oracle CUDA smoke once per
candidate; the CMake build compiles that harness from the trusted source tree, outside candidate snapshots. It covers code widths 1/2/4/8, dimensions 7/33, and FP16/FP32, then
checks forward output, table gradients, occurrence-weight gradients, and
status. The `--performance` validation run also executes the CPU oracle for
the exact large timing fixture before its CUDA-event measurement. The timing gate is intentionally narrower: resident FP32 backward at
4-bit codes, dimension 128, 512 segments, and 16,384 occurrences. It returns
the existing CUDA-event average over 30 calls as one raw sample per engine
round. This provides evidence for that backward workload only; it does not
claim forward or all-shape performance improvement.

## Register and prepare candidates

Run from the Cellerator checkout after the parent has accepted the generated
candidate and task inputs:

```sh
export PYTHONPATH=/home/tumlinson/kda
PY=/home/tumlinson/Software/miniconda3/bin/python
TASK=planning/v100-framework-2026-10-29/kda/packed_embedding/task.json
WORKSPACE=planning/v100-framework-2026-10-29/kda/packed_embedding/workspace-v2
BASE=planning/v100-framework-2026-10-29/kda/packed_embedding/candidate-sources-r2

$PY -m kda_experiments --task "$TASK" --workspace "$WORKSPACE" register baseline "$BASE/baseline" \
  --manifest "$BASE/baseline/baseline.manifest.json" \
  --hypothesis 'The current occurrence-major backward kernel is the incumbent.' \
  --change 'Compile the existing packed embedding implementation without the warp-backward define.' \
  --baseline --experiment-group packed-embedding-v100-sm70-warp-backward-v2
$PY -m kda_experiments --task "$TASK" --workspace "$WORKSPACE" register warp_backward "$BASE/warp_backward" \
  --manifest "$BASE/warp_backward/warp_backward.manifest.json" \
  --hypothesis 'One warp per sparse occurrence may improve backward locality and reduce repeated segment/override resolution.' \
  --change 'Compile the same source snapshot with CELLERATOR_PACKED_EMBEDDING_WARP_BACKWARD=1.' \
  --parent baseline --experiment-group packed-embedding-v100-sm70-warp-backward-v2
$PY -m kda_experiments --task "$TASK" --workspace "$WORKSPACE" prepare baseline warp_backward --jobs 1
```

Preparation invokes the CPU-only build contract and does not run GPU code.
The root operator explicitly registers
`autobench-definition.toml`; it is deliberately outside the live
`.autobench/registry` directory.

## Run the admitted comparison

The registered Autobench definition prepares the cached KDA candidates in its
CPU build phase. Its single GPU packet runs this command only after Scheduler
admission:

```sh
/home/tumlinson/Software/miniconda3/bin/python \
  '{snapshot}/scripts/validation/kda_packed_autobench.py' \
  --source '{snapshot}' \
  --workspace /home/tumlinson/Cellerator/planning/v100-framework-2026-10-29/kda/packed_embedding/workspace-v2 \
  --mode evaluate
```

The wrapper requires `DEVICE_SCHEDULER_JOB_ID`,
`DEVICE_SCHEDULER_ATTEMPT_ID`, and exactly one Scheduler-bound GPU UUID in
`CUDA_VISIBLE_DEVICES`. The KDA probe obtains visible device zero's CUDA UUID
and requires it to match that binding; it also records the Autobench job and
attempt IDs. Root acceptance must verify the outer Autobench receipt's actual
allocation/UUID. KDA runs in direct mode *inside* the admitted packet, so this
path performs no nested admission or direct out-of-packet GPU work.

After a complete run, inspect it and report selection without changing the
preferred pointer:

```sh
$PY -m kda_experiments --task "$TASK" --workspace "$WORKSPACE" inspect
$PY -m kda_experiments --task "$TASK" --workspace "$WORKSPACE" select RUN_ID
```

The promotion gate requires at least 5% weighted benefit, 95% bootstrap
confidence, and no more than 5% regression on the sole measured workload. The
engine leaves the incumbent preferred when those conditions fail. Applying a
preferred-candidate reference is a separate explicit action.
