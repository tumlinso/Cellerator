# Attention row-CTA launch geometry

The benchmark compares the frozen attention source against one isolated source
variant that changes six staged row-kernel launch grids from `blocks(Q)` to
`query_blocks(Q)`. Both launch helpers retain their existing 65,535-block cap.
The candidate preserves the operation, API, kernel implementation, per-row
arithmetic, and CSR semantics.

The task uses the existing full attention smoke matrix for FP16/FP32 staged
and fused forward/backward correctness, plus a trusted host-FP64 oracle for the
exact resident Q=512, N=96, D=32, V=32 FP32 timing fixtures. KDA compares both
staged and mixed-fused routes on all-degree-8 and mixed-degree-8/65 rows, so
the fused route remains an explicit regression guard.

## Prepare the paired KDA experiment

The task and candidate snapshots are prepared locally, with no GPU execution:

```sh
export PYTHONPATH=/home/tumlinson/kda
PY=/home/tumlinson/Software/miniconda3/bin/python
TASK=planning/v100-framework-2026-10-29/attention/launch_geometry/task.json
WORKSPACE=planning/v100-framework-2026-10-29/attention/launch_geometry/workspace
BASE=planning/v100-framework-2026-10-29/attention/launch_geometry/candidate-sources
GROUP=attention-v100-sm70-row-cta-forward-v1

$PY -m kda_experiments --task "$TASK" --workspace "$WORKSPACE" register baseline "$BASE/baseline" \
  --manifest "$BASE/baseline/baseline.manifest.json" \
  --hypothesis 'The frozen staged row CTA implementation is the incumbent.' \
  --change 'Compile the frozen attention.cu with the same shared CUDA source closure.' \
  --baseline --experiment-group "$GROUP"

$PY -m kda_experiments --task "$TASK" --workspace "$WORKSPACE" register one_cta_per_row "$BASE/one_cta_per_row" \
  --manifest "$BASE/one_cta_per_row/one_cta_per_row.manifest.json" \
  --hypothesis 'One CTA per query row should expose row independence when staged kernels currently under-launch the grid.' \
  --change 'Change only the six row-owned staged kernel launch grids to query_blocks(Q), preserving arithmetic and grid-stride behavior.' \
  --parent baseline --experiment-group "$GROUP"

$PY -m kda_experiments --task "$TASK" --workspace "$WORKSPACE" prepare baseline one_cta_per_row --jobs 1
```

Preparation configures an isolated Release build for `sm_70`, compiles with one
job, and does not launch a CUDA kernel. The new Autobench definition remains
under this planning directory; registration in the active Autobench registry,
Scheduler admission, GPU execution, and any promotion are owned by the root
task.

## GPU evaluation contract

The task's Autobench packet runs the KDA evaluation wrapper only after the
root-owned Scheduler admission. The wrapper requires one UUID-bound device
provided by the admitted Autobench attempt. Each candidate must pass
`attention_smoke --device` and the exact-fixture FP64 oracle before timing.
Each of four paired cases returns seven CUDA-event samples, each averaging 25
resident forward calls after four warmups. KDA takes five paired rounds and
four fresh confirmation rounds, with a 5% minimum aggregate improvement and a
5% maximum per-case regression. Preserve all candidate build fingerprints,
samples, device identity, and the outer Autobench/Scheduler receipt.

The initial measured baseline is from the canonical V100 Autobench snapshot at
`planning/v100-framework-2026-10-29/REPORT.md` and
`.autobench/evidence/cellerator-framework-v100/4/f193783e70ece3159f85938570e8d81526ff26cd73268576f90effa6047d3c73/attention-benchmark-none.log`.
