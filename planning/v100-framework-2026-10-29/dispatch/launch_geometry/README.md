# Dispatch weight-gradient launch geometry experiment

## Hypothesis and scope

`grad_weight_kernel` assigns `blockIdx.x` to one routing-weight gradient and
uses a grid-stride loop for any remaining selections. Its two current launch
sites choose `blocks_for(selection_count)`, which divides selections by the
kernel's 256-thread elementwise block size. At 256 selections that launches one
128-thread reduction CTA, which serially visits all 256 selections. The sole
candidate uses up to one CTA per selection, capped by the existing 65,535-block
grid bound; the kernel grid-stride loop still covers larger inputs. It changes
only launch geometry and retains the FP32 reduction arithmetic and order.

The paired resident benchmark uses 64 origins, 256 selections, and FP32
features `{1,32,256,2048}`. Each candidate first runs the full dispatch smoke
(FP16 and FP32 feature tails, empty routes and empty features), then a separate
nonzero-data oracle at all four timed shapes. The exact-timing oracle checks
every expert gradient against `w[d] * grad_origin[origin(d), f]` and every
weight gradient against a host FP64 sum of the represented FP32 inputs. It uses
relative tolerances of `2e-6` for expert gradients and `2e-4` for weight
gradients. The timing driver reruns the relevant oracle before sampling.

Each raw timing sample is a CUDA-event average over 300 resident
`DispatchPlan::combine_backward` calls, after 100 warmups. The API call launches
both backward kernels; immutable plan setup, input upload, allocation and
transfers are outside the measured interval. Seven event samples are returned
per case and engine round. KDA evaluates five paired rounds and four fresh
confirmation rounds, with a 5% minimum geometric improvement and a 5% maximum
per-case regression. This is a narrow backward workload comparison, not a
general dispatch speedup claim.

## Register candidates and prepare

From the Cellerator checkout, register the two immutable source snapshots:

```sh
export PYTHONPATH=/home/tumlinson/kda
PY=/home/tumlinson/Software/miniconda3/bin/python
TASK=planning/v100-framework-2026-10-29/dispatch/launch_geometry/task.json
WORKSPACE=planning/v100-framework-2026-10-29/dispatch/launch_geometry/workspace
BASE=planning/v100-framework-2026-10-29/dispatch/launch_geometry/candidate-sources
GROUP=dispatch-v100-sm70-weight-gradient-grid-v1

$PY -m kda_experiments --task "$TASK" --workspace "$WORKSPACE" register baseline "$BASE/baseline" \
  --manifest "$BASE/baseline/baseline.manifest.json" \
  --hypothesis 'The frozen dispatch implementation is the incumbent.' \
  --change 'Compile the frozen dispatch.cu with the same shared CUDA source closure.' \
  --baseline --experiment-group "$GROUP"

$PY -m kda_experiments --task "$TASK" --workspace "$WORKSPACE" register one_cta_per_selection "$BASE/one_cta_per_selection" \
  --manifest "$BASE/one_cta_per_selection/one_cta_per_selection.manifest.json" \
  --hypothesis 'One reduction CTA per selection should expose the independent routing-weight gradients to more SMs when the current elementwise grid under-launches.' \
  --change 'Change only the two grad_weight_kernel launch grids to min(selection_count, 65535), preserving the kernel grid-stride loop and arithmetic.' \
  --parent baseline --experiment-group "$GROUP"

$PY -m kda_experiments --task "$TASK" --workspace "$WORKSPACE" prepare baseline one_cta_per_selection --jobs 1
```

Preparation uses one build job and compiles only for SM70. It does not launch a
GPU kernel. The parent operator owns registration of `autobench-definition.toml`
and admitted GPU evaluation; this definition is intentionally outside the
live `.autobench/registry` directory.

## Admitted evaluation

The Autobench GPU packet runs
`kda_dispatch_geometry_autobench.py --mode evaluate` only after scheduler
admission. The wrapper requires scheduler job and attempt IDs plus exactly one
scheduler-issued GPU UUID. The KDA probe checks that UUID against the CUDA
runtime device and records tool, driver, runtime, Autobench, and device
identity. Root acceptance must independently verify the outer Autobench
receipt and GPU allocation. KDA uses direct mode inside that admitted packet,
with no nested or out-of-packet launch.

The expected result is either promotion by the declared paired confidence and
regression criteria or retention of baseline. In either case, retain per-case
round samples and candidate binary/source hashes in the KDA workspace.
