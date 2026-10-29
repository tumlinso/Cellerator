# CE-LIVE quantitative forward v1

CE-LIVE-31 validates the checksum-pinned PBMC3K computational fixture through
the planner-backed executable program. This is computational evidence only;
it makes no donor, sample, chemistry, normalization, comparison, or biological
interpretation claim.

The foreground controller verifies the committed manifest hashes, flattens the
local NPZ into a temporary test artifact, binds the feature/gene-source to
cell/row-destination relation, explicitly constructs an FMP1 physical
projection, enumerates the built-in catalog, supplies complete phase costs,
prepares the planner winner once, and runs two mutable value generations over
one immutable topology. Output is compared element-for-element with an
independent destination-row CSR coordinate referee.

Required widths are `1, 16, 17, 31, 32, 48, 64`. Each runs at reuse horizons
`1`, `8`, and `1024`. The current built-in catalog exposes exactly one legal
FMP1 schedule per width: the warp schedule through `N=16` and the CTA schedule
from `N=17` through `N=64`. Planner regret is therefore truthfully zero against
the complete legal candidate set for each request. The retained cuSPARSE
candidate is not currently a legal built-in SpMM catalog entry, so this task
does not misrepresent it as a compared candidate or silently add catalog
integration outside CE-LIVE-31 ownership.

Run correctness and focused timing with:

```bash
python /home/tumlinson/.agents/skills/cuda/scripts/cuda_controller.py run \
  --spec bench/ce_live/forward/cuda_controller.json --json
```

Run memcheck with:

```bash
python /home/tumlinson/.agents/skills/cuda/scripts/cuda_controller.py run \
  --spec bench/ce_live/forward/sanitizer_controller.json --json
```

## Accepted foreground evidence

Evidence `16c72445-6e94-4951-abe4-f57f59705237` passed both the one-repeat
correctness phase and the five-repeat benchmark phase. The controller held the
host benchmark mutex and observed three consecutive idle samples before the
run, with no foreign GPU process or throttle reason.

- Source commit at capture: `ec111b5d9ebb644ade85e99f3d2a03a905b28c4f`
- GPU: Tesla V100-SXM2-16GB, compute capability 7.0, one device
- CUDA compiler: NVIDIA HPC SDK CUDA 12.9 (`sm_70`)
- Driver: 580.173.02
- Fixture: 512 destination cells, 32,738 source features, 433,808 logical edges
- Values/RHS/output/accumulation: f16/f32/f32/f32
- Repeats: one correctness pass, then five timed runs per generation
- Correctness: every output at all 21 width/reuse combinations and both value
  generations matched the independent CSR referee
- Contamination: none reported by the controller

| N | selected legal schedule | median kernel range (ms) | reuse-1 total (ms) | reuse-1024 total (ms) | regret |
|---:|---|---:|---:|---:|---:|
| 1 | FMP1 warp | 2.872-2.873 | 69.898 | 2.941 | 0% |
| 16 | FMP1 warp | 3.141-3.456 | 70.481 | 3.209 | 0% |
| 17 | FMP1 CTA | 3.999-4.241 | 71.266 | 4.067 | 0% |
| 31 | FMP1 CTA | 4.015-4.017 | 71.041 | 4.084 | 0% |
| 32 | FMP1 CTA | 3.901-3.902 | 70.927 | 3.970 | 0% |
| 48 | FMP1 CTA | 4.028-4.031 | 71.056 | 4.097 | 0% |
| 64 | FMP1 CTA | 4.041 | 71.065 | 4.109 | 0% |

The complete 21-record machine-readable result is
`bench/ce_live/forward/pbmc3k_forward_v1.jsonl`. Complete totals include the
explicit host preparation, FMP1 construction, static value packing, transfer,
kernel, and return phases amortized by the declared reuse horizon. These are
foreground validation measurements, not a general performance claim.
