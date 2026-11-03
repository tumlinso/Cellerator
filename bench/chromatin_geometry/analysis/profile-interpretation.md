# Offline profile interpretation at campaign pause

The retained shared-forward diagnostic supports a small-grid,
metadata/synchronization interpretation. It does **not** establish an HBM
bandwidth bottleneck, a useful padding specialization, or general HBM savings.
Final v3 direct / matched two-tile cached / shared profiles are unfinished and
are not used here. No new GPU work was performed for this analysis.

## Evidence boundary

NCU packet [b7f04ad8](../runs/packets/b7f04ad8-b97a-4791-a9cd-0030e810beb1/profile/summary.json)
and Nsys packet [02252219](../runs/packets/02252219-5bed-4350-94a3-ad5feeb47507/profile/summary.json)
share source archive `f9bfbd35…`, binary `02a282c4…`, workload `cf2d7bdf…`
and GPU `GPU-cf22c41f-5b58-77b1-3535-8fadd1ca6505`. They cover the synthetic
2,304-node/64-root complete case, supplied gates, FP16 product, incremental
100% leaf changes, tile64 and one staging buffer. The demanded DAG has 192 nodes.
The selected kernel is shared **forward**, with grid `(2,8,1)` and 128 threads
per CTA. NCU 2025.2.1 profiled one application launch using 81 replay passes,
clock control `none`; its duration is diagnostic, not application throughput.

The first selected forward follows the configured input toggle. The frozen
fixture reconstructs 11 eligible group occurrences out of 16,384, compared with
6,545 before the toggle. Its forward requires 88 distinct participant values
for 88 participant-slot uses: this low-overlap envelope offers no repeated X
values for staging to remove. Coalescing and parameter reuse remain separate
possibilities. These are source/input counts, not NCU-observed array-specific
byte counts.

## Measured NCU evidence

| Quantity | Retained value | Interpretation |
|---|---:|---|
| Selected duration | 58.816 µs | Explicit CSV unit; no speedup inference |
| Grid / device | 16 CTAs / 80 SMs; 0.04 waves/SM | Very small grid |
| Theoretical / achieved occupancy | 31.25% / 5.39% | Capacity differs from achieved concurrency |
| Active / eligible warps per scheduler | 1.000 / 0.099 | Few ready warps |
| Registers / shared allocation per CTA | 80/thread; 16,432 bytes | Shared allocation limits theoretical residency to 5 CTAs/SM |
| DRAM read / write | 71,872 / 0 bytes | Collected launch only; writes may remain in cache |
| DRAM throughput | 0.137% of reported peak | No DRAM saturation evidence |
| L1/TEX / L2 hit rate | 96.06% / 71.57% | Aggregate cache paths, not proof of reuse across evaluations |
| L2 sectors | 6,339 | Measured sectors; no direct L2-byte counter was collected |
| Executed local loads / stores | 0 / 0 | No executed spill traffic observed |
| Shared load / store wavefronts | 114 / 42 | 156 total; 148 ideal |
| Excess shared wavefronts | 8, or 5.13% of 156 | Not 5.13% of kernel time |

Shared load conflicts are 8; store conflicts are 0. NCU's shared-access rule
estimates 0.97% potential kernel improvement, but rule estimates are not
measurements. This evidence does not justify a padding candidate. The 80
registers alone also do not establish spilling; both local instruction and
local request counters are zero in this launch.

NCU source correlation maps 197/197 sampled PCs. Of 566 recorded samples,
124 are barrier samples, 167 waits and 105 long-scoreboard samples. Eighty-eight
barrier samples map to frozen source line 491, immediately after the inter-tile
barrier at 490; another 32 map immediately after the demand barrier. Leading
long-scoreboard locations are the cohort signature scans at 437–438. These
counts support metadata dependency and synchronization hypotheses; they are
not additive critical-path percentages. Average active/predicated-on threads
are 30.58/27.38 per warp, but that includes preparation/control instructions
and does not prove useful numerical lanes or zero divergence. Volta provides
no PM-sampling timeline here, so no measured per-SM tail shape is claimed.

## Nsys phase context

The same-build Nsys run records three forward calls at 53.984–55.968 µs and
three backward calls at 54.656–59.200 µs. Across the whole instrumented process,
45 kernels total 0.520 ms: forward/backward contribute 0.334 ms, cohort
construction 0.074 ms and condition levels/root exposure 0.079 ms. This aggregate
includes validation, measurement and the extra phase probe; it is not one
recurrent evaluation.

The phase probe reports condition 27.94 µs, clears 25.25 µs, preparation 30.53 µs,
forward 60.86 µs, backward 61.50 µs and model update 8.13 µs. Its recurrent host
sample is 0.262 ms. Cold preparation 1,120 ms and output recovery 52.9 ms show
that this instrumented process contains much larger setup/host costs too.
Host API and GPU phase sums overlap and must not be added as independent
wall-clock costs. These single profiled values are not confirmation timings.

## Attribution and strongest counterinterpretation

Source dispatch matters before memory reuse: direct and bitmap consumers
reserve a dense cell×group warp grid; per-cell worklists return early beyond
their count. The original cached cohort owns one tile per CTA, while shared
owns two. A cohort gain can therefore reflect dispatch and eligibility
amortization. The new two-tile cached control fixes that ablation relationship,
but its matched v3 profiles remain unfinished.

Conversely, low utilization can accompany successful work exclusion. Increasing
the grid merely to improve an occupancy percentage could add work and lose
complete latency. This almost-inactive hard-forward launch says little about
dense activity, all-D STE backward, grouped linear blocks, public replay or a
warm multi-evaluation pipeline. Final paired same-device timings and matched
counter profiles are required for those claims. The older direct Nsys packet
on another GPU/build is excluded from speed comparisons.

Exact file hashes, selected counters, source identities and extraction methods
are retained in [profile-interpretation-provenance.json](profile-interpretation-provenance.json).
