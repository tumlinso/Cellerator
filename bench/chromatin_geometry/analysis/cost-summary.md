# Qualified cost summary

This report summarizes only packets listed as accepted by the selected qualification export. The source qualification and raw campaign records remain authoritative for admission, correctness, sanitizer and performance decisions.

Qualification SHA-256: `548758f9b3af3d7667eb8a65aa2bdb29bda65723eae764bd6a4eb239f0cf379c`  
Contract SHA-256: `237da579f5068fc26469f52c79b7ae33beb1ab0af057129a27c186193c5c8147`  
Matrix SHA-256: `c2d40e206fa21692a704a65e029a159f38cb023183e4cc7a28481b4f35749241`

## dag2304-64root-complete-consumer-diverse-dense-low-update-r1-change100

Gate `teacher`, body `linear` / `fp32`, input `10x-pbmc-cgatac01`, update `incremental`, reuse horizon `1`. Device `GPU-6c1cac7f-a360-0aef-ba98-2828bfd1db1a`; job `47b853d3-ddfa-47f4-b1d3-601629c9663f`.

Source archive `f9bfbd356e0fde90ca9defdada2c73e48b974c3e9c02fa44d5b7f9f486f22d6f`; binary `02a282c431888966d9cb5172c6794b441d0464f5ee76d06115fae66d4dcdc8a5`.

### Complete operational costs

Each row summarizes six independent rounds where available. Recurrent and recovery samples inside a round are paired and summarized within that round.

| Arm | Rounds | Cold prep ms (median, range) | Recurrent horizon ms (median, range) | Output recovery ms (median, range) | Complete cost µs (median, IQR) | Net allocated bytes (median) |
|---|---:|---:|---:|---:|---:|---:|
| `bitmap` | 6 | 461.586 (441.735–480.38) | 0.634436 (0.633763–0.676243) | 3.26027 (3.02357–3.7554) | 465300.252 (458202.285–479111.384) | 20971520 |
| `cohort-direct` | 6 | 479.285 (447.096–495.742) | 0.933855 (0.923884–1.04477) | 3.38898 (3.16503–3.85579) | 483803.256 (473453.680–487001.644) | 20971520 |
| `cohort-shared` | 6 | 454.626 (451.51–474.243) | 1.20334 (1.20113–1.24067) | 3.24468 (3.07597–3.64682) | 459261.074 (457032.913–461321.734) | 20971520 |
| `direct` | 6 | 478.842 (454.922–551.057) | 0.576423 (0.573652–0.584024) | 3.20531 (3.13735–3.35623) | 482725.173 (468855.383–498720.253) | 20971520 |
| `library` | 6 | 486.063 (463.059–510.298) | 1.95053 (1.93594–2.18102) | 3.26955 (3.23873–3.33892) | 491446.017 (472582.094–507314.370) | 33554432 |
| `library-batched` | 6 | 495.32 (473.206–510.486) | 0.599425 (0.588997–0.644222) | 3.32859 (3.24567–3.69083) | 499238.681 (493795.314–508927.273) | 33554432 |
| `scalar` | 6 | 459.384 (440.647–483.719) | 0.565408 (0.563561–0.582126) | 3.26141 (3.04375–18.7386) | 470957.603 (458411.486–476528.118) | 20971520 |

### Amortization estimates

These scenario-local line estimates have no confidence interval and do not replace the qualification's paired complete-cost decision.

- `cohort-direct` vs `bitmap`: no positive crossover under this model; estimate not applicable. No confidence interval is available for this model. Measured paired median latency reduction -2.555% (speedup 0.97509, 95% interval [0.93675, 1.00953]); rejected: no median improvement.
- `cohort-shared` vs `bitmap`: linear amortization crossover estimate; not a measured threshold; estimate 11.62 evaluations. The fitted line favors the candidate below the crossing and the control above it; this is an estimated upper reuse cutoff for the candidate. Paired-round lines have positive intersections in 3/6 rounds (range 12.4–49.9); other rounds do not cross at positive reuse. The cold intercept is noisy and this has no confidence interval. Measured paired median latency reduction 0.495% (speedup 1.00497, 95% interval [0.97183, 1.05528]); practically inconclusive: latency reduction does not exceed 5%.

Teacher soft to hard change is recorded separately from circuit equivalence and timing:
- Uniform mean: 0.120562.
- Workload weighted mean: 0.121373.
- Teacher row hardening threshold crossings across validator checks, median per round: 2.
- Observed teacher rows: [5321, 1486, 1466, 1310, 1486, 1290, 1310, 2715].

First-step host and device phase probes are in the JSON under `first_step_phase_probe_diagnostic_not_primary_timing`; they are excluded from the primary complete-cost samples.

## Rebuild and incremental update ablations

When present, v3 update-ablation records are grouped by mode and backend. Rebuild and incremental observations stay in separate groups; only their same-packet, same-round pairs are compared.

| Scenario | Backend | Update mode | Rounds | Cold prep ms | Recurrent horizon ms | Output recovery ms | Complete cost µs |
|---|---|---|---:|---:|---:|---:|---:|
| No qualified update-ablation records | — | — | 0 | — | — | — | — |

## Limits

- This exporter consumes receipt-qualified packets. It rechecks referenced raw record hashes and paired component arithmetic, but does not repeat Scheduler, KDA, archive, binary, interference, or sanitizer qualification.
- The six paired rounds are independent units. Values in timing arrays within each raw record are summarized inside their round, not counted as independent rounds.
- Horizons 1, 16 and 256 occur in distinct frozen workloads. They are not a measured reuse sweep and are never pooled or treated as a crossover.
- Linear amortization is a scenario-local estimate without a confidence interval. It assumes the same workload, update policy, and mean recurrent work continue, with preparation and one output recovery paid per reuse cycle.
- phase_probe timings are first-step diagnostics collected separately from confirmation costs; they are not added to the primary cost.
- Memory fields follow the runner contract. Common condition representation allocations and device capacity are not attributed to an individual kernel; net device bytes is the before/after CUDA allocation delta reported by the runner.
