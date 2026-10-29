# CE-GEO geometry layout evaluation: v13

Status: **complete for the three declared fixtures**. This is bounded evidence about the frozen synthetic workloads; it does not select a production layout or remove any candidate family. The raw v12 partial run and the failed first v13 harness attempt remain preserved.

## Verdict

No topology ordering or hybrid image was faster than the independent SIMT path in the primary same-process paired training-step measurements. For every candidate and fixture, the median `SIMT / packed` synchronized-wall ratio was below 1, and measured paired step savings were negative. A ratio above 1 would favor packed execution.

The large ranked fixture showed the clearest regressions. Reordering reduced the loss relative to canonical packing, and hybrid thresholds reduced both the loss and incremental setup memory, but hybrid8 and hybrid16 remained slower than SIMT. Small and event-local results were closer to parity, with event-local pair distributions wider. Every measured candidate/fixture had no finite setup break-even because positive per-step savings were not supported by paired intervals.

These fixtures are generated synthetic inputs. Per the current direction, all geometry mechanisms remain available for broader real-input evaluation; this result is not a basis for pruning a route.

## Conditions and correctness

- Device: Tesla V100-SXM2-16GB, `sm_70`, UUID `GPU-cf22c41f-5b58-77b1-3535-8fadd1ca6505`; driver 580.178.04; Torch 2.7.0+cu126 and CUDA runtime 12.6.
- GPU work ran through exclusive Device Scheduler admission with three zero-utilization samples and a 60-second quiescence timeout. The successful KDA-only job was `7e1fade6-f295-4b25-8cb4-afa023b7821f`, attempt `7e932a5c-b680-45bd-b6cc-ea1c16ef18c5`; it succeeded with `cleanup_verified: true`.
- The v13 package identity is `4d3b6f0e1a63ea20785975597849bfad0b0f081221c47d9b858d6581b487afd4`; its Torch extension SHA-256 is `51b0838ad3466f4c03f693d620aa6b800e8af9ed87a8a3fceb3bd432a05a4f24`. The v13 change replaces quadratic host-side owner-ID validation; it does not change GPU math.
- Each candidate was checked against SIMT on all three cases for inference output, training parameter gradient, loss, nonzero derivative-only gate gradients, and zero remaining read tickets. All 15 candidate/case validations passed. Hybrid8 and hybrid16 large-ranked paths also passed Compute Sanitizer memcheck with `ERROR SUMMARY: 0 errors`.
- Timing used 3 warmups, 21 within-process SIMT/packed pairs per round, 2 exploration rounds, 2 confirmation rounds, 2 training iterations per step, and 5 balanced setup pairs. The timed region includes zero-grad, gate score/STE, all edge values, relation forward, loss, backward including derivative-only gate gradient, and guarded Adam. The primary ratio and step saving use each same-process pair. Table p10–p90 ranges describe observed paired samples, not confidence intervals. The benchmark's 95% bootstrap intervals are reported in the derived JSON for each 21-pair run.

## Paired training results

Each exploration cell combines 42 same-process pairs (two rounds × 21). `SIMT / packed < 1` means packed was slower. The `extra cold setup` includes host geometry compilation plus the measured packed-only module-setup difference. `extra setup allocation` is the median packed-minus-SIMT setup allocation delta, not the training-step high-water mark.

| Fixture | Candidate | SIMT / packed ratio, median [p10–p90] | Median paired step saving (ms) | Extra cold setup (ms) | Extra setup allocation (MiB) |
|---|---|---:|---:|---:|---:|
| Small, E=96 | canonical | 0.9737 [0.9559–0.9926] | -0.195 | 1.9 | 0.9 |
| Small, E=96 | reordered | 0.9732 [0.9674–0.9778] | -0.194 | 3.0 | 0.9 |
| Small, E=96 | hybrid8 | 0.9731 [0.9490–1.0372] | -0.155 | 3.7 | 0.8 |
| Small, E=96 | hybrid16 | 0.9800 [0.9305–0.9980] | -0.119 | 2.6 | 0.8 |
| Large ranked, E=262,144 | canonical | 0.8404 [0.8386–0.8458] | -2.493 | 398.6 | 631.3 |
| Large ranked, E=262,144 | reordered | 0.8531 [0.8502–0.8558] | -2.260 | 402.8 | 630.0 |
| Large ranked, E=262,144 | hybrid8 | 0.8951 [0.8928–0.8989] | -1.540 | 339.3 | 231.3 |
| Large ranked, E=262,144 | hybrid16 | 0.9065 [0.9017–0.9122] | -1.354 | 348.7 | 149.6 |
| Event local, E=64 | canonical | 0.9733 [0.9684–0.9777] | -0.201 | 3.5 | 0.1 |
| Event local, E=64 | reordered | 0.9756 [0.9566–1.0097] | -0.162 | 2.5 | 0.1 |
| Event local, E=64 | hybrid8 | 0.9766 [0.9521–1.0141] | -0.176 | 1.7 | 0.1 |
| Event local, E=64 | hybrid16 | 0.9744 [0.9478–1.0231] | -0.143 | 2.4 | 0.1 |

For 23 of the 24 exploration runs, the per-run 95% bootstrap interval over 21 paired ratios lay below 1. The exception was the second hybrid16 event-local run, whose interval crossed 1. The paired-saving gate remained false for every candidate/case, so all reported break-even step counts are null. The broader event-local p10–p90 ranges also show more run-to-run variability than large-ranked.

KDA's two-round exploration ranking put hybrid16 first at 1.0416, but its fresh two-round confirmation did not support that initial signal: the aggregate confirmation benefit score was 0.9225 with 95% bootstrap interval [0.9036, 0.9403]. Under KDA's convention, benefit above 1 favors the candidate. The per-case confirmation scores were 1.0139 [0.9652, 1.0660] for small, 0.9085 [0.9072, 0.9099] for large-ranked, and 0.8523 [0.7631, 0.9467] for event-local. KDA retained SIMT and did not promote a candidate. The same-process paired results above remain the primary timing evidence.

Common fixture/workset setup categories were measured separately in each run. Median fixture-owner preparation, shared workset/input preparation, hard-gate freeze/primal-workset construction, and CUDA/Torch initialization were respectively 632/389/162/242 ms for large-ranked, 99/154/121/244 ms for small, and 93/100/157/240 ms for event-local. The table reports the layout-specific extra cold setup separately from those common per-case setup measurements.

## Geometry image footprint

On large-ranked, canonical packing used 58,688 tiles for all 262,144 MMA edges. Topology-neighbor ordering used 48,113 tiles but remained slower. Hybrid8 used 7,687 tiles, routing 165,425 edges to MMA and 96,719 to 256 residual regions. Hybrid16 used 2,573 tiles, routing 109,665 edges to MMA and 152,479 to residuals. Those hybrids reduced image bytes and incremental setup allocation but did not repay residual execution costs in the measured training step.

The small fixture used 16 canonical tiles, 13 reordered tiles, 3 hybrid8 tiles (59 MMA + 37 residual edges), and 2 hybrid16 tiles (44 MMA + 52 residual edges). Event-local used 1 canonical tile and 2 tiles for each other image; all retained 64 edges in the MMA path.

## Provenance and artifacts

- KDA task SHA-256: `76271d12db56990b24cdf3d9fa1cabb4b84973dc08301154ffaf97325b9018dd` (`kda_task_v13_retry1.json`).
- KDA run: `95707422-be66-4b6f-b0f7-cae92e32e30d`; task contract above; engine fingerprint `97af7e5a7002ac6ac09a33532f5acf67bfdfeafee2e42aec62d1df78f01fc277`. The run completed with `tools_unchanged` and `trusted_inputs_unchanged` both true.
- The exact derived values, per-run bootstrap intervals, source/image hashes, candidate metrics, and scheduler receipt are collected in [paired_layout_metrics.json](evidence/gpu1_geometry_layout_20261028_v13_retry1/paired_layout_metrics.json). Raw KDA attempt evidence remains in `evidence/kda_workspace_v13_retry1/`; the successful scheduler wrapper outputs are in `evidence/gpu1_geometry_layout_20261028_v13_retry1/`.
- The first v13 campaign job `b21939de-69a4-4207-99a9-2c49f9df9e77` passed both hybrid memchecks but stopped at an incorrectly task-relative probe path before candidate validation; Scheduler cleanup was verified. Its raw attempt is preserved in `evidence/gpu1_geometry_layout_20261028_v13/`. The corrected KDA-only task uses an absolute adapter path and a fresh workspace.
- Prepared image manifest SHA-256: `6ab55eb2476f306188de3a105c711af7e6072bc37c27e876fe628b9761a86ed7`. Large-ranked and small fixture manifest SHAs are `674d5d4660befcb6ddbcd618b16b136203d3478ddc577d09a037e116bffd6e32` and `c33facc18f330739411c1f293609c237689685bd4003099ca79b331fded144be`.

No profiler was run. No production packer policy, ABI, or numerical kernel change was made by this layout evaluation.
