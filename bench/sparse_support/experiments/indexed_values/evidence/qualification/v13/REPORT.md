# Indexed shared-edge-values v13 probe

## Result

Computing differentiable shared edge values only on `U = unique(P ∪ D)` is a strong win on the sparse event fixture and does not help the denser ranked fixture. It preserves the full-E output vector, native relation call, canonical shared parameter owner, and derivative-only candidates. Values are recomputed after every Adam publication; there is no cross-step cache.

The v13 KDA run validated both candidates on all three cases and selected `indexed_union`. Its composite score is **1.395x speedup** over `full_scan` (95% interval 1.375–1.414). Per-case complete-step medians below pool 12 synchronized wall samples per candidate (3 samples in each of two exploration and two confirmation rounds):

| Case | E | U | U/E | Full-E | Indexed-U | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| Small control | 64 | 16 | 25.0% | 4.339 ms | 4.672 ms | 0.929x |
| Event stream | 1,280,000 | 70,466 | 5.51% | 14.758 ms | 5.028 ms | 2.935x |
| Ranked, 512 tokens | 262,144 | 205,281 | 78.31% | 13.341 ms | 13.488 ms | 0.989x |

Complete-step timing includes zeroing gradients, gate scoring and STE, P/D union construction, shared values, native SIMT relation forward, loss, backward, and guarded Adam publication. Full/indexed parameter-gradient maximum absolute errors were 0 for the small case, 2.56e-13 for events, and 2.62e-14 for ranked; output error was zero in all cases. Each pair used identical workset digests and pruning decisions.

The isolated shared-value-plus-VJP diagnostic for the event case was 11.43 ms full-E versus 2.01 ms indexed-U. It is explanatory only; the complete-step result above is the performance claim. Large-event estimated gather/product storage avoided per forward was 924 MB. The measured peak increase during the *joint comparison process* was 1.79 GB for events and 1.17 GB for ranked, including both resident candidate modules and temporary comparison state; it is not an indexed-only memory delta.

One-time module preparation medians were 2.67 s full / 2.46 s indexed for events and 653 / 545 ms for ranked. These setup values are outside KDA's steady complete-step metric and should remain visible when estimating end-to-end use.

## Interpretation and scope

The result supports a narrow optional integration candidate: score shared edge values for explicit P∪D ordinals, scatter into the existing logical-order E vector, and retain full-E as a fallback. The measured event workload has low U/E and a substantial end-to-end gain. The small case regressed about 7%, and ranked512 was effectively neutral, so the path should be selected only when workset density supports it. No production API or kernel was changed by this probe.

This was one synthetic V100 workload per shape, with repeated samples on frozen inputs. The experiment did not measure CE-GEO packed consumption, multi-event accumulation, dynamic workset construction from raw biological data, or cache reuse. No general biological performance claim follows.

## Reproduction and identities

The admitted command was scheduler job `5c3a61c7-a5e9-45bf-a7dd-04f203909f23` on GPU3 (`GPU-6c1cac7f-a360-0aef-ba98-2828bfd1db1a`); it succeeded and cleanup was verified. Its receipt/log are `scheduler_job.json` and `scheduler_logs.json`. KDA run `d919fe94-0e4c-4437-b086-3e63539dabc9` contains the full requests, responses, validation results, sample timings, and selection record in `../../../../kda_workspace_v13/runs/kda-experiments/attempts/d919fe94-0e4c-4437-b086-3e63539dabc9/`.

Frozen v13 package identity SHA256 is `4d3b6f0e1a63ea20785975597849bfad0b0f081221c47d9b858d6581b487afd4`; extension SHA256 is `51b0838ad3466f4c03f693d620aa6b800e8af9ed87a8a3fceb3bd432a05a4f24`. Fixture and source hashes are in `source_hashes.json`.
