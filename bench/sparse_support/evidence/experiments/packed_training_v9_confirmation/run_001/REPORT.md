# Packed versus SIMT sparse-support qualification

## Result

On the two width-64 fixtures, the current CE-GEO N64 forward plus SIMT VJP did not show a reliable complete-cost advantage over native SIMT. In the driver’s same-process paired measurements, packed was slower in both inference and training. KDA’s separate candidate-request comparison produced a small positive aggregate point score, but its confirmation interval crossed parity and did not promote packed. The differing estimates are retained below as a measurement limitation; neither supports switching the default from SIMT.

The tested packed training path includes the corrected full-candidate gate-gradient scatter and optimizer step. Its backward is still the native SIMT VJP, so this does not qualify a CE-GEO-specific backward kernel.

## Workload and protocol

Both candidates used the same 64-edge support, width 64, frozen initial parameters, hard gate decisions, derivative workset, and inputs within each driver invocation. The support contains four local 4-by-4 blocks inside a 16-by-16 tile. Gate probabilities were rescored on each step; primal and derivative worksets remained frozen for this qualification.

| Fixture | Cells and input | Primal / derivative candidates | Derivative-only candidates |
|---|---|---:|---:|
| `event_local` | 4 cells, 1 positive event per cell | 8 / 16 | 8 |
| `ranked_batch` | 32 cells, 8 selected genes per cell | 449 / 896 | 447 |

Training wall time includes `zero_grad`, gate scoring and STE, edge-value calculation, relation forward, loss, backward including full-candidate gate-gradient scatter, and guarded Adam. Inference includes gate scoring, edge-value calculation, and relation forward. Each KDA confirmation candidate/case request collected 21 paired AB/BA samples, with 5 warmup steps per candidate and 3 iterations per sample. KDA used 5 confirmation rounds after 2 exploration rounds. Thus each case has 10 confirmation driver invocations and 210 same-process paired samples per phase. The input and workset digests are fixed across the two candidate validations:

- `event_local`: input `904b436d252c2a0a9b5bb6be793b58581f10f5bbed8a481d34f93fcb89ce1337`; workset `c0054f9e749a26e670c4f4ebfbff501d1289ee4cb8ed0baf02898ec8a7c37618`.
- `ranked_batch`: input `52c34f0102c525131020591de8fc09182d92b11cec76cabe1e15260582d2e5d2`; workset `bf3d0894f1900d0652d9a8096fdccb067b6972d9ca63b88a31bce6e474b1750f`.

## Steady-step measurements

Times are medians over 210 raw per-step samples; parentheses show the 10th–90th order statistics. Paired ratios are `SIMT / packed`, so values below 1 favor SIMT. Their intervals bootstrap the five outer confirmation-round medians, pooling the two 21-pair driver invocations in each round. This interval is descriptive with five outer clusters; KDA’s separate interval is reported afterward.

| Fixture | Phase | SIMT wall ms | Packed wall ms | Paired ratio (p10–p90) | 95% outer-round bootstrap |
|---|---|---:|---:|---:|---:|
| Event local | Inference | 0.948 (0.930–1.216) | 1.031 (1.009–1.306) | 0.922 (0.905–0.941) | 0.920–0.929 |
| Event local | Training | 6.870 (4.703–7.559) | 7.075 (4.838–7.772) | 0.975 (0.943–1.003) | 0.975–0.977 |
| Ranked batch | Inference | 0.955 (0.933–0.993) | 1.036 (1.012–1.080) | 0.921 (0.900–0.939) | 0.918–0.923 |
| Ranked batch | Training | 7.355 (5.051–7.659) | 7.565 (5.173–7.919) | 0.975 (0.937–1.005) | 0.974–0.983 |

The same-process paired medians correspond to about 8.4–8.6% slower packed inference and 2.5–2.6% slower packed training. The broad per-sample ranges, especially for training, reflect run-to-run GPU timing variation; paired order alternated SIMT-first and packed-first to reduce order bias.

KDA compares the selected candidate’s samples across separate driver invocations. Its confirmation result was packed score `1.0076` with 95% interval `[0.9756, 1.1473]`; a score above 1 favors packed. Per-case scores were `1.0415 [0.9643, 1.1768]` for event-local and `0.9748 [0.9499, 1.1186]` for ranked-batch. The aggregate interval crosses parity, so KDA retained SIMT (`eligible=false`, `promoted=false`). These separate-invocation scores differ from the same-process paired ratios above. Because the adapter launches a fresh driver process for each KDA candidate request, request-to-request timing drift affects the engine’s unpaired candidate medians. Both views are preserved; the discrepancy prevents a positive packed-benefit claim.

## Preparation and memory

Each of the 10 confirmation driver invocations per fixture measured five balanced module-preparation pairs after warming both constructors. CUDA/Torch initialization was timed separately. The fixture-owner module is a common SIMT template used to create identical inputs; it is not a candidate-specific setup comparison. Setup table ranges are over the 10 request-level medians per fixture, not 50 independent preparations.

| Fixture | CUDA/Torch init ms | Input/workset prep ms | Gate freeze ms | SIMT module prep ms | Packed module prep ms | Packed extra incl. geometry compile ms |
|---|---:|---:|---:|---:|---:|---:|
| Event local | 214.6 (210.0–233.5) | 100.9 (100.5–106.0) | 139.3 (137.0–166.2) | 468.7 (459.7–472.4) | 467.6 (460.0–473.4) | -0.22 (-3.25–2.42) |
| Ranked batch | 215.0 (211.9–218.4) | 159.4 (156.1–164.0) | 117.8 (108.2–140.7) | 469.0 (467.4–470.8) | 468.5 (467.0–470.1) | -0.84 (-2.18–1.64) |

CE-GEO host geometry compilation took about 0.142 ms. Balanced module-preparation medians were close; their small differences are within the measured spread. No per-request break-even estimate was supported because none of the driver requests had a positive packed step saving supported by its paired interval. The reported extra-setup medians were slightly negative and noisy, so they do not establish a packed setup advantage either.

Packed module preparation allocated more persistent memory: 72,704 bytes versus 1,536 bytes for event-local, and 495,616 versus 1,536 bytes for ranked-batch. Peak additional allocation during the measured step itself was equal between candidates: event-local 66,560 bytes inference / 352,768 bytes training; ranked-batch 932,352 / 1,565,696 bytes.

## Evidence and limits

- KDA task: `kda_task.json`, SHA-256 `bd4b4d0f41fe72697c5664a41d296ec8eb106d2c67fb84c6a611d3051c815b21`; KDA run `fd675334-5946-48c4-bde9-88dd3f5b5d75`; task used zero minimum improvement to test crossing parity, not a product acceptance threshold. Trusted inputs and command identities were unchanged. `kda_result.json` SHA `509da77ec64ec82673d032f3072079438a82b847b1d5fa5c351b3e26b18abaa1`; raw KDA summary SHA `1b0cab3024aba7d97a1855d5e2c8e4e1ecea5ca43ac7d33ebb3d52b404c5eae2`.
- Frozen v9 identity: SHA-256 `be20f601882e918cdd6c74ace0843234a1b4a5c1fd74e2036cfb476222f86fa2`; Torch extension SHA-256 `93dd66d4559e455d2fda7be4bd95b79d2ca9d460e7cd309d2978706a4ab958d2`. Frozen `sparse_support.py` SHA `002fa4c6c91cb86d7bbf92c647f5f5b74524e5f42afaeb9135ad57f5805f9bb5`; `mechanism.py` SHA `51b79de02496f72997e90d3ee8e00911afc8b81aa21483c0fe8b157e87dbd7b3`; CUDA training test SHA `9077cbd808a258eb04085c3ce346a4e88f04dd5564c03451091b33d70c3a1eac`.
- Benchmark driver SHA `e0267e86a99139c036c36b80b1b8b34d4d54d071cd7eb62718618a93f103ed74`; KDA adapter SHA `26890e943fcfd630c84b0d2b809b24a662a11fce0e922b51e403fe95d943f146`; runner SHA `f64d5b18207b5ac7cdac36439654e7c32966ece45665c239c29ba31c13284aa6`.
- Device Scheduler job `9c00a5a3-fe83-4f02-8e75-51ed08944104`, attempt `0ab599d3-ffc5-4e5f-a6be-b082f6e5884d`, succeeded with `cleanup_verified=true` and exit 0 on `gpu:testlsserver:GPU-d745da9c-7649-8334-41f1-483afa6f3206`. Receipt: `scheduler_job.json`, SHA-256 `cdcdc0230740186f7e7f8077d0049c96fbcd591d18ca594bacf32721b3e96d03`.
- Environment: Tesla V100-SXM2-16GB, SM70, driver 580.178.04; Python 3.13.11, PyTorch 2.7.0+cu126 and CUDA runtime 12.6. The CUDA 12.9 compiler target probe supported `sm_70`.
- The early v9 `run_001` result used only two samples and two confirmation rounds. It is retained separately and is not pooled into this confirmation.
- The fixtures use frozen support and worksets. They do not include evolving event queues, per-step workset reconstruction, full Karyoplast/Geneformer model execution, or a CE-GEO packed backward. No Nsight kernel profile was collected; these measurements compare complete primitive-level inference and training regions, not kernel bottlenecks or another GPU architecture.

Raw engine and driver outputs are retained in this directory and its sibling `kda_workspace`; `derived_analysis.json` (SHA `acc3f0f3deaeda59647705eeb9387bc712c37cdef20f0a32083e7913e9f96fe7`) records the source-bound aggregation. Warmup was 5 steps per candidate per phase.
