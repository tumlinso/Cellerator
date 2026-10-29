# Packed versus SIMT training benchmark (v2 pre-fix)

**Status:** archived pre-binding-fix evidence. This result uses the frozen v2 extension whose gradient binding was later found to return compact derivative gradients without scattering them into the full candidate array. It is not a post-fix performance result and must not be used as the current training-cost claim. The raw files are retained for comparison with the post-fix package.

## Workload and execution

The comparison uses the same 64-edge support, frozen gates, input tensors, workset IDs, trainable owner parameters, and optimizer settings for the SIMT path and the CE-GEO N64 forward path. Both paths use the SIMT VJP. Width is 64; the support is four local 4-by-4 modules inside one 16-by-16 CE-GEO tile. Per-step training includes `zero_grad`, shared gate scoring and sigmoid STE, edge values, relation forward, loss, backward, and guarded Adam. Inference includes gate scoring, edge values, and relation forward.

The two fixture-derived cases are:

| Case | Shape and eligible work |
|---|---|
| Event local | 4 cells, one selected event per cell, 16 derivative candidates, 8 hard-admitted primal edges, 8 derivative-only gate candidates |
| Ranked batch | 32 cells, 8 selected genes per cell, 896 derivative candidates, 449 hard-admitted primal edges, 447 derivative-only gate candidates |

Both cases used seed `20261027`, width 64, and identical workset and frozen-input digests across the two execution paths. Each confirmation request used two in-driver timing samples per path, averaged over two training-step iterations. KDA performed two confirmation rounds. The per-case KDA result therefore compares two round medians and bootstraps those rounds.

The scheduled job completed with verified cleanup on the Tesla V100-SXM2-16GB, SM70, UUID `GPU-d745da9c-7649-8334-41f1-483afa6f3206` (scheduler resource `gpu:testlsserver:GPU-d745da9c-7649-8334-41f1-483afa6f3206`). Environment: driver 580.178.04, Torch 2.7.0+cu126, CUDA runtime 12.6; CUDA 12.9 `nvcc` reported `sm_70` support. Scheduler job `d0cc77b8-18dc-4bae-893e-00d5b5d0e46d`, attempt `30918444-41ac-4bf9-87fb-ead849941faa`, completed at 2026-10-27 19:49:44 UTC; `cleanup_verified=true`.

## Steady-state measurements

The KDA score is the candidate benefit ratio for a minimize objective: a value above 1 favors packed, 1 is parity, and below 1 favors SIMT. The score uses the two confirmation round medians. Lower confidence bound is shown first.

| Case | SIMT training round medians (ms) | Packed training round medians (ms) | Packed score (95% bootstrap interval) | Matched training wall ratio SIMT/packed (8 pairs) | Matched inference wall ratio SIMT/packed (8 pairs) |
|---|---:|---:|---:|---:|---:|
| Event local | 3.282, 4.110 | 4.190, 3.758 | 0.930 [0.783, 1.094] | 0.938 [0.837, 1.074] | 0.918 [0.880, 0.966] |
| Ranked batch | 3.251, 3.725 | 4.225, 3.810 | 0.868 [0.770, 0.977] | 0.959 [0.799, 1.031] | 0.929 [0.835, 0.977] |

The KDA aggregate score was 0.899 [0.776, 1.034], so the aggregate interval includes parity. The ranked-batch confirmation interval was below parity; the event-local interval crossed parity. The matched in-driver medians also favored SIMT in both cases, with noticeable spread and both measurement orders represented four times.

Across the four selected KDA samples per backend and case, event-local training medians were 3.676 ms for SIMT and 3.958 ms for packed; ranked-batch medians were 3.450 ms and 4.022 ms. Across all four confirmation driver invocations per case, matched inference medians were 0.872 ms versus 0.956 ms (event local) and 0.904 ms versus 0.971 ms (ranked batch). CUDA-event and synchronized-wall ratios were close.

Peak per-step extra allocation was identical between paths: 352,768 bytes for event-local training and 1,562,112 bytes for ranked-batch training; inference peaks were 66,560 and 932,352 bytes respectively. Prepared-owner allocation deltas were 1,536 bytes for SIMT and 72,704 bytes for packed in the event case, and 1,536 versus 495,616 bytes in the ranked case.

## Preparation costs and limits

Fixture/workset materialization and gate freezing are common to both paths and are excluded from steady-state step times. Their medians were 111 ms and 158 ms for event local, and 166 ms and 127 ms for ranked batch.

The recorded SIMT preparation medians were 526 ms and 528 ms; packed preparation plus shared host geometry compilation measured 475 ms and 487 ms. These raw deltas appear to favor packed by 41–53 ms, but setup order was not balanced: the SIMT owner was prepared first for gate-state construction, while packed preparation happened after fixture/workset operations and the SIMT preparation. Treat these setup values as descriptive only; they do not establish a packed preparation advantage or a break-even point. The production post-fix binding also adds a gradient scatter and temporary allocation that this v2 run did not measure.

## Correctness, KDA decision, and scope

The v2 numerical checks passed for both cases: output tolerance 5e-3, parameter-gradient tolerance 6e-3, and loss tolerance 5e-4 absolute / 5e-3 relative. The STE produced nonzero probability gradients for all 8 event-local and all 447 ranked-batch derivative-only candidates, and no forward read tickets remained. These checks validate the v2 relation implementation under its old binding contract; they do not cover the later full-candidate gradient scatter fix.

KDA completed validation, exploration, and confirmation with trusted inputs and tool identities unchanged. The task configured `min_improvement=10.0`, which KDA's selection code turns into a benefit-ratio gate above 11; this is an experiment setting, not a user acceptance requirement. No candidate was promoted. The KDA engine retained SIMT as preferred, and its point estimate ranked packed below SIMT.

This is a relation-stage microbenchmark, not a full Karyoplast or Geneformer model run. It has no support-discovery, latent-sequence, or event-queue cost. Timings are CUDA-event and synchronized-wall measurements; this run has no separate Nsight trace or kernel-counter profile. Raw KDA inputs, responses, build identities, failed earlier harness attempts, and scheduler receipts are retained beside this report.

## Source and evidence identities

- Benchmark driver SHA-256: `a7e2034f749b82937b1852fde8116ec6b414d81cd96dc18965b84bbf93ddcf91`
- KDA adapter SHA-256: `a0906c568ad6bf0294ca7ac3310e2333da13a594ca09f36937ea2ee3c9e5e297`
- Frozen v2 Torch extension SHA-256: `b976b2981137d3115f31570b1690dd5142cd973182503e346c7f44e925f341d3`
- KDA task SHA-256: `8a52e90fde0082f3795d34885226a5b77fefd802ed6ca5148d106469e6fea538`
- KDA attempt summary: `r4/kda_workspace/runs/kda-experiments/attempts/bce397ac-4745-4ec2-97e6-899d86e0ec14/summary.json`
- KDA result envelope: `r4/run_004/kda_result.json`
- Scheduler receipt: `r4/run_004/scheduler_job.json`
- Earlier failed attempts remain in `run_001/`, `r2/run_002/`, and `r3/run_003/`; each completed with verified cleanup and failed before producing performance data.
