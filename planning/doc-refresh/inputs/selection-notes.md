# Cellerator evidence selection

## Selected — exact-cover complete cost

Displayed as one verified historical study. Local SHA-256 of `bench/ce_geo/evidence/sm70_forward_complete_cost.jsonl` matches the disposition (`f51964ebb5eccf85fde359105c232ccef071608bef5b3ed42ee176b7177964f7`). The disposition file identity at the current source HEAD is `e09e263a3b82bcbd06f9008a618a5184dd141928af54d1fef67dec54642bada4`; its producer commit is `2fe2db329af1430fb52cf3677bb34be9e97a4216`. Project Control records CE-GEO-113-BENCH and CE-GEO-79-DISPOSITION as historical-valid passed gates, with the disposition receipt bound to evidence ID `56c4e4e8-7df2-4900-ba5e-3583a28dc4e7`. The source record is original historical campaign output, not a new run.

The JSONL contains one provenance row and two aggregate rows for reuse 1 and 16. The benchmark source computes complete cost as cold phase cost divided by reuse plus the median of 11 steady-state samples. Recomputed cold/steady arithmetic matches both rows to the reported 0.001 ns precision. The same file reports relative MAD of 1.507% hybrid and 0.741% sparse. Each of 11 measured iterations ran hybrid first and sparse second; order was fixed, not randomized. It does not include the individual timing samples. Numerical policy differs (hybrid FP16 values/RHS with FP32 accumulation/output; sparse FP32), with zero reported max absolute error against the same fixture reference. Claim remains limited to one synthetic N=64 exact-cover on Tesla V100-SXM2, with the CUDA/cuSPARSE toolchain versions not recovered.

Portable evidence: `docs/results/data/ce-exact-cover-gate-evidence.jsonl` is a byte-for-byte copy of the original aggregate gate file; its companion manifest records hashes, gate identity, source file hashes and the aggregate-only limitation.

## Omitted — normalization

The source summary `bench/ce_geo/evidence/fusion_evaluation.json` is present, but the prepared result record provides no raw evidence path or gate/receipt identity to authenticate the stage-only timing claim. Its two-versus-four launch statement and timing values therefore remain unpublished. The CE-GEO technical documents continue to preserve its non-promotion conclusion.
