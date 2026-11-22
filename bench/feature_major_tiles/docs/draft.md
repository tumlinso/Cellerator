# Feature-major tiled sparse training evaluation

Use the checked-in PBMC3K and GSE147520 CSR support traces without removing
features or changing their row membership. Generate deterministic FP32 master
parameters, weights, and targets. Sparse execution values are their binary16
roundings. Compute FP32 forward, mean-scaled squared loss, the dense weight
gradient, sampled sparse gradients, and SGD updates. Check all products and
updated parameters against an independent FP64 loop over the original CSR
coordinates. The synthetic low-sharing and skewed cases probe where directory
size or reverse load balance matters.

Compare contiguous and locally reordered tiles against native full-width CSR,
CSR with a prepared per-entry transpose directory, and cuSPARSE SpMM/SDDMM.
cuSPARSE receives the exact binary16 value expanded to FP32 and pays for the
expanded plane. Measure both learned sparse values and a CSR-order external
refresh. Report cold build and initial upload, paired CUDA event samples,
refresh time, explicit live array bytes, and mean cost at 1, 8, and 64 steps.
Test each choice before recording timings. Compare against the fastest equivalent
baseline and confirm any claimed improvement with fresh paired measurements.

Build `celleratorFeatureMajorTilesTrainingTest` and
`celleratorFeatureMajorTilesBench` with `sm_70`. Use the test executable for CUDA
correctness and the campaign driver for measurements on an exclusive admitted
V100. Schedule through Autobench with bounded duration and a verified idle
device. Require a 5% complete-cost reduction at 64 steps and a paired 95%
confidence interval above zero before recommending the tiled format. See
the harness README for the measurement contract and exact commands.
