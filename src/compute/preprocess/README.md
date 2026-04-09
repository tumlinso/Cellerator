# `compute/preprocess`

Sparse scRNA preprocessing operators for row-sharded CSR inputs.

This area owns:

- row-local cell metrics
- in-place normalization transforms
- transpose-based gene metric accumulation
- gene filter mask construction
- per-device and fleet workspaces

The split follows the real hot path on V100:

- row-wise kernels stay close to CSR
- feature-wise reductions reuse transpose SpMV
- fleet reduction is explicit and separate from local operators
