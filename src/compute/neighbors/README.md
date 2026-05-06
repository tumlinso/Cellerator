# `compute/neighbors`

Compute-facing home for neighbor search math and forward-neighbor caller policy.

`forward_neighbors/` owns forward-neighbor index/query lifecycle, routing,
caller-facing search configuration, and result semantics for CellShard-backed
matrices.

`exact_search/` owns the reusable CUDA exact top-`k` routines for shard-local routed segment scans and compact device-side result merges. Task-specific policy such as forward-time routing stays above that layer.

`scoring/` owns lower-level KNN scoring and merge helpers, including the older
cuVS sharded KNN implementation.

Do not add new reusable neighbor code under `_rna`.
