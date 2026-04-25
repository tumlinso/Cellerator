# `compute/neighbors`

Compute-facing home for neighbor search operators.

The cuVS sharded KNN implementation, the routed exact-search primitive, and the forward-neighbor index now all live here.

`exact_search/` owns the reusable CUDA exact top-`k` routines for shard-local routed segment scans and compact device-side result merges. Task-specific policy such as forward-time routing stays above that layer.

Do not add new reusable neighbor code under `_rna`.
