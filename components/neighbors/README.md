# Cellerator Neighbors

CUDA builds provide routed search through `Cellerator::neighbors`. Request
the `neighbors` component from an installed Cellerator package and include
headers under `Cellerator/neighbors`.

Forward-neighbor search owns index/query lifecycle, routing,
caller-facing search configuration, and result semantics for CellShard-backed
matrices.

Exact search owns the reusable CUDA top-`k` routines for shard-local routed
segment scans and compact device-side result merges. Task-specific policy
such as forward-time routing stays above that layer.
