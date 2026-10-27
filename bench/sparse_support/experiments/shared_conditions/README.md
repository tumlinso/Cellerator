# Shared-condition bitplane probe

This standalone experiment compares Cellerator's current exact condition
bitmap/workset CUDA path with a prepared condition-major bitplane path. The
candidate transposes binary predicates across 32 cells with warp ballots,
evaluates each DAG level as wordwise operations, uses a prepared edge-to-root
map to materialize row-major edge bitmaps, and updates only condition nodes and
edge ordinals affected by a changed predicate.

A scalar ablation keeps per-cell DAG evaluation and the same current bitmap
and workset semantics, but replaces binding binary searches with a prepared
edge-to-root map. Three-way paired timing separates that map's contribution
from the bitplane condition evaluation.

The comparison includes current full rebuild, current dirty bitmap update,
scalar direct-map full/dirty paths, bitplane full rebuild, bitplane incremental
condition-plane update plus dirty edge update, and the same production workset
compactor in every timed path. The candidate is experimental and assumes
binary predicates. It does not alter production code or the general
condition-program contract.

Configure this directory by itself, with `CELLERATOR_ROOT` set to the source
checkout. Build with `-DCMAKE_CUDA_ARCHITECTURES=70` for the V100 probe.

Example shapes are emitted by `--suite`; use `--edges`, `--cells`, and
`--sharing few|weak` for a single fixture. Output is JSON lines suitable for
retention as experiment evidence. `--check-only` runs independent host-oracle
and candidate-equivalence checks before any timed trials.
