# Later qualification — deliberately outside this run

The creative workshop leaves these obligations visible without forcing them to become prerequisites for every speculative implementation. They are not silently activated TODOs.

## Q01 — Exactness, contracts and safety

Complete required general CPU/CUDA semantics, differential/adversarial tests, stable/unordered output guarantees, relevant sanitizers and public API audit.

## Q02 — Performance and policy selection

Measure end-to-end costs including packing, directory building, replay, bytes and information loss. Select executors from real workloads; preserve failures.

## Q03 — Production stack interfaces

Inspect current Cellerator ABI/runtime/operator ownership, then design producer-owned adapter handoffs and optional Torch views without binding obsolete DeviceMathContext.

## Q04 — Learned biological evaluation

Choose an explicit sequence-to-target objective, split by appropriate biological units, compare fixed/local/dense/learned baselines and investigate leakage, non-identifiability and counterfactual limits.

## Q05 — Distribution and persistent reuse

Only after concrete need, request Cellerator partition and CellShard persistence integration through their separate authorities; account for communication and invalidation.

## Measurements that will matter later

Measure the total path, not just a favored inner kernel: initial sequence read, feature/plane production, representation build, index construction, packing, compaction, useful floating work, output, residual fetch, repeated queries and invalidation. Separate cold and reused paths. Report bytes and losses as well as time. Compare equal semantics or clearly label unequal tasks.

Candidate recall is measured before downstream score quality. A complete equal-key directory is not a complete biological edge set. A warp match against preassembled candidates is not global retrieval. An approximate embedding with an exact source handle is not lossless. A deterministic toy example is not a trained genome model.

Use adversarial fixtures that expose collapsed distinctions: same histogram/different order; equal string/different context; cross-warp equal keys; insertion at a seam; low-surprise but query-relevant sequence; an edit that changes candidate structure; all-positive output that exceeds capacity. Add biological data only under a declared objective and split policy.

Do not overstate negative results either. A custom executor losing to CUB does not invalidate its mathematical operator. A failed small synthetic task does not prove a hierarchy cannot be useful. Keep operator semantics, executor efficiency, model quality and scientific interpretation as separate evidence columns.
