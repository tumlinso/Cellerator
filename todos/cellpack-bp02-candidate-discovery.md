---
slug: "cellpack-bp02-candidate-discovery"
status: "done"
execution: "closed"
owner: "codex-cp-bp-02"
created_at: "2026-08-14T13:00:00Z"
last_heartbeat_at: "2026-08-14T14:12:29Z"
last_reviewed_at: "2026-08-16T15:29:50Z"
stale_after_days: 3
objective: "CP-BP-02 deterministic sketch/LSH candidate generation and deduplication; approximate similarity proposes only."
---

# Current Objective

## Summary
Use MinHash or an equivalent sparse sketch plus LSH/grouping to reduce the candidate space while retaining high-value exact merges.

## Quick Start
- Why this stream exists: exact all-pairs support comparison is unnecessary and scales poorly.
- In scope: `gene_minhash`, `lsh_emit_keys`, CUB/radix grouping, pair compaction/deduplication, deterministic seeds, CPU reference, and recall/reduction evaluation.
- Out of scope / dependencies: final merge decisions, Jaccard-as-objective, CP-BP-01 file edits, CellPack edits, and global optimizer policy.
- Required skills: `cuda`, `todo-orchestrator`.
- Required references: CP-BP-00, completed CP-BP-01 `gene_support_bitset_view`, `AGENTS.md`, `scope.md`, `style_hint.md`, and CUB/library guidance from the CUDA skill.

## Planning Notes
- Approximation proposes candidates only; CP-BP-03 exact codec-cost gain is authoritative.
- The user's current ownership instruction places discovery in new Cellerator compute files. CP-BP-01's host support view is complete and is consumed read-only; CellPack and CP-BP-04 optimizer files are not touched.
- CP-BP-02 freezes only canonical unordered gene endpoints plus discovery provenance. Exact evidence, score kinds, gain, and optimizer acceptance belong to CP-BP-03/04.

## Assumptions
- Candidate output is a deterministic, duplicate-free canonical gene-pair/block-edge list for fixed input/configuration.
- Sketch configuration is explicit and versionable.
- The user's newer instruction overrides the stale CellPack-file suggestion: CP-BP-02 implementation and its frozen candidate-pair contract live in Cellerator compute.
- CP-BP-04 is concurrently claimed but owns CellPack optimizer files; CP-BP-02 will add only new Cellerator compute/test/benchmark files plus shared Cellerator manifests and its own ledger.

## Suggested Skills
- `cuda`
- `todo-orchestrator`

## Useful Reference Files
- `include/Cellerator/compute/gene_support_bitset.hh`
- `include/Cellerator/types.cuh`
- `bench/benchmark_mutex.hh`

## Plan
1. Define a Cellerator pointer-first deterministic sketch/candidate contract without editing CP-BP-01 files.
2. Add CPU exhaustive/high-overlap reference and synthetic fixtures.
3. Implement sketch, key emission, CUB sorting/grouping, and deduplication.
4. Measure candidate-count reduction and deliberately high-overlap recall on tractable matrices without implementing CP-BP-03 scoring.

## Tasks
- [x] Define candidate and scratch/output contracts.
- [x] Implement/reference-test sketches and LSH grouping.
- [x] Add deterministic deduplication and edge-case coverage.
- [x] Report reduction ratio and high-value exact-merge recall.

## Blockers
_None; CP-BP-01 is complete and CP-BP-02 acceptance uses synthetic high-overlap recall without beginning CP-BP-03._

## Progress Notes
- 2026-08-16: Reconciled at worktree `HEAD` `1ebb734`.
  `geneCandidateDiscoveryRuntimeTest` passed CPU/CUDA exact agreement and the
  mutex-serialized V100 benchmark again retained 105,000/105,000 constructed
  cluster pairs from 105,000 candidates versus 449,985,000 exhaustive pairs
  (99.9767% reduction; 58.698 ms minimum, 59.768 ms median). Acceptance is met;
  the implementation was checkpointed in `597a3eb` after this inspection.
- 2026-08-14: Added as a missing ready workstream; no implementation evidence was found.
- 2026-08-14: Claimed by the current serial thread after confirming no live agent owns CP-BP-02. Step 1 sampling, dataset, materialization, support-bitset, and full-size V100 smoke tests passed before edits.
- 2026-08-14: Added the Cellerator pointer-first host candidate contract and fixed SplitMix64-v1 MinHash/LSH provenance. CPU and CUDA paths omit empty genes, use stable CUB radix sorting/scans/unique, cap oversized buckets with a deterministic circular window, and return lexicographically sorted unique canonical pairs.
- 2026-08-14: Focused candidate runtime test passes CPU/GPU exact agreement. The 64-gene exhaustive fixture retained 48/48 deliberately high-overlap pairs, emitted 48/2,016 candidates, and reduced the unordered pair set by 97.619%.
- 2026-08-14: Final Step 1 regressions and CP-BP-02 CPU/CUDA tests passed. The full 65,536-cell support smoke allocated 245,760,000 bytes; no Step 1 API was changed.
- 2026-08-14: Serialized Tesla V100 sm_70 benchmark at 65,536 cells x 30,000 genes produced 105,000 candidates from 449,985,000 exhaustive pairs, retained all 105,000 constructed cluster pairs, and reduced the unordered pair space by 99.9767%. Three timed runs after one warmup measured 59.832 ms minimum and 60.904 ms median.
- 2026-08-14: Benchmark provenance reports 14,060,031 bytes of CUB scratch, 333,424,355 bytes of accounted peak device allocation, and a 534,306,000-byte conservative fixed bound excluding CUB. The result is synthetic correctness smoke, not a production recall threshold.

## Next Actions
- CP-BP-02 is closed. CP-BP-03 now consumes `gene_candidate_pair_view` and its
  immutable provenance without reinterpretation; persistent device-resident
  Step 1 support remains an optional optimization.

## Done Criteria
- Candidate list is deterministic and duplicate-free.
- Tractable fixtures compare against exhaustive pairs and quantify candidate reduction.
- Deliberately high-overlap synthetic pairs meet a documented recall target, explicitly without a production-quality or final-gain claim.
- GPU results match the CPU/reference candidate semantics.
