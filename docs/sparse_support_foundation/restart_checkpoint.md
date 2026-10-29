# Restart checkpoint — 2026-10-27

Historical pause record. Work resumed on 2026-10-28. Subsequent
[v14 qualification](../../bench/sparse_support/evidence/qualification/v14-optional-policies/REPORT.md)
and the [broad evaluation intake](evaluation.md) record the resumed delivery;
the unrun/pending statements below describe the original pause boundary.

Paused at the user's request for a computer restart. Do not launch further jobs
until the user resumes the work. All agents reported no active commands or
queued/running jobs; the Cellerator scheduler audit found zero nonterminal jobs.

## Completed foundation

The original implementation is committed and pushed: native support/indexing/
gating, sparse proposals, sampled gate VJPs, latent transitions, CE-GEO forward
with SIMT backward, two trainable slices, and exact support checkpoint restore.
The final integration suite passed 28 CUDA/Python tests, 5 CPU binding tests,
and the earlier native sanitizer and reference checks. Synthetic learning and
small-workload comparisons retain their exact snapshots and limitations under
`bench/sparse_support/evidence/qualification/`.

User scope subsequently expanded to large **and** small structured workloads.
The v3 generator is committed at `4452dc11`; its SHA is
`887437fc3418e06c25e515487891a3224f8dd56493b689bf41bc70e66ae5a76b`.
It creates unique directed pairs, degree cap 512, overlapping cohorts, separate
observed-gene and shared-latent predicates, and independent temporal windows.
Ranked scale is N=4,096/E=262,144/B=32 with 512 selected genes; event scale is
N=20,000/E=1,280,000/B=8 with full retained window. Earlier duplicate-heavy
fixtures were rejected; their diagnostics remain preserved.

## Large-workload findings and pending work

1. The ordered-disjoint posting specialization is implemented and passed GPU
   correctness/overflow checks. q2048 measured about 251 µs versus 663 µs for
   the full scan on its declared ordered representation. Unsorted, overlapping
   and duplicate inputs retain the general merge; duplicate-heavy fallback is
   still slow. **Memcheck/racecheck and final integrated acceptance are pending.**
   See [its resume note](ordered_postings_fastpath_resume.md). Its public cursor
   scratch layout changed; rebuild consumers before further runs.
2. The CE-GEO image validator's quadratic uniqueness/overlap checks were
   replaced with indexed/sorted checks. CPU differential, malformed-image,
   ASan/UBSan and scaling checks passed; one old malformed duplicate-slot
   acceptance is deliberately rejected. E=262,144 validation is about 17 ms.
   Shared image-builder test passed. The refreshed numerical package is
   `build-sparse-support/evidence/torch-v12-validator/package`, identity
   `99da528db6f4ba14bec983b6dfe31f12a065dbdc8b39c0dfb224ed0e969fca07`.
3. The frozen pre-fastpath native large baseline completed: ranked complete
   rescan/posting paths 9.586/32.100 ms; events 27.860/46.338 ms. These include
   prepared bitmap/workset costs. Large packed/SIMT evaluation is prepared but
   **not submitted**. See
   `bench/sparse_support/evidence/experiments/scaled_sparse_campaign_20261027/resume.md`.
4. The CSR-union bitmap experiment finished. For ordered disjoint postings,
   count/scan/concatenate wins; bitmap remains experimental for overlapping or
   arbitrary-ID cases. Its report and exact overflow/sanitizer evidence are in
   `bench/sparse_support/experiments/csr_union_bitmap/`.
5. The scaled two-predicate condition-cache experiment completed with exact
   outputs and clean memcheck/synccheck, showing large specialized gains.
   It remains isolated: general-DAG behavior and separate preparation
   break-even are not established. See
   `bench/sparse_support/experiments/shared_conditions/scaled/`.
6. Geometry grouping/hybrid images have CPU preparation and validator evidence,
   but **no GPU evaluation yet**. See
   `bench/sparse_support/experiments/geometry_layout/resume.md`.
7. Indexed shared relation-value evaluation has a passing CPU VJP oracle and
   prepared KDA candidates, but **no GPU evaluation yet**. See
   `bench/sparse_support/experiments/indexed_values/ownedresume.md`.

Finish these bounded experiments rather than expanding toward full production
workflow integration. The user explicitly accepts prototype limits. Retain
negative results and measured fallbacks; do not claim a universal layout win.

## Inputs preserved across restart

The latest v3 and earlier a17 raw bundles were copied, without removing the
originals, to `build-sparse-support/evidence/restart-inputs/`. If `/tmp` is
cleared, copy those directories back to `/tmp` before using existing task specs
whose arguments reference those paths. Their per-array bytes and hashes, and
the prepared geometry artifacts, are indexed in
`bench/sparse_support/evidence/restart-checkpoint/local-artifact-manifest.json`.
Raw inputs, built executables, profiler binaries and prepared images remain
local/ignored; generators, manifests, raw timing records, source hashes, and
resume notes are versioned.

Project Control plan registration was unavailable; the proposed CE-SIG1 package
was not imported. No Todo completion is asserted. Do not edit authority state
to substitute for the retained source and qualification evidence.
