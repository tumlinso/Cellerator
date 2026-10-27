# Shared-condition root-cache probe: GPU1 results

Scheduler job `013284ad-cbf8-440e-ae91-6ce717ace79a` succeeded and cleanup
was verified. The target was a Tesla V100-SXM2-16GB, selected by exact resource
ID `gpu:testlsserver:GPU-cf22c41f-5b58-77b1-3535-8fadd1ca6505`.

The candidate interns identical two-predicate AND roots, keeps per-cell root
values, and builds predicate-to-root and root-to-edge postings. Full execution
evaluates each unique root once and ballot-materializes the edge bitmap. Dirty
execution updates roots reached by changed predicates and rewrites their edge
bits. The native comparisons use the existing binding-search and prepared
direct edge-to-root condition-bitmap APIs.

## Correctness and measurement

- An independent scalar CPU oracle matched full bitmaps and both update
  scenarios for all three fixtures.
- Compute Sanitizer memcheck reported zero errors on all three fixtures;
  synccheck reported zero errors on the large event fixture.
- Main timings have 21 samples per path; component timings have 7. Competing
  paths rotate through the same run. The logs retain CUDA-event and host-wall
  samples, medians, and p10–p90 ranges.
- Scheduler admission used a 60-second, three-sample, zero-utilization
  precondition. The successful receipt and cleanup record are retained here.
- The first attempt is kept separately as a pre-timing bookkeeping failure; it
  contributes no performance data.

## Complete GPU operation medians

Values are candidate/native microseconds. The full-scan native baseline is the
prepared direct-map path. For each update, the native baseline is the faster of
prepared full rebuild and prepared dirty update; the candidate is the faster of
root-cache full rebuild and root-local postings. Update timings include applying
the supplied state patches and producing bitmap output. Native dirty timings
include dirty-worklist construction. Reset and synchronization between samples
are outside timed regions.

| Fixture | Full bitmap | Local gene patch | Shared latent toggle |
|---|---:|---:|---:|
| Small (E=96, B=8) | 9.120 / 70.016 (7.7×) | 10.240 / 71.680 (7.0×) | 10.240 / 71.680 (7.0×) |
| Ranked (E=262,144, B=32) | 107.936 / 4,508.288 (41.8×) | 16.384 / 3,404.800 (207.8×) | 36.864 / 4,595.712 (124.7×) |
| Events (E=1,280,000, B=8) | 124.928 / 22,817.823 (182.6×) | 15.360 / 16,292.864 (1,060.7×) | 76.800 / 22,817.823 (297.1×) |

Selected large-case p10–p90 ranges, candidate/native microseconds:

- Ranked full: 102.720–119.360 / 4,156.320–4,613.984.
- Ranked local patch: 15.360–17.408 / 3,403.776–3,411.968.
- Ranked latent toggle: 35.840–37.888 / 4,583.424–4,598.784.
- Event local patch: 14.336–16.384 / 16,291.840–18,495.487. The native dirty
  path had a wider upper tail.
- Event latent toggle: 75.776–77.824 / 22,814.720–22,825.983.

The prepared native root map was itself about 2.2× faster than native binding
search for the two large full scans. The candidate beat that stronger baseline.

## Preparation and storage

| Fixture | Input load (ms) | Host compile plus CPU oracle (ms) | Joint device setup (ms) | Native total bytes | Candidate total bytes |
|---|---:|---:|---:|---:|---:|
| Small | 0.25 | 0.16 | 2,764.42 | 10,032 | 7,040 |
| Ranked | 27.98 | 160.31 | 2,316.07 | 12,419,280 | 6,993,280 |
| Events | 134.37 | 398.24 | 3,900.30 | 35,302,664 | 24,489,656 |

Per-path resident-byte totals include common state/update buffers plus that
path's program/index/cache and work buffers. The measurement process allocates
both paths simultaneously; `joint_probe_setup_bytes` is the summed probe
working set, not either deployment's footprint. Host compile time includes
reference bitmap construction and is not candidate-only preparation. Device
setup includes allocations and copies for both paths and CUDA runtime
initialization. These joint setup numbers do not establish a path-specific
compilation break-even.

## Interpretation and limits

- **Keep this as an isolated follow-up candidate.** It was exact on these
  fixtures and materially faster for every measured complete operation. The
  strongest gains occur when many edges share a condition root and a state
  update touches only a small part of the roots or edges.
- The event latent scenario updates about 6.27% of cell-edge pairs. Root-local
  posting maintenance alone (62.464 µs) was slower than native dirty-worklist
  construction alone (35.840 µs), but the complete candidate update (76.800 µs)
  beat the native prepared dirty update (26.526 ms). The measured gain is
  consistent with avoiding repeated DAG evaluation; it does not show that
  postings construction is always cheaper.
- The ranked fixture uses the frozen v3 512-rank presence view as a set; rank
  order is outside this condition-DAG operation. Of 96 supplied local patches,
  50 changed presence values and 46 were idempotent against that set. The exact
  supplied patch arrays were retained.
- The event fixture exercises local and cohort-synchronized latent predicate
  updates, not event-history windows or temporal retrieval. Every edge has an
  exactly two-participant AND condition, and roots have substantial reuse. No
  general multi-node DAG, low-sharing, all-predicates-changing, or dense-root
  counterexample was measured.
- The old duplicate-pair-heavy fd6 fixture was excluded. These results use the
  frozen unique-pair v2 small/event bundles and v3 ranked bundle. Their
  manifests and selected-array hashes are retained under `evidence/input/`.
- This probe does not benchmark learned gates, autograd, Tensor Cores, the
  downstream worklist consumer, event-history indexing, or a full cell model.
  Nsight Compute was not run: no hot-kernel claim is being made.

## Evidence

- `source_identity.json`, `job_spec.json`, `scheduler_job.json`: source, binary,
  input, toolchain, exact request, admitted resource, and cleanup identities.
- `benchmark_*.jsonl`: raw timings, preparation records, and component probes.
- `correctness_*.jsonl`, `memcheck_*.log`, `synccheck_*.log`: correctness and
  sanitizer results.
- `scaled_condition_probe`: staged executable; its SHA-256 is recorded in
  `source_identity.json`.
- `../gpu1_20261027_01/`: failed-before-timing attempt with its verified-cleanup
  scheduler receipt.
