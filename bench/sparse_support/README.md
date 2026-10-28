# Sparse support qualification and experiments

The retained evidence separates exact indexing, numerical/gradient checks,
fixture learning, and complete-cost timing. CUDA runs require a finite admitted
Device Scheduler job; the commands below describe CPU-only work.

All mechanism families remain available for continued development and
real-input evaluation. The [intake guide](../../docs/sparse_support_foundation/evaluation.md)
describes the 15-family catalog and CPU preparation of caller-supplied ranked
or event records, including reversible native identity maps. Synthetic timing
results describe their declared workloads and do not remove families or set
an automatic runtime policy.

```sh
python -m bench.sparse_support.evaluation catalog
```

- `evidence/qualification/` contains native, Torch lifecycle, and synthetic
  learning receipts and source identities. The corrected v9 learning report
  supersedes the invalid sampled-gradient v8 result.
- `evidence/experiments/posting_singleton_v1/REPORT.md` records the retained
  singleton reverse-posting optimization, rescan control, paired timings, and
  Nsight diagnosis.
- `packed_training_benchmark.py`, `kda_adapter.py`, and
  `run_packed_training_kda.py` compare the complete relation stage with a
  shared optimizer/gating contract. This measures a prepared relation, not a
  full event model with proposal and recurrent-state maintenance.
  The [balanced v9 confirmation](evidence/experiments/packed_training_v9_confirmation/run_001/REPORT.md)
  retains both the same-process paired data and the differing KDA comparison;
  neither establishes a promoted packed regime.
- `experiments/bitmap_retrieval/`, `shared_conditions/`, and
  `counted_conditions/` preserve distinct representations and their limits.
  The shared-condition direct-root map was promoted after CUDA/sanitizer checks.
  The other probes remain experimental. The counted-condition probe is CPU-only
  and loses to full scanning for dense predicate changes.
- `experiments/parallel_workset/` has a host oracle and CUDA build, with no GPU
  timing: the admitted probe never began. Its performance remains unknown.
- The [large native comparison](evidence/experiments/scaled_sparse_campaign_20261027/native_ordered_fastpath_20261028/REPORT.md)
  measures ordered posting discovery plus conditional worksets against serial
  merge and full rescan, with preparation and memory kept explicit.
- The [geometry comparison](experiments/geometry_layout/V13_RESULTS.md) retains
  canonical, reordered, and hybrid physical images. Exact forward/VJP checks
  pass; the measured synthetic steps favored SIMT. All layouts remain available.
- The [indexed-value probe](experiments/indexed_values/evidence/qualification/v13/REPORT.md)
  and [event readout trial](experiments/event_learning_diagnostic/README.md)
  motivate explicit optional policies. The [v14 integration](evidence/qualification/v14-optional-policies/REPORT.md)
  checks those options and the broader intake without changing existing defaults.

## Host indexing benchmark

Deterministic small and scaled support/query/event fixtures, including the raw
little-endian bundle schema shared by host and accelerator harnesses, are
documented in [scaled_fixtures.md](scaled_fixtures.md).

`cellerator_sparse_support_indexing_bench` compares the production CPU
full-rescan, reverse-posting, condition-bitmap, dirty-update, and candidate
workset paths on deterministic base-plus-delta support fixtures. Each record
contains the input digest, shape and density counts, memory estimates, raw
timing samples, and exact-oracle status. The output labels itself
`host_only_no_cuda`; it says nothing about GPU performance.

The benchmark checks that rescan and postings return identical stable edge IDs
and ordinals, that dirty bitmap updates equal full recomputation for the
affected predicate dependency set, and that gated primal and derivative
worksets equal an independent scalar oracle. Derivative work is the eligible
union of primal candidates and derivative proposals; the report separately
counts eligible derivative-only work.

Bitmap timing reports both a full cross-cell rebuild and a one-cell full rebuild
beside a one-cell dirty update. Compare the latter pair for equal state scope;
the full cross-cell timing is retained for complete batch costs.

The matrix anchors at 2,176 edges and 64 cells, then changes batch size, edge
count, source skew, gate probability, or dirty-edge fraction independently. A
separate three-term `all` case covers posting intersection semantics. A
candidate-volume cap can skip workset timing on unusually large batches while
retaining correctness checks; the record reports that boundary. Posting construction is timed
separately and amortized over reuse counts 1 and 16. These are host timings and
do not establish a target speedup.

Run the reproducible CPU matrix with:

```sh
python3 scripts/sparse_support/run_cpu_matrix.py \
  --build-dir /tmp/cellerator_sparse_support_cpu_bench_build \
  --output /tmp/cellerator_sparse_support_cpu_indexing.jsonl
```

The runner configures with CUDA disabled and records source/binary SHA-256,
revision, host, build type, exact commands, and raw samples. It performs no GPU
work. GPU discovery, gating, SIMT, packed forward/backward, derivative-only
candidate, sanitizer, and profiler measurements require the separately
admitted CUDA campaign and must use the same frozen fixtures and oracle.
