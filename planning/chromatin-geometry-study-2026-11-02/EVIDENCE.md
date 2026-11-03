# Evidence index

This index locates evidence; it is not native task status or an assertion that
all study acceptance gates have passed. Mutable admissions and resource use
remain in the study controller and Scheduler receipts. The final report must
identify the exact artifacts supporting each conclusion.

## Protocol and source authority

- [Approved private study contract](../../bench/chromatin_geometry/docs/draft.md)
- [Reviewed evidence and cost protocol](protocol.md)
- [Original source and dirty-file baseline](source-baseline.json)
- [Original baseline closure receipt](baseline-archive.json)
- [Native plan package](native-plan.json) and [application receipt](workflow-apply.json)
- [Authorized workflow exception and unresolved repair](workflow-exception.md)
- [Repair preview with exact-definition conflict](workflow-amend-preview-approved.json)

Candidate source archives and binary identities are separate from the original
baseline. Their hashes belong in each measured result. Protected geometry is
checked against the baseline SHA-256; its definition is not changed by this study.

## Independent contracts and checks

- [Scalar numerical contracts and bounds](../../bench/chromatin_geometry/reference/CONTRACT.md)
- [Independent numerical references](../../bench/chromatin_geometry/reference/reference.cc)
- [Deterministic workload implementation](../../bench/chromatin_geometry/workloads.cc)
- [Frozen workload and validator contract v1](contract-freeze-v1.json), SHA-256
  `df9902560d7532cb8bff27c623906cc0500652d29f1037d5e3d5ed2cc0b0a6a8`
- [Replay-aware complete-cost contract v2](contract-freeze-v2.json), SHA-256
  `237da579f5068fc26469f52c79b7ae33beb1ab0af057129a27c186193c5c8147`
- [v2 revision proof](contract-freeze-v2-delta.json): workload values, public
  inputs, numerical bounds and frozen checks are unchanged; required host/replay
  preparation is separated from oracle work and its tests are added.
- [Matched-ownership contract v3](contract-freeze-v3.json), SHA-256
  `4033eea4a8168f5e898212873252c8782943d12f5325e5a31ca45487f9751f06`
- [v3 revision proof](contract-freeze-v3-delta.json): workload values, public
  inputs, numerical bounds and frozen checks are unchanged. The added cached
  two-tile arm matches shared staging's CTA ownership; prior measurements retain
  their original contract and source identities.
- [Pre-timing measurement matrix](../../bench/chromatin_geometry/docs/measurement-matrix-v2.json)
- [Matched-ownership measurement matrix v3](../../bench/chromatin_geometry/docs/measurement-matrix-v3.json)
- [Host-only freeze exporter](freeze-contract.cc)
- [Existing smooth-product regression receipt](smooth-product-regression.json)
- [Consumed design and workflow source pins](design-sources.json)
- [Native raw-HDF5 replay comparator](../../bench/chromatin_geometry/tests/verify_replay_source.cc)
- [Isolated resource and capture tests](../../bench/chromatin_geometry/tests/test_repro_controls.py)

The raw-source comparator checked every prepared ATAC bit and cell/peak identity
against the downloaded HDF5 input, independently of CellShard conversion. Its
negative probe rejects a flipped bit. This establishes encoding fidelity for the
specific input; it does not establish biological gate validity.

## Review records

- [Initial learning review](reviews/learning-review.md)
- [Initial consumer review](reviews/consumer-review.md)
- [Reproducibility and replay verification](reviews/repro-verification.md)
- [Initial complete-runner review](reviews/campaign-review.md)
- [Cached two-tile source review](reviews/two-tile-review.md)

Reviews bind the source identities they inspected. A later implementation fix
requires fresh evidence; an earlier review is not a verdict on a changed binary.

## GPU and measurement records

- [Joined standalone STE evidence](../../bench/chromatin_geometry/analysis/ste-validation.json)
- [Joined correctness records, including failures](../../bench/chromatin_geometry/analysis/correctness-evidence.json)
- [Early retention loss and its correction](../../bench/chromatin_geometry/analysis/retention-note.md)
- [Receipt-bound measurement qualification](../../bench/chromatin_geometry/analysis/qualification-summary.md)
- [Machine-readable qualification](../../bench/chromatin_geometry/analysis/qualification.json)
- [Sanitizer coverage and exact-binary joins](../../bench/chromatin_geometry/analysis/sanitizer-coverage.json)
- [v3 milestone qualification and rejected timing attempt](../../bench/chromatin_geometry/analysis/qualification-v3-summary.md)
- [v3 raw-record qualification export](../../bench/chromatin_geometry/analysis/qualification-v3.json)
- [v3 exact-binary sanitizer joins](../../bench/chromatin_geometry/analysis/sanitizer-coverage-v3.json)
- [Structural demand counts and interpretation limits](../../bench/chromatin_geometry/analysis/structural-analysis.md)
- [Retained diagnostic profile interpretation](../../bench/chromatin_geometry/analysis/profile-interpretation.md)
- [Qualified cost and memory breakdown](../../bench/chromatin_geometry/analysis/cost-summary.md)
- [Offline cost exporter](../../bench/chromatin_geometry/analysis/cost_summary.py)
- [Native HDF5 ingestion cost](native-ingest-cost.json)
- [Paused interim study report](../../bench/chromatin_geometry/analysis/report.md)
- [User-requested pause and resume frontier](PAUSED.md)
- [2026-11-03 resumption and measurement-capture repair](resumption-2026-11-03.md)
- [Bounded grouping probe CPU checkpoint](../../bench/chromatin_geometry/analysis/grouping-probe-checkpoint.md)
- [Bounded grouping qualification stop and two-row result](../../bench/chromatin_geometry/analysis/grouping-probe-result.md)
- [Authorized bounded probe resumption](bounded-grouping-resume-2026-11-03.json)
- [Pre-timing resumed measurement schedule](measurement-plan-resume-2026-11-03.json)
- [Independent old-packet process-attribution review](reviews/pid-attribution-reassessment.md)
- [Independent future process-observation review](reviews/process-observation-v3-review.md)
- [Final pause resource and cleanup receipt](pause-resources.json)

The standalone learning and consumer fixtures provide bounded component evidence.
Complete condition-to-update comparisons, public replay execution, sanitizer
coverage, performance, and profiler interpretation each require their own exact
source/input/cost boundary. Historical passes with missing raw artifacts must
remain distinguishable from retained, independently auditable packets.

Raw immutable archives, packet logs, Scheduler receipts, and the aggregate budget
ledger are retained under the study's ignored `runs/` and enrollment directories.
Portable evidence exports retain their cryptographic identities and paths.
