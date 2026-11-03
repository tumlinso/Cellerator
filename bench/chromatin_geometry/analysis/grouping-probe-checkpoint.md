# Bounded grouping probe: preparation checkpoint

Paused at the agreed CPU-readiness limit. No GPU qualification or timing job
was submitted, and this continuation consumed zero GPU-seconds. The original
study remains at 2,332.120188475 of 14,400 aggregate GPU-seconds used.

| Reuse horizon | Paired resident result | GPU preparation cost and uncertainty |
| --- | --- | --- |
| 1 evaluation | Not run | Not measured |
| 16 evaluations | Not run | Not measured |

The two private fixtures and focused qualification executable are implemented.
Host checks establish identical inputs and mappings across horizons, 12,160
active memberships out of 16,384, and 128 local cohorts across 64 neighborhood
tiles. All ten original workload records and the numerical bounds remain
unchanged. The immutable archive and CPU-built binaries are retained.

The final adapter preflight completed at 14:57:06.922 UTC, after the
14:56:59 UTC readiness deadline. All new probe lanes and the parent workspace
are paused, and no GPU reservation remains open.

The independent review identified two unresolved pre-timing discrepancies:
the programmed order differs from the exact frozen AB/BA sequence, and the
displayed greater-than-5% flag uses separate arm medians rather than the
approved paired estimator. The next continuation must resolve these before
using timing results; they are packet/protocol issues, not observed numerical
or performance failures of either kernel.

Evidence: [CPU checkpoint and exact identities](../../../planning/chromatin-geometry-study-2026-11-02/grouping-probe-cpu-checkpoint-2026-11-03.json),
[host proof](grouping-probe-host-proof.json), and
[independent review](../../../planning/chromatin-geometry-study-2026-11-02/reviews/grouping-probe-review.md).
