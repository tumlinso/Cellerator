# Contiguous masked tiles versus cohorts: source audit

Status: the required ablation was missing from the inspected snapshot. Root
accepted the bounded addition below; `masked-direct-two-tile` is now implemented
in live source. SM70 compilation/linking passed; separate GPU qualification
is pending.
The ready `6aec…` / `2404…` measurement artifacts are unaffected and do not
contain this arm. This audit is source evidence, not performance evidence.

The inspected consumer source has SHA-256
`28ef73728405f9d8dcb10d81ffdd80fd1ea869ebce899aac5ed396e5a6897245`;
the private contract has SHA-256
`714106c5a18f99132d41878e41beee5f7e3604bc38020ac585bff21f14207794`.

## What the current arms compare

[`direct_consumer`](../backends/consumers.cu#L291) assigns one warp to each
cell/group pair. Its scalar and plane arms reserve the dense `B×G` warp grid;
the bitmap arm maps those slots through per-cell worklists and returns beyond
each work count. Lanes cooperate on arguments or outputs. For products of
arity at most 32 it uses warp prefix/suffix scans. These are useful
conventional controls, but they are not contiguous cell-lane tiles.

[`build_cohorts`](../backends/consumers.cu#L171) constructs exact local `(P,D)`
signatures, then stably sorts each cell tile by those keys. All three cohort
consumers traverse the resulting equal-key runs. A warp handles a cohort and
each participating lane owns one cell; even `cohort-direct-two-tile` retains
the sort and cohort-start/end scanning. Their product body is the sequential
per-cell [`product_cell`](../backends/consumers.cu#L217), with exclusion
products recomputed without division. Their grouped-linear body likewise
uses the per-cell helper. Thus direct/bitmap versus cohorts changes body
algorithm, lane ownership, dispatch capacity, and preparation together.

The retained one-tile cached cohort versus the new two-tile cached cohort
tests tile ownership. Two-tile cached cohort versus two-tile shared cohort
tests staging at matched coarse ownership. Neither supplies the requested
contiguous masked tiles versus cohorts ablation.

## Accepted private arm

Append `Backend::masked_direct_two_tile`, with CLI name
`masked-direct-two-tile`; preserve all existing enum values and arms. Compare
it primarily with `cohort-direct-two-tile`, with `stage_count=1` and the same
32/64/128-cell tile, body, gate route, precision and update/reuse case.

Prepare the same per-neighborhood/tile `P,D` signature arrays and an identity
cell order, but omit sorting. Use the same `ceil(B/(2T)) × neighborhoods` grid,
128-thread CTA, two serial cell tiles, and inter-tile barrier as the cached
two-tile cohort arm. Thread `j` handles contiguous cell `base+j` when `j` is
inside the tile; each group takes its own lane-local mask. Call the existing
`product_cell` and `linear_cell` helpers with cached global participant and
parameter loads. The existing grouped-linear weight-gradient kernel remains
common. No shared staging or padding is introduced.

This fixes the body implementation, lane-as-cell ownership, CTA extent and
memory tier. The intended changed factor is contiguous masked traversal
versus exact-program preparation/traversal. Warp work distribution changes
because that is part of cohort scheduling; this is not a sort-only experiment.
Report preparation and complete recurrent cost as well as numerical phases.
Do not predict a winner from the number of active warps or logical loads.

## Files, ownership and acceptance

- Integrator: append the enum in `contract.hh`, add the parser/help and backend
  name in `main.cc`/`body_runner.cc`, and retain existing enum assertions in
  `contract_smoke.cc`. No installed API or numerical contract changes.
- Consumer owner: add the unsorted preparation and contiguous two-tile kernel
  in `backends/consumers.cu`. Extend workspace/layout/revision guards and the
  existing debug inspection for the unsorted program rows. Preserve existing
  kernel bodies so old source attribution remains inspectable.
- Runner owner: extend `campaign_runner.cu` parsing, neighborhood layout,
  descriptor checks and metadata; label identity cell order and per-lane
  masking explicitly, rather than reporting this arm as cohorts. Keep the
  backend-independent workload digest unchanged.
- Consumer tests: verify exact identity cell order, raw group occurrences and
  per-cell `(P,D)` signatures against the independent fixture; do not apply
  the sorted-key expectation to this arm. Include zero/empty/tail and mixed
  signatures at every tile boundary, repeated/zero participants, overlapping
  outputs/shared parameters, hard-route zero `dq`, poisoned excluded forward
  participants, dormant STE backward candidates and stale revisions. Reuse
  the frozen numerical oracle and product/linear compound bounds unchanged.

The helpers preserve ordered participants, FP16 storage/FP32 product arithmetic,
and the grouped-linear local rounding boundaries. Changed canonical atomic
arrival order remains subject to the existing compound bounds. No participant
load may precede the forward `P` check, while STE backward must retain all
declared `D` work. All CTA threads must reach the inter-tile barrier, including
tail lanes.

Keep the ready immutable snapshots/binaries and their measurements intact; do not
replace or relabel their source identities. Qualify the new arm in a new
source/binary capture, then measure its matched cached-cohort comparator in
that same capture/device protocol. Existing evidence remains valid for its
named sources, but cannot be retroactively credited with the missing ablation.
