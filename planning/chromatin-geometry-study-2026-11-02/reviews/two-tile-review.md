# Cached two-tile consumer review

Reviewed 2026-11-02 against the study contract and the fixed source identities
below. This is a source review, not a GPU correctness, sanitizer, or
performance result.

## Findings

1. The qualification exporter at the reviewed revision does not accept the
   new arm: `qualify_results.py:26` omits `cohort-direct-two-tile` from
   `EXPERIMENTAL`, and line 353 rejects unknown candidate identities. More
   importantly, the comparison at lines 560–575 pairs each experimental arm
   with the fastest scalar, bitmap, direct, or applicable library control; it
   does not report the required same-round cached-two-tile versus shared
   staging pair. Adding the identity alone is insufficient to establish that
   ablation. Preserve the strongest-control comparison and add an explicit
   matched staging comparison bound to the v3 contract and matrix.
2. The interference probe's backend allowlist at `interference.py:133` omits
   the new arm and rejects a probe configuration naming it. The campaign must
   update this route or establish its applicable interference evidence through
   another supported route; a rejected probe is not clean timing evidence.

Both findings concern evidence integration. They do not invalidate the older
v2 result, which remains bound to its original source and matrix. The owners
have been notified; later fixes and their tests require their own identities.

## Source assessment

- `consumers.cu:597–610` launches the cached-two-tile and shared variants with
  the same grid, 128-thread blocks, two cell tiles per CTA, and the same
  neighborhood index. `cached_two_tile_consumer` at lines 447–455 processes the
  two successive tiles with a CTA barrier between them. The shared arm at
  lines 504–541 uses the same two-tile ownership with additional staging and
  synchronization. For two buffers it also overlaps a warp's next-tile loads
  with other warps' current-tile work. Thus this is a comparison of complete
  staging strategies at matched work ownership; it does not isolate a single
  cache instruction or equalize all scheduling and synchronization costs.
- The cached helper at lines 420–445 uses the same exact `(P,D)` signatures,
  sorted cell order, group iteration, `product_cell` and `linear_cell` bodies
  as the shared consumer. It passes null staged pointers; participant and
  weight reads therefore use the global-load paths. The arithmetic bodies
  return before participant and parameter reads for excluded forward work,
  and include derivative-only work in backward. The grouped-linear body uses
  the same local rounding helpers as the shared arm.
- The test loop at `consumer_tests.cu:208–265` includes the new backend for both
  bodies, both dtypes and all three profiles; its 65-cell cases exercise a
  second tile and, for tile 32, a trailing CTA. It compares independent
  numerical references and reads the exact device descriptor and identities.
  Poison, empty-arity, empty-batch and stale-revision checks include this arm
  at lines 267–325. Source inclusion is not evidence those tests passed on a
  GPU or that all revision dimensions were independently exercised here.
- The campaign classifies the new arm as cohort and emits two cell tiles per
  CTA, no feature staging, and zero staging buffers
  (`campaign_runner.cu:98–107, 1057–1082`). The v3 matrix declares the matched
  staging pair. The shared path's additional staged loads, barriers, and
  prefetch behavior remain part of the measured strategy difference.

No additional source-level defect was identified in the reviewed new consumer.
Acceptance still requires the planned GPU correctness, sanitizer, and paired
measurement evidence, including exact source, binary and device identities.

## Inspected SHA-256 identities

| Artifact | SHA-256 |
| --- | --- |
| `contract.hh` | `714106c5a18f99132d41878e41beee5f7e3604bc38020ac585bff21f14207794` |
| `backends/consumers.cu` | `28ef73728405f9d8dcb10d81ffdd80fd1ea869ebce899aac5ed396e5a6897245` |
| `backends/consumer_tests.cu` | `a60a6d0983376d630895c2fe09f9b81767b395cead8788277d42e3aaadba4214` |
| `campaign_runner.cu` | `cb4530807e5efb70c27c5ba76a5e35ff9164db267274fa0f9eb23bfce76fa904` |
| `contract-freeze-v3.json` | `4033eea4a8168f5e898212873252c8782943d12f5325e5a31ca45487f9751f06` |
| `measurement-matrix-v3.json` | `0d814998d7a7927f82ad49e10cb95d5d7d09af3e98ff391e0b0ccd6491cebd26` |
| `scripts/qualify_results.py` | `9851a4503fe5ae5876dc7963b75b308f3b2270b349dabe963da26bb095543d9d` |
| `scripts/interference.py` | `291c82869b56d7ab1cfcde56ffff9617f5e41f9c76705ff38ecbc81062131cc7` |

## Follow-up disposition

The two evidence-path findings above are addressed in the following fixed
revisions, independently inspected on 2026-11-02:

- `scripts/qualify_results.py`, SHA-256
  `ca71af915c77fbc946a3e3eb1b5c460f3c107c062d29a929e18acf841cd2b5a6`,
  accepts the added identity and separately computes the matched shared versus
  cached-two-tile ablation. It uses same-packet, same-run, same supplemental
  round pairs, checks all recorded semantic fields including device, input,
  gate, body, precision, update policy, tile and stage count, and preserves the
  conventional strongest-control comparison separately. Packet validation
  checks six complete rounds, reversed paired orders, raw hashes, per-arm
  validation, Scheduler identity, and matching source/binary identities. V2
  and V3 selections have distinct pinned hashes; mixed selections fail.
- `scripts/interference.py`, SHA-256
  `15a8822d3579983bf7b3c298411de9ae0b99b812682ca7cf487a40e02fe3c7fd`,
  accepts the new backend in its configuration allowlist.

The focused qualification and interference unit suites passed independently:
11 and 14 tests respectively, including missing-arm and mismatched-stage
rejection, separation of the ablation from promotion, and backend acceptance.
The retained archive `3cc430c2eccde4fb6d145376ab721875cabf950467f240df28c98138763b8eb2`
has that SHA-256; independently reading its members confirmed all 50 checked
frozen source, candidate, contract and matrix hashes, and the reviewed
interference script plus expected wrapper and driver hashes. This checks
source closure, not a built binary or a completed measurement packet. Actual
new-arm GPU qualification and the paired ablation remain required before any
result can be accepted.
