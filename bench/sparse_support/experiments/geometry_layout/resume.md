# Geometry probe checkpoint (pause requested 2026-10-27)

## Pause and execution state

The user requested a computer-restart pause. No GPU job was submitted by this
workstream, no local execution session is active, and there is no queued job to
cancel. GPU1's shared-condition probe completed and verified cleanup before this
checkpoint. No profiler was launched.

This checkpoint is source and CPU preparation only. Do not treat the layout
comparison as measured or validated on GPU.

## Production validator fix

Owned production edit: `src/compute/projection/mma_physical_cover_validation.cc`
(SHA-256 `03532ce1bfc632312ae0802381c6cd06ec6c01f91718cc41238b6c94d5328254`).
This is the validator source used by root's immutable v12 package; root reported
that the shared native image-builder test passed.

The edit replaces pairwise logical-ID duplicate checks and residual row/column/
value range overlap checks with byte ownership arrays and sorted intervals. It
also rejects duplicate physical tile slots, which the frozen original validator
accepted in a crafted image with an orphaned occupancy bit. Allocations are
inside a catch-all boundary so allocation failure returns `false` from the
`noexcept` validator. No on-disk fields or ABI changed.

Standalone differential probe source and frozen baseline are in this folder.
The baseline source is the validator from HEAD `4452dc117ad4b5240602d754506431f5f859d5f0`,
with only its exported symbol renamed. It passed 96 randomized valid images,
288 exact malformed-image outcomes (duplicate logical IDs, reserved header,
bad projection slot), and 96 ignored alignment-padding cases with identical
old/new outcomes. The separate crafted physical-slot case was accepted by old
and rejected by new in all 96 trials. Normal and ASan/UBSan standalone builds
passed.

`evidence/validator_scaling.jsonl` records an unpinned host `-O2` run (15 new
validator samples, 3 warmups; old source median over 5 samples for E<=10k):

- E=1,000: new 0.0496 ms; old 6.224 ms; 125.5x.
- E=10,000: new 0.582 ms; old 605.9 ms; 1,041x.
- E=262,144: new 17.08 ms. The old quadratic path was intentionally not run at
  this size.
- New builder+validator E=262,144: 165.3 ms.

Final-source syntax-only compile, Python syntax checks for the geometry tools,
JSON parsing for the geometry task and prepared-image manifest, and
`git diff --check` passed immediately before the pause. The source hash above
matches root's v12 validator package source identity.

## CPU-only geometry preparation

Candidate and harness files are isolated under this directory. No production
packer interface or universal packing policy was added. The grouping feature
uses only graph-neighbor co-occurrence; fixture module IDs remain diagnostic
metadata and are not candidate input. Physical group member arrays are mapped
back to the original logical entity IDs before Torch binding.

The frozen unique-pair v3 ranked fixture is
`/tmp/cellerator_scaled_sparse_fixtures_887437fc/large-ranked/manifest.json`
(SHA-256 `674d5d4660befcb6ddbcd618b16b136203d3478ddc577d09a037e116bffd6e32`),
E=262,144/N=4,096/B=32/rank_count=512/degree cap 512. The small fixture manifest
SHA-256 is `c33facc18f330739411c1f293609c237689685bd4003099ca79b331fded144be`
(E=96/N=64). The third case is the deterministic local event support
(E=64/N=32) from the existing benchmark helper.

Prepared CE-GEO images and their hashes are recorded in
`evidence/prepared_images/prepared_images.json` (manifest SHA-256
`6ab55eb2476f306188de3a105c711af7e6072bc37c27e876fe628b9761a86ed7`). The
last preparation used immutable package
`build-sparse-support/evidence/torch-v12-validator/package`, identity SHA-256
`99da528db6f4ba14bec983b6dfe31f12a065dbdc8b39c0dfb224ed0e969fca07`; its
extension SHA-256 is
`6a85eebb37d26bf7ddc94a74026a67723d5592470a30b3a55349f2fa95b6071c`.

Image counts show the structural signal but are not runtime evidence:

| Workload | Canonical all-MMA | Graph reorder all-MMA | Reordered threshold 8 | Reordered threshold 16 |
|---|---:|---:|---:|---:|
| Small, E=96/N=64 | 16 tiles / 96 MMA edges | 13 / 96 | 3 / 59 MMA + 37 residual | 2 / 44 MMA + 52 residual |
| Ranked, E=262,144/N=4,096 | 58,688 / 262,144 | 48,113 / 262,144 | 7,687 / 165,425 MMA + 96,719 residual | 2,573 / 109,665 MMA + 152,479 residual |
| Local event, E=64/N=32 | 1 / 64 | 2 / 64 | 2 / 64 | 2 / 64 |

The generated local hybrid builder differs from the current native builder only
by its local function name, minimum-per-tile threshold input, tile-count map,
and residual routing for rejected tiles. This remains a benchmark-local
prototype. The graph reorder gains are modest on the large ranked case and
reverse direction on the event-local case; hybrid thresholding is the main
unmeasured candidate.

## Staged GPU comparison, not yet run

`geometry_training_benchmark.py`, `geometry_kda_adapter.py`, `run_geometry_kda.py`,
`kda_task.json`, and five candidate config/manifests are staged. They are not
registered or prepared in KDA yet, and the adapter was syntax-checked but not
runtime-tested. The task compares SIMT, canonical all-MMA, graph-reordered
all-MMA, and graph-reordered hybrid thresholds 8/16 on small, large-ranked, and
event-local workloads. It reuses the qualified Torch v12 package and the
existing complete training benchmark's paired SIMT/Torch output, parameter
gradient, loss, optimizer, setup, and memory checks. It binds the physical image
against the original logical SupportSpec through the existing native binder.

Remaining authorized work after the restart:

1. Verify the new KDA adapter/task against the local KDA command schema; register
   candidates and CPU-prepare their selectors outside the GPU job.
2. Recheck fixture/image/package hashes and the exact GPU1 scheduler state.
3. In one bounded GPU1 scheduler job, run candidate semantic/oracle validation,
   Compute Sanitizer memcheck for representative new hybrid/residual image
   paths, then paired 21-sample complete training comparisons. Keep GPU2/3
   untouched and use the 60-second zero-utilization scheduler precondition.
4. Report performance only after those gates. If layout/hybrid effects do not
   repay preparation and residual costs, retain this as negative/limited
   experimental evidence and do not add a production packer policy.

No KDA candidate registrations, scheduler job specs, job IDs, or results exist
for this geometry comparison at this checkpoint.
