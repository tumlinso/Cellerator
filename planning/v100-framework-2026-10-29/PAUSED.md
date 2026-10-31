# Paused at the user's request

Checkpoint recorded 2026-10-29T20:37:29.923553+00:00 (UTC).

The user requested “Pause at next availability.” Engineering and new workload
submission are stopped. This file is a continuation checkpoint, not Todo
authority or a claim that optimization is complete.

## Completed

- All eleven CUDA smoke families and the nine-test Python binding suite passed
  on V100, including memcheck, racecheck, initcheck and synccheck. The 60 accepted
  family/tool checks, source/binary/log hashes and 22 authoritative Scheduler
  receipts are in `evidence/validation.json`; every contributing job has verified
  cleanup. Some early parent jobs failed other cases; only individually passing
  case checks were accepted, and later complete suites passed.
- Fixed the fused-attention divergent full-warp shuffle and the zero-feature
  dispatch weight-gradient initialization. Corrected the independent fixtures
  before acceptance, including initialized-prefix downloads.
- Updated the CUDA/framework documentation; all 36 Quarto pages rendered to
  `/tmp/cellerator-v100-docs` with a temporary XDG cache.
- Retained 69 source-bound baseline measurement files under
  `evidence/autobench-baseline/`. The packed warp prototype measured about
  0.141 ms versus 1.268 ms backward in the specific 16,384-occurrence, width-128
  FP32 fixture. This is preliminary evidence, not a completed KDA promotion.

## Work remains

- Complete paired KDA evaluation and selection for packed embedding, dispatch
  reduction launch geometry, and attention row launch geometry. No optimization
  candidate has yet completed this formal GPU evaluation or been promoted.
- Dispatch's isolated candidate changes two reduction grids from an
  element-count division to one CTA per selection. Attention's isolated
  candidate changes six row-owned launch grids to one CTA per query. Shared
  production sources have not yet adopted these launch changes.
- A grouped-linear cached-pointer/batched-forward prototype is isolated under
  `grouped_linear/batched_candidate/`; it is CPU-built, not GPU-qualified or
  integrated. It is optional follow-up, not an acceptance gate.
- Frontier baseline is device/sanitizer-qualified. Its optional three-way
  benchmark failed; the latest correctness-only diagnostic identifies line 119
  of `frontier_benchmark.cu`: a host-to-device `cudaMemcpy` returns invalid
  argument. No frontier candidate performance claim is accepted.

## Benchmark integration checkpoint

Two infrastructure issues were diagnosed without changing installed services:

1. Broad `include/Cellerator/**` scopes trigger a deterministic Path-versus-string
   ordering mismatch in the installed Autobench snapshot check. Narrow source
   closures using CUDA-modeling and conditional headers pass actual capture.
2. KDA preparation writes a persistent workspace; Autobench also requires actual
   prepared artifacts under its managed `{build}`. The new shared
   `scripts/validation/kda_autobench_artifacts.py` binds real prepared binaries,
   metadata and hashes. All three outer wrappers now accept `--autobench-build`.

Packed's prepared revision-3 definition passed snapshot-based CPU preparation
and artifact verification using `/tmp/cellerator-kda-managed-builds/packed-final`;
evaluation verification correctly stopped at the Scheduler admission gate.
It has not been published to the live registry or evaluated on a GPU.

Dispatch still needs a narrowed base manifest and a revisioned KDA workspace:
its existing manifest covers 1,069 headers while its new captured definition
contains only the narrower closure. Preserve old immutable KDA records and use
the KDA registration/preparation APIs for the new version.

Attention CPU preparation session 54677 was interrupted at the pause. Preserve
partial workspace/API artifacts and inspect them before resuming. Its output
target was `/tmp/cellerator-kda-managed-builds/attention`; the isolated captured
source was under `/tmp/cellerator-kda-final-snapshots/attention`.

## Submission state and resumption

This task's four registry definitions are preserved as `.toml.paused` files in
`.autobench/registry/`, outside the coordinator's `*.toml` discovery. The installed
parser ignores the `enabled` key, so earlier attempted `enabled = false`
revisions alone did not suspend submissions. The only remaining live definition
is the pre-existing sparse-support indexing campaign; it was not altered.
Autobench reported no unresolved attempts at this checkpoint.

All configured subagents are stopped or complete. Resume only after the user
asks. Before restoring any definition, reconcile its revision, actual source
closure, managed artifact binding, and prepared workspace. Verify captured
CPU preparation before admitting GPU evaluation. Use canonical Autobench for
measurements and the existing Cellerator-scoped Scheduler client for finite
correctness checks; preserve UUID grants and cleanup receipts. Never bypass
admission or edit scheduler/Todo databases.

All current changes remain uncommitted. The last published commit is
`a5190c168b25e2764ccc4e09e406472ab845cae6`. The pre-existing local missing
`external/highway` and `external/htslib` directories remain untouched, with
their Git references preserved.
