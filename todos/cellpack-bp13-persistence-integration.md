---
slug: "cellpack-bp13-persistence-integration"
status: "done"
execution: "closed"
owner: "codex-cp-bp13"
created_at: "2026-08-14T13:00:00Z"
last_heartbeat_at: "2026-08-18T13:05:00Z"
last_reviewed_at: "2026-08-18T13:05:00Z"
stale_after_days: 7
objective: "CP-BP-13: Integrate stable Cellerator packing semantics with CellShard-owned durable .cspack publication and direct execution loading."
---

# Current Objective

## Summary

Define the final ownership/lifecycle boundary: Cellerator compiles and consumes packed execution payloads; CellShard serializes, validates, fetches, and uploads those payloads while preserving canonical partition/gene identity.

## Quick Start

- Why this stream exists: stable plans and packed execution data must be reusable across runs rather than recomputed per partition/minibatch.
- In scope: versioned pointer-free descriptors, compatibility validation, canonical identity linkage, build/compile transform lifecycle, CellShard archive boundary, load/fetch/upload, and direct runtime handoff.
- Out of scope / dependencies: moving discovery semantics into CellShard, per-minibatch relearning/repacking, canonical source mutation, and premature ABI freeze.
- Required skills: `todo-orchestrator`, `cuda` for execution/residency boundaries.
- Required references: CP-BP-00, CP-BP-04, CP-BP-06, CP-BP-08, CP-BP-09, Cellerator and CellShard `AGENTS.md`, `scope.md`, and current `.cspack` contracts.

## Planning Notes

- Intended lifecycle: preserve canonical CellShard identities -> materialize normalized data as appropriate -> invoke Cellerator compile/pack -> persist through CellShard -> later load/upload already-packed payload -> execute directly.
- Exact interface details intentionally wait for evolving Cellerator/CellShard formats. This stream must not freeze incomplete masks/offsets.
- Audit result: retain the current `CSPACK01` shard/partition-offset container.
  CellShard will add a versioned, checksummed execution-envelope payload that
  treats the caller image as opaque bytes while validating generations,
  partition/dataset identities, dimensions, feature-axis identity, payload
  kind/schema, bounds, and checksum. Cellerator will own the inner pointer-free
  CellPack execution image, its plan/order/tile section meanings, semantic
  validation, and host/device runtime rebinding.
- The persisted steady-state image contains the frozen plan arrays required to
  reconstruct canonical identities, the local execution-row permutation, and
  CP-BP-08 tiles. CP-BP-06 records remain a compile-time intermediate and are
  not redundantly persisted beside tiles; their schema version remains bound
  through the tile header.

## Compatibility And Ownership Matrix

| Concern | Owner | v1 rule |
| --- | --- | --- |
| CSPACK container, atomic publication, generation envelope, checksum | CellShard | `CSPACK01`, 64-bit container offsets/counts, little-endian host contract |
| Dataset/shard/partition identity and fetch compatibility | CellShard | explicit nonzero expected identity; mismatch rejects before payload exposure |
| CellPack image sections and semantic versions | Cellerator | pointer-free aligned offsets; plan/order/tile schema versions validated independently |
| Canonical row/feature recovery | Cellerator | full feature permutation/inverse/block maps plus row permutation/inverse remain exact |
| Device staging | CellShard | one contiguous caller-stream H2D transfer into an owned device allocation |
| Native execution | Cellerator | rebind offsets to device pointers and call CP-BP-09 directly; no record/tile rebuild or CSR/BELL reconstruction |

## Claim And File Lease

Claimed by `codex-cp-bp13` from pushed CellStack `711580c`, Cellerator
`d3567ca`, and CellShard `7fcb2cd` after the mandatory read-only audit. The
serial lease is:

- CellShard: new `include/CellShard/io/pack/execution_payload.cuh`,
  `src/io/pack/execution_payload.cu`, `tests/execution_payload_test.cu`, labelled
  CMake blocks, and CSPACK/runtime support documentation;
- Cellerator: new `components/CellPack/include/CellPack/persistent_packing_payload.hh`,
  `components/CellPack/src/persistent_packing_payload.cc`,
  `components/CellPack/tests/persistent_packing_payload_test.cu`, and labelled
  root-CMake blocks;
- CP-BP-13 entries in the root/status/parent ledgers while holding the shared
  ledger lock.

All CP-BP-04/06/07/08/09/10/11/12 source ABIs and kernels are read-only inputs.
No plan inference, record/tile rebuild in the load path, `.csh5` mutation,
Python surface, performance tuning, or unrelated CSPACK family rewrite is in
scope. Build separately in `CellShard/build-cp-bp13` and
`Cellerator/build-cp-bp13`; GPU runs use `/tmp/cellerator-cp-bp13-gpu.lock`.

## Assumptions

- Cellerator is authoritative for `PackingPlan`, packed logical/physical semantics, transformation, CUDA views, and consumers.
- CellShard is authoritative for durable archive validation and I/O integration only.

## Suggested Skills

- `todo-orchestrator`
- `cuda`

## Useful Reference Files

- `AGENTS.md`
- `scope.md`
- `components/CellPack/AGENTS.md`
- `../CellShard/AGENTS.md`
- `../CellShard/include/CellShard/`

## Plan

1. Inventory current `.cspack` versioning, descriptor, validation, and upload boundaries in both repositories.
2. Reuse CP-BP-04's frozen semantic plan identity; freeze physical record/tile descriptors only after CP-BP-06/08/09 correctness.
3. Add CellShard serialization/validation/fetch/upload adapters without semantic plan discovery.
4. Validate archive round trip, compatibility rejection, canonical identity, direct device upload, and no-repack execution.

## Tasks

- [x] Wait for stable v1 plan, block-record, tile, and runtime contracts.
- [x] Define cross-repo ownership and version compatibility matrix.
- [x] Integrate durable `.cspack` payload serialization and validation in CellShard.
- [x] Validate direct load/upload/execute lifecycle and pointer updates in repository order.

## Blockers

- None. CP-BP-04/06/08/09 v1 semantic, record, tile, and direct-runtime
  contracts, CP-BP-10/11 validation, and CP-BP-12's versioned replaceable
  hardware-aware plan-selection policy are integrated inputs.
- Cross-repo edits require reading current CellShard guidance and reconciling its evolving interfaces at pickup time.

## Progress Notes

- 2026-08-18: Completed CP-BP-13. CellShard `197d268` adds the generic
  `CPEXEC01` envelope, exact generation/dataset/partition/feature-axis
  compatibility, atomic CSPACK publication, contiguous host ownership, and one
  caller-stream H2D upload. Cellerator adds a checksummed pointer-free image of
  frozen plan maps, local row order, and CP-BP-08 tiles; validates versions,
  dimensions, offsets, identities, geometry, masks, and permutations; rebinds
  the loaded image to a device base; and invokes CP-BP-09 without record/tile
  rebuild or CSR/BELL reconstruction.
- 2026-08-18: Fresh CUDA 12.9.86/GNU 13.3.0 V100 `sm_70` validation passed the
  CellShard envelope test, the archive/load/upload/direct-execution test,
  CUDA 12.9 memcheck with zero errors, and focused CP-BP-05/06/07/08/09 plus
  optimizer/inferred-pipeline regressions. `git diff --check` passed in both
  repositories. A standalone CellShard CPU-only configure exposed the existing
  `cellshard_cellerator_runtime` no-link-language baseline; the new envelope
  source separately passed warning-clean `CELLSHARD_ENABLE_CUDA=0` syntax.
  No benchmark, profiler, kernel change, or aggressive optimization was added.
- 2026-08-18: Claimed after a read-only audit of the current `CSPACK01`
  container, raw payload codecs, access-adapter contract, generation metadata,
  fetch/upload path, and Cellerator plan/order/tile/runtime views. The current
  CSPACK family has no CellPack image or canonical feature-axis identity in the
  payload, so existing Blocked-ELL/Sliced-ELL codecs cannot be relabeled as
  CP-BP-13. The frozen boundary above adds an opaque CellShard envelope around
  a Cellerator-owned image and preserves both repositories' scope.
- 2026-08-17: CP-BP-12 completed its stable v1 measured-cost/autotune input and
  released all leases. CP-BP-13 is now `planned/ready`, unclaimed, and begins
  with a read-only Cellerator/CellShard compatibility and ownership inventory;
  readiness is not authority to begin serialization before that audit.
- 2026-08-17: Barrier F pushed `2cfa5c8` and removed the shared-worktree/git
  blocker. CP-BP-13 remains deliberately closed behind CP-BP-12; no cross-repo
  inventory or serialization implementation was started.
- 2026-08-17: Barrier E source checkpoint
  `0334f954b1b9e04366f2e2ce191e098c1d476597` satisfies the original Cellerator
  v1 contract prerequisite. CP-BP-13 remains coordination-blocked until Barrier
  F, then begins with the recorded cross-repo inventory rather than premature
  serialization edits.
- 2026-08-14: Added as a missing blocked workstream; existing `.cspack` support remains distinct from this not-yet-stable packed representation.
- 2026-08-16: Reconciliation found existing CellShard `.cspack` artifacts and
  older CellPack coordinate/layout scaffolding, but no serializer/validator or
  direct-load path for `frozen_packing_plan`, CP-BP-06 records, or CP-BP-08
  tiles. Existing durable files do not satisfy this new lifecycle contract.

## Next Actions

- Complete and closed. A later aggressive optimization workflow may profile
  compile, storage, staging, and direct execution, but must preserve these
  versioned ownership and identity contracts unless it deliberately versions
  their replacement.

## Done Criteria

- Cellerator and CellShard ownership is explicit and no discovery/semantic definition moved into CellShard.
- Plan/payload versions, dimensions, offsets, canonical identities, and compatibility failures are validated.
- Archive -> load/fetch -> device upload -> native execution round trip passes without relearning or repacking.
- Cross-repo commits and root submodule pointers follow CellStack ordering and record exact validation commands.
