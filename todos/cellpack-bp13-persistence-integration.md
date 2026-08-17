---
slug: "cellpack-bp13-persistence-integration"
status: "planned"
execution: "ready"
owner: "unassigned"
created_at: "2026-08-14T13:00:00Z"
last_heartbeat_at: "2026-08-17T14:44:32Z"
last_reviewed_at: "2026-08-17T14:44:32Z"
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
- [ ] Define cross-repo ownership and version compatibility matrix.
- [ ] Integrate durable `.cspack` payload serialization and validation in CellShard.
- [ ] Validate direct load/upload/execute lifecycle and pointer updates in repository order.

## Blockers

- None. CP-BP-04/06/08/09 v1 semantic, record, tile, and direct-runtime
  contracts, CP-BP-10/11 validation, and CP-BP-12's versioned replaceable
  hardware-aware plan-selection policy are integrated inputs.
- Cross-repo edits require reading current CellShard guidance and reconciling its evolving interfaces at pickup time.

## Progress Notes

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

- Claim from current pushed Cellerator and CellStack `origin/main`, then perform
  the read-only Cellerator/CellShard compatibility and ownership inventory.
  Do not start serialization before that pickup audit resolves exact versions,
  pointer-free descriptors, identity linkage, and repository ownership.

## Done Criteria

- Cellerator and CellShard ownership is explicit and no discovery/semantic definition moved into CellShard.
- Plan/payload versions, dimensions, offsets, canonical identities, and compatibility failures are validated.
- Archive -> load/fetch -> device upload -> native execution round trip passes without relearning or repacking.
- Cross-repo commits and root submodule pointers follow CellStack ordering and record exact validation commands.
