---
slug: "cellpack-bp13-persistence-integration"
status: "blocked"
execution: "closed"
owner: "unassigned"
created_at: "2026-08-14T13:00:00Z"
last_heartbeat_at: "2026-08-16T14:38:44Z"
last_reviewed_at: "2026-08-16T14:38:44Z"
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

- [!] Wait for stable plan, block-record, tile, and runtime contracts.
- [ ] Define cross-repo ownership and version compatibility matrix.
- [ ] Integrate durable `.cspack` payload serialization and validation in CellShard.
- [ ] Validate direct load/upload/execute lifecycle and pointer updates in repository order.

## Blockers

- CP-BP-04 semantic plan identity is complete. This remains blocked on
  CP-BP-06, CP-BP-08, and CP-BP-09 physical representation/runtime decisions.
- Cross-repo edits require reading current CellShard guidance and reconciling its evolving interfaces at pickup time.

## Progress Notes

- 2026-08-14: Added as a missing blocked workstream; existing `.cspack` support remains distinct from this not-yet-stable packed representation.
- 2026-08-16: Reconciliation found existing CellShard `.cspack` artifacts and
  older CellPack coordinate/layout scaffolding, but no serializer/validator or
  direct-load path for `frozen_packing_plan`, CP-BP-06 records, or CP-BP-08
  tiles. Existing durable files do not satisfy this new lifecycle contract.

## Next Actions

- Do not start serialization until the Cellerator logical ABI is complete enough to reject incompatible payloads and decode exactly.

## Done Criteria

- Cellerator and CellShard ownership is explicit and no discovery/semantic definition moved into CellShard.
- Plan/payload versions, dimensions, offsets, canonical identities, and compatibility failures are validated.
- Archive -> load/fetch -> device upload -> native execution round trip passes without relearning or repacking.
- Cross-repo commits and root submodule pointers follow CellStack ordering and record exact validation commands.
