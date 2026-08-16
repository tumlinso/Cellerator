---
slug: "cellpack-bp00-05-integration-audit"
status: "done"
execution: "closed"
owner: "codex-cp-bp00-05-integration"
created_at: "2026-08-16T15:50:59Z"
last_heartbeat_at: "2026-08-16T15:59:27Z"
last_reviewed_at: "2026-08-16T15:59:27Z"
stale_after_days: 3
objective: "Verify that CP-BP-00 through CP-BP-05 compose as one explicit inference-and-application pipeline without entering CP-BP-06."
---

# Current Objective

## Summary

Trace and validate the real public contracts from deterministic representative
sampling through exact scored candidates, full-domain optimization, and frozen-
plan application. Repair only missing cross-step integration evidence or narrow
adapter defects.

## Quick Start

- Why this stream exists: isolated acceptance tests do not prove that the
  completed CP-BP-01 through CP-BP-05 APIs compose with consistent canonical
  identity, provenance, row-domain, and cost semantics.
- In scope: source-level handoff audit, one permanent end-to-end contract test,
  build wiring, and ledger reconciliation.
- Out of scope: convenience workflow APIs, new inference algorithms, compact
  CP-BP-06 records, row reordering, runtime kernels, persistence, or tuning.
- Required skills: `todo-orchestrator`, `cuda`.
- Required references: CP-BP-00 through CP-BP-05 child ledgers,
  `components/CellPack/AGENTS.md`, and `optimization.md`.

## Planning Notes

- Preserve the explicit low-level boundaries. Prove composition in a test
  rather than hiding ownership, allocation, or synchronization in a new
  convenience pipeline.
- A CP-BP-05-applicable optimizer result must be frozen against the full
  dataset row domain. A sample-scoped plan must continue to be rejected.
- CP-BP-03 singleton-pair gain nominates/ranks optimizer work; CP-BP-04's exact
  whole-plan oracle remains authoritative for accepted later block geometry.

## Assumptions

- The canonical feature axis and full row-domain identities remain caller-owned
  opaque identities because dataset naming/versioning lies outside CellPack.
- Host/reference chaining is sufficient to prove semantic composition; the
  individual CUDA tests continue to own CPU/GPU equivalence.

## File Lease

_Released._ The audit no longer owns source, build, test, or ledger paths.

## Suggested Skills

- `todo-orchestrator`
- `cuda`

## Useful Reference Files

- `include/Cellerator/compute/sampling*.hh`
- `include/Cellerator/compute/gene_support_bitset.hh`
- `include/Cellerator/compute/gene_candidate_discovery.hh`
- `components/CellPack/include/CellPack/merge_cost.hh`
- `components/CellPack/include/CellPack/optimizer.hh`
- `components/CellPack/include/CellPack/packing_plan.hh`
- `components/CellPack/include/CellPack/apply_plan.hh`

## Plan

1. Audit every adjacent handoff and its identity/provenance validation.
2. Add one deterministic host end-to-end test over the real public APIs.
3. Require a full-domain optimizer freeze and exact canonical/value round trip.
4. Run the new test plus focused CP-BP-01 through CP-BP-05 regressions.
5. Record remaining deliberate manual responsibilities and close the audit.

## Tasks

- [x] Inspect CP-BP-00 through CP-BP-05 ledgers and public contracts.
- [x] Confirm CP-BP-01 support provenance reaches CP-BP-02/03 validation.
- [x] Prove CP-BP-03 exact relations feed CP-BP-04 without reinterpretation.
- [x] Prove a full-domain CP-BP-04 plan is accepted by CP-BP-05 and round-trips.
- [x] Run focused regressions and reconcile the parent ledger.

## Blockers

_None._ This is a serial integration audit; CP-BP-06 remains unclaimed and out
of scope.

## Progress Notes

- 2026-08-16: Source tracing found strong CP-BP-01→02→03 provenance checks and
  structurally compatible CP-BP-03 relations/CP-BP-04 candidate views, but no
  permanent test exercises the complete 01→05 chain. Existing optimizer tests
  use synthetic relations and existing application tests hand-freeze plans.
- 2026-08-16: Added a named zero-copy CP-BP-03 relation-view adapter and a
  permanent public-API integration test covering deterministic sampling,
  sampled CSR, support bitsets, candidates, exact scores, full-domain
  optimization, plan application, and exact canonical/value reconstruction.
- 2026-08-16: Preserved CP-BP-01 sample provenance in the CP-BP-04 support view,
  added a versioned deterministic sample/mapping identity, rejected identity
  mismatches, and prevented a partial exact-evaluator source from being labeled
  as the full dataset row domain.
- 2026-08-16: The focused `sm_70` targets and runtime tests passed for sampled
  materialization, CPU/CUDA support, CPU/CUDA candidates, CPU/CUDA merge cost,
  evaluator, optimizer, plan application, end-to-end inference/application,
  and reconstruction. Rebuilding `samplingRuntimeTest` alone remains blocked by
  the pre-existing unrelated `nccl_communicator.cuh` `local_context` failure.
  `git diff --check` passed.

## Next Actions

- None for this audit. CP-BP-06 remains the next representation workstream;
  this audit deliberately added no compact records or workflow façade.

## Done Criteria

- One test calls sampling, sampled CSR materialization, support extraction,
  candidate discovery, exact scoring, optimization, and plan application in
  order using public APIs.
- The optimizer result is explicitly full-domain and CP-BP-05 accepts it.
- Canonical row/feature/value tuples reconstruct exactly after application.
- Existing focused tests still pass and the ledger states any deliberate
  caller-owned identity responsibilities.
