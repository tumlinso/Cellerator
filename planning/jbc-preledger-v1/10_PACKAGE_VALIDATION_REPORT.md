# Package validation and Todo-Orchestrator handoff report

## Scope

This package is a non-mutating pre-ledger plan. It contains proposed Todos and planning metadata only. It did not alter either Todo authority, claim a lane, create an implementation worktree, begin implementation, or release CE-AMP.

## Live snapshot used

| Item | Value |
| --- | --- |
| Cellerator commit | `4e41d2a3726ca428869a99b58bcaa2e5fc3b5b6c` |
| Cellerator Todo UUID | `0ccaac37-dbbf-448e-a5f8-def197a70aba` |
| Cellerator Todo revision | `3600` |
| CellShard commit | `9187e86c476ecc1014bc3db597cf7b1d1a04a561` |
| CellShard Todo UUID | `a52537a5-20db-4aeb-a126-dd0128c71fda` |
| CellShard Todo revision | `311` |
| CellShard location | `/home/tumlinson/Cellerator/components/CellShard` |

Project Control observations across the two authorities were separate snapshots, not a globally atomic transaction. The ledger-construction agent must revalidate both cursors before applying plans.

## Structural validation

- Proposed Todos: **415**.
- Cellerator Todos: **100**.
- CellShard Todos: **315**.
- Workstreams: **34**.
- Interfaces: **20**.
- Promotion gates: **18**.
- Suggested lanes: **34**.
- Explicit task-to-task dependency edges: **434**.
- Cross-authority/interface receipt dependencies: **8**.
- Task IDs are unique.
- Task-only dependencies form an acyclic graph.
- The only dependency-free roots are `CE-JBC-B01` and `CS-JBC-B01`.
- Every task contains repository, subsystem, lane, write scope, prerequisites, biological motivation, architecture rationale, concrete mechanism, invariants, failure/fallback behavior, validation, performance evidence, and completion criteria.
- Every task appears exactly once in the complete Markdown catalog and exactly once in the JSON/CSV catalog.
- All workstream barrier labels resolve to one of the eight declared major barriers.
- Every deliverable listed in `MANIFEST.sha256` passes its SHA-256 check.
- The ZIP archive passes integrity testing.

## Major interface barriers

1. `JBC-G0-LIVE-BASELINE` — exact source/authority snapshot and charter supersession.
2. `JBC-G1-ATOM-THIN-WAIST` — identity, exact coverage, atom envelope/planes, requirements, affordances, and partial algebra.
3. `JBC-G2-EVIDENCE-CERTIFICATION` — uncertain proposal evidence and independent exact certification are operationally separate.
4. `JBC-G3-CELLERATOR-FRAGMENT` — atom-bound local compilation and candidate frontier.
5. `JBC-G4-COMPOSITION-BASIS` — typed composition/grammar and complete no-basis fallback.
6. `JBC-G5-PARTIAL-ARTIFACT` — persistent partials and lowering-resumption artifacts.
7. `JBC-G6-RUNTIME-SUBSTRATE` — portable schedule, topology, I/O, transport, residency, and exact distributed combine.
8. `JBC-G7-VERTICAL-SLICES` — all required integrated slices before final acceptance.

## Parallelism

- Expected immediately safe first-class fan-out after the minimum thin waist: **24 lanes**.
- Expected peak safe first-class fan-out: **27 lanes**.
- Provider implementations use source-linked fragments; central registries, umbrella headers, package exports, root CMake files, and the parent submodule pointer remain integration-lane-only.
- Cross-authority dependencies are represented as explicit version/hash receipts rather than pretending the two Todo databases provide one atomic dependency graph.

## Live architectural contradictions surfaced rather than hidden

1. `components/README.md` currently states that components do not own planning or runtime; CellShard requires a new privileged compiler-component category while ordinary adapter-component rules remain intact.
2. CellShard `AGENTS.md` and `docs/FORMAT_ROLES.md` still center `.csh5 → .cspack` storage/staging. A versioned successor charter must supersede this as the native compiler direction while keeping CSH5/CSPACK compatibility.
3. CSG1 is an exact disjoint semantic execution cover and cannot also serve as the overlapping uncertain atom-evidence atlas. The plan keeps CSG1 exact and creates a sibling CellShard evidence layer.
4. `program_v2` is useful but narrow. The first integration uses a prepared atom-fragment wrapper instead of prematurely forcing `program_v3`.
5. Cellerator uses 128-bit stable identities while CellShard strong IDs are 64-bit. The plan requires explicit namespace-qualified adapters rather than silent truncation or mixed identity domains.
6. The two Todo authorities currently provide no observed native cross-project interface edge. The plan uses mirrored producer/consumer receipts.
7. CellShard's historical access-adapter record has a raw-closed/effective-ready lifecycle anomaly. It is preserved and reconciled as historical evidence, not reopened or duplicated.

## Decisions intentionally left to the Todo-Orchestrator planning agent

- Exact priorities, run identifiers, and plan-schema fields after reading the live Todo Orchestrator skill and validating collisions.
- Final atom-store format name and magic values after live format collision audit; the logical sections and semantics are already fixed.
- Canonical exported numaBraid CMake target/package name after inspecting its live build.
- Numeric promotion thresholds after local calibration; the ledger should encode required metrics and dispositions, not invented numbers.

No further biological, compiler-architectural, or algorithmic invention should be needed to map the proposed catalog into Todo Orchestrator.
