# Live authority and research snapshot

## Observation

The package records a source and authority observation at `2026-09-03T17:34:41+00:00`.

| Item | Observed value |
| --- | --- |
| Workspace | `cellerator` |
| Project UUID | `0ccaac37-dbbf-448e-a5f8-def197a70aba` |
| Todo revision | `3894` |
| Todo semantic fingerprint | `2247877ebdb63a131bb671b82f789c34bedfe80c2a4e6b741f29a45455bc5899` |
| Workflow revision | `3894` |
| Workflow fingerprint | `2247877ebdb63a131bb671b82f789c34bedfe80c2a4e6b741f29a45455bc5899` |
| Cellerator main | `31efdb245f41263acd4432d78fa9e228e21fd444` |
| Cellerator clean | `false` |
| CellShard source observation | `unavailable` |
| Apply-plan schema | `3` |

The detailed machine snapshot is `evidence/live_snapshot.json`. It includes every discoverable worktree and distinguishes the current local observation from the last explicit Project Control observation used as a fallback for fields absent from local snapshots.

## Current authority constraints

The existing Todo authority contains historical and still-observable runs, including JBC records. The new program must be additive:

- do not hijack `CE-JBC-RUN-V1`;
- do not reopen or rewrite completed JBC tasks;
- do not reuse historical lane IDs;
- do not infer cross-project authority from query grouping;
- use one new Cellerator run, `CE-CCP1-RUN-V1`;
- preserve source and Todo provenance for migrated work.

## Source research performed

The planning package inventories and reasons from:

- current root `CMakeLists.txt`, including unconditional CUDA language enablement, provider/toolchain options, Baseplane integration, and optional embedded CellShard;
- `scope.md`, which already places biological execution semantics in Cellerator and storage/distribution in CellShard;
- `include/Cellerator/execution`, `compute`, `geometry`, `planner`, `profiling`, `runtime`, and JBC interfaces;
- operation core and relation algebra;
- CSG1 and CPE2 boundaries;
- candidate catalog, planner, external cost, connected operation, prepared program, training, readiness, graph capture, and lowering resumption;
- current `docs/language/` documents and externally supplied IR design documents when present;
- `planning/jbc-preledger-v1/`, its source map, interfaces, task graph, validation style, and manual-bootstrap precedent;
- current Cellerator and CellShard JBC worktrees;
- current Todo schema-v3 plan files and generated Todo projections;
- installed Todo/Project Control CLI help where discoverable.

Machine inventories:

- `evidence/language_document_inventory.json`
- `evidence/plan_precedent_inventory.json`
- `evidence/observed_plan_key_union.json`
- `evidence/todo_cli_discovery.json`
- `inventories/worktrees.json`
- `inventories/jbc_source_migration.csv`

## Revalidation rule

Immediately before manual apply:

1. re-read project UUID, Todo revision/fingerprint, workflow revision/fingerprint, and active claims;
2. verify `HEAD`, submodule gitlink, and worktree cleanliness;
3. validate and preview the apply plan against the live authority;
4. regenerate the package if task/interface/checkpoint/run IDs collide or if source changes invalidate the migration map;
5. never force stale preconditions merely because this package once validated.
