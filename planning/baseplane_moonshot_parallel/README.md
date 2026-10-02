# Parallel implementation plans

These schema 3 plans derive from the immutable [supplied bootstrap](../baseplane_moonshot_bootstrap/START_HERE.md). They were applied on 2 October 2026 following the user’s instruction. See the [adoption receipt](../baseplane_moonshot_adoption/README.md) for actual policy updates, Baseplane supersession and preserved Cellerator runs. No implementation was dispatched. The validation receipts below record the earlier preparation pass.

## Derivation

Task IDs, task records, dependencies, exclusive scopes, run IDs and scientific boundaries are preserved. Every embedded `planning/bp-moonshot-20261002` reference becomes `planning/baseplane_moonshot_bootstrap`. The charter's delegation guidance now requests dependency-driven parallel work with a bounded total active agent budget.

| Plan | Foundation | Independent family/provider lanes | Integration queue | Total lanes |
| --- | --- | --- | --- | --- |
| [Baseplane](baseplane.todo-plan.json) | BP-MOON-010 | BP-MOON-020 through BP-MOON-130: 12 lanes | BP-MOON-140, BP-MOON-150, BP-MOON-000 | 14 |
| [Cellerator](cellerator.todo-plan.json) | CE-MOON-010 | CE-MOON-020 through CE-MOON-050: 4 lanes | CE-MOON-090, CE-MOON-000 | 6 |

Foundation lanes use role `implementer` and workspace mode `exclusive`. Family/provider lanes use role `implementer`, mode `isolated_merge`, the foundation lane as `parent_lane_id`, and the first integration task as `integration_task_id`. Final lanes use role `integrator`, mode `exclusive`, and the same foundation parent. Native schema 3 explicitly supports these fields and modes.

All family tasks depend on their foundation. Composition/provider handoff tasks depend on every family in their authority. Aggregate closure stays at the end of each integration queue. Parent relationships describe lane structure; task dependencies enforce readiness.

## Execution constraints

Integrate and commit the foundation artifacts before capturing family worktree bases. Family task completion alone does not prove a patch was integrated: the controller must accept family patches into the integration workspace before compositions or provider handoff run. Each family owns its existing distinct directory. Foundation/integration owns shared include files, build files, host/CUDA/Python seams and results aggregation; family workers return requested shared changes to that owner.

The 16 family/provider lanes describe potential independent work. Dispatch in waves within the configured total concurrency ceiling, counting the root, any supervisory heads and all workers. Use configured bounded Codex subagents; root retains claims, workflow lifecycle, cross-authority decisions and final acceptance. GPU work still requires assigned resources and existing timing rules. Native plan validation does not qualify execution or lease behavior.

Before Cellerator adoption, stage the supplied package at `planning/baseplane_moonshot_bootstrap` in the Cellerator workspace so its relative source references resolve. This preparation did not modify Cellerator. Coordinate Baseplane consumers and Cellerator providers using the original cross-authority contract and explicit provider receipts; neither local task DAG automatically waits for another authority's completion. Preserve CE ML2 and use the supplied reviewed supersession procedure for the deferred Baseplane run.

## Verification

Run `python3 planning/baseplane_moonshot_parallel/check_parallel.py` from the Baseplane repository. [Static receipt](evidence/static-checks.json) checks exact task-record preservation after path correction, complete unique queue coverage, lane counts and parent relationships, disjoint family write scopes, unchanged acyclic dependencies, and equality between delivered plans and native preview requests.

[Baseplane native receipt](evidence/baseplane.native-preview.json): `valid: true`, no dependency/scope/interface errors or warnings, `mutation_guard: unchanged`, authority revision 216. Native canonical plan digest: `11658d62c072e08183948eb6531d64b8683a3e97570652bdcd413fcd625d64ab`.

[Cellerator native receipt](evidence/cellerator.native-preview.json): `internal_error`, warning `bounded_read_failed`. Cellerator native validation is unavailable; its static checks pass. Revalidate against the live Cellerator authority before adoption. No native pass is claimed for that payload.

[Baseplane request](evidence/baseplane.native-request.json) and [Cellerator request](evidence/cellerator.native-request.json) record the exact payloads sent to read-only `plan_preview`, mode `validate`, detail `standard`. The attached archive's checker expects a single lane and remains appropriate for the supplied package; this directory uses its own read-only static check.
