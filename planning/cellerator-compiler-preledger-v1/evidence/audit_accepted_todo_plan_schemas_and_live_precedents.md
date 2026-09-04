# Accepted Todo plan schemas and live precedents

Todo: `CE-CCP1-A01-004`

This audit separates native plans, generated Todo projections, and historical
pre-ledger catalogs. Every listed JSON source was parsed before its keys were
inventoried. The installed Todo validator was then run against each native
plan precedent.

## Installed schema contract

- database migration version: 10
- project schema version: 2
- accepted legacy plan schema version: 2
- accepted first-class workflow plan schema version: 3
- workflow snapshot section version: 1
- workflow protocol version: 2

`validate_plan` accepts plan `schema_version` 2 or 3. Schema 2 is the
compatibility plan form and may not declare first-class `runs`; schema 3 owns
explicit runs, lanes, workspaces, rendezvous, and integration topology. The
validator consumes known keys but does not implement a closed JSON schema that
rejects every unknown key.

## Native plan precedents

| Path | Schema | Installed-validator result |
|---|---:|---|
| `ce-exop-plan.json` | 3 | valid: 291 tasks, 38 checkpoints, 98 gates, 12 barriers, 28 interfaces |
| `.todo-orchestrator/ce-ptr-plan.json` | 3 | valid: 17 tasks, 3 checkpoints, 0 gates, 2 barriers, 1 interface |
| `planning/cellerator-compiler-preledger-v1/machine/cellerator-compiler-part1.todo-plan.json` | 3 | valid: 557 tasks, 44 checkpoints, 512 gates, 10 barriers, 41 interfaces |

All three roots are JSON objects with `schema_version`, `project`, `tasks`,
`invariants`, `locks`, `interfaces`, `barriers`, `resource_classes`, and
`runs`; the observed plans may additionally carry `decisions`.

## Supported and observed keys

Project metadata observed: `baseline_commit`, `name`, `objective`,
`program_document`, `repository`, `workspace`.

Task keys consumed or present in accepted precedents:

`id`, `parent_id`, `kind`, `title`, `objective`, `status`, `priority`, `tags`,
`parallel_policy`, `result`, `next_action`, `notes`, `result_policy`, `scope`,
`claim_locks`, `invariants`, `depends_on`, `consumes_interfaces`,
`publishes_interfaces`, `checkpoints`, `gates`, `resource_requests`,
`produced_artifacts`, and `completion_contract`.

Nested key shapes:

- `scope`: `exclusive_paths`, `read_paths`, `forbidden_paths`, `shared_locks`.
- `result_policy`: `allowed_dispositions`; accepted dispositions are
  `implemented`, `validated`, `evaluated_not_promoted`, `no_change_required`,
  `superseded`, and `failed`.
- dependencies: `type` plus the matching `task_id`, `checkpoint_id`,
  `interface_id`, `barrier_id`, or `decision_id`; conditional keys are
  `operator`, `value`, `state`, `version`, and `dispositions` where applicable.
- produced artifacts: `path`, `kind`, and optionally `sha256`.
- resource requests: `id`, `selector`, `amount`, `mode`, `phase`, `required`.
- checkpoints: `id`, `title`, `state`, `metadata`, `publishes_interfaces`.
- invariants: `id`, `rule`, `scope`, `severity`, with optional enforcement in
  the installed implementation.
- decisions: `id`, `title`, `value`, `allowed`.
- locks: `name`, `capacity`, `metadata`.
- interfaces: `id`, `owner_task_id`, `state`, `version`, `contract_paths`.
- barriers: `id`, `title`, `mode`, `requirements`, with optional `quorum`;
  requirements use `type`, `id`, `state`, and `dispositions`.
- resource classes: `id`, `mode`, `metadata`, `instances`; instances use `id`,
  `capacity`, `hostname`, and `metadata` where supplied.
- runs: `id`, `root_task_id`, `charter`, `lanes`, `rendezvous`,
  `context_fragments`.
- lanes: `id`, `parent_lane_id`, `role`, `tasks`, `workspace`.
- workspaces: `mode`, optionally `integration_task_id`.
- rendezvous: `id`, `mode`, `participants`, `join_task_id`, `barrier_id`, with
  observed optional `producers` and `required_roles`.

Exactly one lane per run must omit `parent_lane_id`. Accepted roles are
`coordinator`, `implementer`, `validator`, `integrator`, and `specialist`.
Accepted workspace modes are `exclusive`, `read_shared`, `isolated_merge`, and
`contract_split`. A workspace `integration_task_id` must name a known task.

## Gate shapes

Accepted precedents use `command` and `benchmark` gates:

- command: required `id`, `type`, nonempty `argv`; observed optional `cwd`,
  `required`, `checkpoint_id`, `input_paths`, and `resources`;
- benchmark: required `id`, `type`, nonempty `argv`; observed optional `cwd`,
  `required`, `resources`, `locks`, `metric_path`, `operator`, `threshold`, and
  `evaluation_required`.

The installed executor also implements static `file_exists`, `pattern`,
`task_state`, `checkpoint`, `interface`, and `manual` gate types, plus
`json_predicate`. Plan validation requires a nonempty `argv` for `command`,
`benchmark`, and `json_predicate`, validates repository-relative gate paths,
and requires referenced checkpoints/resources to exist.

## Interface publication rules

- Every interface must name a known `owner_task_id`; contract paths must be
  safe repository-relative paths.
- Consumers declare `consumes_interfaces` and default to requiring `frozen`
  state. Checkpoints may publish only known interfaces.
- Reapplying a plan cannot silently demote an already frozen or revised
  interface back to draft.
- Runtime publication is capability- and role-checked. The active task must be
  the declared interface owner. Project Control hashes the current contract
  files; a caller-supplied content hash must match that authoritative hash.
- First publication freezes the interface. Publishing an already frozen
  interface revises it and marks active consumers for attention.

## Generated projections and historical catalogs

- `.todo-orchestrator/state.snapshot.json` parsed as a schema-2 generated
  snapshot with snapshot version 3, workflow section version 1, and 40 named
  table projections. It is a projection, not a plan input. The lane-local copy
  records project revision 3895 because the isolated workspace began at that
  plan baseline; live Project Control revision at claim was 3918.
- `planning/jbc-preledger-v1/proposed_todos.json` parsed as a schema-1
  pre-ledger catalog with historical `tasks`, `workstreams`, `lanes`,
  `interfaces`, `barriers`, and `promotion_gates`. It is not a native Todo plan.
- `planning/jbc-preledger-v1/interface_catalog.json` parsed as a JSON array and
  is an auxiliary catalog, not a native plan root.
- `planning/jbc-preledger-v1/plan_summary.json` parsed as a schema-1 summary
  projection (415 tasks, 20 interfaces, 34 lanes, 18 promotion gates), not a
  native plan.
- `planning/jbc-preledger-v1/._proposed_todos.json` is an AppleDouble metadata
  sidecar whose bytes are not UTF-8 JSON. It was detected and rejected as an
  input rather than misclassified as a plan precedent.

The native plan results above were produced by `json.loads` followed by the
installed `todo_orchestrator.plan.validate_plan` with the current repository
root, thereby checking both structure and repository-path safety.
