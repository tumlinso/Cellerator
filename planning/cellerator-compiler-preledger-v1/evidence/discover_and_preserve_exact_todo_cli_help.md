# Exact Project Control and Todo CLI discovery

Todo: `CE-CCP1-A01-005`

Only `--help` and Project Control's read-only native-plan validation were
executed. No apply, activation, recovery, or other write command was run.

## Resolved entrypoints

- Project Control:
  `/home/tumlinson/.local/state/project-control/venvs/project-control-ccp1-workspaces-9127b2d/bin/project-control`
- Installed Todo helper used behind Project Control:
  `/home/tumlinson/.agents/skills/todo-orchestrator/scripts/todo.py`
- Required Project Control runtime binding for this installation:
  `PROJECT_CONTROL_SKILLS_ROOT=/home/tumlinson/.agents/skills`

Project Control remains the workflow and mutation front door. The Todo helper
commands below document the installed lower-level contract; they are not an
authorization to bypass Project Control.

## Project Control plan commands

Top-level plan help:

```text
usage: project-control plan [-h] {compile,validate,apply} ...
```

Validation and preview:

```text
usage: project-control plan validate [-h] --project PROJECT --file FILE
```

Exact read-only invocation:

```text
PROJECT_CONTROL_SKILLS_ROOT=/home/tumlinson/.agents/skills \
  project-control plan validate \
  --project cellerator \
  --file planning/cellerator-compiler-preledger-v1/machine/cellerator-compiler-part1.todo-plan.json
```

This command performs native-plan validation and a collision/diff preview. At
Todo revision 3922 it returned `status: validated`, `valid: true`, plan digest
`af157ca6e4b0662bb77cd8f87daded1756f4d5e6c4971ed314afc4e9b3aa5cf6`,
zero `would_add` entries, and 557 `would_modify` entries. The observation
preconditions also reported zero provider skew. Revision remained 3922, proving
that the preview did not write.

Apply help (captured, not executed):

```text
usage: project-control plan apply [-h] --project PROJECT --file FILE
```

Exact write invocation contract:

```text
PROJECT_CONTROL_SKILLS_ROOT=/home/tumlinson/.agents/skills \
  project-control plan apply --project PROJECT --file FILE
```

Project Control constructs a freshness-bound proposal, reruns validation and
diff, rejects stale preconditions, and delegates the transaction to Todo.

## Installed Todo validate, collision audit, and apply help

```text
usage: todo plan validate [-h] [--repo-root REPO_ROOT] [--json] [--pretty]
                          --file FILE

usage: todo plan diff [-h] [--repo-root REPO_ROOT] [--json] [--pretty]
                      --file FILE

usage: todo plan apply [-h] [--repo-root REPO_ROOT] [--json] [--pretty]
                       --file FILE
```

The exact argv shapes are:

```text
todo.py plan validate --repo-root REPO_ROOT --file FILE --json --pretty
todo.py plan diff     --repo-root REPO_ROOT --file FILE --json --pretty
todo.py plan apply    --repo-root REPO_ROOT --file FILE --json --pretty
```

`plan validate` checks schema, references, paths, roles, workspaces, and graph
acyclicity. `plan diff` is the installed collision audit and reports additions,
updates, and existing entities omitted from the incoming plan. Direct
`todo.py plan apply` is a lower-level write and was not executed.

## Workflow verification help

```text
usage: todo semantic workflow [-h] [--repo-root REPO_ROOT] [--json] [--pretty]
```

Exact read-only argv:

```text
todo.py semantic workflow --repo-root REPO_ROOT --json --pretty
```

For ordinary model-facing operation, the corresponding Project Control reads
are `project_overview`, `project_frontier`, and `coordination_view`.

## Run activation and managed-workspace preparation

The installed CLI has no separate `plan activate-run` or
`plan verify-workflow` subcommand; both candidates were rejected by argparse
during help discovery. In schema 3, `project-control plan apply` creates the
declared run and its lane topology transactionally, and the run is active on
creation. Therefore run activation uses the same freshness-bound apply argv,
not a second mutation command.

Writable `isolated_merge` and `contract_split` lanes additionally require
managed workspaces before `next_task` can claim them. Exact help:

```text
usage: project-control admin prepare-run-workspaces [-h] --repo REPO
                                                    --plan PLAN --run RUN
                                                    [--apply]
                                                    [--confirm CONFIRM]
```

Preview is the default. The exact apply contract is:

```text
project-control admin prepare-run-workspaces \
  --repo REPO --plan PLAN --run RUN \
  --apply --confirm PREPARE-RUN-WORKSPACES
```

After activation and required workspace preparation, model-facing work begins
with Project Control `next_task`. There is no separate `claim_task` tool.

## Discovery safety result

The first read-only validation attempt without `PROJECT_CONTROL_SKILLS_ROOT`
failed closed with `Todo runtime root is not configured`. Supplying the
installed Skills root enabled the read without changing authority. No command
candidate that might write was executed merely to discover its behavior.
