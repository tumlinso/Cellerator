# Manual Todo bootstrap for Cellerator Compiler Part One

## Status and boundary

This document describes how **you**, from a terminal, may validate and then
manually apply the proposed Todo program. Package construction did not apply the
plan, activate the run, claim a task, create an implementation worktree, or
modify Todo Orchestrator authority.

The one apply-ready file is:

```text
planning/cellerator-compiler-preledger-v1/machine/cellerator-compiler-part1.todo-plan.json
```

`machine/proposed_todos.json` is a richer non-authoritative planning catalog.
It is **not** the file to apply.

## Install the planning package in the repository

Place this directory at:

```text
<Cellerator repository>/planning/cellerator-compiler-preledger-v1/
```

Run every command below from the Cellerator repository root. The recorded
baseline is `31efdb245f41263acd4432d78fa9e228e21fd444`.
The package records the full authority cursor in
`planning/cellerator-compiler-preledger-v1/evidence/live_snapshot.json`.

## Safe, read-only sequence

### 1. Inspect the current repository

```bash
git status --short
git rev-parse HEAD
```

Do not apply from a dirty implementation worktree. A changed clean HEAD is not
automatically wrong, but it invalidates the recorded source cursor and requires
a fresh preview. If the architecture, task IDs, interfaces, or compiler source
moved materially, regenerate the package rather than forcing it.

### 2. Validate package integrity and graph consistency

```bash
python3 planning/cellerator-compiler-preledger-v1/scripts/validate_package.py \
  --package-root planning/cellerator-compiler-preledger-v1 \
  --source-root . \
  --require-manifest
```

This checks JSON, task IDs, parents, dependencies, checkpoint expansion, DAG
acyclicity, interface ownership, barrier references, lane coverage, live ID
collisions, human/machine counts, CSV equivalence, and SHA-256 integrity. It
does not contact a mutating Todo API.

The canonical wrapper runs the same check:

```bash
python3 planning/cellerator-compiler-preledger-v1/scripts/todo_bootstrap.py validate --source-root .
```

### 3. Preview the apply-ready plan against live authority

```bash
python3 planning/cellerator-compiler-preledger-v1/scripts/todo_bootstrap.py preview --source-root .
```

This uses only the exact non-mutating preview command captured in
`planning/cellerator-compiler-preledger-v1/evidence/todo_cli_resolution.json`. The wrapper refuses to invent a command when the installed
tooling no longer matches the captured interface.

To print, but not execute, the underlying command:

```bash
python3 planning/cellerator-compiler-preledger-v1/scripts/todo_bootstrap.py preview --source-root . --show-command
```

### 4. Check collisions independently

```bash
python3 planning/cellerator-compiler-preledger-v1/scripts/todo_bootstrap.py collisions --source-root .
```

The package validator already scans live task IDs. This extra command invokes
the resolved Todo/Project Control collision or audit surface when the installed
tool exposes one.

### 5. Inspect the machine plan directly

```bash
python3 -m json.tool planning/cellerator-compiler-preledger-v1/machine/cellerator-compiler-part1.todo-plan.json >/dev/null
python3 - <<'PY'
import json
from pathlib import Path
p = Path("planning/cellerator-compiler-preledger-v1/machine/cellerator-compiler-part1.todo-plan.json")
plan = json.loads(p.read_text())
print("schema_version:", plan["schema_version"])
print("workspace:", plan["project"]["workspace"])
print("baseline_commit:", plan["project"]["baseline_commit"])
print("tasks:", len(plan["tasks"]))
print("interfaces:", len(plan.get("interfaces", [])))
print("barriers:", len(plan.get("barriers", [])))
print("runs:", [run["id"] for run in plan.get("runs", [])])
PY
```

## Mandatory live-precondition decision

Immediately before mutation, compare:

- current `git rev-parse HEAD` with the plan baseline;
- current Todo revision and semantic fingerprint with
  `evidence/live_snapshot.json`;
- Project Control preview collision results;
- current run/lane state, especially historical `CE-JBC-RUN-V1`;
- current interface/checkpoint namespace.

Proceed only when the preview reports that the additive plan is valid and
collision-free. Regenerate the package when any of the following is true:

1. a proposed `CE-CCP1-*` ID now exists;
2. a source migration named by the package was already performed differently;
3. the language/IR specifications changed semantically;
4. an existing interface path or frozen ABI changed;
5. the active authority rejects schema version 3;
6. task dependencies or central-file ownership no longer match live source.

Do not edit the generated plan by hand merely to silence one of these changes.
Update the human architecture, regenerate all machine projections, and rerun
the package validator so the package remains one coherent artifact.

## Mutating sequence

The following operations are deliberately separated from validation. Package
construction stopped before this section.

### 6. Apply/import the plan

First print the exact captured mutation command:

```bash
python3 planning/cellerator-compiler-preledger-v1/scripts/todo_bootstrap.py apply --source-root . --show-command
```

After the safe preview passes and you have deliberately chosen to mutate Todo
authority:

```bash
python3 planning/cellerator-compiler-preledger-v1/scripts/todo_bootstrap.py apply \
  --source-root . \
  --confirm APPLY-CELLERATOR-COMPILER-PART1
```

The confirmation string is a local package interlock. It is not a Todo
Orchestrator semantic feature.

### 7. Verify the imported graph before activating it

```bash
python3 planning/cellerator-compiler-preledger-v1/scripts/todo_bootstrap.py verify --source-root .
```

Verify at least:

- root task `CE-CCP1-0000` exists;
- run `CE-CCP1-RUN-V1` exists but no implementation task is claimed;
- all 557 task records imported;
- all 41 proposed interfaces have exactly one owner;
- all 10 program barriers resolve;
- lane membership is complete;
- no old JBC lane was reused or altered;
- no Part Two task became a Part One prerequisite.

### 8. Activate the new run, if activation is separate

Print the exact command first:

```bash
python3 planning/cellerator-compiler-preledger-v1/scripts/todo_bootstrap.py activate --source-root . --show-command
```

Then, only after graph verification:

```bash
python3 planning/cellerator-compiler-preledger-v1/scripts/todo_bootstrap.py activate \
  --source-root . \
  --confirm ACTIVATE-CE-CCP1-RUN-V1
```

Activation is not a claim. Do not claim implementation work until the imported
architecture/supersession bootstrap tasks have been reviewed.

## Captured underlying command resolution

The package captured exact command forms from installed tooling and the accepted
manual-bootstrap precedent. The machine-readable record is
`planning/cellerator-compiler-preledger-v1/evidence/todo_cli_resolution.json`.

- `validate`: unresolved by live discovery; the wrapper refuses to guess.
- `preview`: unresolved by live discovery; the wrapper refuses to guess.
- `collisions`: unresolved by live discovery; the wrapper refuses to guess.
- `apply`: unresolved by live discovery; the wrapper refuses to guess.
- `verify`: unresolved by live discovery; the wrapper refuses to guess.
- `activate`: unresolved by live discovery; the wrapper refuses to guess.

The wrapper executes only recorded commands. It never falls back to a guessed
Todo-Orchestrator verb. If a command is unresolved or the installed CLI has
changed, rerun discovery against current help and regenerate
`evidence/todo_cli_resolution.json` before applying.

## Canonical order

```text
package validation
→ live source/authority comparison
→ Project Control/Todo plan preview
→ collision audit
→ manual apply
→ imported-graph verification
→ optional run activation
→ later, a separate deliberate claim
```

Stopping at any arrow before `manual apply` is read-only with respect to Todo
authority.
