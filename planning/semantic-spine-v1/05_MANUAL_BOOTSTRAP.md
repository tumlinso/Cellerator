# 5. Manual Todo bootstrap

## Two independent permissions

The present authorization permits creating this package and demonstration program. It does not permit mutating the live project. **You manually apply the plan; execution/activation is a separate subsequent permission.** The supplied wrapper has no activation, agent-launch, worktree-creation, source-patching or run-start operation.

## Current readiness and exact native file

The native input is `machine/semantic-spine-v1.todo-plan.json`, schema 3. The rich `proposed_todos.json`, lane catalog and CSVs are not apply inputs. The local package validator checks their consistency but is not Todo's installed native validator.

At construction, Cellerator source remained at `b3340736c8c7b17266bd825c9f079d86e42a5639` with two library edits. Cellerator's Todo revision/workflow/state/export were unavailable. **A live native validation and collision-free additive diff must be obtained locally before applying.** The wrapper stops if those are unavailable. Do not patch the plan to schema 2: first-class lanes would be lost.

The inspected local Project Control CLI provides:

```text
project-control plan validate --project cellerator --file <native-plan>
project-control plan apply    --project cellerator --file <native-plan>
```

It prints JSON automatically; the inspected parser does not accept a `--json` flag on these subcommands. `plan validate` already invokes native validation and diff. A distinct local `plan diff` was not verified and is not invented here [S13].

## 1. Place, protect and commit the inputs

Copy the archive's `planning/semantic-spine-v1/` and `examples/semantic_spine_v1/` into the repository. Do not overwrite existing different files without inspection. Preserve the two pre-existing library edits, especially the unread `.ceh` body. Review and commit them together with, or before, the supplied new files. No implementation change is required merely to install the plan.

The wrapper requires a clean checkout. It allows only the new package/demo and those two already-reported library paths to differ from the inspected source baseline. Any other implementation changes require a new source review and plan reconciliation. This is intentionally not a `--force` workflow.

```bash
git status --short
git diff -- library/cellerator.cell library/cellerator/cellerator.ceh
# Review staged content and commit the preserved inputs using your normal Git workflow.
git rev-parse HEAD
```

The commit containing the installed package is not the same as the source baseline. Both identities are recorded: the plan's design baseline stays fixed, while the reviewed bootstrap HEAD is supplied explicitly to the wrapper. The user must actually review this HEAD, not treat a fresh hash as evidence that arbitrary code drift is safe.

## 2. Validate package integrity and demo placement

```bash
python3 planning/semantic-spine-v1/scripts/todo_bootstrap.py validate --source-root .
```

This verifies the SHA-256 manifest, native/rich projections, CSVs, ID uniqueness, parentage, checkpoints, interfaces, scopes, dependencies, serial lane order, all 30 records assigned exactly once, four-way fan-out, the strict task cap and the supplied demonstration's bytes. It performs no Todo mutation. Do not use `--without-manifest` to justify application.

## 3. Resolve the installed authority, not stale command guesses

Check the installed command help locally:

```bash
project-control plan validate --help
project-control plan apply --help
```

If the executable or native runtime cannot reach Cellerator's existing authority, repair that connection using the project's normal administration. Do not create a new Todo database, initialize a replacement project, edit SQLite or a generated snapshot, change the UUID, reuse an old run, or downgrade this plan. The package records observed CLI source, not a claim that your current installation is healthy.

## 4. Obtain a fresh native preview

Set `REVIEWED_HEAD` to the literal 40-character commit you inspected, and choose a new receipt path outside the package and authority directories. The `--acknowledge-preserved-library-edits` flag is required only when those original dirty paths differ from the source baseline; it acknowledges their review, not permission to discard them.

```bash
REVIEWED_HEAD="$(git rev-parse HEAD)"
RECEIPT="$HOME/.cache/cellerator/ss1-native-preview.json"
python3 planning/semantic-spine-v1/scripts/todo_bootstrap.py preview   --source-root . --review-head "$REVIEWED_HEAD"   --acknowledge-preserved-library-edits --receipt "$RECEIPT"
```

The wrapper calls the native validator/diff and requires `valid=true`, the correct Cellerator UUID, a concrete revision, coherent first-class workflow authority and **no modifications of existing records**. It records the source, plan, package, executable and authority identities. Read `would_add`, `would_modify`, warnings and the native validation details. Use a fresh filename rather than overwrite an earlier preview receipt.

Any unexpected existing `CE-SS1-*` record is a collision to inspect. This package does not silently reset done tasks, demote interfaces or reuse an active run. A true no-op means no application is needed. Cross-authority dependencies are not required.

## 5. Explicitly apply, then stop

Only after reviewing the preview:

```bash
python3 planning/semantic-spine-v1/scripts/todo_bootstrap.py apply   --source-root . --review-head "$REVIEWED_HEAD"   --acknowledge-preserved-library-edits --receipt "$RECEIPT"   --confirm APPLY-CE-SS1-RUN-V1
```

The wrapper revalidates immediately, compares material source/authority/diff identities to the reviewed receipt, checks that the preview is recent, and invokes the one observed native apply command. It does not retry automatically. A failure after a possible write requires inspecting current authority before trying again.

The current CLI apply command builds its own fresh proposal and checks its own transaction preconditions [S13]. The wrapper's preview and apply are separate processes, not one externally locked transaction. Keep other Todo writers quiescent during the manual sequence. The wrapper checks for material drift immediately before apply; it does not claim to hold an authority lock across these commands or to provide a stronger atomic reviewed-revision guarantee than the installed service.

## 6. Verify the inert result

Using Project Control's read-only overview/coordination/inspection surfaces, verify the new run, all seven lanes, 30 records, draft interface and unreached checkpoints. No leaf should be claimed, no implementation agent should be launched and no workspace should have been created by this package. The root starts planned. Do not treat unavailable workflow inspection as a successful verification.

Then stop. Only a separate execution authorization enables first-class lane setup and implementation. Follow `handoff/coordinator.md` and `03_PARALLEL_LANES_AND_INTEGRATION.md` at that point. There is no hidden “apply and start” helper.

## Revisions and integrity

`scripts/compile_plan.py --check` verifies that the native plan matches the richer catalog. Regeneration changes an artifact, not Todo. Any approved package edit requires regenerating consistent views and the integrity manifest before previewing again. Do not hand-edit only the native JSON and leave the richer task catalog, task sheets or lane map contradictory.

`evidence/` separates package checks performed here from native authority validation and future runtime evidence. Runtime preview/apply receipts must be stored outside the immutable package. A passed reference-only demo is never the GPU acceptance receipt.
