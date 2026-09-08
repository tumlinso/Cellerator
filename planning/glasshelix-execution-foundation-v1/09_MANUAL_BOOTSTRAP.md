# Manual bootstrap, from delivery to verified native graph

## What this procedure does

It installs planning/demo files, validates their integrity and two-project graph, obtains a fresh native schema-3 validation/collision diff, explicitly imports each additive plan, and checks the imported records. It does not implement source or dispatch workers.

The native runtime observed by the predecessor creates **active run metadata** during import. That does not mean an agent is running. Do not use or invent a separate `activate` command. The later user launch of `handoff/START_CONTROLLER.md` authorizes first-class execution.

## 1. Install both overlays and inspect

Copy the CONTENTS of `GlassHelix/` into the actual GlassHelix root and the CONTENTS of `Cellerator/` into the actual Cellerator root. The overlays contain only a new planning directory and example directory for each repository. They do not overwrite root CMake or production implementations.

The combined archive includes `install_overlay.py`: it previews by default and writes only after `--apply`. It refuses changed existing files and unknown destinations; it never runs Git or Todo mutation. Normal reviewed manual copying is also fine.

Set `GH` and `CE` to the actual canonical repository roots. Do not assume the old `extern/Cellerator` gitlink is the canonical Cellerator authority. Set `PCPY` to the Python interpreter that actually imports the installed, configured Project Control runtime. No local Python path is guessed by the package.

```sh
export GH=/actual/GlassHelix
export CE=/actual/Cellerator
export PCPY=/actual/configured/project-control/python
export NF1_EVIDENCE="$(mktemp -d "${TMPDIR:-/tmp}/nf1-bootstrap.XXXXXX")"
```

The path examples above are intentionally placeholders. Verify roots and inspect dirty files before substituting real paths. Review/commit only the new package and demo paths as desired. Inspect already-staged files before committing; never use a broad cleanup or blindly stage unrelated files. The manifest covers package bytes independently of whether they are committed.

## 2. Offline validation and tests

```sh
python3 -B "$CE/planning/glasshelix-execution-foundation-v1/scripts/validate_package.py" \
  --peer-package "$GH/planning/native-foundation-v1"
python3 -B "$GH/planning/native-foundation-v1/scripts/validate_package.py" \
  --peer-package "$CE/planning/glasshelix-execution-foundation-v1"
python3 -B "$GH/planning/native-foundation-v1/scripts/test_package.py"
python3 -B "$CE/planning/glasshelix-execution-foundation-v1/scripts/test_package.py"
```

Package tests use temporary local fixtures and isolated doubles. They do not import, apply or dispatch through a live authority. Their success does not replace native validation below.

## 3. Inspect the native runtime

```sh
python3 -B "$CE/planning/glasshelix-execution-foundation-v1/scripts/todo_bootstrap.py" inspect-runtime \
  --repo "$CE" --runtime-python "$PCPY"
python3 -B "$GH/planning/native-foundation-v1/scripts/todo_bootstrap.py" inspect-runtime \
  --repo "$GH" --runtime-python "$PCPY"
```

If the installed mutation-module pin differs, stop and inspect the actual installed source. The corrected RU1 pin and observed repository file identity differ and are recorded separately. Do not bypass the check or downgrade to schema 2. A deliberate tooling update requires review of the native API, regenerated manifest and fresh preview. This is a compatibility check, not evidence that the host runtime has already been executed here.

## 4. Reconcile the baseline and save reviewed heads

Inspect live project-control source/authority and Git changes since the recorded package baselines. Confirm no CE-NF1/GH-NF1 namespace already exists and that no relevant contract/source change supersedes the plan. Existing JBC/CCP/SS1/RU1 runs and worktrees are not reused.

After review:

```sh
export CE_REVIEW_HEAD="$(git -C "$CE" rev-parse HEAD)"
export GH_REVIEW_HEAD="$(git -C "$GH" rev-parse HEAD)"
```

Supplying a head is your explicit assertion that this source was reviewed. It is not an automatic architectural audit. A changed source/API may require revising the catalog and revalidating all projections. Do not merely edit the native plan to silence a conflict.

The wrapper refuses dirty work by default. If reviewed dirty work is expected, such as the existing `.DS_Store`, newly installed uncommitted overlay or generated Todo projections, add `--acknowledge-dirty` to preview and apply. The snapshot binds those exact dirty bytes. It never cleans them. Runtime receipts must remain outside both repositories.

## 5. Preview Cellerator, review, then explicitly import

```sh
python3 -B "$CE/planning/glasshelix-execution-foundation-v1/scripts/todo_bootstrap.py" preview \
  --repo "$CE" --peer-repo "$GH" --runtime-python "$PCPY" \
  --review-head "$CE_REVIEW_HEAD" --peer-review-head "$GH_REVIEW_HEAD" \
  --receipt "$NF1_EVIDENCE/ce-preview.json"
```

Read the receipt. It must validate the complete exact plan at schema 3, identify the expected project UUID, propose all 80 new CE-NF1 records and modify no existing task. Review native warnings; the wrapper treats them as requiring reconciliation. The saved native observation preconditions, source pair, package manifests and runtime identity are the approval basis.

```sh
python3 -B "$CE/planning/glasshelix-execution-foundation-v1/scripts/todo_bootstrap.py" apply \
  --repo "$CE" --peer-repo "$GH" --runtime-python "$PCPY" \
  --review-head "$CE_REVIEW_HEAD" --peer-review-head "$GH_REVIEW_HEAD" \
  --preview "$NF1_EVIDENCE/ce-preview.json" \
  --receipt "$NF1_EVIDENCE/ce-apply.json" \
  --confirm APPLY-CE-NF1-RUN-V1
python3 -B "$CE/planning/glasshelix-execution-foundation-v1/scripts/todo_bootstrap.py" verify \
  --repo "$CE" --peer-repo "$GH" --runtime-python "$PCPY" \
  --applied-receipt "$NF1_EVIDENCE/ce-apply.json" \
  --receipt "$NF1_EVIDENCE/ce-verify.json"
```

A fresh native validation is checked against the SAVED cursor immediately before apply. The native transaction checks the expected authority revision. It does not replace the approved cursor with a newer one and apply silently. A changed package/source/peer/runtime/cursor requires a new reviewed preview. A preview expires after one hour.

If an apply outcome is ambiguous, inspect actual authority and the durable attempt receipt before any retry. There is no automatic retry and no automatic rollback.

## 6. Fresh GlassHelix preview and import

Cellerator import may update generated repository files. Reinspect those files, refresh reviewed heads if a reviewed commit was made, and explicitly acknowledge expected dirty projections if they remain. Do not reuse a GlassHelix preview taken before the Cellerator import.

```sh
python3 -B "$GH/planning/native-foundation-v1/scripts/todo_bootstrap.py" preview \
  --repo "$GH" --peer-repo "$CE" --runtime-python "$PCPY" \
  --review-head "$GH_REVIEW_HEAD" --peer-review-head "$CE_REVIEW_HEAD" \
  --receipt "$NF1_EVIDENCE/gh-preview.json"
```

Review the complete 90-record additive plan and all 16 local lanes. Then:

```sh
python3 -B "$GH/planning/native-foundation-v1/scripts/todo_bootstrap.py" apply \
  --repo "$GH" --peer-repo "$CE" --runtime-python "$PCPY" \
  --review-head "$GH_REVIEW_HEAD" --peer-review-head "$CE_REVIEW_HEAD" \
  --preview "$NF1_EVIDENCE/gh-preview.json" \
  --receipt "$NF1_EVIDENCE/gh-apply.json" \
  --confirm APPLY-GH-NF1-RUN-V1
python3 -B "$GH/planning/native-foundation-v1/scripts/todo_bootstrap.py" verify \
  --repo "$GH" --peer-repo "$CE" --runtime-python "$PCPY" \
  --applied-receipt "$NF1_EVIDENCE/gh-apply.json" \
  --receipt "$NF1_EVIDENCE/gh-verify.json"
```

Only the native `.todo-plan.json` is applied. Do not apply `proposed_todos.json`, schemas, context fragments or a compiled schema-2 substitute.

## 7. Confirm the imported workflow

Use current project-control read tools to inspect both new runs, all lane queues, 170 task ids, local dependencies, 14 checkpoints/barriers, 12 interface owners and the absence of new claims/dispatches. If the runtime snapshot omits lane/run tables, the wrapper reports partial verification; perform this explicit read rather than interpreting missing evidence as a pass.

Check that existing projects, tasks, runs, pending patches and worktrees were preserved. Native metadata may say active. Worker dispatch must still be absent.

## 8. Start the controller only when ready to execute

Launch one controller in GlassHelix with `planning/native-foundation-v1/handoff/START_CONTROLLER.md`. That launch gives the autonomy described there to execute the accepted program in both authorities. The controller discovers exact current first-class dispatch and GPU services in the A tasks, configures external execution bindings, and starts ready lanes.

Neither package ships a guessed Codex command nor silently begins this step.

## Manual revisions

The rich catalog is the authored planning source; the native plan and CSV/JSON views are derived. `compile_plan.py --check` checks native projection. For a deliberate catalog revision, use `refresh_projections.py --confirm REFRESH-NF1-PROJECTIONS` after updating authored content/task sheets, then rerun package tests, review changed gates and reseal with `seal_package.py --confirm SEAL-NF1-PACKAGE`. No script edits live Todo state during regeneration. Preserve prior delivered receipts as historical evidence and take a fresh native preview before import.
