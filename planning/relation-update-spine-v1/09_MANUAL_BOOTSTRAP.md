# Manual Todo bootstrap

## What this package authorizes

Nothing in this package applies itself. Package construction, installation as repository files, native plan validation, deliberate Todo ingestion and implementation are separate actions. Native schema-3 ingestion creates the run with status `active`, even when the payload omits status; it does not dispatch tasks. The supplied plan has no started tasks. The delivered default is all 35 records planned. No invocation below has been run against the remote authority during package construction, except the explicitly read-only final `plan_preview` validation recorded in evidence.

`machine/relation-update-spine-v1.todo-plan.json` is the **only apply input**. It is native schema 3. Do not apply `proposed_todos.json`, CSV projections or the source snapshot. Do not run the generic preledger compiler over this package: that can discard native run/lane information. `scripts/compile_plan.py --check` is the package's pure deterministic projection checker.

The package requires Python 3.10 or newer for its local validation scripts. C++ reference checks require C++20. The future GPU build uses the project's supported CUDA 12.x sm70 configuration. The manual native bridge additionally requires the actual configured Project Control Python environment; a random system Python need not contain those modules.

## 1. Inspect and stage the overlay, without touching authority

The ZIP has repository-relative `planning/` and `examples/` paths. Extract it into a separate staging directory, inspect it, and run:

```sh
python3 -B STAGING/planning/relation-update-spine-v1/scripts/validate_package.py
python3 -B STAGING/planning/relation-update-spine-v1/scripts/compile_plan.py --check
python3 -B STAGING/planning/relation-update-spine-v1/scripts/test_package.py
```

Replace `STAGING` with the actual extraction directory. Validation checks the complete package manifest and the separately hashed demo files. It neither reads nor writes Todo. A missing/mismatched file is not a reason to disable verification.

To copy the inspected files, ensure neither destination directory already exists. The only additions should be `planning/relation-update-spine-v1/` and `examples/relation_update_spine_v1/`. Do not overwrite an existing package or demo. No production source file is included. Review the resulting Git diff, then commit those two directories as a planning-only commit through your ordinary workflow. No branch/worktree creation is necessary for ingestion. Keep the repository clean before native preview.

The reviewed bootstrap baseline is `45b965123d1d7ae4d10b0ac59e1714fcb7990d02`, the package-install commit. Reinspection covered its removal of the old top-level `semantic_spine_v1/` example; the canonical `examples/semantic_spine_v1/` remains present. The original source evidence retains its historical baseline.

The bootstrap checks that the baseline commit is an ancestor of the reviewed HEAD and that intervening content changes are restricted to those two directories. Other implementation changes require re-review/rebase of the package, not a force flag. Paths should not be symlinks.

## 2. Select the actual Project Control runtime

Set `PCPY` to the Python interpreter used by the installed Project Control environment. This is intentionally explicit rather than guessing a historical candidate directory. Use the same `PROJECT_CONTROL_SKILLS_ROOT`, `PROJECT_CONTROL_RELEASE_MANIFEST` and `PROJECT_CONTROL_RELEASE_DIGEST` as its installed release launcher. Confirm it imports `project_control` and inspect the bridge's runtime identity:

```sh
export PCPY=/absolute/path/to/the/configured/project-control/bin/python
"$PCPY" -B planning/relation-update-spine-v1/scripts/native_bridge.py inspect-runtime
```

The bridge pins the reviewed `project_control.mutation` source identity. The 2026-09-06 reinspection corrected the recorded hash to `3d8558670bb66a3d03614126236582f27649d91b16589a615e379fa4f74282a9`; the installed module is byte-identical to the cited Project Control commit. See `evidence/tooling_contract.json` for the review. It records paths/hashes for the relevant modules and interpreter. A mismatch means the runtime needs reinspection and a coherent package revision. It does not justify editing just the pin or bypassing freshness checks. No native apply was exercised while constructing this package; its public Python API signatures and transaction preconditions were source-verified, and local approval behavior was tested with isolated doubles.

The current CLI also accepts `project-control plan validate --project cellerator --file PATH`. However, this package uses its bridge for application so that the exact previously reviewed preconditions, rather than a silently refreshed snapshot, are supplied to the atomic transaction.

## 3. Fresh native preview and review

After installing/committing the package, run from the Cellerator repository root:

```sh
export REPO="$(git rev-parse --show-toplevel)"
export REVIEW_HEAD="$(git rev-parse HEAD)"
export RECEIPTS="$(mktemp -d "${TMPDIR:-/tmp}/ce-ru1-bootstrap.XXXXXX")"
python3 -B planning/relation-update-spine-v1/scripts/todo_bootstrap.py preview \
  --repo "$REPO" --review-head "$REVIEW_HEAD" --runtime-python "$PCPY" \
  --receipt "$RECEIPTS/preview.json"
```

This checks complete package integrity, a clean explicit HEAD, baseline ancestry, permitted file changes, actual project UUID, coherent Todo/workflow revisions, native schema validation and exact additive diff. Expected: **35 additions, zero modifications, zero warnings**. Existing CE-RU1 IDs, a partial reapply, or a change to unrelated records stops the wrapper.

Inspect `preview.json`. It contains the native plan digest, source HEAD, authority revision/fingerprints, other worktree preconditions, runtime hashes, and intended additions. The receipt must remain outside the repository to avoid changing the approved source fingerprint. It expires after one hour. An expired or changed preview is recreated and reviewed, never relabelled as current.

The archived `evidence/native_plan_validation.json` is a selected-field receipt from this planning turn. It proves that the delivered native payload passed schema/diff validation at baseline revision 6877. It is not a usable application approval because package installation changes Git state and time/workflow may have advanced.

## 4. Deliberate manual ingestion, only after approval

The following is the **only mutating command** in these instructions. Run it only when you separately decide to ingest the reviewed plan:

```sh
python3 -B planning/relation-update-spine-v1/scripts/todo_bootstrap.py apply \
  --repo "$REPO" --review-head "$REVIEW_HEAD" --runtime-python "$PCPY" \
  --preview "$RECEIPTS/preview.json" --receipt "$RECEIPTS/apply.json" \
  --confirm APPLY-CE-RU1-RUN-V1
```

The wrapper repeats validation and compares the fresh response with the reviewed receipt. The bridge then constructs an inert `ProposalEnvelope` containing the **saved reviewed observation preconditions**, calls the actual `apply_proposal`, and relies on the native mutation service's immediate freshness recheck and expected-revision check inside the Todo transaction. It does not directly edit SQLite, use MCP-to-MCP mutation, reconstruct authority, or dispatch tasks. The native transaction creates an `active` run record.

A receipt file is reserved before mutation. Any ambiguous exception, timeout or disconnect is recorded as requiring inspection. **Do not automatically retry.** Inspect the current authority and apply receipt first. Reapplying a partially imported plan or overwriting an earlier receipt is rejected. Rollback would require its own explicitly reviewed administrative operation, not editing authority files.

## 5. Verify ingestion, do not start work yet

Review the actual apply result and read Project Control again. Confirm the expected project UUID, plan digest, 35 task IDs, planned root/leaves, CE-RU1-RUN-V1, eight lane definitions and their queues, checkpoints and interface ownership. Confirm unrelated records, including prior run records, were not replaced. Expect the new run record to be `active` and the observer active-run selection to identify it. No tasks should be claimed and no implementation workspace should be created. The installed runtime has no planned/paused run state; this package does not promise an inactive run.

Task execution still requires later explicit user authorization; this is an instruction boundary, not a runtime activation gate. The package intentionally does not bundle a dispatch command that could accidentally collapse this boundary. The lane handoffs are prospective instructions for that later step.

## Failure and drift rules

Never substitute a capability table or a schema probe for final native validation. The complete delivered plan was validated live; its pretty-JSON observer digest is recorded separately from the CLI compact-JSON digest, so the two encodings are not accidentally compared as if identical. Regenerating any machine plan requires all projections, human scope, manifest and native validation to agree again.

The runtime source pin is intentionally conservative. A changed runtime, project UUID, revision, source HEAD, dirty fingerprint, run/lane context or interface version stops application. Re-review the relevant change, regenerate a preview, and approve it anew. There is no `--force`, automatic task dispatch or schema-downgrade escape hatch.
