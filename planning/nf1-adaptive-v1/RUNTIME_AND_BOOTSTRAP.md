# Runtime binding and manual bootstrap

The delivered package is inert. It adds a successor plan to each existing authority; it does not change live source, recover old claims, retire records, prepare worktrees or launch workers by extraction. Source implementation remains future work. Do not use the old NF1 launch instruction after adopting this amendment.

## 1. Install the two overlays without overwriting old work

From the combined bundle copy only `Cellerator/planning/nf1-adaptive-v1` into the actual registered Cellerator repository and `GlassHelix/planning/nf1-adaptive-v1` into the actual registered GlassHelix repository. Each destination must be new. Do not replace either repository, its existing planning directory, the old NF1 package, `.todo-orchestrator`, Codex configuration or the nested historical Cellerator gitlink in GlassHelix.

An ordinary commit of the new planning directory in each repository before preview is appropriate. Commit only these known added files; do not use a broad staging command that absorbs other work. Package files are sealed for delivery. The controller may later amend implementation guidance through explicit source/authority changes, but it must not pretend altered bytes are still the delivered package.

The shell variables below are operator bindings, not guessed host paths:

```bash
# CE/GH: actual registered authority repository roots.
# PY: Python executable from the verified installed Project Control release.
# Run with the same release/Skills identity environment as its launcher.
# PC: that installed Project Control executable, for canonical admin operations.
# STATE: new external directory, outside both repositories.
REL=planning/nf1-adaptive-v1
```

Resolve PY/PC and their environment from the existing installed launcher and Codex configuration. Do not substitute arbitrary system Python, mix a maintained Skills checkout with an unrelated frozen release, or strip release-manifest identity settings. The preview validates the configured runtime; unavailable or skewed authority fails closed. Secrets are never pasted into the prompt or package.

## 2. Verify construction and inspect the actual native diff

```bash
"$PY" -B "$CE/$REL/scripts/validate_package.py" --peer-package "$GH/$REL"
"$PY" -B "$GH/$REL/scripts/validate_package.py" --peer-package "$CE/$REL"
"$PY" -B "$CE/$REL/scripts/test_package.py"
"$PY" -B "$GH/$REL/scripts/test_package.py"
"$PY" -B "$CE/$REL/scripts/bootstrap.py" preview --state-dir "$STATE"
```

Preview is read-only with respect to repository/authority state. It creates private external review/binding files, resolves both registered roots, checks both package seals, checks successor ID collisions, validates/diffs the exact native plan using the installed kernel, and saves its exact ProposalEnvelope and full observation preconditions. Its only plan rendering is the verified Python executable and the external binding-file path in gate argv. Profiles, alternatives and explanatory JSON are not silently promoted to unsupported native fields.

Review the resulting file and SHA-256. Expected CE diff: nine new records, seven local lanes and one local run, no old task modifications. The exact plan also introduces only its namespaced invariants, gates/checkpoints and generated context. A changed old record, conflicting new identifier, incorrect authority UUID or unexpected runtime warning must be resolved before applying.

## 3. Apply exactly the reviewed proposal, then verify

```bash
"$PY" -B "$CE/$REL/scripts/bootstrap.py" apply --review "$CE_REVIEW" --confirm "$CE_REVIEW_SHA256"
"$PY" -B "$CE/$REL/scripts/bootstrap.py" verify --review "$CE_REVIEW"
"$PY" -B "$GH/$REL/scripts/bootstrap.py" preview --state-dir "$STATE"
# Inspect the GH review: again nine new records, seven lanes, no old modifications.
"$PY" -B "$GH/$REL/scripts/bootstrap.py" apply --review "$GH_REVIEW" --confirm "$GH_REVIEW_SHA256"
"$PY" -B "$GH/$REL/scripts/bootstrap.py" verify --review "$GH_REVIEW"
```

Use the actual emitted filenames and digests for CE_REVIEW/GH_REVIEW. Reviews expire after one hour and also fail if relevant state, packages, interpreter or runtime changed. The wrapper calls Project Control's canonical `apply_proposal` once; it does not directly write SQLite, reapply old task catalogs, or manufacture fresh preconditions at apply time. It verifies actual task identity, queue/role/workspace mode, bound gate argv, and preservation of preexisting task rows. Native validation still owns schema and transaction semantics.

**Stop here for manual bootstrap.** Both imports should be verified; no worker, claim recovery, retirement or source editing is started. Run metadata saying active is not a worker launch. Keep the external review and applied receipts.

A failure after reserving an attempt has potentially ambiguous outcome. The wrapper retains `.attempt.json` and `.uncertain.json` and refuses automatic replay. Use `verify` and exact live inspection. Do not delete the attempt marker merely to retry. If the first project imported successfully and the second did not, preserve the first and create a fresh reviewed proposal only for the incomplete import after resolving its actual state. There is no cross-project atomic transaction or automatic rollback.

## 4. The explicit controller launch authorizes ADOPT

Select Astra for the root session and supply `handoff/START_CONTROLLER.md` with the external receipt/binding location. The intended starts are `GH-NF1A-ADOPT` and `CE-NF1A-ADOPT`, not the aggregates or old NF1 frontier. Use a Cellerator coordinator delegate only if the same native agent cannot occupy both qualified seats. Keep both databases separate.

ADOPT is a compact recovery/handoff outcome. Verify both imports, record actual child model/effort once, preserve pending work, map inherited source to new ownership, recover only stale old NF1 authority through current canonical operations, and retire the exact superseded unfinished records. Do not commission a repeat full architecture audit or rebuild the delegation tooling.

The read-only preservation helper is available:

```bash
"$PY" -B "$GH/$REL/scripts/capture_worktrees.py" --output "$STATE/preservation-1"
```

It records all Git worktrees, captures dirty NF1 source bytes/symlinks plus staged/unstaged patches into an external archive, checks for races, and records source/workspace/patch identifiers without exporting claim secrets. It does not stash/reset/clean, move branches, integrate commits or certify tests. Review the H02 bytes, N ancestry and rejected X01 artifact explicitly. Missing old worktree coverage needs local reconciliation, not an assumption that the work vanished.

Attach a short disposition for each inherited workspace: preserved source and successor owner; already integrated with source evidence; or historical with no remaining source to import. A donor commit can span new owners. Transfer necessary scope once or use the proper integrator rather than duplicating code to fit a directory. Leave old worktrees in place until their source is accounted for and verified merged; retirement is not cleanup.

Generate a refreshed allowlisted retirement intent for each project:

```bash
"$PY" -B "$CE/$REL/scripts/bootstrap.py" retirement-intent --output "$STATE/ce-retirement-intent.json"
"$PC" admin prepare-retire-run-batch --repo "$CE" --intent "$STATE/ce-retirement-intent.json" --output "$STATE/ce-retirement-request.json"
# Root reviews this current request after preservation and old-claim recovery.
"$PC" admin retire-run-batch --repo "$CE" --request "$STATE/ce-retirement-request.json" --apply --confirm RETIRE-RUN-BATCH
```

Repeat in GH with distinct output paths. These are observed canonical admin command forms; verify the installed help if runtime identity changed. The actual native transaction checks UUID/revision/fingerprint/task rows, both run identities, quiescence and nonterminal outside consumers. It supersedes selected unfinished records, skips their queued entries, and cancels the old run. It does not migrate artifacts or complete old capability checkpoints. Already completed records are excluded. A newly completed or additional old task changes the review; do not use the shipped 44/76 counts as permission to overwrite it.

Recovery authority belongs to the root or an explicitly authorized bounded delegate. Do not invent an `authorize-delegated-recovery` CLI: the inspected model-facing CLI intentionally lacks that issuer. A true owner/TTY boundary may require operator action; that is a real authority boundary, not routine engineering indecision.

Only after both ADOPT outcomes are honestly complete should successor implementation workers be dispatched. The root then uses CONTROL as its ordinary seat and closes the aggregate last.

## 5. Prepare the useful frontier and integrate early

All implementation producers use isolated workspaces with explicit local integration targets. The single CE QUALIFY and GH ACCEPT destinations use exclusive integrator lanes. Their claim prerequisites are only CORE or SYSTEM, respectively; their final gates wait for the remaining producer outcomes and required peer acceptance. Thus a final gate does not prevent early integration.

Use the installed scoped `prepare-run-workspaces` operation with the saved rendered native plan (export the `proposed_change` object from the reviewed file). This is an ordinary root administrative action, not another Todo. Its current implementation may require a second idempotent call after producer workspaces exist before the exclusive destination becomes preparable. Do not misdiagnose that as a need for a new workflow framework.

The current materializer enforces a common integration base. Deliver prerequisite commits into the already owned branches or advance a wave deliberately using the native lifecycle. Do not rebase one producer and leave the declared bases inconsistent. Keep an explicit receiver and actual source/test receipt for an early integrated capability. A completed producer branch alone is not a qualified integrated dependency.

Run affected tests during development. Use the outcome gate for a coherent producer result and integrated gates for the combined source pair. A final release record reconciles existing valid evidence; it is not another mandatory full test campaign. Additional legitimate tasks may be created through native replanning when they earn an ownership/dependency/acceptance boundary; the shipped 18 records are not a ceiling or a script.

## 6. Bind real acceptance, never placeholders

`preview` creates deliberately empty external execution bindings. The future implementation is not present, so the package does not fabricate test names, binary paths or GPU verifier commands. ADOPT/root binds them using the inspected existing test/runtime infrastructure. The complete binding contract is `GATE_BINDINGS.md`. The empty template cannot pass a gate.

GPU gates use native `cuda` metadata for a real foreground lease. The runner verifies its live owner, visible devices and project worktree, then holds one shared host lock named identically in both binding files. It supplies compatibility context for retained inner leased tests. Never put an old outer `run_gpu_gate.py` or `run_gate.py` inside this locked CTest inventory: that would reacquire the lock and deadlock. Bind the inner executable/test wrappers instead.

For timing, use actual quiescence/topology controls and report interference; the host lock alone cannot make unrelated GPU work disappear. Do not acquire GPUs for the entire coding lifetime of an outcome. Build/source/binary identity, exact nonempty test inventories and actual JUnit execution are checked; semantic completeness still depends on the declared independent tests and review.

The final order is CE QUALIFY → GH ACCEPT → CE RELEASE → GH RELEASE. Publish the qualified source to both actual origin/main branches. Changed code requires an explicitly source-bound requalification, retaining the original historical completion. A new authorized qualification invocation may evaluate the same outcome contract under a fresh native lease; its receipt is new evidence, not a rewrite of the original done record. Retain the relevant native authorization/attempt reference alongside that receipt. Stop before a biological experiment.
