# NF1A recovery quiescence repair proposal

This is a review artifact only. It does not modify Project Control, the installed runtime, authority state, services, claims, or NF1 source.

## Verified defect

The installed Todo runtime is sourced from `/home/tumlinson/.agents/skills/todo-orchestrator` at skills commit `688dcc2e92ffc2af690c29734ef9d09c67d8e091`. Its installed `todo_orchestrator/workflow/recovery.py` SHA-256 is `c94664084c27fe7cfde5ba0baf8afd02de00fc25f20940c9908341931019b145`, matching that source. `retirement.py` SHA-256 is `bba229e5ab84c23bc490c744c1fe4513c4a7a8570c71e9fc37d6cb8c76bac6be`.

`WorkflowRecoveryEngine.inspect` emits `retire_dispatch` at `todo_orchestrator/workflow/recovery.py:289-307`. Its transaction branch at lines 533-534 marks only `workflow_dispatches` recovered. The later `release_claim` branch at 535-560 can requeue a lane row only when its selected claim/dispatch path reaches that code. `retire_run_batch_in_transaction` deliberately refuses every active lane row at `todo_orchestrator/retirement.py:132-145`, and only skips queued rows at line 161. Thus an active lane task with no live claimant can leave recovery reporting `already_recovered` while retirement correctly refuses the non-quiescent run.

## Minimal correction

Extend the existing recovery operation; do not add an administrative API or weaken retirement quiescence.

1. In the `retire_dispatch` transaction branch, reconcile only the exact `(lane_id, task_id)` action row to `queued` in the same revisioned transaction. Guard it with no other active/orphaned claimant for that task and no active dispatch anywhere on that lane, including a dispatch for a mismatched task. Do not update workspaces, task completion rows, other lane rows, or transition the lane to `ready` while any active lane entry or dispatch remains. Existing dirty-scope handling and workspace quarantine stay in `release_claim`/`quarantine_workspace`.

2. In `inspect`, add a task-scoped query for an `active` `workflow_lane_tasks` row on an `active` lane. It accepts only `planned`, `in_progress`, or `attention_required` task status, and requires neither an active/orphaned claim nor any active dispatch on its lane. An otherwise eligible orphan held by an active lane dispatch emits an `orphan_lane_dispatch` blocker with the exact task, lane, and dispatch IDs, making the plan `refused`; it never reports `already_recovered`. Emit `requeue_orphan_lane_task` only when that blocker is absent. Apply it atomically with the same fresh-inspection, active-lane, claimant, and lane-dispatch guards, then mark only that lane ready when it has no remaining active row or dispatch. This makes already-orphaned H02/N02/S01-style state repairable and makes a repeated recovery idempotently return `already_recovered` after the row is queued.

The new orphan action deliberately does not recover non-`active` lane states (including `attention_required` or `cancelled` lanes), and does not target tasks in review states. Those rows retain their existing handling and are not generalized into lane cleanup by this correction.

The repair must refuse a live claimant, active dispatch, live child, running gate, or sibling lane activity. It must preserve dirty/quarantined workspaces and completed tasks. Retirement remains the only operation that supersedes tasks/cancels the run.

## Required regression coverage

Add to `tests/test_workflow_recovery.py` and `tests/test_retirement.py`:

- A stopped first-class dispatch with an active matching lane row: recovery atomically marks dispatch recovered, claim released, and the exact lane row queued; a freshly prepared retirement then succeeds.
- An already-orphaned active lane row with a planned task and no claim/dispatch: task-specific inspect yields `recovery_needed`; execute queues only that row; repeat is `already_recovered`; retirement then succeeds.
- A live claimant/dispatch, including a mismatched live dispatch on the candidate lane: inspection is `refused` with exact task/lane/dispatch IDs; execution raises `recovery_live_work_refused`, leaves the row untouched, and retirement remains blocked. A foreign sibling-lane active row is also untouched.
- An `attention_required` or otherwise non-active lane and an unsupported review-state task: inspection leaves each unchanged.
- A completed historical task and a dirty/quarantined workspace: recovery leaves completed state/result and workspace preservation/quarantine unchanged.

Run only the focused recovery and retirement tests in an isolated source copy. Root review, source integration, release construction, and deployment are separate actions.

## Candidate evidence

The full Git-generated patch is [RECOVERY_FIX_CANDIDATE.patch](RECOVERY_FIX_CANDIDATE.patch), SHA-256 `cfc89ecd221c8316caf09efd62357ae8493362b14adf36b4e27c3549cd5c8ac0`. It clean-applied with `git apply --check` and `git apply` to a fresh copy at `/tmp/nf1a-recovery-cleanapply-blocker.wSWpPN/skills` from the matching baseline. Using the installed release Python with that fresh copy and its tests directory on `PYTHONPATH`, `python -m unittest tests.test_workflow_recovery tests.test_retirement` passed all 34 tests.

The durable isolated candidate is `/home/tumlinson/.local/state/project-control/nf1a-bootstrap-20260916/recovery-candidate-skills`, cloned from the clean original skills source at `688dcc2e92ffc2af690c29734ef9d09c67d8e091`. It has only the three patched paths and two local, unpushed commits, ending at `18e4f05c234d82dcff47d0dc0cc7ccc0edd2caa0` (`HEAD^{tree}` `0e609172c793f14aaf460b397f210ce8da310519`). The active launcher, service, original source checkout, and authority were not modified.
