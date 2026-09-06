# Semantic Spine v1 recovery log

This log records repairs made while resuming the interrupted CE-SS1-RUN-V1.
Todo remains the task authority; this log does not declare unfinished tasks complete.

## 2026-09-06: preserve completed work

- Verified main and all five Semantic Spine lane tips against `origin` before
  reconciliation. Kept the unrelated JBC branches and worktrees untouched.
- Committed the interrupted session's authoritative A04/F04 completion
  projections as `9d834ccc` and pushed main. The four unintegrated lanes retain
  their original implementation commits.

## Project Control runtime mismatch

The installed Codex profile refused startup because its packaged Todo source
fingerprint differed from the configured Skills checkout. The supported
installer built isolated candidates in `/tmp`; the active installation and
registration were not replaced. The replacement passed the runtime identity
check and exposed the expected 20 Codex tools. Restricted-sandbox stdio attempts
timed out; the unrestricted client successfully read the same Cellerator
authority, including revisions 6830 and 6831.

## Native lane stale ownership and workspace recovery

The authority reported N06's stale dispatch and session heartbeat. The supported
owner recovery preview found no live-work blockers. Recovery audit
`9ba2f9f7-4546-4a19-b469-cadddc7a2597` retired stale ownership at revision 6831
and preserved the native worktree in quarantine.

The first attempted supported reprovisioning failed the unique run/lane
constraint. Its transaction rolled back. Temporary branch/worktree renames were
reversed: the original path, branch, and `26aa12ed` tip were restored.

External source inspection established a Todo Orchestrator defect:
`reconcile_workspace_base` selected only `isolated_merge` participants, so a
quarantined `contract_split` workspace could not reactivate. Three isolated
regression tests failed against the original implementation. A narrow repair in
Skills makes reconciliation select only the requested `contract_split`
workspace, retaining the existing isolated-merge behavior and ancestry/cleanliness
checks. All 27 workspace regression tests passed after application. Existing
unrelated Skills and Project Control edits were preserved.

The repaired candidate at `/tmp/project-control-ss1-reactivation-20260906`
successfully reactivated the original native workspace at revision 6832, with
base `6543a911` and tip `26aa12ed`. It changed no files or commits, superseded no
artifacts, and did not repair or modify another lane's integration destination.

## Remaining acceptance work

N06 completed at authoritative revision 6837. Its source and raw evidence hashes
were verified, and its matched direct-provider overhead comparison ran on the
leased V100. The full native suite passed (evidence
`5d6e4132-1356-4108-b03f-718dec4fcdd4`); eight sanitizer runs passed
(`41eeef30-f0dc-497f-8bee-a56b655a2cdf`). Native source/report commits
`30f62f72` and `0498be36` are pushed. The automatic handoff captured the
authority checkout's main HEAD instead of the lane HEAD; its completion note
records the real lane commits, which integration must verify directly.
All accepted lane branches were merged into main and pushed at `f6eae363`.
I04 build integration, I05-I07 execution validation, root closure, and final
cleanup remain pending.

## Internal acceptance audit corrections

- Native preparation: a valid structure identity could XOR into a zero CTP1
  projection identity. The native lane reproduced rejection on a leased V100
  and corrected the derived ID under N06, with public forward/transpose
  regression coverage for both zero-forward and zero-transpose derivations. This does not change the biological identity.
- Frontend lowering: canonical arithmetic retained f16 relation storage, but
  legacy operation/algebra transport incorrectly copied f32 input storage into
  that field and dropped nonfinite rejection. The source-level discrepancy was
  found before integration. Project Control accepted supplemental task
  `CE-SS1-F05` in `CE-SS1-L-FRONTEND-REPAIR`; I04 now depends on it. The original
  frontend lane is closed, and the kernel forbids appending to closed lanes.
  This correction restores the original single-contract requirement, adds no
  operation family, and does not start another epic. The original regression failed before the fix; all five host tests then passed
  with assertions enabled, including numeric/nonfinite parity, independent output
  dtype, copy/move, and unavailable legacy FMA/reassociation transport. Canonical
  mathematics stays valid when legacy transport cannot represent its restrictions.
  Commit `57c6a60c` was pushed and F05 closed at revision 6839.

## Required-gate completion lease

The conformance gate passed under the live CUDA controller lease, but the first
completion call ran outside that lease. Source inspection confirmed that
`finish_task` reruns required executable gates; its rerun correctly rejected the
missing lease receipt. Completion calls for I05-I07 therefore run inside the
controller and benchmark mutex too. This is an invocation correction, with no
workflow-source edit, fabricated lease, skipped gate, or weakened test.

## Integrated acceptance

All six producer branches (the original five plus the frontend correction) are
verified ancestors of main. The root Release build passed with assertions
enabled. I05 completed at revision 6851 after real cross-origin conformance and
missing-device/wrong-output controls. I06 completed at revision 6857 after the
unchanged supplied demo and Compute Sanitizer reported zero errors. Final suite
evidence `6e0f663c-cc34-47c3-ae1f-9d8e022a07ea` passed all required targets
and demo memcheck. The completion call will validate the committed final report
under the same lease discipline before authoritative closure and safe cleanup.
No active project-owned `.ceh` remains; the umbrella content hash is unchanged.

## Closure and cleanup

I07 completed at revision 6863 with final committed-source gate evidence
`18fd6964-3815-447b-aae7-dcdc5223eb39` (validation HEAD `a66a64de`).
The root and run closed at revision 6865: all 29 original leaves plus F05 are
done, all eight lanes closed, and no run claim remains active.

Owner maintenance recorded the six contract-split integration receipts at
revisions 6866-6871, verifying the real producer commits against main `3db12f07`
and the recorded final executable gate. Cleanup eligibility followed at
6872-6877. All six clean, merged worktrees and local/remote Semantic Spine
branches were removed. Remote deletion was atomic and guarded by exact expected
tips. The nine JBC worktrees and their branches were preserved.

The final audit found I05 had no directory at its declared integration artifact
path, although the integrated tests themselves had executed from the owning
verification directory. Added `tests/semantic_spine/integration/README.md` as
the truthful entry point to those real targets, source files, lease requirements
and recorded results. No test implementation or expected result changed.
Regenerated snapshot and Markdown through the supported Todo projection service
after owner maintenance; no database or generated projection was hand-edited.
The database integrity and foreign-key checks pass. General historical audit
notices about tasks without structured per-task gates are not new failed tests;
this epic retains the independent lane evidence and required integrated gates.
