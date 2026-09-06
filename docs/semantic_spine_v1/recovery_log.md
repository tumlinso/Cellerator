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
I04-I07 integration/validation, root closure, and final cleanup remain pending.

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
  operation family, and does not start another epic. Implementation and numeric
  parity regressions remain required before F05 closure.
