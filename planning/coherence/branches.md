# Branch reconciliation

Audit baseline is `main` at `2f851ddb9719a85064f7140864c614245424f4cc`.
The captured branch inventory is [`live-branches.tsv`](live-branches.tsv), SHA-256
`3f41d22324ca1c0c0ef8564fe6db995d4cdb0889bf99a32975eb4500da1e23a8`; its
remote-tracking tips match the refs inspected in this checkout.

The original branch-by-branch comparison is preserved in [`branches.json`](branches.json).
Its disposition records describe the audit at the baseline above; closure was
performed later and is recorded separately in [`branch-closure.json`](branch-closure.json).

## Disposition

No production source needs to be merged before ordinary remote working refs can
close. The two branch-only non-ancestor source commits in `ce-mom-integrator`
and `ce-moon-l-diff` are patch-equivalent to main commits `07765d87` and
`038645f5`, respectively. Current main contains their source trees.

Other non-ancestor CE-MOON tips have corresponding source paths in main:
effects, learning, state, and trajectory are byte-identical; mechanisms and
tensor contain later numerical guards; rewrite has a refreshed capability
receipt; MMA has expanded implementation/evidence provenance and historical
receipts. The compared source prefixes contain no branch-only path absent from
main. Preserve the exact `archive/worktree-cleanup-20261009/codex/...` tags as
the historical source/evidence versions. Do not import their stale generated
Todo projections.

The `ce-moon-l-integrate-v1`, `ce-moon-l-integration`, `ce-mom-producer`,
`resident-moments-20261007`, and all nine `jbc/ce-*` remote tips are ancestors
of main. Their ordinary remote working refs can close without losing commits.

Keep both `backup/todo-cutover-20261019...` refs, all archive/worktree-cleanup
tags, the managed `ce-is1-integrator-l` worktree and branch, the Todo
integration ref, and `main`. The backup `wt-0` commit is graph-unique and
contains an older workspace snapshot; it is retained as backup, not merged as
current source. The IS1 branch tip is already an ancestor of main, but the
managed worktree remains protected because supported cleanup is unavailable.

The full tip-by-tip record, blob comparisons, exact archive tags, and audit-time
closure list are in [branches.json](branches.json). After that audit, 23 ordinary
remote working refs were archived locally and deleted remotely using exact
expected-tip leases. The exact command result, closed tips, retained remote
heads, and durable bundle identity are in [branch-closure.json](branch-closure.json).
The bundle is stored at
`/mnt/block/cellerator/preservation/20261028T204231Z/branch-reconciliation/audited-working-refs.bundle`
with SHA-256
`f5b85451f05133290161a6c283c535b7b484dd78a338fe01ecb533b784f8bd2f`.
Both backup refs, all archive refs, `main`, and the managed IS1 worktree/branch
remain preserved. The supported workflow is still required to close the IS1
worktree and reconcile its registered obligation.
