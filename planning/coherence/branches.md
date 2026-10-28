# Branch reconciliation

Audit baseline is `main` at `2f851ddb9719a85064f7140864c614245424f4cc`.
The captured branch inventory is [`live-branches.tsv`](live-branches.tsv), SHA-256
`3f41d22324ca1c0c0ef8564fe6db995d4cdb0889bf99a32975eb4500da1e23a8`; its
remote-tracking tips match the refs inspected in this checkout.

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

The full tip-by-tip record, blob comparisons, exact archive tags, and closure
list are in [branches.json](branches.json). No refs were changed. Project
Control rejected plan apply with `native_operation_rejected`; its authority
revision remained 10, so branch closure and subsequent cleanup remain pending
the supported workflow.
