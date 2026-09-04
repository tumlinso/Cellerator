# CellShard JBC Branch-Tip and Worktree Inventory

This is the preservation receipt for `CE-CCP1-A02-002`. It records the live
CellShard repository at `2026-09-04T04:23:06Z`, before any Part One migration.
The repository was inspected read-only at
`/home/tumlinson/Cellerator/components/CellShard`.

## Reference state

- `main`: `b9749ad3e5146a04f847533d8c6f1a54146aed20`
- `origin/main`: `b9749ad3e5146a04f847533d8c6f1a54146aed20`
- main worktree: clean and synchronized (`ahead 0`, `behind 0`)
- common ancestry base of all 24 local `jbc/*` tips:
  `7762a5925fe18b2ca45ab8a436f3461804ed2ad9`
- preserving octopus merge: `1efc4df57c728aa04383fee8d2acc4f8451c6ffc`
- terminal projection merge: `b9749ad3e5146a04f847533d8c6f1a54146aed20`

`merge-base(main, tip)` equals the tip in every row. Therefore every JBC tip is
already reachable from `main`; there are no preservation-critical unreachable
commits. The changed-path count is the exact `git diff --name-only` count from
the common ancestry base to the branch tip. The immutable tip and base are the
authoritative changed-path enumeration; the command below materializes every
path without copying a large, duplicated projection into this receipt.

```sh
git diff --name-only 7762a5925fe18b2ca45ab8a436f3461804ed2ad9..<branch-tip>
```

## Branch tips

| Branch | Tip (also merge base with `main`) | Changed paths | `origin` | Worktree | Clean | Preservation |
|---|---|---:|---|---|---|---|
| `jbc/cs-atom-core` | `09e324f4bff4b759400e6489a0625154d388d682` | 41 | equal | `cs-jbc-l-atom-core` | yes | reachable |
| `jbc/cs-basis` | `0927aebcbc59571dad02d82ec64548dfc3364f00` | 34 | equal | `cs-jbc-l-basis` | yes | reachable |
| `jbc/cs-certification` | `2df82289e03bc8763ceeef8433828c2a82a7dae0` | 32 | equal | `cs-jbc-l-certification` | yes | reachable |
| `jbc/cs-composition` | `5030e73ba63ef03df4d1691da866d4eadb432cf1` | 48 | equal | `cs-jbc-l-composition` | yes | reachable |
| `jbc/cs-disc-bicluster` | `483dddb0ab56dbf76d60db20149bb8fa5f4f6f0a` | 17 | equal | `cs-jbc-l-disc-bicluster` | yes | reachable |
| `jbc/cs-disc-cosupport` | `68010879022a0f32ad95607d885d6f8b0444a7ad` | 21 | equal | `cs-jbc-l-disc-cosupport` | yes | reachable |
| `jbc/cs-disc-factor` | `0e11cda6dc5f6af20668e86f7a3d6bebce50c78d` | 12 | equal | `cs-jbc-l-disc-factor` | yes | reachable |
| `jbc/cs-disc-motif` | `e1487965d0ee3ced30c1970ec00453946ac57716` | 15 | equal | `cs-jbc-l-disc-motif` | yes | reachable |
| `jbc/cs-disc-multimodal` | `8b80a05784d4ac86a5f322ee3e4b4b4d82191d65` | 20 | equal | `cs-jbc-l-disc-multimodal` | yes | reachable |
| `jbc/cs-disc-optrace` | `38f15b2f7275d2fac004537d2837bce7a85f237b` | 16 | equal | `cs-jbc-l-disc-optrace` | yes | reachable |
| `jbc/cs-disc-overlap` | `3b4d9184f2f5ad14576ef5ee5c83be01e7976688` | 12 | equal | `cs-jbc-l-disc-overlap` | yes | reachable |
| `jbc/cs-disc-sequence` | `30012b4860793e6f06a42df7f07481c4434a6531` | 12 | equal | `cs-jbc-l-disc-sequence` | yes | reachable |
| `jbc/cs-disc-signature` | `d064cb9d950693fcf33e6d8dc6cd602c6a0aa021` | 19 | equal | `cs-jbc-l-disc-signature` | yes | reachable |
| `jbc/cs-disc-trajectory` | `e1816a2498b42c1fb02007a845d6d704f99bb3e9` | 25 | equal | `cs-jbc-l-disc-trajectory` | yes | reachable |
| `jbc/cs-evidence-core` | `2115d742a65e25f5ee034e1d3f824fb9b8366ec1` | 35 | equal | `cs-jbc-l-evidence-core` | yes | reachable |
| `jbc/cs-explicit-grammar` | `764f643bbbd9a97eb4ada84c076158df18d04cde` | 20 | equal | `cs-jbc-l-explicit-grammar` | yes | reachable |
| `jbc/cs-global-ir` | `b4dd91db9700a092159dc2f550eb8300826fa8bc` | 33 | equal | `cs-jbc-l-global-ir` | yes | reachable |
| `jbc/cs-induced-grammar` | `f1774b1c5ea831081e85f7a382145bfff144b6f1` | 22 | equal | `cs-jbc-l-induced-grammar` | yes | reachable |
| `jbc/cs-integrate` | `9f6527276ae53367fe6b699bcdf48467d40f8ab3` | 661 | equal | `cs-jbc-l-validation-integration` | yes | reachable |
| `jbc/cs-partials` | `a971bc28d275842b201417c8279deb627070faf3` | 35 | equal | `cs-jbc-l-partials` | yes | reachable |
| `jbc/cs-persistence` | `27de0b3a1b2083793b678dbf6b5c495efec081ea` | 73 | equal | `cs-jbc-l-persistence` | yes | reachable |
| `jbc/cs-projections-final` | `45aa4bb5ccb4d98a5d54b76663a9d5d05a620591` | 318 | absent | none | n/a | reachable |
| `jbc/cs-runtime` | `da73cf20031d9d56c74da1f11276dfe7560725f1` | 63 | equal | `cs-jbc-l-runtime` | yes | reachable |
| `jbc/cs-superatom` | `ceb108c214b366af60d29d92f6ca61cca7c85154` | 15 | equal | `cs-jbc-l-superatom` | yes | reachable |

## Worktree reconciliation

There are 23 live JBC worktrees: one for every branch except the local-only
`jbc/cs-projections-final`. Each live worktree is attached to the branch and
tip recorded above and has no staged, modified, conflicted, or untracked files.
The integration branch is intentionally checked out in the worktree named
`cs-jbc-l-validation-integration`.

One additional administrative entry is prunable:
`/tmp/cs-jbc-main-delivery-20260901`. It is detached at current `main`, its path
is absent, and Git reports its gitdir as prunable. It is not a JBC branch
worktree and contains no unique commit.

## Preservation disposition

All producer histories are retained by `main`. The only local branch without a
same-named remote tracking ref is `jbc/cs-projections-final`; its tip is the
second parent of the terminal merge and is therefore preserved by both local
and remote `main`. No branch or worktree may be deleted merely on the strength
of this inventory; cleanup remains an explicit later workflow action.
