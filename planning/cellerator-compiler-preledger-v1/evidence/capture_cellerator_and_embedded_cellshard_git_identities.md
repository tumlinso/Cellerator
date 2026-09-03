# Cellerator and CellShard Git identities

Todo: `CE-CCP1-A01-002`

Observation time: `2026-09-03T19:39:48Z`

The Cellerator and CellShard observations below are independent Git reads.
They are not an atomic cross-repository snapshot.

## Canonical Cellerator checkout

- path: `/home/tumlinson/Cellerator`
- branch: `main`
- HEAD: `31e491ed29de0fcde70259cbeab8c5c7ad353485`
- `main`: `31e491ed29de0fcde70259cbeab8c5c7ad353485`
- `origin/main`: `31e491ed29de0fcde70259cbeab8c5c7ad353485`
- clean: `false`
- reported dirty paths:
  - `M .todo-orchestrator/state.snapshot.json`
  - `M todo-status.md`
  - `M todos.md`
  - `M todos/ce-ccp1-a01-001.md`
  - `M todos/ce-ccp1-a01-002.md`

These paths are Project Control projections produced while completing and
advancing the active run. They are reported as observed and are not absorbed
into this lane Todo.

## Embedded CellShard gitlink

- path: `components/CellShard`
- gitlink: `b9749ad3e5146a04f847533d8c6f1a54146aed20`
- `git submodule status` marker: `-` (not initialized in this A01 worktree)
- embedded working-tree identity: unavailable in this worktree

The other uninitialized gitlinks reported by `git submodule status` were
`external/highway` at `d92d33d2f4f1b0fecd6183d4626bac08f878ebe0`
and `external/htslib` at `57d5baf4483dd4747222b7ec4ea65258464e530a`.

## Independent CellShard sibling checkout

- path: `/home/tumlinson/CellShard`
- branch: `main`
- HEAD: `b9749ad3e5146a04f847533d8c6f1a54146aed20`
- clean: `true`
- relation to embedded gitlink: exact commit match at observation time

This equality does not make the two repository reads atomic.

## Registered Cellerator worktrees

| Path suffix or path | Branch | HEAD | State |
|---|---|---|---|
| `/home/tumlinson/Cellerator` | `main` | `31e491ed29de0fcde70259cbeab8c5c7ad353485` | dirty; five paths reported above |
| `ce-ccp1-l-a01` | `codex/ce-ccp1-l-a01` | `d72316a4db21f5ab24fd982a6bf42dcf5b823a99` | clean |
| `ce-ccp1-l-a02` | `codex/ce-ccp1-l-a02` | `31e491ed29de0fcde70259cbeab8c5c7ad353485` | clean |
| `ce-ccp1-l-a03` | `codex/ce-ccp1-l-a03` | `31e491ed29de0fcde70259cbeab8c5c7ad353485` | clean |
| `ce-ccp1-l-a04` | `codex/ce-ccp1-l-a04` | `31e491ed29de0fcde70259cbeab8c5c7ad353485` | clean |
| `ce-jbc-l-crossop` | `jbc/ce-crossop` | `a8a6eeb3320bde7a8c0c6c135bcee1f9f0dd3822` | clean |
| `ce-jbc-l-decomposition` | `jbc/ce-decomposition` | `4702972026f85bcf9ba47a252ac2fffca2a1a6bc` | clean |
| `ce-jbc-l-external-cost` | `jbc/ce-external-cost` | `d4268d3f46d80e941fe4e5819fcc4cbd3803f219` | clean |
| `ce-jbc-l-fragment` | `jbc/ce-fragment` | `8a9376598f86e2df33008726f40b7fbe4dc0c2c4` | clean |
| `ce-jbc-l-interfaces` | `jbc/ce-interfaces` | `f4afd1da624f23a6a9428a2b48bbcfe5fe99f3ba` | clean |
| `ce-jbc-l-multiatom` | `jbc/ce-multiatom` | `9dd38f34a25a37c5686bac095c783ed26bf7bb4e` | clean |
| `ce-jbc-l-planes` | `jbc/ce-planes` | `774a41977eeea757816af019e4f4c772106481ef` | clean |
| `ce-jbc-l-resumption` | `jbc/ce-resumption` | `f7f2015be3ea8a44d12cd4a9d6db37199f4509a1` | clean |
| `ce-jbc-l-verify-integrate` | `jbc/ce-integrate` | `3e3c674780b39b00843c9cd1fafbcc44f79ad3c5` | dirty: Project Control projections `.todo-orchestrator/state.snapshot.json`, `todo-status.md`, `todos.md`, and untracked `todos/ce-ccp1-a01-001.md` |
| `/tmp/ce-jbc-main-delivery-20260901` | detached | `71cab986f8e7922a745865cead439b86483cda8c` | registered prunable entry; path unavailable, cleanliness unavailable |

## Commands and disposition

The evidence was collected with `git rev-parse`, `git branch --show-current`,
`git status --porcelain=v1`, `git submodule status`, `git ls-files -s --
components/CellShard`, and `git worktree list --porcelain`. Every registered
worktree was then checked individually; the two dirty worktrees and the one
unavailable prunable entry are all reported above. No Git or Project Control
state was mutated by these observation commands.
