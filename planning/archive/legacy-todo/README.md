# Legacy Todo archive

The seven legacy paths below were archived and removed from the working checkout on 2026-10-29 with explicit user authorization. Their data is preserved outside the repository; the source-bound index here resolves their former paths without retaining obsolete projections as current guidance. This is preservation evidence, not live task status or an authority migration.

| Original path | Allocated size | Tracked files |
|---|---:|---:|
| `todos` | 5.62 MiB | 1341 |
| `todos.md` | 0.33 MiB | 1 |
| `todo-status.md` | 0.28 MiB | 1 |
| `.todo-orchestrator/state.snapshot.json` | 26.63 MiB | 1 |
| `.todo-orchestrator/ce-ptr-plan.json` | 0.06 MiB | 1 |
| `.todo-orchestrator/runtime` | 74.47 MiB | 0 |
| `.git/todo-orchestrator` | 876.83 MiB | 0 |

The verified archive is `/mnt/block/cellerator/preservation/legacy-todo-20261029T132818Z/legacy-todo.tar.zst` (86,342,582 bytes). Its SHA-256 is `9264dc1f1faaaa7acfb31d7e9190e550ec809b5baef57bd34cde20a374d55aa9`. The [manifest](manifest.json) records every formerly tracked file's original path, Git blob and content hash. The private durable manifest records all 16,333 filesystem entries, including ignored runtime data and the old worktree's Git administration files. Archive verification checked file contents, modes and link targets before retirement.

## Current authority remains in place

- `.todo-orchestrator/authority.sqlite3`
- `.todo-orchestrator/authority.sqlite3-wal` and `authority.sqlite3-shm`, whenever present
- `.todo-orchestrator/project.json`

These files were excluded from the archive and kept at their original paths. Their before/after content fingerprints and identities were unchanged. No current database, project metadata, native lease, acceptance or completion state was edited. Current workspaces under `~/.local/state/project-control/` were not moved.

## Git worktree preservation

The old worktree under `.git/todo-orchestrator/` was clean and was removed through `git worktree remove` without force. Its branch `codex/ce-is1-integrator-l` at `41952561a1fa9b2aab7d6d563fae2e2006b0850e` is retained. Source bytes and the original Git administration files are archived. The branch is also preserved in the earlier full recovery bundle under `/mnt/block/cellerator/preservation/cleanup-final-20261029T113734Z/`.

One retired legacy directory was read-only. Its original mode is recorded in the archive; owner write permission was added only to that retired directory to complete removal. This did not alter legacy database contents.

## Recover historical files

Verify the archive checksum, then extract into an **empty staging directory** and compare contents with the durable manifest. Restore individual old documents deliberately. Do not extract `.git` snapshots or old authority files over a live checkout or the current database. Reconstruct a worktree through Git from its retained branch before deliberately overlaying any archived source. External symlink targets were recorded without following or copying them.

Historical plans, reports and source-bound records may still name the old paths. Resolve those paths with this index or read their original Git revision; do not replay an old plan, regenerate old projections, or restore an old authority to make a current task appear complete.
