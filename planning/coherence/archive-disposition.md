# Archive and cleanup disposition

This record separates completed archival work from items that remain in place
because their owners or consumers have not been reconciled. It supplements the
approved plan; Project Control remains the live status authority.

## Completed branch archive

The branch audit and later closure are separate records:

- [`branches.json`](branches.json) and [`live-branches.tsv`](live-branches.tsv)
  capture the source comparison at `main` commit
  `2f851ddb9719a85064f7140864c614245424f4cc`.
- [`branch-closure.json`](branch-closure.json) is the exact closure receipt.
  It records 23 ordinary working refs deleted remotely after exact-tip
  verification, their retained local archive tags, and surviving remote refs.
- The bundle remains at
  `/mnt/block/cellerator/preservation/20261028T204231Z/branch-reconciliation/audited-working-refs.bundle`
  with SHA-256
  `f5b85451f05133290161a6c283c535b7b484dd78a338fe01ecb533b784f8bd2f`.

The two backup refs, archive refs, `main`, managed IS1 worktree/branch, and Todo
integration ref were retained. The IS1 worktree still needs its supported
workflow closeout; ancestry alone does not satisfy that obligation.

## Root plan snapshots retained pending path reconciliation

These tracked root snapshots are completed historical workflow inputs, not the
current approved Cellerator plan:

| Path | Historical ownership record | Disposition |
|---|---|---|
| `ce-live-plan.json` | `todos/ce-live-01.md` | Keep at its recorded path until historical path compatibility is reconciled through the supported workflow. |
| `cellerator-remap-plan.json` | `todos/ce-remap-01.md` | Keep at its recorded path until historical path compatibility is reconciled through the supported workflow. |
| `cellerator-post-remap-plan.json` | `todos/ce-post-remap-01.md`, `todos/ce-post-remap-04.md` | Keep at its recorded path until historical path compatibility is reconciled through the supported workflow. |

Their managed task records mark the work closed, but also retain these paths as
exclusive historical inputs. No non-Todo source consumer was found. Moving or
renaming them now would invalidate recorded path identity without improving the
current plan's discoverability. `planning/coherence/README.md` is the current
approved plan; these older snapshots remain preserved history.

The broader source-map inventory contains many other historical plans and
experiment artifacts. They stay with their initiative until implementation,
evidence, and cross-project consumers are reconciled. In particular, linked
experiment source was extracted only as part of a complete build/consumer unit;
unfinished unique operations remain available for later qualification.

## Payload and workflow holds

`external-payloads.json` records ignored datasets and Git metadata symlinks from
the verified preservation snapshot. The source archive preserves symlink bytes
and paths; external target contents were not relocated. One dataset symlink was
already dangling during backup review. These files remain untouched pending a
separate owner and storage decision.

The supported Project Control workflow still has a pending preservation-task
handback caused by an expired claim. No authority state was edited to make it
appear complete. Its source and lifecycle evidence remain with the registered
workflow repair records and private preservation archive.
