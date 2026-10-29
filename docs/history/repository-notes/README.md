# Archived repository notes and root guides

The files listed here were removed from the repository root on 2026-10-29 and preserved byte-for-byte. Their former paths and SHA-256 values are recorded below. Historical task records and snapshots may still name the old root paths; use this index to resolve them.

These notes preserve the checkout context and links that existed when they were written. Relative links inside the archived files retain their original text and should be interpreted relative to the former repository root. The source-context links below point to maintained current guidance.

| Former root path | Archived path | Bytes | SHA-256 |
|---|---|---:|---|
| `ARCHITECTURE_FOLLOWUPS.md` | `docs/history/repository-notes/ARCHITECTURE_FOLLOWUPS.md` | 2697 | `c86fb4391e9858b91da52fdb68011f17f5e8e5cd90695d9416d0fdc9cd56d0ef` |
| `ARCHITECTURE_INVESTIGATION.md` | `docs/history/repository-notes/ARCHITECTURE_INVESTIGATION.md` | 23112 | `1003e0ac98e7196b2fb1b814cb79f8ffccde97646759a02ba609eda581dba63d` |
| `optimization.md` | `docs/history/repository-notes/optimization.md` | 48179 | `c4542b0e0a9cc8bb8ec00bfa2de0309e22c7d2e8fc6206730933b1f1b141174c` |
| `out_of_scope_inventory.md` | `docs/history/repository-notes/out_of_scope_inventory.md` | 4607 | `7c60edc8102e176b5f1b65b38585e99975d2f8871a334c83eae268bba79c68e0` |
| `planning_strategy.md` | `docs/history/repository-notes/planning_strategy.md` | 7551 | `98231e44408e93bedac2efbd3bd08e5812b5aac5536ccca5f59d657e49818487` |
| `ENVIRONMENT_NOTES.md` | `docs/history/repository-notes/ENVIRONMENT_NOTES.md` | 3876 | `2011fefbfce8dba34df176acb2c9d01aff82fa6b5eb30c96d834044fd374cff4` |
| `custom_torch_ops.md` | `docs/history/repository-notes/custom_torch_ops.md` | 434 | `00e1d53420cd45693d1196bd8f86a7cede750aa4897925bcfe15a9ae5d30ba4d` |
| `style_hint.md` | `docs/history/repository-notes/style_hint.md` | 374 | `1faecae9a19c6ddbee958ffa517c5a70841ecc99193905c84633c630e3dc6167` |

## Historical migration assertion

`tests/architecture/test_migration_status.py` was unregistered and asserted obsolete migration-status text that is no longer present in the maintained current-implementation page. Its original bytes are retained at `docs/history/repository-notes/tests/test_migration_status.py.txt` (1710 bytes; SHA-256 recorded below). Do not treat this archived assertion as a current test contract.

| Former path | Archived path | Bytes | SHA-256 |
|---|---|---:|---|
| `tests/architecture/test_migration_status.py` | `docs/history/repository-notes/tests/test_migration_status.py.txt` | 1710 | `de42a69c915f3dcda2c35e1ddeeef2bd91e2cdbce124ee0d71a460855813d742` |

## Current source context

### `ARCHITECTURE_FOLLOWUPS.md`
- [architecture.qmd](../../../docs/architecture.qmd)
- [current_implementation.qmd](../../../docs/current_implementation.qmd)
- [routing.md](../../../docs/development/routing.md)

### `ARCHITECTURE_INVESTIGATION.md`
- [architecture.qmd](../../../docs/architecture.qmd)
- [model_building.qmd](../../../docs/model_building.qmd)
- [experimental-fields.md](../../../docs/development/experimental-fields.md)

### `optimization.md`
- [layout_execution.qmd](../../../docs/layout_execution.qmd)
- [operations.qmd](../../../docs/operations.qmd)
- [current_implementation.qmd](../../../docs/current_implementation.qmd)
- [developer_reference.qmd](../../../docs/developer_reference.qmd)

### `out_of_scope_inventory.md`
- [scope.qmd](../../../docs/scope.qmd)
- [routing.md](../../../docs/development/routing.md)

### `planning_strategy.md`
- [architecture.qmd](../../../docs/architecture.qmd)
- [model_building.qmd](../../../docs/model_building.qmd)
- [layout_execution.qmd](../../../docs/layout_execution.qmd)

### `ENVIRONMENT_NOTES.md`
- [developer_reference.qmd](../../../docs/developer_reference.qmd)
- [architecture.qmd](../../../docs/architecture.qmd)

### `custom_torch_ops.md`
- [operation_registry.md](../../../docs/development/operation_registry.md)

### `style_hint.md`
- [implementation_style.md](../../../docs/development/implementation_style.md)
- [routing.md](../../../docs/development/routing.md)

The two former root compatibility guides, `custom_torch_ops.md` and `style_hint.md`, are archived with their original bytes. Their links above resolve the maintained operation registry and implementation style guide.

Historical test source uses a `.py.txt` suffix so ordinary pytest discovery does not collect obsolete completion assertions. The original source bytes are preserved.
