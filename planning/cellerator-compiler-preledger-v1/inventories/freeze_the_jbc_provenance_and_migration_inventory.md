# Frozen JBC provenance and migration inventory v1

This receipt publishes `CE-CCP1-I02-JBC-MIGRATION-MANIFEST` version 1 for
consumption by `CE-CCP1-A03-014` and `CE-CCP1-E02-018`. It freezes source
identity and migration intent; it does not move code, advance the CellShard
gitlink, or authorize work outside a future Todo's scope.

## Frozen source

- Repository: `git@github.com:tumlinso/CellShard.git`
- Branch: `main`
- Commit: `b9749ad3e5146a04f847533d8c6f1a54146aed20`
- Common JBC ancestry base: `7762a5925fe18b2ca45ab8a436f3461804ed2ad9`
- Source branches: 24 local `jbc/*` tips, all reachable from the frozen main
- Unique branch-union paths: 979
- Sorted source-path SHA-256:
  `af783b7c35be048289a8da5798e8b11c7895846f0d42d938dc6a235e73a5aee9`

Every source path exists at the frozen main commit. The inventory stores a
SHA-256 and byte count of that committed content, not of a mutable branch
worktree.

## Published machine artifacts

| Artifact | Schema | Rows | SHA-256 |
|---|---:|---:|---|
| `planning/cellerator-compiler-preledger-v1/inventories/jbc_source_migration.csv` | 12 columns | 979 plus header | `76f9c5fd93c25598fe4bc6d1bba7587868e1ea08da95d965219ec13e6c99ad5f` |
| `planning/cellerator-compiler-preledger-v1/inventories/jbc_source_migration.json` | version 1 | 979 | `39d55b70070c22239326bf28d44a7aa9a77ce50365379d519ca28a87c9083595` |

CSV columns are `source_repository`, `source_branch`, `source_commit`,
`source_path`, `sha256`, `bytes`, `disposition`, `proposed_target`,
`migration_task`, `required_gate`, `provenance_rule`, and `status`. JSON carries
the same path records plus the common source envelope. All records have status
`source-frozen`.

The checked-in A02-012 evidence test is also the deterministic generator:

```sh
freeze_the_jbc_provenance_and_migration_inventory_test \
  --generate <CellShard-root> \
  planning/cellerator-compiler-preledger-v1/inventories/jbc_source_migration.csv \
  planning/cellerator-compiler-preledger-v1/inventories/jbc_source_migration.json
```

Regeneration is permitted only when a new version intentionally selects a new
source commit. It must update both artifact hashes and cannot silently mutate
this frozen version.

## Lifecycle reconciliation

| Disposition | Rows | Required outcome |
|---|---:|---|
| preserve in place | 457 | CellShard storage/runtime/evidence/history remains reachable and keeps its concrete owner. |
| move | 220 | Rehome compiler-semantic source with source export and commit trailers. |
| adapt | 242 | Port tests/benchmarks and preserve expected results or explicit experimental disposition. |
| split | 52 | Separate compiler semantics from CellShard application/runtime or central integration. |
| wrap temporarily | 1 | Keep the narrow evidence adapter only through compatibility proof. |
| retain as compatibility | 4 | Keep frozen readers/adapters without parallel semantic authority. |
| retire after replacement proof | 3 | Remove only after canonical replacement, migrated evidence, and consumer proof. |
| **Total** | 979 | Unique source paths; no dangling or duplicate record. |

## Proposed destination and gate map

The machine rows carry the exact target for each path. This human aggregation
shows which Project Control task owns the application and which terminal gate
must validate it.

| Applying task or retained seam | Rows | Proposed Cellerator destination / retained boundary | Required migration gate |
|---|---:|---|---|
| `CE-CCP1-A02-012` | 322 | Frozen evidence/history remains CellShard-addressed in the inventory | `CE-CCP1-A02-012-GATE` |
| `CE-CCP1-A03-014` | 2 | Ownership/umbrella/build split assigned by the architecture integration task | `CE-CCP1-A03-014-GATE` |
| `CE-CCP1-E02-002` | 17 | `compiler/discovery/evidence/` | `CE-CCP1-E02-018-GATE` |
| `CE-CCP1-E02-003` | 9 | support-signature discovery | `CE-CCP1-E02-018-GATE` |
| `CE-CCP1-E02-004` | 16 | co-support and overlap discovery | `CE-CCP1-E02-018-GATE` |
| `CE-CCP1-E02-005` | 14 | motif and operation-trace discovery | `CE-CCP1-E02-018-GATE` |
| `CE-CCP1-E02-006` | 12 | trajectory/lineage discovery | `CE-CCP1-E02-018-GATE` |
| `CE-CCP1-E02-007` | 16 | multimodal and sequence-compatible discovery | `CE-CCP1-E02-018-GATE` |
| `CE-CCP1-E02-008` | 14 | factor/topic and bicluster discovery | `CE-CCP1-E02-018-GATE` |
| `CE-CCP1-E02-009` | 16 | exact certification | `CE-CCP1-E02-018-GATE` |
| `CE-CCP1-E02-010` | 19 | discovery atom envelope | `CE-CCP1-E02-018-GATE` |
| `CE-CCP1-E02-015` | 5 | temporary CellShard compatibility adapters | `CE-CCP1-E02-018-GATE` |
| `CE-CCP1-E02-016` | 139 | discovery/atom/certification tests and non-runtime evidence | `CE-CCP1-E02-018-GATE` |
| `CE-CCP1-E03-001` | 23 | composition productions | `CE-CCP1-E03-018-GATE` |
| `CE-CCP1-E03-002` | 2 | canonical derivation DAG | `CE-CCP1-E03-018-GATE` |
| `CE-CCP1-E03-003` | 9 | explicit grammar | `CE-CCP1-E03-018-GATE` |
| `CE-CCP1-E03-004` | 10 | experimental induced grammar | `CE-CCP1-E03-018-GATE` |
| `CE-CCP1-E03-006` | 17 | basis semantics | `CE-CCP1-E03-018-GATE` |
| `CE-CCP1-E03-008` | 7 | superatom promotion evidence | `CE-CCP1-E03-018-GATE` |
| `CE-CCP1-E03-009` | 17 | planning partial semantics | `CE-CCP1-E03-018-GATE` |
| `CE-CCP1-E03-010` | 15 | program/global graph | `CE-CCP1-E03-018-GATE` |
| `CE-CCP1-E03-012` | 4 | portable semantic schedule/ruleset | `CE-CCP1-E03-018-GATE` |
| `CE-CCP1-E03-013` | 135 | retained CellShard atom-store/materialization/runtime seam | `CE-CCP1-E03-018-GATE` |
| `CE-CCP1-E03-014` | 103 | composition/basis/graph/schedule tests and benchmarks | `CE-CCP1-E03-018-GATE` |
| `CE-CCP1-E03-015` | 36 | split semantic differential and CellShard application validation | `CE-CCP1-E03-018-GATE` |

These row counts total 979. A future applying task must select only its assigned
rows, refresh Project Control scope, preserve the A02-009 source trailers, and
satisfy the listed terminal gate. A target beginning `CellShard:` is an
intentional retained owner, not a missing Cellerator path; a target beginning
`Cellerator+CellShard:` is an explicit split to resolve in the named task.

## Freeze validation

The focused gate performs all of the following against live immutable Git
objects and the checked-in artifacts:

- reconstructs the 24-branch union and requires 979 unique paths;
- requires every row's committed source to exist at frozen main;
- recomputes all 979 SHA-256 values and byte counts;
- rejects duplicate paths, unknown dispositions, empty destinations, missing
  applying Todos/gates, and provenance-rule mismatches;
- reconciles all seven disposition counts;
- requires 979 JSON records and the exact source commit; and
- recomputes the CSV and JSON hashes recorded above.

The interface is frozen only with those checks passing and the resulting commit
pushed. Later evolution publishes an adjacent manifest version rather than
rewriting this source envelope.
