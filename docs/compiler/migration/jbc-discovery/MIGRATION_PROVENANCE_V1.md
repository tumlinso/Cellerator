# JBC discovery migration provenance v1

This cold manifest is the source-of-truth provenance companion for the E02
discovery migration. It does not enter hot Planning IR or runtime records.

All entries were migrated from `tumlinso/CellShard` commit
`b9749ad3e5146a04f847533d8c6f1a54146aed20`. The exact destination-to-source
and legacy-Todo mapping is compiled in
`preserve_migration_provenance_in_source_and_artifacts_v1.hh` and checked by
`CE-CCP1-E02-014-GATE` for every E02-001 through E02-013 header, source, and
test file.

Commits that materially alter a migrated mechanism should retain these cold
fields in their commit message or review receipt:

```text
Migrated-From-Repository: tumlinso/CellShard
Migrated-From-Commit: b9749ad3e5146a04f847533d8c6f1a54146aed20
Migrated-From-Path: <manifest path>
Migrated-From-Todo: <manifest Todo set>
```

The frozen source families are `compiler/atom`, `compiler/evidence`,
`compiler/certification`, and the `compiler/discovery/{support_signature,
co_support,overlap,motif,operation_trace,trajectory,multimodal,factor_topic,
bicluster}` provider trees. Cellerator owns the migrated compiler behavior;
CellShard remains the recorded historical source and a runtime/storage
consumer.
