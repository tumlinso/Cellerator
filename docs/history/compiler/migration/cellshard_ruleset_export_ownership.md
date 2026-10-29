# Cellerator ruleset export consumed by CellShard

Todo `CE-CCP1-A03-012` reserves a narrow, immutable Part Two-facing export. It
contains only schema/size, compiled ruleset identity, representative profile
identity, exact coverage identity, realization-requirement identity, and
structure generation. It is pointer-free and trivially copyable.

CellShard may later persist, place, stage, deliver and bind concrete data under
this export. It may not reinterpret the ruleset, substitute approximate
coverage, infer profile identity, or select a different realization. Concrete
paths, object keys, device ordinals, leases, routes and resident addresses are
separate application bindings and never part of the export.

This Todo intentionally does not implement deep CellShard materialization,
runtime orchestration or Part Two JIT. Existing CPE2 opaque-artifact behavior
and JBC portable schedule artifacts are compatibility evidence. The public
header includes only standard C++ headers, so standalone Cellerator has no
required CellShard compile or link dependency.

Gate `CE-CCP1-A03-012-GATE` compiles and links a standalone consumer using only
the new header, validates every identity/generation, and scans out CellShard
includes and concrete runtime/storage vocabulary.
