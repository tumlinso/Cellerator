# CellPack compatibility includes

This directory is a temporary source-compatibility surface. Headers here only
forward to canonical declarations under `Cellerator/geometry/`.

New Cellerator code must include `Cellerator/geometry/...` directly. Existing
external consumers may use these paths until their owning migration phase cuts
them over; no implementation or new API belongs here.
