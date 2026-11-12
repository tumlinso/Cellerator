# Compute ownership

- `operation/` owns operation contracts, preparation, and the built-in catalog.
- `projection/` owns physical representations and explicit conversions.
- `candidate/` owns native, vendor, and evaluated hardware implementations.
- `operators/` owns reusable mathematical primitives.
- `training/` owns native forward/backward training primitives.

Numerical references and sparse operations use their canonical owner paths. Numerical
reference helpers live under `Cellerator/math/reference`; physical packing and
projection contracts live under `Cellerator/layout/projection`. The original
CP-Math v1 evidence is archived under `docs/history/compat`.
