# Compute ownership

- `operation/` owns operation contracts, preparation, and the built-in catalog.
- `projection/` owns physical representations and explicit conversions.
- `candidate/` owns native, vendor, and evaluated hardware implementations.
- `operators/` owns reusable mathematical primitives.
- `training/` owns native forward/backward training primitives.

The `math/` and `sparse/` paths are forwarding compatibility surfaces. CP-Math
v1 evidence is physically owned by `compat/cp_math_v1`.
