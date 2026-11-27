# Native operation ownership

Operation implementations, derivatives, and their backend variants live under
`src/ops/` by mathematical contract. Public contracts follow the same ownership
under `include/Cellerator/`.

- Execution planning owns candidate catalogs, capability checks, and exact cost
  comparisons. Include conversion, preparation, epilogues, and synchronization
  in the measured cost.
- Execution preparation owns operand and projection binding. Prepared state
  retains its existing lifetimes, and launch bindings supply per-call inputs.
- Execution programs own staged submission, including training. Runtime sessions
  own streams, library handles, and device resources.
- Layout owns projection construction, operand packing, format conversion, and
  physical order. Geometry owns sampled support and grouping. Preserve logical
  identity when realizing a layout.
- Independent numerical references live under `components/reference/`.

See [the source layout guide](../../../docs/development/source-layout-cleanup.md)
and [the operation-core design](../../../docs/core_execution_cp_math.qmd). The
CP-Math v1 evidence remains in [the compatibility archive](../../../docs/history/compat/).
