# Deferred interop modules

The future host-only module interfaces are `cellerator.baseplane` and
`cellerator.cellshard`. Their underlying conventional contracts now have
canonical homes under `include/Cellerator/interop`.

These interop modules remain deferred. The separate optional
[`cellerator.nn`](../native/README.md) host surface has a local Clang/Ninja
dependency-scanning route; that does not qualify the interop boundaries above.
CUDA stays outside the named-module graph. `cellerator.glasshelix` is additionally
deferred because no frozen scientific contract exists.
