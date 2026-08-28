# Deferred interop modules

The future host-only module interfaces are `cellerator.baseplane` and
`cellerator.cellshard`. Their underlying conventional contracts now have
canonical homes under `include/Cellerator/interop`.

Native CMake module scanning remains deferred because this host has CMake 3.28
with Unix Makefiles and no `clang-scan-deps` or Ninja. No unbuilt `.ccm` facade
is introduced, and CUDA stays outside the module graph. `cellerator.glasshelix`
is additionally deferred because no frozen scientific contract exists.
