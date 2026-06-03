# Cellerator

Cellerator is the umbrella workspace for the core CellStack genomics GPU
libraries. It is replacing the old `CellStack` coordination checkout as the
primary repository.

## Layout

```text
Cellerator/
+-- components/
|   +-- Baseplane/
|   `-- CellShard/
+-- include/Cellerator/
+-- src/foundation/
`-- Archive/current-cellerator/
```

- `components/Baseplane/` owns compact DNA/RNA sequence bit primitives and
  backend-aware CPU/CUDA kernels.
- `components/CellShard/` owns biology-centric storage, metadata, pack
  generation, and runtime delivery.
- `include/Cellerator/` and `src/foundation/` own only the foundation pieces
  currently needed by the components: core sparse layout ABI, CUDA runtime
  substrate, distributed runtime declarations, and matrix conversion helpers.
- `Archive/current-cellerator/` contains the previous Cellerator implementation
  as source-only archive material for later reclassification into new
  subprojects.

The root Python package is intentionally disabled in this first umbrella cut.
The old preprocessing Python package is archived with the old implementation.

## Build

Configure and build from the repository root:

```bash
cmake -S . -B build
cmake --build build -j 4
```

Useful focused targets:

```bash
cmake --build build --target baseplaneDna2Test cellShardInspectPackageTest celleratorFoundationRuntimeTest -j 4
```

The old implementation is not part of the default build graph. To expose the
archive as source-only CMake metadata:

```bash
cmake -S . -B build-archive -DCELLERATOR_ENABLE_ARCHIVE=ON
```

## CellStack Deprecation

The old `CellStack` repository should now be treated as a temporary wrapper that
contains only this `Cellerator/` checkout. New cross-project coordination,
component layout, and helper scripts belong here.
