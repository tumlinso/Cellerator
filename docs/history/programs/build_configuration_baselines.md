# Standalone and embedded build baselines

## Authority and environment

This is the `CE-JBC-B04` build observation. The Cellerator Git cursor was
`3559d0fe18401b76e6f674e185c42fbaf554b6f9`; its committed CellShard gitlink
was `5f6a502b4355732c4ed3cc873a25b8aec66d8338`. The registered CellShard source
checkout used by the final embedded reconfiguration was
`96a691e4a271fabd738ff5819eef6349ac3621a0`. The separately observed
Cellerator Todo cursor was revision `3610`.

The observations are not globally atomic and the cursors are intentionally not
normalized. Both configurations used:

- CMake `3.28.3`;
- GNU C++ `13.3.0` through `/usr/bin/c++` with nvcc host compatibility set by
  the project to `/usr/bin/g++-12`;
- NVIDIA CUDA compiler `12.9.86` from the HPC SDK;
- Release mode, detected `sm_70`, and `-j 80` from `nproc`;
- Tesla V100-SXM2-16GB, 16144 MiB reported by the configure-time probe.

No implementation, CMake input, wire format, runtime setting, or source-owned
generated file was changed to obtain either result. Build products and captured
graphs are in fresh `/tmp` directories.

## Standalone baseline

Configuration directory:
`/tmp/cellerator-jbc-b04-off.PqoLrU`.

```bash
cmake -S . -B /tmp/cellerator-jbc-b04-off.PqoLrU \
  -DCELLERATOR_ENABLE_CELLSHARD=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  --graphviz=/tmp/cellerator-jbc-b04-off.PqoLrU/targets.dot
cmake --build /tmp/cellerator-jbc-b04-off.PqoLrU \
  --target cellerator_runtime cellerator_operation_core_v2_test \
  -j "$(nproc)"
/tmp/cellerator-jbc-b04-off.PqoLrU/cellerator_operation_core_v2_test
```

Result: configure, both builds, and the focused operation-core v2 executable
passed with exit status zero. The build produced
`libcellerator_runtime.a` without CellShard configuration or linkage.

## Embedded baseline

Configuration directory:
`/tmp/cellerator-jbc-b04-on.I8y8Ii`.

```bash
cmake -S . -B /tmp/cellerator-jbc-b04-on.I8y8Ii \
  -DCELLERATOR_ENABLE_CELLSHARD=ON \
  -DCMAKE_BUILD_TYPE=Release \
  --graphviz=/tmp/cellerator-jbc-b04-on.I8y8Ii/targets.dot
cmake --build /tmp/cellerator-jbc-b04-on.I8y8Ii \
  --target cellerator_runtime cellshard_runtime \
           cellerator_operation_core_v2_test \
  -j "$(nproc)"
/tmp/cellerator-jbc-b04-on.I8y8Ii/cellerator_operation_core_v2_test
cmake --build /tmp/cellerator-jbc-b04-on.I8y8Ii \
  --target cellshardAccessAdapterCompileTest -j "$(nproc)"
/tmp/cellerator-jbc-b04-on.I8y8Ii/cellshardAccessAdapterCompileTest
```

Result: configure, the canonical Cellerator runtime, registered CellShard
runtime, operation-core v2 smoke, and CellShard access-adapter compile/run smoke
all passed with exit status zero. The embedded build adds CellShard targets; it
does not replace or mutate the canonical Cellerator runtime target.

The embedded configuration was first built at the preceding clean source
cursor `6ab8932704ac5988ac64853b3cf43e41e991ee98`. After the documentation-only
`CS-JBC-B02` commit advanced the registered checkout, the same configure,
targets, and executables were rerun at `96a691e4a271fabd738ff5819eef6349ac3621a0`.
All targets were current or rebuilt successfully and both executables again
returned zero; the nested worktree was clean.

## Canonical runtime target proof

The direct target-graph edges emitted for `cellerator_runtime
(Cellerator::runtime)` are identical in both configurations:

```text
cellerator_runtime -> CUDA::cublas
cellerator_runtime -> CUDA::cudart
cellerator_runtime -> CUDA::cusparse
cellerator_runtime -> cellerator_cuda_mode
```

The normalized direct-edge manifests have the same SHA-256 digest:

```text
975e4409711103d83784cd058c63b11e22f380ceac92235139b76e6db00f49f9
```

Sorted externally defined symbol manifests from the two independently built
`libcellerator_runtime.a` archives also compare byte-for-byte and have the same
SHA-256 digest:

```text
9cfb4288a82974f08fdad14507cf53c79eb429e41258d715b3314adbe8371700
```

Therefore enabling the privileged CellShard component expands the combined
target graph but leaves the canonical `Cellerator::runtime` dependency surface
and exported implementation symbols unchanged. This baseline classifies the
native runtime as **preserve** and the access-adapter smoke as
**compatibility-only** pending its named JBC migration gate.

## Diagnostics and limits

The builds emitted NVIDIA's expected warning that offline compilation below
`sm_75` will be removed in a future release. The embedded dependency build also
emitted an existing cuSPARSE deprecation warning for
`cusparseCreateIdentityPermutation`. Neither warning is a new failure or a
performance claim. No benchmark or profiler run was performed, no GPU execution
resource was leased, and these compile/smoke results do not promote any kernel
or reinterpret historical performance evidence.
