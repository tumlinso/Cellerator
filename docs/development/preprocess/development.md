# Development

Build from the Cellerator root with `CELLERATOR_ENABLE_PREPROCESS=ON`.
Enable `CELLERATOR_BUILD_TESTS` for native checks and examples. See the
[single-cell methods guide](../single-cell-methods.md)
for build and server validation settings.

The numerical module accepts native Cellerator views. Consumers keep input
storage alive, provision output and workspace, and control parallelism.
Checked calls report status and required capacity. Trusted calls use the
documented preconditions. The reference allocates no memory during sizing,
validation, preparation, or execution.

The native qualification drivers use `CELLERATOR_BUILD_METHOD_QUALIFICATION`.
The Scanpy comparison driver also needs the optional AnnData component.
The historical [CCell source archive](https://github.com/tumlinso/CCell/tree/codex/cellerator-native)
records the original qualification scripts and their source context.

## Retired duplicate source

`components/preprocess` is the sole preprocessing implementation. The removed
`src/preprocess` tree is preserved in Git at commit
[`15281fd666a5270c6896b9c15cdb108908e7544b`](https://github.com/tumlinso/Cellerator/tree/15281fd666a5270c6896b9c15cdb108908e7544b/src/preprocess),
with tree object `9cd7c3e125b412a53f056d4a320fcb651fa679a6`.

| Removed source | Canonical component source |
| --- | --- |
| `src/preprocess/qc.cpp` | [QC, filtering and statistics](../../../components/preprocess/src/qc.cpp) |
| `src/preprocess/qc.cppm` | [QC interface](../../../components/preprocess/src/qc.cppm) |
| `src/preprocess/preprocess.cpp` | [Normalization and transforms](../../../components/preprocess/src/preprocess.cpp) |
| `src/preprocess/preprocess.cppm` | [Primary module interface](../../../components/preprocess/src/cellerator.preprocess.cppm) |
| `src/preprocess/hvg.cpp` | [HVG accumulation and scoring](../../../components/preprocess/src/hvg.cpp) |
| `src/preprocess/hvg.cppm` | [HVG interface](../../../components/preprocess/src/hvg.cppm) |

The old CMake definition and its local `AGENTS.md` and `STYLE.md` were retired
with the duplicate implementation. The supported module and target are
`cellerator.preprocess` and `Cellerator::preprocess`. The old `preprocess`,
`preprocess.qc`, and `preprocess.hvg` modules have no compatibility facades.

The component retains the old QC, mergeable statistics, normalization,
transformation, and staged Seurat and Cell Ranger HVG algorithms. It accepts
native Cellerator views and validates signed strides and sparse index types at
that boundary. The old raw dense HVG API also accepted arbitrary byte strides;
no in-tree consumer required that representation. File acquisition and any
format conversion belong at the caller's adapter boundary.

Historical Scanpy, runtime, and build reports remain unchanged and identify
their original source snapshots. The Python qualification helper still uses
the former source and test layout; it requires a separate tooling update before
it can record provenance for the current component.
