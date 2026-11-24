# CCell and Cellerator component ownership

CCell owns the native preprocessing and RNA velocity implementations under
[`components/CCell`](../../components/CCell/README.md). Cellerator supplies
their numerical and modeling primitives and view contracts. AnnData++ owns
arrays and file I/O. The optional Cellerator AnnData component reads selected
backed arrays into Cellerator buffers for models or numerical methods.

The standard build installs CCell and AnnData++ together. `find_package(ccell
CONFIG REQUIRED)` loads the method targets. Request the Cellerator `anndata`
component for native buffer reads. AnnData++ itself supplies
`import anndatapp;`. See the [AnnData buffer guide](anndata.md).

The former [AnnData++ and Velocity source archives](https://github.com/tumlinso/Cellerator/tree/d7414e7f830fcd25cac585399a0535b72e75ff01/quarantine)
remain available in Git history. Local cutover recovery files and the retired
Python AnnData source snapshot with its checksum manifest were backed up
outside the checkout.

The former `src/velocity` checkout was archived outside the repository with
its unpublished changes, Git metadata, and validation artifacts. CCell owns
direct binary16 input and caller-scheduled gene blocks.

## Supported substrate surfaces

The native core module, computational operands, generic feature bindings,
modeling, numerical operations, and selectable substrate SDK retain their
existing owners. Callers supply acquired numerical data and explicit identity,
layout, ownership, and execution contracts. File selection and scientific IO
belong to the consumer. Generic binary16 conversion remains in
`src/ops/numeric/binary16` and is independently buildable and installable as
`CelleratorBinary16`, with target `Cellerator::binary16`.

The Python AnnData adapter and its public API are retired. Generic NumPy-based
modeling and optional Torch, CUDA, and distributed adapters remain. Original
adapter, support-script, and affected test/example sources remain in the
backup described above.

## Retained inactive source

| Source | Current status |
|---|---|
| `components/CellShard` | Preserved in place; detached from supported Cellerator builds and package installs. No new CellShard development or IO transfer is included. |
| `include/Cellerator/interop/cellshard` | Preserved source only; excluded from native and SDK header installs. |
| `src/ops/velocity` and `tests/velocity` | Inactive native adapter and tests. The supported velocity method is owned by `components/CCell/src/velocity` and supplied by `ccell::velocity`. |
| `cmake/CelleratorVelocity.cmake` and `cmake/CelleratorVelocityModule.cmake` | Retained former integration definitions, no longer included or installed by supported builds. |
| `src/preprocess` | Dormant reference copy preserved in place; CCell owns the active preprocessing implementation. |
| `examples/torch` | Existing CellShard-oriented example sources retained without a supported Cellerator build option. |

Enabled retired velocity/CellShard options and required package components fail
explicitly. Their preserved sources can contain historical standalone build
files; those files do not re-establish supported Cellerator integration.

Source datasets, fixtures, recorded outputs, local histories, recovery trees,
and historical scientific receipts remain at their original paths. Historical
records retain their original dependency and qualification statements; they
are not evidence for the changed source. No dataset migration or deletion is
part of this ownership change.
