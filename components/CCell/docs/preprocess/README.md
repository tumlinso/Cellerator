# ccell preprocessing

The preprocessing methods under `src/preprocess/` provide direct C++20
numerical operations over native Cellerator views.
The standard implementation makes traversal explicit and performs no
allocation. Callers acquire storage, supply outputs and scratch, and choose
threading and synchronization. See the [numerical contract](operations.md).

Use `import ccell.preprocess;` and call methods in `ccell::preprocess`.
For AnnData++ input, acquire storage through `import anndata;` and bind it with
the optional `ccell.interop.anndata` adapter. The input owner stays with the
caller.
The API exposes reference operations and permits replacement of individual
operations or larger regions with caller defined algorithms and layouts.

The scope includes quality control and filters, feature statistics, total
normalization, log transforms, scaling, and Seurat or Cell Ranger highly
variable gene calculations. The CPU method target builds with the data adapter
disabled. Scanpy 1.11.5 supplies independent comparison cases for optional
qualification.

## Build and validate

Build and install preprocessing through the Cellerator root CMake project.
Follow the [CCell build instructions](../../README.md). The root build provides
the `ccell::preprocess`
target and optional native tests, example, Scanpy comparison driver, and
runtime benchmark.

The historical Scanpy report records 302 cases for an earlier source snapshot.
It is preserved as history and does not qualify the relocated or current code.
See [development](development.md) for current commands and
[validation](validation.md) for retained fixtures and evidence.

The original code is MIT licensed. See the ccell root [LICENSE](../../LICENSE)
and [third-party notices](../../THIRD_PARTY_NOTICES.md).
