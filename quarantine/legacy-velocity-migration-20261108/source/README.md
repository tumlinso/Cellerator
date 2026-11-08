# velocity

`velocity` is Cellerator's native, portable C++20 reference path for RNA
velocity calculations. It is an independently buildable component with named
modules `velocity` and `velocity.precision`; Cellerator can call the same
reference operations while supplying its own scheduling and surrounding
execution policy.

The active implementation is CPU-only. It does not select an accelerator or
ISA backend. CUDA and AVX2/F16C sources and their historical support files are
preserved in the parent Cellerator quarantine and do not participate in this
library's build or package.

The reference accepts explicit numerical views for FP16, FP32, and FP64 input
storage. `NumericView` and `NumericEdit` distinguish borrowed memory from
retained owners. Ordinary rank-one and rank-two metadata is stored inline;
acquired numerical descriptors are plain views over already-validated
storage. Affine subsets use `subview(axis, start, count, step)`. Irregular gene
selection belongs to `Input::genes`, which retains the selected coordinates
for the run.

The caller owns block scheduling. A run validates its fixed inputs and
promotion availability once, performs initialization blocks, joins them,
resolves one global selection, performs refinement blocks, then joins again.
Workers use separate workspaces, callbacks, and providers and write only to
disjoint outputs and scratch. The reference has no scheduler or internal
locks. The serial workspace cap defaults to 64 MiB and includes worker
numerical and mask scratch plus requested sink-conversion and global-selection
scratch. Promotion has a separate 64 MiB default cap for workspace scratch
plus declared provider staging; caller-managed parallel runs account for
aggregate worker scratch and provider staging.

AnnData++ supplies independent array, axis, and dataset interfaces. Its
integer arrays are not used as FP16 storage. Half conversion uses the small,
CPU-only `Cellerator::binary16` helper; velocity does not require the full
Cellerator runtime.

## Build

The standalone component is currently supported with Clang 18, its matching
`clang-scan-deps`, CMake 3.28 or newer, and Ninja. Install AnnData++ and the
binary16 helper separately, then configure from this directory:

```sh
cmake -S . -B build/standalone -G Ninja \
  -DCMAKE_CXX_COMPILER=clang++-18 \
  -DCMAKE_PREFIX_PATH="/path/to/anndatapp/install;/path/to/binary16/install" \
  -DVELOCITY_BUILD_TESTS=OFF \
  -DVELOCITY_BUILD_EXAMPLES=OFF
cmake --build build/standalone
```

Current source and numerical qualification for the thin reference
implementation is recorded separately from earlier receipts in
[validation](docs/validation.md) and the [thin-reference receipt](tests/results/thin-reference.json).
Historical measurements and their qualification exceptions remain documented
there; they do not qualify the current caller-scheduled API.

## Documents

- [API and ownership](docs/api.md)
- [Architecture](docs/architecture.md)
- [Precision and input promotion](docs/precision.md)
- [Development and build support](docs/development.md)
- [Portable source boundary](docs/segmentation.md)
- [Validation history and current receipt](docs/validation.md)
- [Include, abstraction, and compactness review](docs/reference-review.md)
- [scVelo source foundation](docs/scvelo-foundation.md)
- [Coding style](STYLE.md) and [agent guidance](AGENTS.md)
