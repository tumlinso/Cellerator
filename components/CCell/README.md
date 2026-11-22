# CCell

CCell provides native C++20 single-cell preprocessing and RNA velocity over
Cellerator views. Preprocessing includes quality control, filtering, feature
statistics, normalization, log transforms, scaling, and highly variable gene
selection. Velocity provides CPU reference methods for steady-state fitting
and stochastic refinement.

## Build and consume

Configure from the Cellerator root with CMake 3.28 or newer, Ninja 1.11 or newer,
Clang 18 or newer, and matching `clang-scan-deps`. The standard build includes
CCell, the AnnData++ adapter, and the pinned AnnData++ source with native H5AD
and Zarr support. Install the system dependencies described by
[AnnData++](../../extern/anndatapp/docs/storage-dependencies.md).

```sh
git submodule update --init -- extern/anndatapp
cmake -S . -B build/standard -G Ninja \
  -DCMAKE_CXX_COMPILER=clang++-18 \
  -DCMAKE_CXX_COMPILER_CLANG_SCAN_DEPS=clang-scan-deps-18 \
  -DCELLERATOR_ENABLE_CUDA=OFF
cmake --build build/standard
ctest --test-dir build/standard --output-on-failure
cmake --install build/standard --prefix "$PWD/build/install"
```

For a new build directory, set `CELLERATOR_ENABLE_CCELL_ANNDATA=OFF` to build
both CCell methods without the data dependency. Set `CELLERATOR_BUILD_CCELL=OFF`
for the Cellerator core alone. `CCELL_BUILD_TESTS` defaults to the root test
setting. AnnData++ backend switches are `ANNDATAPP_ENABLE_HDF5` and
`ANNDATAPP_ENABLE_ZARR`.

Create a source archive from a committed revision with
`tools/maintenance/archive-cellerator-source.sh /absolute/output.tar.gz REVISION`.
Initialize the submodule first so the command can include the exact commit
recorded by that revision. An extracted archive can be configured directly.

| Import | Installed target | Purpose |
|---|---|---|
| `ccell` | `ccell::ccell` | Numerical method umbrella |
| `ccell.preprocess` | `ccell::preprocess` | Preprocessing methods under `ccell::preprocess` |
| `ccell.velocity` | `ccell::velocity` | Velocity methods under `ccell::velocity` |
| `anndata` | `anndatapp::anndatapp` | Upstream AnnData++ storage and file access |
| `ccell.interop.anndata` | `ccell::anndata_interop` | Checked borrowing helpers under `ccell::io` |

Installed applications can set `CMAKE_PREFIX_PATH` to the install prefix.
Use `find_package(ccell CONFIG REQUIRED)` for methods and request the
`anndata_interop` component when binding AnnData++ storage. Applications that
use AnnData++ directly can discover it with `find_package(anndatapp CONFIG
REQUIRED)`. Set C++ extensions off on module-consuming targets.

```cmake
find_package(ccell CONFIG REQUIRED COMPONENTS anndata_interop)
target_link_libraries(my_app PRIVATE ccell::preprocess ccell::anndata_interop)
set_target_properties(my_app PROPERTIES CXX_EXTENSIONS OFF)
```

## Native views and data access

Include the relevant Cellerator view header and import the method module.
Pass typed host views, plus explicit output and scratch storage. The dense
view at `Cellerator/memory/view.hh` stores element strides. Use a signed
`ptrdiff_t` stride type when a pointer at the first logical element addresses
reversed storage. The compressed view at `Cellerator/tensor/base/compressed.hh`
keeps CSR/CSC orientation explicit. Specify the offset type separately when
it differs from the index type.

A Cellerator view describes storage and layout. Maintain cell and feature IDs
and their correspondence to the view's axes at the caller boundary. Keep array
owners alive through every use of the view and any prepared object that
retains it. Use the same provisioned outputs and scratch on repeated calls
when the capacities suffice.

Data-facing code uses the upstream module and requests the adapter only when
borrowing views. The namespace alias is local to the translation unit.

```cpp
import anndata;
import ccell.interop.anndata;

namespace anndata = ::anndatapp;
```

Use `ccell::io::borrow_dense` for dense arrays and
`ccell::io::borrow_compressed` for CSR or CSC. Check the returned
`ccell::io::BorrowCode` before using the view. The adapter borrows the existing
pointers and preserves the cell and feature order. Acquire and retain the
source owner in the caller. Materialize non-addressable selections explicitly
before borrowing.

The [native example](examples/modeling.cpp) normalizes caller-owned counts and
passes the result to a model. The [AnnData++ test](tests/anndata_modeling.cpp)
uses the same model path with acquired H5AD and Zarr data and checks feature
order and borrowed pointer identity.

## Source and provenance

- `src/preprocess/` owns the [preprocessing methods](docs/preprocess/README.md).
- `src/velocity/` owns the [Velocity methods](src/velocity/README.md) as tracked
  CCell source.
- `src/ccell.interop.anndata.cppm` owns the optional checked borrowing adapter.
- `extern/anndatapp/` at the Cellerator root pins AnnData++ v1.1.0 at
  `c65af7de3c8e273bb0d3c2f5243774728fed452e`.

The initial tracked imports came from CCell
`62438b8b1b76b798009745a6d7c8a66370d3b859` and Velocity++
`570055b0f5ad8b491890ffcdf35aac74d3d541cb`. See the component [license](LICENSE)
and [third-party notices](THIRD_PARTY_NOTICES.md). Imported qualification reports
record their original sources and toolchains.
