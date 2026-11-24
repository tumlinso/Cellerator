# Native AnnData buffer reads

The optional `Cellerator::anndata` component exposes the existing AnnData++
API and reads selected rank-two backed arrays into Cellerator host buffers.
Use the AnnData++ API to open H5AD or Zarr, inspect annotations, and choose a
selection. Standalone consumers can use `import anndatapp;` and link
`anndatapp::anndatapp`.

## Build and import

Initialize the `packages/anndatapp` submodule for the standard native build.
Set `CELLERATOR_ENABLE_ANNDATA=OFF` to omit the component. To build it
independently of CCell, set `CELLERATOR_BUILD_CCELL=OFF` and
`CELLERATOR_ENABLE_ANNDATA=ON`. AnnData++ controls its individual backends
through `ANNDATAPP_ENABLE_HDF5` and `ANNDATAPP_ENABLE_ZARR`.

```cmake
find_package(Cellerator CONFIG REQUIRED COMPONENTS anndata)
target_link_libraries(my_app PRIVATE Cellerator::anndata)
set_target_properties(my_app PROPERTIES CXX_EXTENSIONS OFF)
```

## Read into supplied storage

Prepare a selection once, then reuse the same or compatible output storage.
For a dense destination, supply a typed matrix view and the backing allocation
record. The record bounds the addressed byte span even with negative strides.
The sparse overload accepts three typed array views for values, minor indices,
and major offsets, and returns a compressed view pointing at those buffers.
Use the stored dtype and both stored index widths. A later I/O failure can leave
caller-provided buffers partially written.

```cpp
#include <Cellerator/memory/allocation.hh>
#include <Cellerator/memory/view.hh>

import cellerator.anndata;

void read_counts(const char* path, cellerator::memory::matrix_view<float> dst,
                 const cellerator::memory::allocation& backing) {
    auto file = anndatapp::open_h5ad(path);
    auto selected = file.slice(anndatapp::indices{3, 1}).x().prepare_read();
    cellerator::anndata::read_into(selected, dst, backing);
}
```

`read_into` uses stack descriptors and transfers into the supplied storage.
AnnData++ and its backends can still allocate metadata and use scratch for
decoding or gathering. The overload taking `BackedArray` prepares the
selection each time. Retain a `PreparedRead` for repeated I/O.

## Allocate explicitly

`cellerator::anndata::read_allocated(selection)` sizes and allocates a single
aligned host block for a nonempty dense payload or all three sparse arrays.
Keep the returned move-only owner alive while using its checked
`dense<T>()` or `compressed<I, T, O>()` view. The materialized storage
survives closure of the source file.
