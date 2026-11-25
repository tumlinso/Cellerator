# Native RDS reader

`Cellerator::rds` reads a single matrix saved with R `saveRDS()`. It calls base R
`readRDS()` through libR, then exposes read-only Cellerator views over the R
storage. Use it with local files from a trusted producer.

Build Cellerator with `-DCELLERATOR_ENABLE_RDS=ON` and an R installation with
development headers and shared libR. CMake finds `R` on `PATH`. Set
`CELLERATOR_R_EXECUTABLE` to an absolute path to select another installation.
Installed consumers request `find_package(Cellerator CONFIG REQUIRED COMPONENTS rds)`
and link `Cellerator::rds`. A host that embeds R can also link `LibR::LibR` for
the embedding headers and library.

The host initializes R before reading. An embedding host sets `R_HOME` for
the linked R installation. On the R main thread, create a
`cellerator::rds::Matrix` from `<Cellerator/rds/matrix.hh>` and call `read(path)`.
On success, `storage()` identifies which view to use.

| R object | Native view |
| --- | --- |
| Plain integer matrix | `dense_integer()`, signed 32-bit values and column-major strides |
| Plain double matrix | `dense_real()`, FP64 values and column-major strides |
| `Matrix::dgCMatrix` | `csc_real()`, FP64 values and signed 32-bit row indices and column offsets |

The view preserves R axis order, values, and missing-value encodings. `row_name()`
and `column_name()` expose the dimension names and encodings restored by R.
The reader accepts ordinary in-memory storage. Package containers such as Seurat
and lazy ALTREP storage return `unsupported_matrix` when R restores them.

The move-only `Matrix` keeps the R object reachable. A successful replacement,
`reset()`, or destruction invalidates the previous views and names. A failed
read preserves them. Complete all uses before releasing the matrix on the R
main thread, and release it before R shuts down. Reset a prior matrix before
starting a new read if retaining both briefly would exceed the memory budget.

R materializes the object during `readRDS()`. The adapter adds no full payload
copy and validates CSC indices before publishing the view. A future gzip backend
could feed ordered bytes from rapidgzip into `R_InitInPStream` and `R_Unserialize`,
keeping R's deserializer on the R main thread. This could improve throughput
when gzip decompression limits the read. A gzip index addresses byte ranges.
Selective reads need an object-aware index or a backed format such as H5AD
or Zarr, which the AnnData component already reads.
