# Indexed RDS reads

`Cellerator::rds` streams whole-cell batches from a local gzip-compressed
R XDR v3 `Matrix::dgCMatrix`. Source columns are cells and source rows are
features for the embryo fixtures. This reader accepts a single gzip member
with ordinary serialized slots, optional character dimension names, and an
empty `factors` list. The returned Cellerator CSC view preserves source
orientation, double values, and 32-bit row indices and offsets. Dimension
names can be read separately.

## Build and open

Use Clang with its matching `clang-scan-deps` and configure with
`-DCELLERATOR_ENABLE_RDS=ON`. The reader uses zlib. The focused fixture test
uses R and Matrix to generate an independent input.

Installed callers request `find_package(Cellerator CONFIG REQUIRED COMPONENTS rds)`,
link `Cellerator::rds`, and use `import cellerator.rds;`.

Call `build_index(source, index_path, stats)` once for each physical source
file. The call reads the source sequentially with bounded working storage and
publishes a complete sidecar at a new path. `RdsFile::open(source, index_path)`
validates the sidecar in fixed-size chunks, then maps it for on-demand access.
The open reads the full sidecar and checks the source identity. Keep both files
immutable while a session is open. Rebuild to a new index path after replacing
or copying the source.

The three-argument `build_index` uses an 8 MiB checkpoint spacing. For small,
widely scattered reads, the overload taking `checkpoint_spacing_bytes` can
trade sidecar size and open time for shorter seeks. The spacing is the minimum
distance between checkpoints. The reader can only place them at eligible
DEFLATE block boundaries, so spacing does not cap the bytes a read may decode.

## Stream into supplied storage

1. Pass a list of `ColumnRange` values to `RdsFile::prepare_read`. A
   `PreparedRead` borrows the file and range list. Keep the list alive and
   unchanged across reads. Output columns retain the requested order and repeats.
   For a shuffled or overlapping batch, use the overload with caller-supplied
   `ReadRun` storage. Supply one slot per input range, then keep the plan storage
   alive, unchanged, and disjoint from the output arrays. The input list can be
   reused after preparation. The reader decodes in source order and copies
   overlapping results from columns already written in the same batch.
2. Use its `requirements()` to provision host buffers for double values,
   `int32_t` row indices, and `int32_t` column offsets.
3. Each worker calls `ReadContext::prepare_transfer(file, TransferMode::numeric)`
   for matrix reads or `prepare_transfer(file)` to also provision any names.
   Then call `read_into` for one batch at a time. Independent worker contexts and
   disjoint destination buffers allow parallel reads. Drain all workers before
   closing or reopening the file. Prepare requests again after reopening.
   A context can attach to a new open session while retaining decoder storage.
   Keep output buffers alive through their use. A numeric-only context can later
   provision names explicitly. Switching it back to numeric retains the
   allocated capacity for reuse.

Capacity, placement, and overlap checks run before writing the numeric
destination. After a decoding or source error, a destination can be partially
written. Consume a result only after `Status::ok`.

`ReadContext::read_names_into` reads one range from the feature or cell axis.
Pass empty descriptors to size the byte buffer, then supply an `AxisName` array
and the byte buffer. Names borrow the supplied bytes. Null text represents an R
missing string. An empty string has a nonnull pointer and size zero. The byte
requirement includes a terminator for each present string. Short byte capacity
returns the required size and can leave partial names and bytes. The reader
exposes the saved bytes and R encoding mark. The native encoding mark denotes
the source R session's encoding.

See the [storage comparison harness](bench/README.md) for the embryo
qualification workflow.
