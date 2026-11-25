# CellShard server handoff

CellShard is detached from supported Cellerator builds and installs. The
tracked subtree contains the legacy formats and tools, sparse packing
experiments, and the distributed runtime and JBC experiments.

## Preserve the source and recover useful data

1. Archive `components/CellShard` on a branch of the
   [standalone CellShard repository](https://github.com/tumlinso/CellShard).
   The audited Cellerator revision is `17cb31cc946ccc2f54e892619435db484c2e96d1`
   and its CellShard tree is `8d227d570486336c5f3dc477451b628f2243e76e`.
   Verify the archived branch has the same tree. Record the matching
   Cellerator revision because the standalone build uses its headers. If the
   server has newer CellShard work, preserve it with its own source identity.
2. Inventory legacy files on the server. For `.csh5` datasets worth keeping,
   preserve the reader, CSR materializer, and a representative fixture. Use
   the existing `cellshardH5adExport` utility if useful. It uses Python for the
   final H5AD write. Check shapes, sparse values, ordered observation and
   feature IDs, assay row maps where applicable, and needed annotations in
   the recovered data. Keep representative `.cspack` fixtures and any packs
   that would be expensive to regenerate.
3. At cutover, remove the tracked subtree and retire the detached CellShard
   interop header and Torch example hooks. Preserve source-linked format
   specs, fixtures, and JBC evidence on the archive branch. JBC's exact
   coverage tests and its recorded no-promotion result belong with that
   experiment, along with placement, residency, transport, and publication.

## Reuse the existing owners

AnnData++ already owns H5AD and Zarr reads, writes, and backed block access.
CellShard delegated H5AD input to it. The optional
[Cellerator AnnData component](anndata.md) reads backed selections into native
host buffers. Keep the specialized 10x, Matrix Market, and Loom importers in
the archive and revisit individual readers when a caller needs them.

Cellerator owns sparse geometry planning, native layouts and execution images,
and routed CUDA search. Evaluate CellShard's byte-cost scoring for Blocked-ELL,
Sliced-ELL bucketing, and compact reorder maps individually against the native
packers when a workload warrants it.

The first candidate to carry forward is reusable packing by biological identity.
Store the assembly, annotation release, and ordered canonical feature IDs with
the packing. Compare them once against the dataset's feature metadata, prepare
a reusable column map when the order differs, and validate sparse topology when
the packing depends on it. The [reference catalog](../../components/annotation/catalogs/README.md#why-this-exists)
supplies the metadata. Binding a dataset to a prepared packing is proposed work.
