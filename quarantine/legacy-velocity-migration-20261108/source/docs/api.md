# Native API and execution contracts

`import velocity;` exports the reference types and functions and re-exports
`anndatapp.core`. The library is C++20 and CPU-only. `velocity::NumericView`
and `NumericEdit` carry storage descriptions; the reference operates on
acquired numerical descriptors and caller-provided result storage.

## Numeric storage and ownership

Storage type is explicit: FP16, FP32, or FP64. Ordinary AnnData++ FP32/FP64
array views adapt directly. FP16 uses `std::uint16_t` binary16 encodings, not
an AnnData++ integer array. `NumericView` is read-only and `NumericEdit` is
mutable. Their `borrow_*` factories retain no owner: the caller keeps the
underlying bytes stable and alive until every operation using the view has
completed. Their `retain_*` factories also keep the supplied
`std::shared_ptr<const void>` alive. A retained descriptor extends storage
lifetime; it does not make concurrent mutation safe.

Rank-one and rank-two shape and stride metadata fit inline in the descriptor.
`bytes()` and `dense<T>()` return small borrowed descriptors with no owner,
allocation, or reference count. They describe the existing storage; they do
not copy or convert it. `dense<T>()` checks the requested element type and
returns strides in elements. `bytes()` reports byte strides. Use
`sparse_bytes()` for sparse values, indices, offsets, widths, and element
counts. `SparseBuffers<T>` describes value storage plus independent raw index
and offset buffers, capacities, and their integer types. `borrow_sparse()` and
`retain_sparse()` cover FP32 and FP64; `SparseF16` aliases
`SparseBuffers<std::uint16_t>` and uses explicit `*_sparse_f16()` factories.
The raw factories construct inline descriptors without allocating. Values
must meet their element type's alignment. Index and offset byte buffers may be
unaligned; acquisition reads them without assuming alignment.

`NumericView::subview(axis, start, count, step)` expresses an affine dense
subset. The signed step may be negative; a zero step is invalid. It carries the
same underlying owner/lifetime and changes only the view metadata. Sparse
views do not have affine subviews. Irregular gene selection is represented by
`Input::genes` and acquired by `PreparedInput`; it is not a numeric view
operation. `GeneRange{first,count}` names a contiguous range of positions in
that prepared gene order. `PreparedInput::gene_index(position)` maps a
prepared position back to the source gene index.

`Input` groups spliced, unspliced, and weight views with the selected gene
axis and optional row/gene masks. `from_anndata()` is an adapter for the
default AnnData++ layer and pairwise names; it does not normalize the supplied
graph. `reference::prepare_neighborhood()` is an explicit operation for
positive-support, self-linked, row-normalized weights.

`PreparedInput` acquires the fixed shapes, layout, sparse structure, masks,
and gene order. Keep those structural facts unchanged during the run. Value
content may be updated between runs; do not mutate any input or mask while
workers are active. A workspace caches the fit-row mask in worker-sized
scratch: call `Workspace::refresh_masks()` or
`StochasticWorkspace::refresh_masks()` before scheduling after changing mask
values. For caller-scheduled range operations, call `validate_run(prepared)`
once before starting workers. It performs the value scan when the prepared
input uses full validation; `Validation::none` omits optional content scans.
Use the workspace-taking block overloads such as
`reference::moments(prepared, workspace, range)` and
`reference::second_moments(prepared, stochastic_workspace, range)` for this
schedule. They use prepared structures without rescanning the full input for
each block. The separate raw-pointer convenience overloads taking
`first,count` validate full-checked input on each call. Replacement operations
validate their supplied alternate batches according to the prepared
validation mode. Serial stage and convenience drivers perform one full scan
during setup; callers do not add a separate scan before those drivers.
`validate_values()` is available when a caller needs to repeat the scan
explicitly.

## Caller-scheduled deterministic blocks

The caller owns scheduling, synchronization, and worker lifetime. Each worker
has its own `Workspace`; all workers share immutable prepared input and write
disjoint gene ranges in caller-owned result buffers. The library has no
scheduler or internal locks. Numerical ranges need not be submitted in order.
Output, input, scratch, and worker result spans must obey their documented
non-aliasing requirements.

Allocate diagnostics once. `make_fit_result<T>(n_genes)` allocates owning
AnnData++ result arrays; `fit_buffers(result)` returns a `FitBuffers<T>` of
writable pointers in prepared-gene order. For lower-level storage control, a
caller can construct the same plain descriptor around disjoint arrays.
`BasicMomentBlock<T>`, `BasicFitBlock<T>`, and `DenseOutput` are borrowed
descriptors. Reference moment pointers alias workspace; their validity is
bounded by that worker's next use of the scratch. Manually supplied block
buffers and output storage remain caller-owned and must have the documented
capacity and lifetime.

The deterministic range path is:

1. Construct `PreparedInput`, create worker workspaces, refresh cached masks
   if their values changed, and call `validate_run()` once.
2. Acquire each destination once with `acquire_output(prepared, edit)`.
3. Give each worker a separate `Workspace` and disjoint `GeneRange` values.
4. For each range, call `reference::moments()`, then `reference::fit()`, and
   write the requested velocity with `reference::write_residuals()`.
5. Join workers before consuming shared results or beginning another phase.
6. Call `reference::finalize_selection()` once after all fit ranges if global
   low-signal recovery is requested.

`reference::moments()` and `fit()` perform the calculations for one assigned
range. `write_residuals()` writes through the acquired `DenseOutput` directly;
it does not create a sink conversion block. The convenience
`reference::steady_state_into()` accepts a caller-owned `NumericEdit` for
dense velocity output. When moments are requested from a sink, only fields
matching the computation type can be exposed as zero-copy workspace views;
any requested storage conversion is a separate cost.

[`examples/parallel_blocks.cpp`](../examples/parallel_blocks.cpp) shows a
small thread schedule with per-worker workspaces, disjoint ranges, one
validation scan, and a join before global selection. It does not introduce a
velocity scheduler abstraction.

`FitBuffers<T>` contains pointers for `gamma`, `offset`, `r2`, selected flags,
and status. The arrays have `size` entries, are in prepared-gene order, and
must be mutually disjoint. Each worker writes only entries for its assigned
range. The optional global `scvelo_percentile` selection pass needs caller
scratch for `n_genes()` elements; other selection modes do not need that
gene-sized score buffer.

## Stochastic blocks

Stochastic refinement is globally staged because the refinement population
depends on completed deterministic initialization. The required synchronization
order is:

1. Create worker workspaces, refresh cached masks if needed, and call
   `validate_run()` once before any worker starts. The serial
   `stochastic_into()` driver performs this validation during stage setup.
2. Run `reference::initial_moments()` and `reference::fit()` over disjoint
   ranges, filling the initialization portion of `StochasticFitBuffers<T>`.
3. Join all initialization workers.
4. Call `reference::finalize_selection()` once to produce one immutable
   `SelectionResult` for the run.
5. Run `reference::second_moments()`, `scales()`, and `fit()` over disjoint
   refinement ranges using that same selection.
6. Join all refinement workers before consuming the final results.

`make_stochastic_fit_result<T>()` allocates owning diagnostics and
`fit_buffers(result)` returns a `StochasticFitBuffers<T>` containing its
initialization arrays and stochastic arrays. Blocks expose borrowed
row-major pointers plus an explicit stride. A range's moment storage and fit
arrays must remain valid until the operation that consumes them completes.
For direct velocity or second-residual outputs, acquire the destination once
as `DenseOutput` and call the stochastic `reference::write_residuals()`;
output arrays may not overlap each other, the input, or live scratch.

The high-level `reference::stochastic_into()` writes directly to caller-owned
dense outputs. `StochasticBlockSink` fields alias reusable workspace and must
be consumed before its callback returns. Matching-type moment fields are
zero-copy borrowed views. `StochasticFitBuffers<T>` covers initialization,
final fit parameters, scales, statuses, and objective values in prepared-gene
order. Workers write disjoint ranges; the global selection remains fixed
through refinement.

The serial convenience path's default 64 MiB workspace cap includes worker
numerical and mask scratch, requested sink conversion buffers, and the
gene-sized low-signal scratch when required. A workspace reports its actual
`scratch_bytes()` and selected `block_genes()`. In caller-scheduled parallel
execution, the caller budgets the sum of worker scratch, one global selection
buffer when needed, and provider staging. Optional original-source promotion
has a separate `PromotionOptions::workspace_bytes` cap (64 MiB by default)
that covers workspace scratch plus declared `OriginalSource` staging bytes
for requested genes. Reduce block width if optional per-block storage must fit
the serial workspace cap. During reconfiguration, obsolete or shrinking sink
buffers are released before selection and worker scratch are resized, avoiding
a transient peak that holds those old buffers alongside their replacements.

## Optional original-source promotion

`velocity.precision` provides advisory sensitivity reports and explicit
promotion operations. A caller-supplied source-gene mask identifies which
originals may be requested. `OriginalSource::acquire` returns one dense
spliced/unspliced N-by-1 pair or reports it unavailable. Promotion requires
FP32/FP64 originals; widening a stored FP16 value does not recover prior
precision. Declare provider staging bytes. The worker-local provider and its
returned owners must remain valid through consumption.

`PromotionBuffers` holds one caller-owned status byte per prepared-gene
occurrence. During deterministic processing, call
`velocity::precision::promote()` on
each computed moment block before fitting it. Stochastic processing uses the
same promotion during initialization and `velocity::precision::reapply()` on its
refinement blocks. Each worker writes only its assigned status and block;
status and availability are frozen before global selection and refinement.
This keeps source acquisition in the caller's block schedule. The
`SelectiveStages`/`SelectiveStages32` and
`SelectiveStochasticStages`/`SelectiveStochasticStages32` aliases remain
available for serial stage use; their underlying templates are
`BasicSelectiveStages<T>` and `BasicSelectiveStochasticStages<T>`.

Availability is fixed for the whole run. A requested gene unavailable during
initialization stays on the baseline source. In stochastic execution, each
previously promoted gene must still be available with a valid FP32/FP64 pair
during refinement; otherwise reapplication throws. The caller must keep
original source values stable across phases. The implementation checks
availability and shape/type, but does not compare values to detect mutation.
`OriginalArraySource` returns zero-copy columns from retained dense arrays.
Caller-owned backed providers may materialize requested columns and must
account for that staging separately.

## Custom stages and standalone primitives

`ReferenceStages<T>` and `StochasticReferenceStages<T>` expose the reference
pipeline as direct stage operations. `steady_state_with()` and
`stochastic_with()` let callers replace stages or fuse `process_block()` while
keeping custom state opaque. The custom path does not intercept calls in an
element loop; reference functions remain explicitly callable.

The `reference` namespace also exposes scalar per-gene fitting, moment
construction, second-order statistics, stochastic scales and fitting, and
residual writers over caller-provided pointers. These low-level functions do
not acquire owners. Pointer capacity, lifetime, stride, and required scratch
are caller preconditions; checks at the call boundary do not discover buffer
capacity. No checked array acquisition occurs inside numerical element loops.

For equations, masks, fallback rules, and numerical tolerances, see
[architecture](architecture.md) and [validation](validation.md).
