# Cellerator velocity half-storage adapter

`import cellerator.velocity;` reexports `velocitypp` and
`velocitypp.precision`. The adapter accepts raw IEEE binary16 storage and
already acquired AnnData++ FP32/FP64 descriptors independently for W, S, and
U. `Matrix` borrows exactly one of `HalfMatrix` or `AcquiredMatrix`; it does
not own or construct storage. Dense half strides are signed bytes. Sparse
half descriptors describe CSR or CSC with explicitly typed indices and offsets.
All descriptors and buffers remain alive and stable throughout use.

Call `validate(Input)` once after acquiring inputs, then use the validated
`FitDomain` with Velocity++ `workspace_size`, `bind_workspace` and
`bind_stages`. Select `float` or `double` explicitly for arithmetic and result
arrays. For each selected-gene block, call `moments` into the bound workspace,
construct a `BasicMomentBlock` over it, and call the fit stages' `fit` and
`emit`; call `finish` after every block and retain its returned result, including scalar flags. For stochastic fitting, use the same
initial blocks with `initial_fit` and `accept_initialization`, call
`resolve_selection` once, then form `BasicStochasticMomentBlock` values with
`second_moments` and call `scales`, `fit`, `emit` and `finish` in order. The
Velocity++ stage contract governs masks, selection, results and fallback.

Moment calls are trusted: first/count, output capacities, pointer extents,
non-overlap, topology stability and thread synchronization belong to the
caller. Each output is rows times block-count elements; `source_s` and
`source_u` require rows elements each. No full widened matrix or graph is
allocated. Sparse expression duplicates are summed in stored order; weight
entries contribute individually. CSR and CSC weights are traversed by their stored entries; no performance
qualification is claimed. Prepare/reuse separate workspace per concurrent worker.

For half residual output, `residuals_f16` and `stochastic_residuals_f16` invoke
the upstream residual calculation using one caller-provided rows-by-block
compute scratch and round once to binary16. Output strides count half elements;
the caller validates capacities, signed spans and disjointness. Normal upstream
FP32/FP64 output remains available directly.

Velocity++ sensitivity operates on the computed moment block. For selective
promotion, configure `velocitypp::precision::Promotion` with `replace32` or
`replace64` and a `PromotionMoments` context pointing at the active initial or
second block, input, and two rows-sized scratch buffers. Original-column
providers must supply FP32/FP64 columns; widening half does not restore original
precision. The promotion controller validates columns and maintains pass/status
semantics. All calls are synchronous and allocate no storage.
