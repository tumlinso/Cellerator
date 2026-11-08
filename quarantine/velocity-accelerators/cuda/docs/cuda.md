# Standard CUDA reference

`import velocity.cuda;` adds the standard deterministic and stochastic methods
on CUDA. It reuses the scientific arithmetic in the CPU implementation:
percentiles, masks, intercept constraints, residual scales, rank checks, joint
fitting, and deterministic fallback. It does not dispatch through Cellerator.

The public module contains no CUDA runtime headers. Its implementation uses
ordinary NVCC translation units behind Clang C++20 module interfaces. The
optional CMake target is `velocity::cuda`; CPU consumers do not require CUDA.

## Precision and input storage

Compute precision is independent of device and storage. FP16 storage is
converted to FP32 or FP64 at numerical acquisition; it does not select another
velocity algorithm. The reference provides:

| Stored values | Compute and accumulation | Diagnostics |
| --- | --- | --- |
| FP16 | FP32 | FP32 |
| FP32 | FP32 | FP32 |
| FP64 | FP64 | FP64 |
| Mixed FP16/FP32/FP64 | Explicit FP32 or FP64 | Selected compute type |

`ComputePrecision::automatic` resolves once at workspace construction. If any
W/S/U input uses FP16 and none uses FP64, it chooses FP32. Otherwise it chooses
FP64, preserving existing FP32/FP64 CPU defaults. Explicit `float32` or
`float64` wins. An original-data source does not change that decision.

W, S, and U have independent storage types and layouts. Dense matrices have
explicit signed element strides. CSR and CSC matrices have independent index
and offset integer types. Logical expression duplicates are summed in compute
precision before multiplication or nonlinear products. Graph contributions
remain additive. Products, moments, fitting, and scales are never narrowed
back to FP16. Output storage is independently declared by each output view.

The uniform FP16/FP32 path performs no screening, promotion, or original-data
acquisition. It remains available when originals exist. There is no automatic
CPU fallback, FP64 retry, TF32 conversion, or fast-math path.

## Borrowing and execution

`MatrixView` borrows `anndata::cuda::DenseView` or `SparseView`. These are the
same POD contracts installed in `<anndata/device_abi.h>` for ordinary NVCC
consumers. They describe allocation spans, dtype, device ordinal, shape,
origin, strides, mutability, and sparse capacities. An external owner can
supply them directly; neither anndata nor velocity must adopt its allocation.
Device pointers must not be wrapped as host `anndata::Array` objects.

`PreparedInput` validates metadata and, by default, sparse contents and finite
values. It retains structural row/column indexes and the selected gene map,
not another copy of numerical inputs. Preparation completes its supplied
stream. Numeric values may subsequently change between runs; changed shapes,
indices, offsets, or layouts require re-preparation. `validate_values(stream)`
is an explicit acquisition-boundary check that completes the stream.

Numerical reference calls enqueue work on the supplied `Stream`. Keep input
owners, workspace, result, outputs, and stream alive until completion. Use
explicit stream/event ordering before another stream consumes them. A
`FitResult` owns diagnostics; `ResultView` also accepts caller-owned device
storage. Its scalar diagnostics are contiguous compute-type columns,
selection/status arrays are uint8 columns, and global `low_signal` and
`refit_all` flags are uint8 device scalars. Downloads are explicit.

```cpp
#include <cstdint>

import velocity.cuda;

namespace ac = anndata::cuda;
namespace vc = velocity::cuda;

// w, s, u are borrowed device matrix views. The application owns their buffers
// and has ordered their readiness on stream.
vc::Input input{w, s, u};
vc::PreparedInput prepared(input, stream);
velocity::StochasticRunOptions options;
options.compute_precision = velocity::ComputePrecision::float32;
vc::Workspace workspace(prepared, options);

auto n = prepared.n_cells(), g = prepared.n_genes();
ac::DeviceBuffer storage(n*g*sizeof(float), prepared.device());
ac::DenseView output{storage.span(), 0, n, g, std::int64_t(g), 1,
                     ac::ScalarType::float32};
auto fit = vc::reference::stochastic(prepared, workspace,
    velocity::StochasticOptions{}, vc::Outputs{.velocity = output}, stream);
ac::synchronize(stream);
```

Use `steady_state` with `RunOptions` for deterministic fitting. The `_into`
forms accept a preallocated `ResultView`, allowing repeated execution without
another result allocation.

The explicitly requested `prepare_neighborhood` helper produces dense
FP32/FP64 support, self edges, and row normalization in caller-owned output.
Its requested output precision controls duplicate summation. It completes
the stream before releasing temporary structural preparation. Ordinary
moments apply W literally and never normalize it implicitly.

## Bounded batches and replaceable stages

The default numeric budget is 64 MiB, with at most 64 genes per block. Source
S/U acquisition, moment fields, score sorting, and a packed fit mask share
that budget. A block shrinks to fit; inability to fit one gene is an error.

Workspace construction uses the memory, block-size, and compute options, plus
`unspliced_second` to reserve optional Muu storage. CUDA output requests and
dtypes are defined by `Outputs`; CPU output-selection flags do not allocate
device outputs.

`scratch_bytes()` reports internal numeric device storage. `staging_bytes()`
reports declared base-provider staging, and `budget_bytes()` reports the
aggregate limit. Prepared structural indexes, input owners, O(genes) result
diagnostics, and retained outputs are separate reported storage.

For backed or host data, set `Input::source` and declare `source_genes`,
`source_spliced_dtype`, and `source_unspliced_dtype`. The callback receives
selected host gene IDs and returns dense N-by-count device views in that
order. Its host ID list is valid during the callback only. It may perform I/O
or enqueue transfers using existing anndata filling primitives; velocity adds
no I/O scheduler. Returned storage must remain valid through its consumers on
the supplied stream. Declare the total temporary host/device numerical
materialization capacity in `BlockSource::staging_bytes`; borrowed resident
inputs have zero staging. This declaration is charged when choosing block
size, and a base provider must supply every requested pair.

Set `Outputs::block_local = true` to use reusable N-by-block device output
buffers. A `BlockSink` receives the actual first/count, sliced output views,
and stream after each block has been submitted. Enqueue consumption on that
stream, or establish explicit event ordering, before the next block reuses
the buffers. No full velocity matrix is allocated by the library.

`ReferenceStages` exposes moments, initialization, global selection, scales,
fitting, emission, and completion. It composes with `steady_state_with` and
`stochastic_with`; replacement stages may keep opaque device state. Moment
blocks are borrowed column-major N-by-count views in compute precision.
Custom moment views with that layout are accepted after alias checks. A
`FitBlock` must describe the result views supplied when constructing the
stage; a custom kernel can write those caller-owned result views directly.
No public allocator or backend registry is required.

## Explicit promotion and advisory screening

`ExpressionSource` prepares alternate S/U column access while reusing the
base graph and gene mapping. It can borrow resident FP32/FP64 matrices or use
a `BlockSource` callback. Keep it, its numerical owners/provider, and staging
alive and unmoved through both stochastic passes and stream completion.
Values must stay fixed during a run.

`initial_moments_from` and `moments_from` take a borrowed host uint8 promotion
mask over the full source-gene axis. A null mask requests no promotion and
does not acquire originals. Only explicitly requested columns are acquired;
they overwrite the corresponding source columns in existing compute scratch
before the one W application. No unpromoted original columns or full FP32
shadow are materialized. Optional host block statuses use
`precision::PromotionStatus`: not requested, promoted, or unavailable.

An original callback may decline individual pairs through
`SourceBlock::available`. The host availability bytes remain valid until the
acquiring stage method returns. All-declined requests need no dummy device
allocation. Missing sources, declined pairs, and pairs still in FP16 retain
base FP16 values. The original provider's declared staging is charged only
when requested. Original data never activates promotion by its presence.

Stochastic initialization records the applied choices before global selection.
The second pass reuses those choices, including through ordinary `moments()`.
If a promoted original disappears, execution reports an error rather than
mixing original-data initialization with FP16 second moments. Shared W is
unchanged; S/U promotion cannot restore graph-weight precision.

`ReferenceStages::sensitivity` is an independent optional diagnostic. Invoke
it after an initialization fit and before global stochastic selection. Its
`SensitivityView` arrays use the prepared gene axis; `state` is required,
other fields are optional, and output columns may have signed row strides.
All output fields must be disjoint from one another, live moments, scratch,
masks, inputs, and fit diagnostics.

The screen reuses existing moments and score scratch for two initialization
stress fits. Defaults use a `0.5*u16` trimming band and report a changed local
selection/status, R² change above 0.05, or prediction change above 10% of
baseline residual L2 with the declared numerical floor. Thresholds are
configurable. With trimming disabled it reports existing diagnostics only.
The report does not certify actual FP16 error and does not promote genes.
Disabled screening allocates no report storage and performs no extra fits.

## Qualification boundary

Build with `VELOCITY_ENABLE_CUDA=ON` against an anndata installation built with
`ANNDATA_ENABLE_CUDA=ON`. Both are disabled by default. CUDA implementation
edits remain behind compiled module boundaries; vendor and intrinsic headers
are private. The initial native target is SM70 with CUDA 12.9 and Clang 18.

On the shared host, execute CUDA tests, Compute Sanitizer, and measurements
only through an admitted Device Scheduler job. `test_cuda`,
`test_precision_cuda`, and the CUDA precision interop driver are built as
explicit executables; CPU CTest invocation grants no GPU allocation.

The reference uses ordered scalar scientific reductions and per-gene sorting
on the device, with parallel moment evaluation. This favors a clear numerical
comparison; it does not establish a GPU speedup. Compare matching stored bytes
and compute precision first. Report original-to-FP16 quantization separately,
and include preparation, staging, transfers, result/output storage, allocation,
and completion in end-to-end measurements. See the milestone validation record
for actual source-bound qualification and its remaining limits.
