# Source-bound reference review

## Scope and inventory method

The baseline is `ff7014fd579a26038c5dc86ccb072f0d41d232cc` (`HEAD` for this
isolated checkout). It was read with `git show HEAD`, not from the parent
Cellerator worktree. At baseline, `src/**/*.{cpp,cppm,hh,hpp}` contained 94
direct standard-library include occurrences across 16 header names. Four
additional include occurrences named Cellerator's binary16 helper and are not
standard headers.

The current library source list is:

```text
src/detail/fit.cpp          src/detail/fit.hh
src/detail/neighborhood.cpp src/detail/prepared.cpp
src/detail/prepared.hh      src/detail/scientific.hh
src/detail/scientific_adapters.hh
src/detail/sensitivity.hh   src/detail/stochastic_fit.cpp
src/detail/stochastic_fit.hh
src/numeric.cpp             src/precision.cpp
src/velocity.cpp            src/velocity.cppm
src/velocity.numeric.cppm   src/velocity.precision.cppm
```

The current library has 83 direct standard-header occurrences across the same
16 header names. The 11-occurrence reduction removes obsolete per-view vector,
backend, and unused direct includes; it does not remove the `std::vector`
owners used for selected gene indices, variable-size numerical scratch, and
optional promotion status. The other C++ sources contain 172 standard-header
occurrences in tests, examples, and benchmarks. Those are listed separately
below; they are not library dependencies. Across all four scopes the current
inventory is 255 standard-header occurrences across 31 names. The only
non-standard helper included by active library source is
`<Cellerator/ops/numeric/binary16.hh>`.

Each occurrence appears in the location column as `file:line`. The purpose and
disposition columns apply to every listed location. `Keep` means the header
provides a facility used by that translation unit or its explicitly included
private adapter. Test, example, and benchmark headers remain local to those
targets and do not imply library requirements.

## Library standard-header occurrences

| Header | Occurrences | Use and disposition |
| --- | --- | --- |
| `<algorithm>` | `src/detail/fit.cpp:2`, `src/detail/neighborhood.cpp:3`, `src/detail/prepared.cpp:3`, `src/detail/stochastic_fit.cpp:2`, `src/numeric.cpp:2`, `src/precision.cpp:2`, `src/velocity.cpp:3` | Sorting, filling, copying, and range helpers. In `fit.cpp` and `stochastic_fit.cpp`, it currently supplies `std::sort` for `StandardSort` in `scientific_adapters.hh`; see the header-hygiene note below. **Keep.** |
| `<array>` | `src/velocity.cpp:4` | Fixed-size span/alias-check collections. **Keep.** |
| `<cmath>` | `src/detail/fit.cpp:3`, `src/detail/neighborhood.cpp:4`, `src/detail/prepared.cpp:4`, `src/detail/scientific.hh:4`, `src/precision.cpp:3`, `src/velocity.cpp:5` | Finite checks and numerical functions such as absolute value and square root. **Keep.** |
| `<cstddef>` | `src/detail/fit.cpp:4`, `src/detail/neighborhood.cpp:5`, `src/detail/prepared.cpp:5`, `src/detail/scientific.hh:5`, `src/detail/stochastic_fit.cpp:3`, `src/numeric.cpp:3`, `src/precision.cpp:4`, `src/velocity.cpp:6`, `src/velocity.cppm:2`, `src/velocity.numeric.cppm:2`, `src/velocity.precision.cppm:2` | `std::size_t`, `std::ptrdiff_t`, and byte-sized buffer descriptions, including public declarations. **Keep.** |
| `<cstdint>` | `src/detail/fit.cpp:5`, `src/detail/neighborhood.cpp:6`, `src/detail/prepared.cpp:6`, `src/detail/scientific.hh:6`, `src/detail/stochastic_fit.cpp:4`, `src/numeric.cpp:4`, `src/precision.cpp:5`, `src/velocity.cpp:7`, `src/velocity.cppm:3`, `src/velocity.numeric.cppm:3`, `src/velocity.precision.cppm:3` | Fixed-width status, index, and binary-storage types. **Keep.** |
| `<cstring>` | `src/detail/neighborhood.cpp:7`, `src/detail/prepared.cpp:7`, `src/numeric.cpp:5`, `src/velocity.cpp:8` | `std::memcpy` for output bytes and unaligned sparse structural values. **Keep.** |
| `<initializer_list>` | `src/velocity.numeric.cppm:4` | Dense factory shape/stride arguments. **Keep.** |
| `<limits>` | `src/detail/fit.cpp:6`, `src/detail/neighborhood.cpp:8`, `src/detail/prepared.cpp:8`, `src/detail/scientific.hh:7`, `src/numeric.cpp:6`, `src/precision.cpp:6`, `src/velocity.cpp:9`, `src/velocity.cppm:4` | Numeric limits, infinities, and checked dimension/offset arithmetic. **Keep.** |
| `<memory>` | `src/numeric.cpp:7`, `src/precision.cpp:7`, `src/velocity.cpp:10`, `src/velocity.cppm:5`, `src/velocity.numeric.cppm:5`, `src/velocity.precision.cppm:4` | Explicit shared owners and cold PIMPL/state allocation. **Keep.** |
| `<numeric>` | `src/velocity.cpp:11` | `std::gcd` in dense-output overlap validation. **Keep.** |
| `<optional>` | `src/velocity.cpp:12`, `src/velocity.cppm:6` | Optional input masks and optional outputs/options. **Keep.** |
| `<stdexcept>` | `src/detail/fit.cpp:7`, `src/detail/neighborhood.cpp:9`, `src/detail/prepared.cpp:9`, `src/numeric.cpp:8`, `src/precision.cpp:8`, `src/velocity.cpp:13`, `src/velocity.cppm:7`, `src/velocity.numeric.cppm:6` | Contract and range failures at acquisition or operation boundaries. **Keep.** |
| `<string_view>` | `src/velocity.cpp:14`, `src/velocity.cppm:8` | Non-owning AnnData layer/observation key names. **Keep.** |
| `<type_traits>` | `src/detail/neighborhood.cpp:10`, `src/detail/prepared.cpp:12`, `src/numeric.cpp:9`, `src/velocity.cpp:17`, `src/velocity.cppm:9`, `src/velocity.numeric.cppm:7` | Compile-time storage-type selection and signed/index traits. **Keep.** |
| `<utility>` | `src/detail/neighborhood.cpp:11`, `src/detail/prepared.cpp:10`, `src/numeric.cpp:10`, `src/precision.cpp:9`, `src/velocity.cpp:15` | Move, forward, and swap operations at ownership/configuration boundaries. **Keep.** |
| `<vector>` | `src/detail/neighborhood.cpp:12`, `src/detail/prepared.cpp:11`, `src/precision.cpp:10`, `src/velocity.cpp:16` | Variable-length sparse traversal buffers, prepared source-gene indices, numerical workspace, and optional status storage. **Keep.** |

The private `scientific_adapters.hh` uses `std::sort` and `std::size_t` but does
not include `<algorithm>` or `<cstddef>` itself. Its two inclusion sites
currently provide those headers through their translation-unit global
fragments. This is the deliberate private implementation boundary: each
translation unit lists the standard facilities needed before including the
adapter after the module declaration. These headers do not become public
module dependencies, and the arrangement has no per-element runtime cost.

## Test, example, and benchmark standard-header occurrences

These occurrences support test fixtures, file drivers, user examples, and
measurement/report tooling. They are deliberately separated from the library
inventory.

| Header | Occurrences | Harness use and disposition |
| --- | --- | --- |
| `<algorithm>` | `tests/precision_interop_driver.cpp:2`, `tests/test_precision.cpp:2`, `tests/test_stochastic.cpp:1`, `benchmarks/qualification.cpp:1`, `benchmarks/stochastic_qualification.cpp:1` | Fixture copying/sorting/filling and benchmark range operations. **Keep in harness targets.** |
| `<array>` | `tests/test_blocks.cpp:1`, `tests/test_precision.cpp:3`, `tests/test_reference.cpp:1`, `tests/test_stochastic.cpp:2`, `examples/parallel_blocks.cpp:1` | Fixed-size fixtures, disjoint schedules, and small scratch arrays. **Keep in harness targets.** |
| `<atomic>` | `tests/test_blocks.cpp:2`, `benchmarks/precision_qualification.cpp:1`, `benchmarks/qualification.cpp:2`, `benchmarks/stochastic_qualification.cpp:2` | Thread-test state and allocation/measurement counters. **Keep in harness targets.** |
| `<bit>` | `benchmarks/precision_qualification.cpp:2`, `benchmarks/stochastic_qualification.cpp:3` | Endianness and binary representation checks for benchmark fixtures. **Keep in benchmark targets.** |
| `<cassert>` | `tests/test_numeric_views.cpp:1` | Assertion macro in the numeric-view test. **Keep in test target.** |
| `<charconv>` | `benchmarks/precision_qualification.cpp:3`, `benchmarks/qualification.cpp:3`, `benchmarks/stochastic_qualification.cpp:4` | Numeric command-line/record parsing with `from_chars`. **Keep in benchmark targets.** |
| `<chrono>` | `benchmarks/precision_qualification.cpp:4`, `benchmarks/qualification.cpp:4`, `benchmarks/stochastic_qualification.cpp:5` | Elapsed-time measurements. **Keep in benchmark targets.** |
| `<cmath>` | `tests/test_blocks.cpp:3`, `tests/test_customization.cpp:1`, `tests/test_numeric_views.cpp:2`, `tests/test_precision.cpp:4`, `tests/test_precision_profiles.cpp:2`, `tests/test_reference.cpp:2`, `tests/test_stochastic.cpp:3`, `tests/test_stochastic_customization.cpp:1`, `benchmarks/precision_qualification.cpp:5`, `benchmarks/qualification.cpp:5`, `benchmarks/stochastic_qualification.cpp:6` | Numeric fixtures, finite checks, and reference/oracle comparisons. **Keep in harness targets.** |
| `<cstddef>` | `tests/precision_interop_driver.cpp:3`, `tests/sensitivity_driver.cpp:1`, `tests/test_blocks.cpp:4`, `tests/test_numeric_views.cpp:7`, `tests/test_precision.cpp:5`, `tests/test_precision_profiles.cpp:3`, `tests/test_reference.cpp:3`, `tests/test_stochastic.cpp:4`, `examples/selective_precision.cpp:1`, `benchmarks/precision_qualification.cpp:7`, `benchmarks/qualification.cpp:7`, `benchmarks/stochastic_qualification.cpp:8` | Fixture dimensions, strides, and byte counts. **Keep in harness targets.** |
| `<cstdint>` | `tests/consumer/main.cpp:2`, `tests/interop_driver.cpp:1`, `tests/precision_interop_driver.cpp:4`, `tests/stochastic_interop_driver.cpp:1`, `tests/test_blocks.cpp:5`, `tests/test_customization.cpp:2`, `tests/test_numeric_views.cpp:8`, `tests/test_precision.cpp:6`, `tests/test_precision_profiles.cpp:4`, `tests/test_reference.cpp:4`, `tests/test_stochastic.cpp:5`, `tests/test_stochastic_customization.cpp:2`, `examples/selective_precision.cpp:3`, `benchmarks/precision_qualification.cpp:8`, `benchmarks/qualification.cpp:8`, `benchmarks/stochastic_qualification.cpp:9` | Numeric storage bytes and fixed-width serialized/test fields. **Keep in harness targets.** |
| `<cstdio>` | `tests/test_customization.cpp:3`, `tests/test_stochastic_customization.cpp:4`, `examples/parallel_blocks.cpp:2`, `examples/stochastic.cpp:1` | Small C-style diagnostic output. **Keep in harness targets.** |
| `<cstdlib>` | `tests/test_blocks.cpp:6`, `tests/test_numeric_views.cpp:4`, `benchmarks/precision_qualification.cpp:9`, `benchmarks/qualification.cpp:9`, `benchmarks/stochastic_qualification.cpp:10` | Test allocation hooks (`malloc`/`free`) and process utilities. **Keep in harness targets.** |
| `<cstring>` | `tests/precision_interop_driver.cpp:5`, `tests/test_numeric_views.cpp:3`, `tests/test_precision_profiles.cpp:5`, `tests/test_reference.cpp:5` | Buffer copy and bytewise checks in fixtures. **Keep in test targets.** |
| `<exception>` | `tests/test_blocks.cpp:7`, `examples/parallel_blocks.cpp:3` | `exception_ptr` storage/propagation across worker threads. **Keep in harness targets.** |
| `<filesystem>` | `benchmarks/precision_qualification.cpp:10`, `benchmarks/qualification.cpp:10`, `benchmarks/stochastic_qualification.cpp:11` | Qualification input/output path handling. **Keep in benchmark targets.** |
| `<fstream>` | `tests/interop_driver.cpp:2`, `tests/precision_interop_driver.cpp:6`, `tests/stochastic_interop_driver.cpp:2`, `benchmarks/precision_qualification.cpp:11`, `benchmarks/qualification.cpp:11`, `benchmarks/stochastic_qualification.cpp:12` | Fixture and result file streams. **Keep in harness targets.** |
| `<iomanip>` | `tests/sensitivity_driver.cpp:2`, `benchmarks/precision_qualification.cpp:12`, `benchmarks/qualification.cpp:12`, `benchmarks/stochastic_qualification.cpp:13` | Stable diagnostic/report formatting. **Keep in harness targets.** |
| `<iostream>` | `tests/interop_driver.cpp:3`, `tests/precision_interop_driver.cpp:7`, `tests/sensitivity_driver.cpp:3`, `tests/stochastic_interop_driver.cpp:3`, `tests/test_blocks.cpp:8`, `tests/test_precision.cpp:7`, `tests/test_precision_profiles.cpp:6`, `tests/test_reference.cpp:6`, `tests/test_stochastic.cpp:6`, `examples/selective_precision.cpp:5`, `benchmarks/precision_qualification.cpp:13`, `benchmarks/qualification.cpp:13`, `benchmarks/stochastic_qualification.cpp:14` | Test diagnostics, examples, and qualification console output. **Keep in harness targets.** |
| `<limits>` | `tests/test_numeric_views.cpp:9`, `tests/test_precision.cpp:8`, `tests/test_precision_profiles.cpp:7`, `tests/test_reference.cpp:7`, `tests/test_stochastic.cpp:7`, `benchmarks/precision_qualification.cpp:14`, `benchmarks/qualification.cpp:14`, `benchmarks/stochastic_qualification.cpp:15` | Boundary and non-finite fixtures, plus serialized-size checks. **Keep in harness targets.** |
| `<memory>` | `tests/test_numeric_views.cpp:10`, `benchmarks/precision_qualification.cpp:15`, `benchmarks/qualification.cpp:15`, `benchmarks/stochastic_qualification.cpp:16` | Shared owners in lifetime tests and benchmark source fixtures. **Keep in harness targets.** |
| `<new>` | `tests/test_blocks.cpp:9`, `tests/test_numeric_views.cpp:5`, `benchmarks/precision_qualification.cpp:16`, `benchmarks/qualification.cpp:16`, `benchmarks/stochastic_qualification.cpp:17` | Allocation-failure hooks and aligned allocation instrumentation. **Keep in harness targets.** |
| `<optional>` | `benchmarks/qualification.cpp:17`, `benchmarks/stochastic_qualification.cpp:18` | Optional benchmark input/output switches. **Keep in benchmark targets.** |
| `<stdexcept>` | `tests/interop_driver.cpp:4`, `tests/precision_interop_driver.cpp:8`, `tests/stochastic_interop_driver.cpp:4`, `tests/test_customization.cpp:4`, `tests/test_numeric_views.cpp:11`, `tests/test_precision.cpp:9`, `tests/test_precision_profiles.cpp:8`, `tests/test_reference.cpp:8`, `tests/test_stochastic.cpp:8`, `tests/test_stochastic_customization.cpp:3`, `benchmarks/precision_qualification.cpp:17`, `benchmarks/qualification.cpp:18`, `benchmarks/stochastic_qualification.cpp:19` | Expected-error paths and test/driver contract failures. **Keep in harness targets.** |
| `<string>` | `tests/precision_interop_driver.cpp:9`, `tests/stochastic_interop_driver.cpp:5`, `benchmarks/precision_qualification.cpp:18`, `benchmarks/qualification.cpp:19`, `benchmarks/stochastic_qualification.cpp:20` | Owning parsed paths, names, and serialized driver fields. **Keep in harness targets.** |
| `<string_view>` | `benchmarks/precision_qualification.cpp:19`, `benchmarks/qualification.cpp:20`, `benchmarks/stochastic_qualification.cpp:21` | Non-owning command-line field parsing. **Keep in benchmark targets.** |
| `<thread>` | `tests/test_blocks.cpp:10`, `examples/parallel_blocks.cpp:4` | Explicit worker creation in the block test/example. **Keep in harness targets; not a library scheduler dependency.** |
| `<type_traits>` | `tests/precision_interop_driver.cpp:10`, `tests/test_numeric_views.cpp:6`, `tests/test_precision.cpp:10`, `tests/test_precision_profiles.cpp:9`, `benchmarks/precision_qualification.cpp:21` | Compile-time storage/profile selection in test and benchmark templates. **Keep in harness targets.** |
| `<utility>` | `tests/test_blocks.cpp:11`, `tests/test_numeric_views.cpp:12`, `tests/test_precision_profiles.cpp:10`, `tests/test_reference.cpp:9`, `benchmarks/precision_qualification.cpp:20`, `benchmarks/qualification.cpp:21`, `benchmarks/stochastic_qualification.cpp:22` | Move, swap, and pair operations in harness code. **Keep in harness targets.** |
| `<vector>` | `tests/consumer/main.cpp:3`, `tests/interop_driver.cpp:5`, `tests/precision_interop_driver.cpp:11`, `tests/sensitivity_driver.cpp:4`, `tests/stochastic_interop_driver.cpp:6`, `tests/test_blocks.cpp:12`, `tests/test_numeric_views.cpp:13`, `tests/test_precision.cpp:11`, `tests/test_precision_profiles.cpp:11`, `tests/test_reference.cpp:10`, `tests/test_stochastic.cpp:9`, `examples/selective_precision.cpp:4`, `benchmarks/precision_qualification.cpp:22`, `benchmarks/qualification.cpp:22`, `benchmarks/stochastic_qualification.cpp:23` | Dynamic fixtures, transfer buffers, and driver-owned results. **Keep in harness targets.** |

The non-standard binary16 helper occurs at `src/detail/neighborhood.cpp:2`,
`src/detail/prepared.cpp:2`, `src/velocity.cpp:2`, `tests/consumer/main.cpp:1`,
`tests/precision_interop_driver.cpp:1`, `tests/test_blocks.cpp:14`,
`tests/test_precision.cpp:1`, `tests/test_precision_profiles.cpp:1`,
`examples/selective_precision.cpp:2`, and the three qualification benchmarks
at `benchmarks/precision_qualification.cpp:6`, `benchmarks/qualification.cpp:6`,
and `benchmarks/stochastic_qualification.cpp:7`. Only the three `src/` sites
are active library dependencies. AnnData++ remains imported independently as
`anndatapp.core`; its headers are not conflated with the C++ standard include
inventory.

## Public abstraction families

| Family | Ownership and recurring/cold cost | Disposition |
| --- | --- | --- |
| `NumericView`, `NumericEdit`, `SparseBuffers<T>` | Owner-bearing view/edit values distinguish borrowing from explicit retention. Dense rank-one/rank-two shape and stride metadata is inline. `bytes()`, `dense<T>()`, and sparse byte access return borrowed descriptors. Copying a retained view copies a `shared_ptr` reference; borrowed raw factories allocate no metadata owner. Sparse values have an explicit type, independent index/offset integer types and capacities, and validate alignment/ranges at acquisition. | **Keep.** The boundary makes lifetime, mutability, geometry, and numeric type explicit. |
| `NumericBytes`, `NumericDense`, `NumericSparseBytes`, `NumericBlock`, `DenseOutput` | Small non-owning descriptors carry pointer, shape, stride, element width, and type. Their use inside blocks does not create an owner or allocate. `DenseOutput` depends on the caller keeping its `NumericEdit` owner alive. | **Keep.** This is the plain acquired form required by the hot numerical operations. |
| `Input`, `PreparedInput` | `Input` carries source views, masks, and an `anndatapp::Axis` gene selection. `PreparedInput` owns a PIMPL and prepared input views; it stores selected source-gene indices in a variable-size vector, O(selected genes), and validates fixed structure at preparation. | **Keep.** Preparation is the acquisition boundary. The gene vector is deliberate ownership for irregular selections, not a per-view axis wrapper. |
| `Workspace`, `StochasticWorkspace` | Each owns one precision-specific state and variable-size numerical vectors for one worker. Construction/setup allocates moment, score, mask, output-conversion, and optional global-selection scratch; `scratch_bytes()` and `block_genes()` expose the result. The 64 MiB default workspace cap covers worker numerical/mask scratch plus requested serial sink conversion and selection scratch. Promotion separately checks workspace scratch plus declared provider staging against `PromotionOptions::workspace_bytes`. | **Keep.** Per-worker ownership and caller-visible aggregate budget replace hidden shared state. |
| `GeneRange`, `BasicMomentBlock<T>`, `BasicFitBlock<T>`, `BasicStochasticMomentBlock<T>`, scale/fit blocks | Plain by-value range and result descriptors. They contain no owner. Moment pointers alias workspace or caller-supplied storage; capacities and disjoint ranges are caller preconditions. | **Keep.** They let the caller schedule work without a runtime or per-block allocation layer. |
| `FitBuffers<T>`, `StochasticFitBuffers<T>`, `PromotionBuffers` | Caller-owned writable result/status arrays in prepared-gene occurrence order. `fit_buffers()` returns pointers into one previously allocated result. Separate workers write disjoint ranges. | **Keep.** This makes output allocation, indexing, and thread disjointness inspectable. |
| `FitResult`, `StochasticFitResult` | AnnData++ arrays own O(G) diagnostics and final values; they are separate from worker scratch and caller-owned dense outputs. `make_*_result()` allocates them once before block scheduling. | **Keep.** These are useful retained results; their cost is explicit and does not grow with cell-by-gene output unless requested. |
| `BasicReferenceStages<T>`, `BasicStochasticReferenceStages<T>` | Serial stage owners use a PIMPL and allocate result arrays plus a small amount of stage state at setup. They enforce ordered stage use and provide sink callbacks. They are not shared among concurrent workers. | **Keep as serial composition.** The caller-scheduled free functions are the parallel surface; the stage owner adds no per-element dispatch. |
| `steady_state_with()` and `stochastic_with()` | Small templates express the serial stage order and allow a `process_block()` customization. `if constexpr` selects the customization at compile time; no virtual call or generic registry is introduced in numerical loops. | **Keep.** The abstraction represents replaceable scientific stages and a direct fused-block option. |
| `BlockSink`, `StochasticBlockSink`, `OutputBlock`, `StochasticOutputBlock` | A function pointer and `void*` context avoid `std::function` type erasure. Output descriptors alias workspace and are valid only through callback return. Matching-type requested moments can bypass conversion and expose their source block; other requested output conversions use budgeted sink scratch. | **Keep optional.** Direct output uses one acquired `DenseOutput` and no sink conversion scratch. |
| `OriginalSource`, `OriginalArraySource`, `PromotionOptions`, `BasicSelectiveStages<T>`, `BasicSelectiveStochasticStages<T>` | A callback supplies requested source-gene pairs. `OriginalArraySource` retains the supplied views; caller-backed sources own/account for staging. `PromotionBuffers` and low-level `promote()`/`reapply()` make each worker's status and callback context explicit. The serial stage wrappers own convenience state and O(G) status. | **Keep optional.** Use low-level promotion with worker-local providers for caller scheduling; do not make promotion or its provider a mandatory run layer. |
| `NumericView::select` and `velocity::cpu`/compiled backend selection | The baseline view-level `select` accepted `std::vector<Axis>` and stored an axis vector; the baseline compiled path carried a `compiled` switch and `velocity::cpu` entrypoints. Neither is in the active surface. | **Remove from active API.** Dense affine subsets are `.subview(...)`; irregular genes remain `Input::genes`. All active methods use the portable reference path; accelerator provenance stays in quarantine. |

No blanket STL removal is supported by this review. Dynamic vectors remain
where the data is genuinely variable-size or owns numerical scratch. The
removed vectors were per-view shape/stride/axis metadata and a prebuilt set of
output wrapper descriptors; those are now inline or returned by value.

## Compactness and active implementation style

These examples compare the baseline and current source directly. They describe
semantic and ownership changes; source shape alone is not performance evidence.

### 1. Numeric metadata: dynamic wrapper to inline descriptor

Baseline `src/numeric.cpp` stored raw FP16 data in a separately allocated
`HalfStorage` with vector dimensions and strides, and each raw view factory
called `std::make_shared` (baseline `src/numeric.cpp:17-25,133-145`). The
representative baseline shape was:

```cpp
struct HalfStorage {
    const std::uint16_t* values = nullptr;
    std::vector<std::size_t> shape;
    std::vector<std::ptrdiff_t> strides;
    anndatapp::Layout layout = anndatapp::Layout::dense;
    SparseF16 sparse;
    std::size_t nnz = 0;
    std::shared_ptr<const void> owner;
};
auto storage = std::make_shared<HalfStorage>();
```

The current public descriptor is a fixed-shape record:

```cpp
struct NumericStorage {
    const void* data = nullptr;
    std::size_t shape[2] = {}, rank = 0, nnz = 0;
    std::ptrdiff_t strides[2] = {};
    StorageType type = StorageType::float16;
    anndatapp::Layout layout = anndatapp::Layout::dense;
    SparseBuffers<void> sparse;
};
```

The view keeps that record inline; explicit retained views still keep the
provided `shared_ptr` owner. The change removes the per-view control-block and
dimension/stride-vector allocations for raw factories. The representation
also generalizes raw numeric storage from the prior FP16-only structure to
typed FP16/FP32/FP64 dense and sparse buffers. It does not change the stored
values or infer ownership. See [the descriptor](../src/velocity.numeric.cppm#L30)
and [raw factory implementation](../src/numeric.cpp#L140).

### 2. Affine view versus irregular gene selection

The baseline API accepted general axis objects through vector overloads:

```cpp
NumericView select(const std::vector<anndatapp::Axis>& axes) const;
NumericView select(std::initializer_list<anndatapp::Axis> axes) const {
    return select(std::vector<anndatapp::Axis>(axes));
}
```

The active numeric view instead exposes
`subview(axis, start, count, step)` for dense affine coordinates. A negative
step is allowed; sparse `NumericView`s do not expose affine subviews. Arbitrary
gene ordering or repeated/irregular selection remains represented by
`Input::genes` and is resolved into prepared source-gene indices once. Thus the
view operation is narrower, while the run-level selection capability remains.
The current declaration is a scalar-coordinate operation:

```cpp
NumericView subview(std::size_t axis, std::size_t start,
                    std::size_t count, std::ptrdiff_t step = 1) const;
```

The cost change is removal of per-view `std::vector<Axis>` composition and
gather-to-stride inspection; preparation still owns an O(G) index list for an
irregular selected order. See [the public signature](../src/velocity.numeric.cppm#L108)
and [subview arithmetic](../src/numeric.cpp#L200).

### 3. Workspace output wrappers to borrowed block descriptors

The baseline workspace reserved `slots * (block + 1)` `NumericView`s and
constructed borrowed AnnData views for every slot and block width
(`src/velocity.cpp` at baseline lines 127-135). The current workspace computes
the data pointer and returns one `NumericBlock` by value
([`view()`](../src/velocity.cpp#L134)); the descriptor has no retained owner,
allocation, or reference count. Matching-computation-type moment fields are
returned directly from their existing moment block; fields requiring a
different requested dtype still use explicitly budgeted conversion storage.

The baseline storage/access pattern was a prepared table:

```cpp
views.reserve(mul(slots, block + 1));
for (std::size_t slot = 0; slot < slots; ++slot) {
    // populate one owner-bearing NumericView for each count in this slot
}
NumericView view(std::size_t slot, std::size_t count) const {
    return views[slot * (block + 1) + count];
}
```

The current `view()` derives a pointer and returns the plain descriptor by
value. The values and row-major order are unchanged; the owner-bearing wrapper
table is gone. This is a descriptor-construction/storage distinction, not an
arithmetic change.

The moment values and their row-major order do not change. The changed cost is
removal of a persistent O(slots × block widths) vector of high-level wrappers
and owner-bearing views. A direct output is acquired once as `DenseOutput` and
written in place with no sink conversion buffers
([single acquisition and block loop](../src/velocity.cpp#L779)); the sink path
remains optional. Both contracts preserve borrowed lifetimes: sink blocks are
consumed before callback return, and a `DenseOutput` owner outlives the run.

### 4. Internal serial stage loop to caller-owned block schedule

The high-level convenience drivers still run serially. The added range API
exposes `GeneRange`, caller-owned fit buffers, per-worker `Workspace`,
`validate_run()`, and allocation-free `reference::` block functions in the
public module ([range declarations](../src/velocity.cppm#L442)). For stochastic
fitting, the caller submits initialization blocks, joins, calls
`finalize_selection()` once, then submits refinement blocks and joins. The
same per-gene equations and global selection rule apply; the external join is
what permits safe parallel writes to disjoint ranges. No scheduler, locks, or
thread-pool policy is introduced. Parallel memory is the caller's sum of
worker scratch plus global selection/source staging.

The public use is a two-phase caller schedule; a deterministic excerpt from
[`parallel_blocks.cpp`](../examples/parallel_blocks.cpp#L37) is:

```cpp
validate_run(input);
auto moments = reference::moments(input, worker, range);
auto fit = reference::fit(input, worker, moments, buffers, options);
reference::write_residuals(moments, fit, output);
// join workers before finalize_selection(...)
```

Stochastic scheduling adds the initialization join and one global selection
before second-moment/refinement blocks. The block functions take preallocated
workspace and result descriptors; no scheduler or per-block owner is created.

The serial `stochastic_into()` code makes the barrier explicit in the same
order: initialization loop, `resolve_selection()`, then refinement loop
([serial implementation](../src/velocity.cpp#L888)). It acquires its dense
output once and does not allocate sink output scratch. This is an ownership
and scheduling change, not a speed claim.

### 5. Portable checked stride arithmetic

The baseline used compiler overflow builtins (`__builtin_mul_overflow` and
`__builtin_add_overflow`) in `src/numeric.cpp`, for example:

```cpp
if (__builtin_mul_overflow(a, b, &c))
    throw std::length_error("numeric stride overflow");
```

The current code checks standard signed bounds before evaluating the
operation:

```cpp
if ((a > 0 && (b > 0 ? a > max/b : b < min/a)) ||
    (a < 0 && (b > 0 ? a < min/b : b < 0 && a < max/b)))
    throw std::length_error("numeric stride overflow");
return a*b;
```

Representable offsets and strides keep the same arithmetic result; overflow
continues to throw at acquisition rather than invoke signed overflow. The
semantic change is portability away from compiler-specific builtins. The new
path has explicit comparisons and branches; no runtime improvement or
regression is claimed without a measurement. See
[`mul`/`add`](../src/numeric.cpp#L19).

### 6. One active portable moment path

The baseline prepared-moment implementation carried a runtime `compiled`
hint. For a qualifying contiguous/aligned block, it entered a CPU helper:

```cpp
if (compiled && x.column_stride == sizeof(C) &&
    reinterpret_cast<std::uintptr_t>(source) % alignof(C) == 0) {
    cpu::weighted_add(reinterpret_cast<const C*>(source), w,
                      out + i * stride, count);
    continue;
}
```

The active block overload calls the prepared kernel without that flag, and the
source contains no `cpu::` helper or backend selector. The equations, selected
gene order, and output values are still defined by the reference calculation;
the changed cost is removal of the alternate helper/dispatch path and its
support files from this package. This does not imply a measured speed change.
The baseline fast path is at `HEAD:src/detail/prepared.cpp:121-143`; the
current [range operation](../src/velocity.cpp#L1081) omits the flag.

### 7. Compile-time numerical types and cold runtime boundaries

The public block algorithms are templated on FP32/FP64 computation and receive
plain typed block pointers. `Workspace` resolves compute precision before
blocks run, while output storage type is checked at acquisition or sink
boundary. Type-specific code uses `if constexpr`; storage-type conversion is
dispatched outside the arithmetic expression. The remaining dynamic owners
and PIMPL allocation occur during `PreparedInput`, `Workspace`, result, or
stage setup. The inner per-gene and per-row calculations remain direct loops.

The private scientific adapter still has two tiny sort callables—
`StandardSort` in `scientific_adapters.hh` and `Sort` in `precision.cpp`—both
calling `std::sort(p, p+n)`. They are local to different compiled paths and
introduce no distinct numerical semantics. Sharing them could remove a few
source lines but add a private-header dependency; keeping them is the simpler
boundary at this scope. No generated-code or runtime difference is claimed.

## Audit outcome

The current portable reference path keeps explicit owners where data is retained and
plain descriptors where blocks are consumed. It removes vector-backed view
metadata, per-width wrapper tables, and backend selection from the active
path while retaining numerical arrays, caller-owned outputs, and useful
scientific stage composition. The serial workspace cap defaults to 64 MiB;
caller-scheduled parallel runs must account for aggregate per-worker scratch.
Promotion staging is charged against its separate 64 MiB default cap when
declared.
No speedup, allocation reduction ratio, or code-count claim is made here.
