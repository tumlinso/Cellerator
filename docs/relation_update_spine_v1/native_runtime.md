# Relation Update Spine v1 native runtime

This internal implementation extends the existing prepared biological relation;
it does not introduce a second training object or an installed SDK. Mathematical
identity remains in the [architecture spine](../architecture.qmd) and the
value-owned relation calculus. Compiler recipes and native requests share these
operations.

## Physical ownership and arithmetic

The pair owns one mutable f16 plane in FMP1 physical edge order. Its independently
prepared CTP1 traversal references those same values. N1 and N16 forward remain
direct FMP1 operations; the typed N16 transpose consumes source-owned CTP1 rows
without importing training bias, ReLU or RMS behavior. Source/destination domain,
order, geometry, partition, structure epoch and value generation are checked.

Cold preparation retains bijective logical/physical maps and physical endpoint
references. Their host and device storage is charged to the reported preparation
budget. `inspect_edge_layout` exposes the immutable host logical-to-physical map
for explicit observation. No map is applied to each gradient or update.

N16 full-f32 edge VJP consumes f32 operands directly. The explicit half-rounded
profile refreshes both RNE f16 operand packs every invocation and uses the retained
half sparse contraction. Forced hybrid additionally uses a bounded exact cover,
gathered panels, actual WMMA scores, physical-slot extraction and disjoint sparse
residuals. Automatic selection remains sparse without preparing unused hybrid
panels; there is no end-to-end WMMA promotion claim. See [WMMA policy](wmma_policy.md).

Delta update computes `RNE_f16(f32(w) + delta)`; gradient step computes
`RNE_f16(fma(-alpha, g, f32(w)))`. Alpha is finite and nonnegative. Alpha zero
still executes the stated arithmetic and publishes the requested generation;
nonfinite gradients can therefore propagate NaN. There are no hidden f32 master
weights, optimizer states, hot allocations or implicit canonicalizations.

## Binding and lifetime

Callers externally serialize host API entrypoints. Execution and mutation belong
to the owner's stream. Buffers are pointer-plus-capacity views: callers promise
that the declared capacity lies in a live allocation and remains valid through
asynchronous consumption. Validation checks required element counts, overflow,
alignment, CUDA residency/device and forbidden overlap; it does not rediscover
an external allocation's extent through a driver query on each launch.

Gradient stamps bind pair incarnation, structure epoch, physical order, forward
generation, input/cotangent versions, arithmetic and producer serial. A step must
also use the latest produced gradient buffer. Another accepted gradient supersedes
the old stamp; a value update or replacement publication invalidates it. Callers
must not overwrite or release an outstanding gradient buffer before consumption.

Initial/replacement publication and updates validate the exact current generation
and reader state before writes. Successful metadata advancement means that the
ready event was enqueued after the writes, not that execution completed. One
external const value lease may be outstanding. Begin-read waits on ready; return
records consumer-done and makes the owner wait before the storage can be mutated.
No historical snapshot is promised. See [readiness protocol](readiness.md).

Any failure after partial submission poisons the pair. The old generation remains
diagnostic; in-place numeric writes cannot roll back. Failed lease return retains
the exact lease and runtime ticket. A matching return can retry for cleanup while
poisoned, but no new read, gradient or update becomes usable.

`close_relation_pair` refuses an unreturned borrow and drains owned work before
freeing buffers. Legacy `destroy` refuses to delete a borrowed pair; mutable callers
should use checked close. Events close on the owning device. If cleanup succeeds
but restoring the caller's previous device fails, checked close can return failure
with a null slot: storage has already been consumed. An asynchronous context fault
can instead preserve the poisoned pair because safe cleanup could not finish.

New mutable/lease APIs reject capture before submission. Owner-stream read-only
N1 graph replay retains its existing behavior; it does not call the mutable
readiness wait API. Reports count accepted preparation, publication, gradient,
update, sparse/WMMA/residual and pack-refresh branches. They are not performance
promotion evidence.

## Validation evidence

All native device evidence uses Tesla V100 `sm_70`, CUDA 12.9.86, nvcc host GCC12,
release compilation and the CUDA controller's GPU reservation and
`/tmp/cuda_v100_benchmark.lock`. Focused build scripts live in `/tmp/ce-ru1-n01`
through `/tmp/ce-ru1-n06`; controller receipts contain exact source revision,
commands, compiler paths and hardware metadata. Final integrated gates must
supersede these lane-local receipts before workspace cleanup.

- N01 forward/adapter/N1 lifecycle: `57fd959d-ea08-45a8-98f2-9652d3925a32`.
- N02 independent N16 transpose/adjoint and regressions:
  `ee3239b7-1658-4adf-9c21-f11b5c74b2a9`.
- N03 physical mapping, two numeric profiles and provenance rejection, memcheck:
  `52f2ad78-b861-4787-a27b-71de7eea9a0a`, zero errors.
- N04 independent arithmetic half oracle, rounding/nonfinite/update rejection,
  memcheck: `43e36446-5305-4ad6-b1f2-45d433afaf27`, zero errors.
- N05 public hybrid gradient, two update styles, three generations and delayed
  cross-stream readers, memcheck: `10e396c9-c9d7-484d-a367-be46840f9e9c`, zero errors.

`native_failure_test` normal mode covers capture refusal, busy close, stale
pointers, empty support, failed publication after an actual numeric write, and
poisoned lease cleanup. `native_failure_test --async-error` is a separate process
mode that deliberately traps its CUDA context and checks failure observation and
pointer preservation. That intentional-fault mode is run without sanitizer;
normal mode and the retained N1 lifecycle binary must remain memcheck-clean.
N06 first execution passed in `3ca7e0ff-9e9f-4dd6-9d4c-402eac55ae37`.
After review corrections, `0b187474-974a-4fca-8bb3-9a9713b1828f` passed
normal failure tests and the retained N1 lifecycle under memcheck with zero errors,
and the separate intentional asynchronous-fault mode passed. The reviewed source
`379bd1de` adds preallocation hybrid scratch-cap enforcement, budget-specific
rejection, opaque-storage overlap protection, empty half-sparse no-op support,
and overflow-safe N16 grid rounding. The capacity regression rejects a budget
that covers only whole-operand scratch and accepts the exact additional panel
budget. These focused receipts are correctness evidence, not final integrated
acceptance or a performance claim.
