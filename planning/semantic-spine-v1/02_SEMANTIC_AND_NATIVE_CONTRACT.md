# 2. Semantic authority and provisional native contract

## The mathematical object

For logical edge e from source s(e) to destination d(e), forward application is

```text
Y[d,n] = sum over edges e with d(e)=d of W[e] * X[s(e),n]
```

Transpose application is

```text
Z[s,n] = sum over edges e with s(e)=s of W[e] * U[d(e),n]
```

Transpose changes the direction of application, not the underlying biological edge identity. It is not a matrix inverse, and not the complete meaning of a gradient composition. Every logical edge contributes once. Different edge identities with identical endpoints are distinct additive contributions; no silent deduplication or f16 weight coalescing is legal merely because a sparse format stores unique endpoints. A provider that cannot preserve that case may reject it explicitly.

Zero-degree destinations or sources produce zero under overwrite. Empty support remains well-defined. Native realization may use device zero-fill/no-op for empty support; it may not call an existing kernel outside its preconditions. Wider shapes and update modes remain representable only where their meaning is explicit, with capability rejection rather than fake execution.

## Three different classes of information

**Mathematical contract:** topology identity and epoch; source and destination identities and extents; logical edge order/count; direction; input, relation, multiplication, accumulation and output types; numerical permissions; output update and alias rules.

**Source provenance:** source location, field/function symbol, AST/IR ownership and diagnostic context. These can differ between a C++-origin and source-origin operation without making their mathematics unequal.

**Prepared/launch state:** geometry/projection, candidate, device, caller stream, actual pointers and capacities, current value-generation publication and readiness. This changes at its own lifetime. Do not bake a changing generation into the immutable topology key or compare pointer addresses as biological identity.

Reuse `execution::persistent_identity<Tag>`, `persistent_axis_identity`, `structure_epoch` and `value_generation` where they already carry the right information [S04]. The descriptor is value-owned without pointers back into itself. Comparison is fieldwise; serialization/hash must not include C++ padding. No new monolithic operation-core v3 is needed just to provide a small canonical relation descriptor.

Semantic identity must distinguish an identity/order change from merely rebinding equivalent data at another address. The initial supported physical order can be the input's declared order; this epic need not create a layout optimizer. Explicitly recover biological order when a projection differs.

## Numerical meaning versus acceptance tolerance

The required witness uses binary16 relation values and float32 state/multiply/accumulation/output, round-to-nearest conventions and separately specified FMA/reassociation permission. The oracle uses the same rounded/stored values, then accumulates small fixtures in double. For the provided finite toy fixture, compare with `abs(actual-expected) <= 1e-5 + 1e-5*abs(expected)`.

This tolerance is a test criterion for that fixture, not a universal scientific error budget, permission to change dtypes, or a waiver of support/identity correctness. Additional cancellation or high-degree tests must justify their numerical bounds from the accumulation length/data and selected policy rather than blindly reusing one threshold. Check NaN/Inf deliberately: a false comparison involving NaN must never pass a finite expected result. A nonfinite rejection policy that requires an unsupported device scan should be rejected as a capability, not pretend to have validated values on the host. FMA and association can legitimately change floating-point results [R01].

## Concrete provisional API

`contracts/relation_semantics.hh` and `contracts/prepared_relation.hh` are declaration sketches for C01/C03. They are planning artifacts, not a replacement native library. The demo targets their proposed eventual public paths:

```text
include/Cellerator/compute/operation/relation_semantics.hh
include/Cellerator/compute/operation/prepared_relation.hh
```

The internal published interface can be refined before I02; changes then require coordinated updates to the demo and all lane consumers. It does not freeze the final public SDK.

The principal functions are `validate`, `equivalent`, `prepare_relation_pair`, `publish_values`, `enqueue`, `inspect` and `destroy` in `cellerator::compute::relation`.

`prepare_relation_pair` takes canonical forward and transpose descriptors for the same topology/arithmetic, a bounded host-CSR view and explicit device/stream. It builds using existing packing/projection machinery, including FMP1 and CTP1 where applicable [S05–S07]. It must work from the supplied support, not manufacture a prevalidated fixture-specific image. Cold host work is allowed; the hot math remains accelerator execution. Check CSR extents, monotonic offsets, terminal nnz count, source bounds and duplicate limitations before narrowing into physical indices.

Preparation failure leaves the output handle null and releases any partial allocation. Host CSR arrays are consumed during preparation; any asynchronous upload must either retain owned staging or document that the caller keeps those arrays alive until the preparation stream completes. No later hot enqueue may depend on caller-owned host CSR remaining alive.

The narrow native implementation owns its preparation state and fixed-capacity projection/value buffers. Input/output views borrow storage. No generic allocator/container framework is required. A caller may replace pointers between launches under the declared ordering/lifetime contract.

## Generation and stream protocol

A pair begins with no published values. `publish_values` accepts device binary16 data, logical edge order, structure identity/epoch, count, device ordinal and a **strictly increasing nonzero generation**. Re-publishing the same or older generation is an explicit stale-generation error in this first version. This deliberately small protocol is not a universal future API restriction.

The publication operation can enqueue value packing on the pair's caller stream and update its latest **enqueued** generation after successful submission. Same-stream consumers execute after those writes. The report is not a claim that the GPU has already completed the work. Caller buffers remain valid until completion. If enqueue partially fails, make the pair's error state explicit; do not report a new generation ready with incomplete device storage.

`enqueue` validates the full relevant descriptor, axes, capacities, device, stream, alias prohibition and expected generation before changing output. Stale/mismatched requests leave output and accepted-launch counters unchanged. Native entry points do not silently drop an unsupported numerical or update policy.

For this epic, one pair belongs to one device and one caller stream. Concurrent use through another stream is rejected rather than given a falsely safe event protocol. The implementation may use existing readiness events [S08]; it must not invent global synchronization to cover missing ownership reasoning. Destruction may fence that pair's stream to release its own memory safely. Host readback in the demo is an explicit observation, outside the resident hot path [R02].

## Honest reporting and reuse

The inspection report identifies actual bound candidates and projections, number of topology preparations, value refreshes, accepted directional launches and latest enqueued generation. The native demo expects one preparation, two refreshes, three forward launches and two transpose launches. Counters are cross-checked against actual calls and runtime tests; strings/counters alone are not hardware evidence.

No performance-selection promise is attached to this initial pair API. N1 sparse execution is a deliberate small witness. The existing narrow candidates are computationally real [S05–S07]; this epic does not need to force Tensor Cores into a width regime where they are not the point.

## Compiler-origin path

Use the existing parser/Sema/IR and `lower_relation_apply_operation_v1` seam [S09]. The test environment may supply runtime identities and loaded relation metadata, but may not simply return the descriptor expected by the test. Changing the source direction/symbol/type must change the parsed meaning or trigger a diagnostic.

The compiler-origin descriptor is compared fieldwise with an independently constructed C++ descriptor, then passed through the **same** prepared accelerator path in integrated tests. No separate host-reference launch or hand-coded demo interpreter can satisfy this requirement. This proves a bounded embedded source-origin route, not the installed `.cell` command-line pipeline.

## Bounded algebra repairs

Scalar support dot: `dot[e] = sum_k A[s(e),k]*B[d(e),k]`; splitting k produces partial sums. Edge-channel product: `product[e,k] = A[s(e),k]*B[d(e),k]`; splitting k produces disjoint output panels. The current support-embedding validator accepts concatenation for the ambiguous contraction kind [S10]. Correct the law and its consumers, not only the comment.

A primitive/composition/effect classification must not carry a valid substitute primitive which a caller can accidentally execute. Preserve the operation's real meaning even when lowering is unavailable. Publication is an effect, chain/exchange/hierarchy/gradient remain explicit compositions, and unsupported lowering returns unsupported [S11]. This is not authorization to implement their full execution in this epic.
