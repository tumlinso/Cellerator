# Mathematical, numeric and native contract

## One operation and its derivatives

Let the immutable support be logical edges `e=(destination,source)`. For N16:

```
Y[d,k] = sum(e.dst==d) float(w[e]) * X[e.src,k]
dX[s,k] = sum(e.src==s) float(w[e]) * dY[e.dst,k]
g[e] = sum(k=0..15) X[e.src,k] * dY[e.dst,k]
```

The transpose is an adjoint, not an inverse. The edge VJP is one scalar per logical edge. It is neither an edge-channel product nor a mean over channels. A caller that wants a mean must supply the corresponding scaled cotangent. Splitting K sums partials; splitting disjoint edges concatenates edge outputs. N1 remains the original operation, not a special reinterpretation.

Topology identity, epoch, exact source/destination axes, edge identities/order and edge count remain the shared structural contract. Forward and transpose preserve independent physical projection identities. The gradient output order is a binding/realization fact recoverable through an explicit logical mapping, not a new meaning of the relation. Inputs with the same shape but different biological axes are not equivalent.

## Two explicit operand policies, not an invisible precision change

The inspected FMP1 forward and old N16 transpose use f32 dense inputs; the current sparse/rectangular contraction portfolio takes f16 dense inputs. Directly plumbing these together would silently change the requested VJP. This package therefore chooses:

**Full-f32 edge VJP:** consume X and dY as f32 and accumulate f32. A bounded sparse realization supplies this path. WMMA is ineligible for this policy in this epic.

**Half-rounded edge VJP:** define Q(x)=binary16 nearest-even conversion, then widen for mathematical comparison. The operation is explicitly `g[e]=sum_k float(Q(X[s,k]))*float(Q(dY[d,k]))`, accumulated as permitted by its f32 policy. Both sparse fallback and WMMA use the same rounded operands. The conversion may be hoisted/cached only with valid identity, version, layout and lifetime information. It is not licensed by pointer equality or an undocumented assumption about biological precision.

The second policy is an explicitly mixed-precision approximation to the full-f32 VJP on arbitrary inputs. It must not be advertised as the exact derivative of a rounding function or silently substituted for full-f32 semantics. The demo's X is already half-representable; it shows the quantized-cotangent difference separately. Finite-difference checks evaluate the continuous real-valued model at the stored weights with no rounding of perturbed weights. Half weight-update rounding is tested independently.

The user selected low precision for this bounded work, not a universal scientific claim that all biological datasets have half-precision uncertainty. Numeric behavior remains an explicit model/caller contract.

## Update effects

There is one physical mutable value authority. The supported update descriptions are:

```
delta_add:     next[e] = RNE_f16(float(current[e]) + delta[e])
gradient_step: next[e] = RNE_f16(fma(-alpha, g[e], float(current[e])))
```

The step specifies permitted f32 FMA; changing that policy must be represented and tested rather than hidden. A materialized `delta=-alpha*g` can differ slightly because it rounds an extra time, so compare delta and step according to the precise policies, not an unconditional bitwise promise. Alpha is a finite nonnegative scalar for the bounded convenience. Alpha zero is valid and still advances the explicitly requested generation. For finite gradients it contributes a zero update; signed-zero behavior follows the stated FMA/storage policy. With nonfinite gradients, zero multiplication can propagate NaN, so alpha zero must not be silently optimized into an unconditional value-preserving shortcut. Delta values can have either sign.

Nearest-even conversion, signed zeros, underflow and IEEE overflow/nonfinite behavior are documented. The normal numerical policy propagates nonfinites; metadata validation of a scalar is not a promise to scan all device values. A data-level finite-only guarantee would require separate validation cost and is not silently implemented here.

Update consumes an exact expected current generation and requests a strictly greater nonzero generation. Binding mismatch, generation overflow, malformed axes/order, insufficient capacity, stale gradient stamp and illegal aliasing reject before any kernel. Device or event failure after partial submission poisons the prepared pair; no rollback of in-place memory is promised and no new usable generation is reported.

## Value-plane and buffer contracts

The pair keeps the authoritative f16 weights in a persistent physical order. The initial logical f16 values are uploaded/packed once. The transpose maps refer to that plane. Gradient and delta buffers are caller-owned f32 arrays in the same order and carry structure, epoch, order, count and device tags. Logical-order export is explicit observation work; it is not part of every update.

Inputs and cotangents remain valid through asynchronous use. Rebinding a pointer does not change mathematics, but changing its contents invalidates any cached operand pack. The runtime needs an input identity/version token where it caches content. The simplest correct implementation refreshes every invocation; profitable reuse must prove the token discipline first.

The current duplicate-endpoint limitation remains an explicit unsupported case. Do not silently collapse two biological edges into one parameter. Empty support and isolated axes are supported without illegal zero-size launches. Count and byte products use checked 64-bit arithmetic before any bounded local index is cast to 32 bits.

## Prepared state

`prepared_relation_pair` owns or explicitly accounts for immutable mappings, reusable internal scratch and readiness metadata, while external data buffers remain borrowed. Preparation options cap persistent/scratch memory and report all allocations. Default geometry is the existing physical order; a different authoritative order needs measured justification and retained identity recovery, not a second value authority.

Cold initialization may create events and allocate bounded storage. Steady update/gradient execution may launch kernels, perform required pack refresh and enqueue event dependencies. It must not allocate events or buffers, reconstruct support, globally sort, canonicalize all values or fence the whole device. Separate counters distinguish topology preparations, initial value loads, physical updates, operand pack refreshes, logical exports and each actual provider launch.

The proposed declarations are in `contracts/relation_calculus.hh`, `contracts/relation_update.hh` and `contracts/relation_update_spine_bridge.hh`. These are **not installed headers or implemented APIs** at package delivery. The implementing foundation task may revise them coherently; the demo normal build intentionally requires the new core implementation.
