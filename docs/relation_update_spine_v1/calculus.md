# Relation calculus v1

For each immutable logical edge e=(destination,source), forward computes
Y[d,k]=sum_e w[e] X[s,k], transpose computes dX[s,k]=sum_e w[e] dY[d,k],
and the edge VJP overwrites g[e]=sum_k X[s,k] dY[d,k]. Transpose is the
adjoint, not an inverse. Channel partitions sum scalar partials; only disjoint
edge partitions concatenate. There is no mean normalization or edge-channel
output. A mean objective supplies an appropriately scaled cotangent.

The descriptor owns both operation descriptors and every mathematical policy.
Forward and transpose must share topology, epoch, exact axes (including domain,
order, geometry, partition and extent), edge order/count, width and arithmetic.
Their orientations are opposite; output policy may independently overwrite or
accumulate. N1 retains its existing semantics. This bounded calculus supports
N1 and N16, f16 weights, f32 dense inputs and f32 accumulation/output, with
explicit FMA and reassociation permission. Providers still check capability.

Full-f32 gradient consumes f32 X and dY. The half-rounded profile computes
sum_k float(Q(X[s,k])) float(Q(dY[d,k])), where Q is binary16 nearest-even.
Sparse and WMMA must implement the same requested profile. WMMA cannot silently
replace the full-f32 profile. The gradient is scalar f32, overwrite-only, with
explicit FMA/reassociation permissions and propagated nonfinites. It describes
the continuous real-valued VJP at stored weights, or its explicitly rounded
operand approximation; it is not a derivative of storage rounding. Finite
differences perturb real weights without rounding; update rounding has separate
tests. Signed zeros, subnormals, overflow and nonfinites follow IEEE arithmetic;
metadata validation does not promise to scan device data for finiteness.

Delta add stores RNE_f16(float(w)+delta). Gradient step stores
RNE_f16(fma(-alpha,g,float(w))), with finite nonnegative alpha. Alpha zero still
advances generation and must preserve nonfinite propagation (0*NaN is NaN).
Materializing delta=-alpha*g introduces an additional f32 rounding and is not
promised bitwise equal to the FMA step.

Mathematical equivalence compares all fields, never byte padding, compiler
origin, candidate, pointer, stream, current generation or a version token.
Unknown orientations, arithmetic/update enums, edge channels, numeric policies,
and mismatching biological identities reject explicitly before dispatch.

The separate value-owned effect witness has at most 16 identified stages.
Each stage names predecessor index bits and explicit generation reads/writes.
Old-generation forward/transpose/gradient consumers must transitively precede
an update; publication must depend on that update and precede new-generation
consumers. Stage list order alone does not establish these dependencies.
Generations increase strictly without wraparound. A publication is an ordered
effect witness; this host validation never claims actual GPU readiness.

C01 establishes the declaration and mathematical contract. C03 supplies the
shared validator and adversarial host test. Native/runtime execution and sm70
acceptance remain separate tasks; this header alone proves no device capability.
