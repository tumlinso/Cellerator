# CE-MOON-050 numerical mechanisms

`Cellerator::moonshot_mechanisms` exports `<ce_moon/mechanisms.hpp>` in
`ce_moon::mechanisms`. This experimental header is self-contained C++17.
`Matrix(rows,cols,data)` uses row-major FP64 values. These axes and IDs carry
caller-declared numerical meaning; Baseplane supplies sequence interpretation.
Host reference operators return owning vectors. The bounded tuple interface and
CUDA representatives use caller-owned output buffers. CUDA launches, streams,
allocation, capacity and graph validation belong to the caller.

| Card | Entry points and shapes | Executed example |
|---|---|---|
| E41 | `condense_ports(App[p,p],Api[p,i],Aip[i,p],Aii[i,i],bp[p],bi[i])`; `solve_ports`, `reconstruct_interior`; `compose_port_system` sums compatible responses | `[[4,-1],[-1,3]] x=[1,2]`: port `5/11`, interior `9/11`, matches full solve |
| E42 | `coarse_correct(A[n,n],P[n,c],R[c,n],x[n],b[n],alpha)`; `fit_step_size` | Coupled two-variable system: residual loss `1 -> 0`; fitted alpha `1`; bad restriction retains loss `1` |
| E43 | `join_factors(role0,role1,role2,weights[4],output,capacity)`; `aggregate_factors` | Equal opaque keys retain two of 27 tuples with scores `30,36`; factorized sum matches expansion including duplicate postings |
| E44 | `evaluate_worlds(dag,world_deltas)`; `evaluate_independent`; `exact_query_groups(states,query_nodes)` | Five-node graph outputs `5,9,5`; seven shared evaluations versus fifteen independent evaluations |

E41 eliminates interiors using pivoted solves rather than a materialized inverse.
`PortResponse` retains interior reconstruction coefficients. Port composition
requires identical caller-declared port order; it sums response forces and loads.
The singular/near-singular pivot check uses `1e-12` times the largest matrix entry.
It is not a conditioning certificate. Responses are exact only for the specified
linear model, within floating arithmetic; no learned-port basis is implemented.

E42 makes one correction `x += alpha P (R A P)^-1 R (b-Ax)` per call. The explicit
objective is `0.5 ||b-Ax||^2`. `fit_step_size` fits its single trainable scalar
response parameter by the quadratic optimum. A central finite difference checks
its gradient. P and R are fixed caller-supplied maps in this experiment; fitting
those maps is a later variant. A poor, invertible restriction produces no
improvement; zero restriction fails the solve; an excessive step increases loss.
These examples establish neither general convergence nor learned hierarchy.

E43 defines caller-declared role compatibility by exact equal posting keys.
The score is `w0+w1*a+w2*b+w3*c`. `JoinResult` reports required cardinality,
written count and overflow, including a zero-capacity count pass. Enumeration can
still explode. `aggregate_factors` computes the linear score sum from role
counts/sums without tuple expansion, retaining the original input postings as the
factorized provenance. It does not support arbitrary nonlinear factor scores or
infer a biological relation.

E44 requires a topologically ordered immutable affine numerical DAG. A world
delta replaces a node's numeric value; it does not alter the graph. Descendants
are recomputed only while exact state divergence remains. Query equality labels
world outputs without discarding interior states or changing execution under a
different query. No approximate merging or causal interpretation is provided.
This host implementation copies baseline state per world; its scalar evaluation
count does not account for copying, allocations or GPU traffic.

Reusable support: E11 port-load sensitivity has an analytic/finite-difference
fixture. `residual` and `objective` provide E25/E28 numerical diagnostics, without
absence certificates or sequence queue semantics. E45 `bilinear_response` is a
unit-square arithmetic oracle: a bilinear fixture is exact, while a hard step at
x=0.49 has error 0.49. It establishes no texture interpolation precision.

`cuda_probe.cu` provides FP64 sm70 representatives for one-port condensation,
coarse prolongation, selected factor scoring and one-thread-per-world affine DAG
execution. These are compile-tested experimental kernels. The CUDA port kernel
uses an absolute scalar pivot threshold; the general host solver uses a relative
threshold. CUDA graph inputs must already satisfy the host shape/topology checks.
The world CUDA path independently evaluates each world; shared execution is the
host experiment. No GPU comparison or throughput result is claimed here.

Build the family independently or through the parent experimental CMake tree:

```sh
cmake -S experiments/baseplane_moonshot/families/mechanisms -B /tmp/ce-mechanisms \
  -DCE_MOON_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/bin/nvcc
cmake --build /tmp/ce-mechanisms -j1
ctest --test-dir /tmp/ce-mechanisms -V
```

Campaign builds were run through `/tmp/moonshot_build_slot.py` to share four
single-job build slots. `receipts/provider.json` identifies source digests, examples,
checks and the compiled/run boundary. The root controller owns integration and
workflow acceptance.
