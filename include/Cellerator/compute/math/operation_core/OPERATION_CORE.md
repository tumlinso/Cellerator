# Cellerator Operation Core v1

This contract is the reusable operation seam shared by native kernels, vendor
libraries, and composed implementations. It is not a second runtime and it does
not make SpMM the ontology of Cellerator.

`operation_problem` names immutable mathematical semantics and arity.
`structure_key` separates durable structure identity, its runtime-interned
handle, and structure epoch. `projection_key` names concrete physical bytes
without changing biological identity. `numeric_policy` separately declares
sparse, dense, output, multiply, accumulation, scalar, and bias types together
with rounding, saturation, and quantization granularity. `prepare_policy`
contains reuse, determinism, graph, preprocessing, and memory constraints.

`prepared_operation` is a compact direct-dispatch record. It may freeze the
problem, structure epoch, projection, kernel or vendor algorithm, persistent
preprocessing state, transient workspace requirement, and output-axis contract.
The record borrows persistent state whose owner must outlive it. Per-launch
input, output, value, bias, scalar, stream, and transient-workspace bindings are
carried only by `execution::launch_bindings`; changing them never requires
structural preparation. A graph-stable binding instance, if added later, must
be a separately versioned object.

The fixed-capacity `candidate_registry` performs host-side registration and
capability exposure without virtual objects or allocation. Both native and
vendor candidates use the same prepare contract. The prepared hot path calls a
direct function pointer after deterministic binding validation. Discovery,
device queries, descriptor creation, hashing, allocation, workspace growth,
and synchronization are forbidden in that run function.

The operation core consumes `Cellerator::biological_abi` contracts and is
integrated with the sole `Cellerator::runtime` substrate. CE-ARCH-60 retired
the duplicate `DeviceMathContext`, virtual `SpMMBackend`, and bound
`PreparedExecution` implementation. Independent referees, physical-view
validation, the packed dense operand, and the native CPK1 adapter remain as
explicit v1 evidence targets; they are not a second runtime or planner.

Compatibility and evidence policy:

- CP-BP v1 and CPK1 continue through their versioned adapters.
- Retained CP-Math v1 evidence is built only through explicit compatibility
  targets and is not silently wired into `Cellerator::operation_core`.
- Unsupported numeric combinations are rejected before preparation.
- Output order is carried by `execution::output_axis_contract` and is never
  implicitly canonicalized.
- Persistent and transient bytes are declared separately by each candidate;
  later planner evidence accounts for preparation, conversion, launch, order,
  and synchronization independently.

The v1 interface is ready only when the focused target proves that one native
and one vendor candidate coexist, dynamic bindings and streams change without
re-preparation, stale structures and invalid orders are rejected, and no
unsupported numeric contract is advertised.
