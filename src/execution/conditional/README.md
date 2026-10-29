# Native conditional module dispatch

`Cellerator::conditional` supplies a host sparse dispatch substrate, independent
of Python, Torch, CUDA, model architecture and scheduling policy. A selected
module is an opaque `ExpertKey{expert_id, revision}`; IDs/revisions are uint64
values including zero. An origin is `{rank, instance, row}`. Physical placement
does not change this logical identity or the caller's selected route weight.

`DispatchPlan` owns every caller selection occurrence, including repeats,
zero/signed weights and different revisions of the same expert. It groups exact
keys in deterministic expert-major order and exposes compact contiguous buffers:

| Buffer | Meaning |
|---|---|
| `expert_offsets` | One contiguous dispatch segment per exact expert/revision. |
| `dispatch_to_selection`, `selection_to_dispatch` | Forward/inverse occurrence permutations. |
| `origins` | Unique origin identities in first logical occurrence order. |
| `dispatch_origin_slots`, `dispatch_weights` | Dispatch-order origin-row indices and unchanged weights. |
| `selection_to_origin` | Original occurrence to unique origin slot. |
| `origin_offsets`, `origin_dispatch_indices` | Reverse scatter segments, retaining original logical occurrence order within each origin. |

All counts/indices are host `size_t`; weights are stored as double. Device
runtime owners explicitly validate/convert index and weight precision when
uploading. Expert keys, parameter revisions, origins and original occurrence
indices remain available independently of physical buffers. These buffers
describe selected work; they do not discover candidates or choose placements.

The C++17 templates implement real host execution:

1. `gather_inputs<T>` reads each unique origin once through a caller provider,
   then copies the common row to every dispatch occurrence. Only one row of
   scratch is needed; the provider can read sparse/ragged or external state.
2. `execute<In,Out,Accum,Combined>` invokes a module callback once per exact
   expert key on contiguous rows, then combines weighted outputs by origin.
   Input/output widths can differ. Callbacks get origin and original selection
   indices for provenance or keyed stochastic semantics.
3. `combine<Storage,Accum,Out>` preserves original per-origin summation order,
   even when expert grouping changes execution order. `transpose_combine`
   implements its weighted linear transpose for output adjoints. It does not
   implement expert derivatives or weight gradients.

The callback must fill every output row and must not retain borrowed views.
Equivalence to independent calls requires row-separable module behavior,
including normalization and stochastic semantics. The library cannot make a
batch-dependent callback invariant by changing its physical grouping. There is
no hidden route pruning, expert substitution, transfer, collective operation,
residency cache or placement planner in this host path.

`parameter_store.hh` is the separately owned trainable-store contract linked
into the same native target. `parameter_dispatch.hh` connects it to dispatch:
`execute_parameterized` atomically acquires every exact expert/revision lease
before reading inputs, then passes each callback its pinned
`const ParameterRevision&`. Parameter publication or retirement during one
callback cannot change the revisions seen by the remaining callbacks. An
unavailable required revision fails before any input/provider work; it never
silently refreshes to the latest parameters. This helper is synchronous.
Asynchronous runtime/backward work must retain its own leases until completion.

The optional `cellerator._conditional_native` module exposes keys, origins,
selections, plans and NumPy `gather_inputs`, `combine`, `transpose_combine`, and
`execute`. A plan accepts selection objects or records
`(expert_id, revision, rank, instance, row, weight)`. Origin input arrays use
`plan.origins` order; output arrays use the same origin order. Dispatch arrays
use `plan.dispatch_to_selection` order. NumPy storage is C-contiguous float16,
float32 or float64. Float16 inputs are expanded exactly to float32: callbacks,
multiplication, accumulation and outputs are float32. Float32/float64 preserve
their explicit dtype. This is CPU interoperability, not a half-precision CUDA
claim or a Torch autograd adapter.

The Python callback signature is
`callback(expert_key, input_matrix, origins, selection_indices) -> output_matrix`.
It must return contiguous NumPy output with matching working dtype and declared
output width. Bindings copy buffers, allocate outputs and invoke Python once per
expert segment; no performance claim is made. Native callbacks avoid Python.
`execute_occurrences` has the same invocation/callback signature but returns
unweighted module outputs in original selection order. This allows a shared
execution-engine owner to apply routing weights and combination exactly once.

Independent checks live in `tests/framework/native/conditional_dispatch_test.cc`
and `conditional_binding_test.py`. The native oracle traverses original caller
selections rather than trusting the dispatch mappings. It covers revisions,
rank/instance/row distinctions, repeats, signed/zero weights, permutations,
provider reuse, callback provenance, the transpose dot-product identity,
cancellation-sensitive summation order, invalid shapes/weights and exceptions.
`conditional_dispatch_example.cc` is a runnable generic CPU module example.
`conditional_parameter_execution_test.cc` verifies parameter publication and
retirement during active dispatch, while `parameter_store_test.cc` covers its
store owner's lifecycle and concurrent publication contracts.

```sh
g++ -std=c++17 -Wall -Wextra -Werror -Iinclude \
  tests/framework/native/conditional_dispatch_example.cc \
  src/execution/conditional/dispatch.cc -o /tmp/cellerator_conditional_example
/tmp/cellerator_conditional_example
```

Plan construction allocates/maps/sorts host metadata. Execution allocates
dispatch input/output and origin output vectors. Placement/topology cost,
parameter stores, distributed communication and CUDA execution have separate
owners; this source does not establish their readiness or measured performance.
