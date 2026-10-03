# Native structured host state consumer

`Cellerator::structured_state` supplies borrowed state descriptions and views
through `<Cellerator/state/structured_state.hh>`. The C++20 host target owns no
storage, parameters, optimizer, allocator, streams or publication. Owners retain
all spans and generation metadata for each call, serialize concurrent access,
and advance native structure/value/activity/parameter stamps before rebinding.
A generation mismatch rejects saved views before writes.

A scalar patch declares one scalar per actor without a private coordinate
domain. Actor-private state declares ragged offsets and actor-local coordinate
domains. Uniform storage exposes row/column shape, while private coordinate
meaning stays attached to each actor. Ragged storage rejects a uniform shape.
A physical binding may reorder or replicate canonical coordinates; gather
preserves values, and pullback explicitly sums physical cotangents into a
caller-owned canonical destination. Slot incarnations reject recycled storage.

Support carries universe identity, structure/epoch, semantic type and extent.
Measurement detection and dependency support are separate types even if their
members coincide. Structural state bindings use declared primal dependency
support. A zero value keeps its structural membership; activity and capacity
remain separate metadata. Linear readouts and explicitly named scalar
observables preserve actor identity and output roles; nonlinear readouts report
unsupported. Repeated linear terms contribute in caller order.

```sh
python3 -B tests/substrate/state/check.py --build-dir /tmp/ce-is1-state-host
```

The consumer executes the library in both float and double, checks ragged and
scalar examples, independently predicted readouts/replica gradients, native
axis admission, duplicate IDs, support tags/versions/bounds, stale incarnations,
all generation changes, capacity/activity consistency and output preservation
on rejection. Empty state is valid. Neither execution allocates storage.

Only synchronous host calls are provided. CUDA lowering, readout derivatives,
nonlinear readout operators, state migration, arbitrary physical stride views,
scientific meaning and numerical performance remain subsequent tasks. Floating
readouts and pullbacks use ordinary caller-selected float/double arithmetic;
this target provides no precision or conditioning certificate. The strict test
consumer disables fast math and contraction.

Shared integration request: add `src/state` to the chosen root build component
and export `Cellerator::structured_state` through BUILD's granular installed
package. This leaf intentionally changes only its authorized directories.
