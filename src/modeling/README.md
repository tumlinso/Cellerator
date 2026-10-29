# Host sparse modeling substrate

`Cellerator::modeling` is an additive CPU owner built on existing
`Cellerator::support` and `Cellerator::indexing`. It provides owning conveniences
and numerical execution; it does not replace their borrowed native contracts.

## Native model and cell transitions

`<Cellerator/modeling/model.hh>` is header-only C++17 and can be used without
Python or the sparse owners. `StepContext<Parameters, Inputs, Random, Metadata>`
holds time/dt and explicit borrowed invocation pointers. A model owns shared
parameters; context parameters override them for one invocation. Callback
inputs remain caller-owned and cannot be retained past the invocation.

`StepResult<State, Events, Effects, Metadata>` accepts caller-defined state and
event/effect containers. `make_model<State, Context>(transition, parameters,
observer)` creates shared model behavior. A transition reads `(const State&,
const Context&)` and returns a result, or returns a new state to use the default
result type. The default observer returns a const state view. No tensor or
biological type is required. A state value must be move-constructible or
copy-constructible; handles can represent externally owned resources.

`make_cell(model, state, identity, time)` constructs an identified instance.
`cell.advance(dt)` or `cell.advance(context)` returns
`shared_ptr<const Model::result_type>` and adopts its state without copying.
The cell owns time; an explicit result time overrides `time + dt`.
`cell.state_handle()` and retained results pin prior state even across further
steps. Cells are moveable and cannot be copied to duplicate a persistent
identity. `cell.observe(context)` uses the current state/time. The model's
parameter owner may be replaced for subsequent calls without changing cell
state or identity. Input pointers and RNG/metadata are explicit context fields.

Clock validation and result allocation precede adoption; a failed transition
does not replace the cell's state/time/generation. Effects through externally
mutable state handles remain the callback owner's responsibility. Callers
serialize cell advances and shared parameter replacement; there is no scheduler
or implicit concurrent-state synchronization. Native result snapshots refer to
the caller's state representation, so a shared mutable handle remains shared.

The runnable
[`cell_relation_example.cc`](../../tests/framework/native/cell_relation_example.cc)
combines native models/cells with sparse state, overlapping directed
hyperedges, role reductions/scatter, typed events, an observation, explicit
input, and dynamic relation publication. Two cells share behavior/parameters
while keeping separate state and clocks. `model_contract_test.cc` also checks
move-only state and preservation on failed transitions. Release builds keep
these focused checks' assertions enabled.

```sh
g++ -std=c++17 -Wall -Wextra -Werror -Iinclude \
  tests/framework/native/cell_relation_example.cc src/modeling/sparse_relation.cc \
  src/support/logical_support.cc src/support/adaptive_support.cc \
  src/indexing/reverse_postings.cc -o /tmp/cellerator_cell_relation_example
/tmp/cellerator_cell_relation_example
```

## Sparse relations, state and execution

- `SparseRelation`: immutable owning hyperedge snapshot, canonical positive edge
  IDs, ordered/repeated `(role, entity_id)` incidences, explicit relation and
  structure identity/epoch, indexed any/all incidence lookup.
- `DynamicRelation`: staged additions/removals through the adaptive support
  manager; publication advances structure epoch. Removed edge IDs stay retired.
  Old snapshots remain independently usable. A no-change publication is a no-op.
  The existing adaptive owner requires positive participant entity IDs for
  additions, even though immutable logical support and layouts permit zero.
- `ActivityState`: mutable eligibility with its own generation, immutable
  `ActiveWorklist` snapshots, and identity-preserving `rebase` onto a newer
  structure epoch. New-edge eligibility is explicit. Numerical values do not
  determine gates automatically; the caller supplies decisions.
- `EntityLayout`: unique entity IDs in caller physical order, existing typed
  domain/order identities, lookup and explicit cross-order permutation.
- `IncidencePlan`: compiles selected incidences to physical entity slots,
  keeping a snapshot of topology, activity, and layout. Each plan can serve
  arbitrarily many numerical value generations without recompilation.
- `ValueField<T>`: optional owned field, sparse identity-addressed writes,
  conflict/shape validation and a separate value generation. Existing external
  array owners can bypass this container.

`gather`, `joint_reduce`, `scatter_add`, and `joint_reduce_scatter` are C++17
templates with storage, accumulation and output types separated for arithmetic
operations. Joint reductions are sum, mean, product, minimum and maximum.
Role filters make input/output direction explicit. Repeated incidences count
repeatedly, including scatter contributions; an empty role gives sum=0 and
product=1, while mean/min/max reject it. NaN propagates through min/max.
Operations read one common input and produce a fresh output, including scatter.

The optional `cellerator._modeling_native` module exposes the same owners and
NumPy operations, plus `Float32Field`/`Float64Field`. Arrays must be C-contiguous
float32 or float64 with shape `[entity]`/`[entity, feature]` (scatter instead
accepts `[edge]`/`[edge, feature]`). Dtype and rank are preserved. There is no
implicit conversion from FP16, integers, Torch, or device arrays. These are
explicit CPU precision alternatives; native FP16 and device implementations
remain future work. Fields expose value copies so writes cannot silently
bypass generation tracking.

Preparation owns and allocates host vectors. Bindings copy arrays into native
vectors and allocate output arrays; they release the GIL during arithmetic.
`joint_reduce` directly reads entity storage without materializing a gather;
`joint_reduce_scatter` uses one edge-sized intermediate. No performance claim
or automatic layout/backend selection is made. Mutation/publication is
host-serialized by the caller; these containers are not concurrent writers.

`cmake/CelleratorModeling.cmake` declares the native target, adds the optional
Python module to `cellerator_python_extensions`, and registers focused checks
when `CELLERATOR_BUILD_TESTS` is enabled. `CELLERATOR_ENABLE_MODELING` defaults
to the sparse-support option and requires that owner when explicitly enabled.

Focused standalone native check:

```sh
g++ -std=c++17 -Wall -Wextra -Werror -Iinclude \
  tests/framework/native/sparse_modeling_test.cc src/modeling/sparse_relation.cc \
  src/support/logical_support.cc src/support/adaptive_support.cc \
  src/indexing/reverse_postings.cc -o /tmp/cellerator_sparse_modeling_test
/tmp/cellerator_sparse_modeling_test
```

No GPU execution, profiler measurements, autograd or scientific qualification
is part of this host owner.
