# Alternative host packing strategies

`two_sided` orders destinations by contribution degree, then sources by their
average destination rank and load count. `shared_load_cohorts` groups declared
compatible evaluator/parameter tags and ordered argument shape, then derives
source placement from first use. State and operation placements remain independent.
The strategies invoke the existing PACK validity/ownership contracts, retain every
logical contribution including duplicate pairs, and require caller-registered
physical order identities. Neither algorithm is a Cellpack wrapper.

Both layouts, identity/no-pack, actual Cellpack and a caller callable execute through
the same actual native host relation. Cohort operation order is cold metadata: the
native relation applies source/destination conversion and canonical reduction order;
no fused opcode execution is claimed. The supported fixture has one weighted-relation
evaluator; mixed evaluator execution remains unsupported. Exact tile support counts
nominate dense/sparse regions while counting duplicate numerical contributions
separately. The active numerical route remains native sparse host relation; there is
no new dense kernel, Tensor Core path, invented padded dimension or runtime registry.

`prepare_routes` prepares the native forward and input-VJP relations. The VJP route
reverses endpoints under an explicit distinct response structure identity. Both retain
logical coefficient order, duplicate edges and the existing arithmetic policy. This
is the mathematical input adjoint at caller-supplied coefficients, not a derivative
through floating rounding. Coefficient VJP, JVP, generic process/port packet lowering,
GPU execution and asynchronous stream semantics are unsupported. The lowerer's
existing limitations still apply, including nonidentity contribution schedules.

## Repair and selection

`repair` supports fixed domains/extents and stable operation IDs. It scans old/new
metadata, retains untouched source/destination positions and stable operation order,
repairs touched rows and replaces the full canonical contribution ledger. Removed
and added operations are explicit. It constructs a scratch candidate and performs
full ownership validation; it does not promise sublinear preparation. Changed jobs,
removed/added jobs, declaration order changes and logical contribution-order changes
require a later structure epoch. Declaration reordering can retain every stable job
while invalidating coefficient addressing; old value identities must fail on new
routes. Native immutable routes are prepared again before publication.

`choose` compares continuing with a valid incumbent against preparation, migration,
publication and horizon-weighted forward/input-VJP/JVP/expected-repair cost, plus
hysteresis and a minimum reuse horizon. Incumbent preparation is sunk. Strict
break-even retains the incumbent. Costs must be finite and nonnegative. A stale
incumbent requests mandatory repair regardless of economic advantage; hysteresis
cannot authorize stale topology. Unsupported derivative costs cannot be used to
claim a valid candidate. No benchmark winner is automatically published.

## Native checks

```
python3 -B tests/substrate/strategies/check.py --build-dir /tmp/ce-is1-strategies-host --sdk-prefix /tmp/ce-is1-sdk-a
```

172 checks cover five plans, direct native vector multiplication control, forward,
input VJP/duality, zero primal with nonzero response, duplicates, mixed region
nominations, invalid cohorts/order IDs, fixed-universe repair, added/removed work,
same-epoch reorder rejection, old-value-identity rejection, and horizon guards.
The fixture links the accepted installed PACK/native SDK and compiles only this
new strategy owner; no sibling private source is compiled.

## Integration recipe for the root

The new source target is `Cellerator::packing_strategies` from
`src/packing/strategies/CMakeLists.txt`, depending solely on actual
`Cellerator::packing_strategy`. For the selected SDK module, add the component name
`packing_strategies` to the known components, set its dependency to `packing_strategy`,
and declare `_ce_owner(packing_strategies SOURCES src/packing/strategies/placement.cc
CAPABILITIES "host_two_sided_cohort_fixed_universe_repair,input_vjp_routes")` in the
component dispatch. Request it explicitly when building/installing the next SDK.
The existing public-header install includes `include/Cellerator/packing/strategies`.
No shared CMake or native owner was changed by this leaf.
