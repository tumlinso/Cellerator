# Native cold packing contract

`<Cellerator/packing/strategy.hh>` and `Cellerator::packing_strategy` provide
identity, caller-supplied and actual Cellpack strategies through one validated
cold `propose` call. Callables inject further strategies without a registry.
Metadata owns its order arrays; source declarations are borrowed during calls.
Failed proposals and lowerings preserve existing accepted outputs.

The problem reuses native relation/axis descriptors and indexed mechanism
incidence. Ordered slots, repeated arguments, output roles, assembly owners and
one contribution per `(instance, output slot)` remain explicit. Source,
destination, state, operation and contribution orders are independent exact
permutations. Nonidentity input/output maps require explicit physical order
identities. Structure epochs and domains are validated against the problem.

Cellpack proposal calls its real support preparation, exact occupancy evaluator
and hypothetical cost estimator on caller-provided or planner-produced generic
two-sided geometry. Unique structural coordinates are scored; repeated
numerical contributions retain their complete ownership. Occupancy is exact
for that supplied support and geometry; cost is an explicitly labelled proxy.
The adapter does not claim to optimize supplied geometry or measure latency.
Cellpack's frozen feature-only image is narrower than its generic two-sided
view. No universal frozen-image conversion is introduced.

The host lowerer calls existing `native_numeric::host_relation::prepare`, with
canonical coefficient ordering and explicit source/destination maps. Caller
storage and `convert_rows` make gather/scatter visible. Numeric evaluation
executes the existing native owner directly; no second executor or parameter
master is added. A caller-declared `relation_evaluator` token binds work to the
weighted-relation semantics, and other evaluator identities are rejected.
Nary/state-argument work and nonidentity contribution schedules are represented
but rejected by this narrow lowerer. Output aggregate overwrite is supported;
local contributions use validated overwrite/accumulation ownership.

```sh
python3 -B tests/substrate/packing/check.py --build-dir /tmp/ce-is1-pack-host
```

The native host consumer compares identity, supplied, injected and actual
Cellpack paths (including Cellpack planner-generated geometry) against an
independent weighted-relation oracle. It exercises repeated structural pairs,
independent orders, exact contribution coverage, stale epochs, physical order
admission, duplicate identities/writers, invalid permutations, workspace
limits, aliases, empty work and explicit unsupported mechanisms/projections.
The existing CPK1 N=1 header compiles unchanged; its CUDA execution is not
claimed. That candidate retains its own prebound image, numeric, axis, output
and lifetime contract. Host lowering explicitly declines the CPK1 family.

Shared integration request: add `src/packing/core` once existing
`Cellerator::native_numeric`, `Cellerator::indexed_mechanism` and `cellpack`
targets are available; export the target through BUILD. The standalone fixture
compiles actual host implementations for qualification and does not qualify
installed packages, CUDA, capture, derivatives, native CPK1 execution or
performance. It does not reimplement Cellpack or native arithmetic.
