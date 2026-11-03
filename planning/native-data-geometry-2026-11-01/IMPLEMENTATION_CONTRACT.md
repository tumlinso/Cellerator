# Native cellular-geometry data implementation

Implementation contract, 2026-11-02. The user authorized implementation,
commit/push and multiple principal owners, and explicitly waived the task-handle
requirement while Project Control's stale adapter is repaired. Todo task
`CE-NDG-IMPLEMENT` is registered. This waiver does not waive GPU admission,
correctness, evidence or preservation requirements.

The governing [cellular geometry definition](../../cellular_geometry.md) is
protected and must remain byte-identical. Its initial SHA-256 is
`d37f504e5c1ab629334f4127dcaa60d008afe94e560d41aa936d8ba161124254`.
The [reviewed scope](SCOPE.md) and
[design](DESIGN.md) supply the data
boundaries; this contract supplies the executable numerical and qualification cut.

## Operation and numeric contract

For one logical sample, ordered input incidences I_m and scaled output
incidences (i,s_mi), compute

```
u_m = z[c_m,0] + phi[k_m] * z[c_m,1] - tau[k_m]
y_i = -0.25*x_i + sum_m s_mi*theta[k_m]*gate(u_m)*product(x[r] for r in I_m)
gate(u) = 0 if u <= 0, otherwise u*u
```

The inactive case is an explicit operation-defined branch: it does not read
the mechanism's participant values and contributes zero local first-order
VJPs. It does not remove the independently defined baseline. Generic realized
numerical zeros do not authorize this shortcut. Preserve input multiplicity,
semantic slot order, overlapping outputs and shared parameter reduction.
Use the existing division-free product VJP construction so zeros/repeats are
correct. First-order input, condition and theta/phi/tau VJPs are required;
higher-order derivatives are unsupported by this first operation contract.

Use FP16 expression/state storage by default, FP32 regulators, FP32
multiply/accumulation for this product operation, FP32 parameters and
output/gradient buffers, and an explicit FP32 expression-storage alternative.
Reference comparison quantizes expression x identically and retains FP32 z;
the storage conversion is declared separately. Require finite active-path
inputs in the bounded numerical fixture; test inactive short-circuit and
generic-zero counterexamples separately. Exact dyadic/rational fixtures must
match their representable expected results; expanded numerical cases use
predeclared absolute/relative tolerances, not adjusted by candidate
implementations. Before the first GPU run, root fixed `atol=2e-5, rtol=2e-5`
for every output/VJP/updated-parameter/next-forward field on the small fixture
and both timing workloads, with hard finite, inactive-zero and baseline checks.
The independent verifier reviews the validator and reports per-field maximum
absolute/relative errors. Existing tighter host checks remain unchanged.

## Representation and distributed cut

Start with immutable block-masked structure plus bulk mutable value instances,
canonical recovery maps and explicit retained-content read leases. Allocated
masks describe valid coordinate slots, not numerical nonzeros. A stored value
may become zero without changing structure. A required unallocated or missing
remote coordinate is an admission error.

The initial distributed fixture uses two V100/SM70 devices on one host. State
and canonical parameters have declared home devices; regulator values have a
different home. Split overlapping mechanisms across devices, including a
shared parameter and participants. Both devices consume specialist views
directly. Explicit packed transport messages are permitted; a whole-state
dense decode does not establish the proposed direct representation path.

Retain actual forward contents through backward, not only allocations and
generation counters. Derive transfers from separate condition/primal/VJP
demands. Begin with correct staged transfer and prepared per-device operations;
qualify direct transfer only on a capable admitted pair. Compare the reduced
gradient, one deterministic parameter update, replica generation refresh and
the next forward with the same logical single-device reference.

## Owners and first implementation steps

- Semantic principal: canonical IDs, applicability/pattern/read contracts,
  exact omission, operation meaning and host exemplar; Luna migrates established
  support/index/identity callers.
- Representation principal: specialist views/owners, maps, retained contents
  and direct local kernels; Luna extracts resources, codecs/codebooks and
  implements bounded physical candidates.
- Distributed principal: placed bindings, transfer/readiness/tapes, reductions,
  updates and the connected two-device operation.
- Luna integrator owns shared CMake/install/export wiring; the independent
  verifier owns reference fixtures. Root owns cross-cut acceptance and Git.

First compile host contracts, then connect direct local execution and the
distributed spine while migration proceeds. Keep Torch/CUDA out of semantic
headers. Conventional dense/CSR/COO use existing libraries or caller storage;
there is no replacement tensor/autograd engine or mandatory Cell superclass.
Preserve the existing dirty Cell/model redesign and integrate it deliberately.

## Reference, baseline and informative alternatives

Independent Fraction reference:
[`tests/native_data/reference/exact_oracle.py`](../../tests/native_data/reference/exact_oracle.py). Its active fixture, repeated/zero
incidences, inactive branch, shared reductions and one-update expectations
already passed before implementation. Existing indexed-mechanism input saving,
division-free VJPs and reader/write guards are reusable source precedents.

The baseline must compute the same operation and derivative request using a
conventional/native or library representation. Compare direct implicit reads
with maintained projections, condition-first transport with replication, and
fused reduction/scatter with the existing intermediate-producing path where
these are compatible. Charge preparation, maintenance, memory, communication,
fences, forward/backward and canonical output conversion. No speedup is assumed.

## Validation and evaluation

Host build entry point (integrator owns exact target registration):

```sh
cmake -S . -B /tmp/cellerator-native-data-host \
  -DCELLERATOR_ENABLE_CUDA=OFF -DCELLERATOR_ENABLE_PYTHON=OFF \
  -DCELLERATOR_ENABLE_TORCH=OFF -DCELLERATOR_ENABLE_NN=OFF \
  -DCELLERATOR_ENABLE_CELLSHARD=OFF -DCELLERATOR_BUILD_TESTS=OFF \
  -DCELLERATOR_BUILD_NATIVE_DATA_TESTS=ON
cmake --build /tmp/cellerator-native-data-host --target cellerator_native_data
ctest --test-dir /tmp/cellerator-native-data-host -R '^native_data_' --output-on-failure
```

CUDA builds use SDK 12.9, g++-12 and SM70 with hardware probing disabled during
configuration. GPU correctness, sanitizers and timing run only in finite
Scheduler-admitted jobs on refreshed UUID-selected resources. Root coordinates
the shared interval; no unmanaged CUDA launches. Registered measurements use
Autobench snapshots and depend on independent correctness. Initial correctness
jobs have a 120-second timeout; timing packets have a 120-second envelope;
sanitizer/profiling packets are separate and bounded at 600 seconds. Expand a
budget only for a concrete diagnostic need and preserve all receipts/cleanup.

Promotion requires meaningful host/installed-consumer checks, same-operation
two-device output/VJP/update correctness, direct specialist consumption and
valid ownership/publication. Performance promotion additionally requires
complete-cost evidence against the strongest compatible baseline. Keep useful
counterexamples and alternatives when a specialist path loses. Neither a host
simulation nor two disconnected device computations closes the milestone.

## Qualification execution policy

The Scheduler orders visible GPU UUIDs lexicographically. The selected NVLink
pair is visible as `6c1cac7f...` then `cf22c41f...`; logical owner slots retain
the opposite order by explicit device ordinals `[1,0]`. Jobs do not override
Scheduler visibility. Placement and numerical identity are independent.

Correctness-only Scheduler jobs allow at most 5 percent observed utilization
with three process-free samples, positive memory accounting and an exclusive
grant. This accommodates observed 0-3 percent idle fluctuations. The installed
Autobench interface uses its stricter supported 0 percent/two-sample gate.
Numerical tolerances and finite-value checks do not change. Timing additionally
requires three zero-utilization process-free observations before each backend.
Failed admission/startup attempts are retained as such and do not count as
numerical results. The adapter's maximum observation timeout is 60 seconds.

Autobench can snapshot externally built artifacts through `data.paths`.
Conventional and specialist artifacts are built from immutable source snapshots,
with compiler/source/binary manifests checked in their execution snapshot.
These are externally built snapshot artifacts, not claimed Autobench build
results. Ordinary local CPU compilation is preparation; every CUDA execution
remains supervised by the Scheduler.
