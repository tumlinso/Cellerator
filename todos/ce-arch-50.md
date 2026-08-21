# CE-ARCH-50: Forward-compatibility contracts without premature implementation

Status: validated compatibility review

Repository owner: Cellerator. This review consumes the biological execution
contracts, the single Cellerator runtime, the operation core, and the
Cellerator-owned CellPack execution image. It does not assign work to Baseplane
or CellShard and does not add an implementation task for any deferred feature.

## Objective and method

This review tests whether the foundations delivered by CE-ARCH-11,
CE-ARCH-21, and CE-ARCH-22 preserve a versionable route to transpose and
backward execution, multiple numerical policies, CUDA Graph capture,
persistent scheduling, nested partitions, multiple devices, and later GPU
architectures. It distinguishes fields that must be present now from optional
cold sections or prepared state that can be added only when a measured consumer
exists.

Canonical source at review time was Cellerator
`28925ec3b95a1e87d8b5c154821ba4c435367054`. The review used:

- `include/Cellerator/execution/identity.hh`, `operands.hh`, `lifetimes.hh`,
  `execution_order.hh`, and `launch_bindings.hh`;
- `include/Cellerator/runtime/session.cuh` and `runtime/SESSION.md`;
- `include/Cellerator/compute/math/operation_core/operation_core.hh` and
  `operation_core/OPERATION_CORE.md`;
- `components/CellPack/include/CellPack/semantic_geometry.hh`;
- `components/CellPack/include/CellPack/persistence/execution_image_v2.hh`
  and `EXECUTION_IMAGE_V2.md`;
- the focused biological-ABI, operation-core, runtime-session, execution-order,
  and execution-image tests already owned by their foundation workstreams.

No source, public header, CMake file, test, benchmark, persistence record, or
runtime behavior changed in this validation task.

## Compatibility matrix

| Future capability | Required now and already carried | Optional mechanism activated later | ABI break avoided | Disposition |
| --- | --- | --- | --- | --- |
| Transpose and backward | source/destination `axis_identity`, immutable relation identity and epoch, explicit output order, `value_position_map_view::direction`, projection identity | `transpose_value_map` section, `transpose_backward` projection, backward operation kinds and kernels | A logical edge remains independent of projection-local value order and forward-only bytes | Safely deferred after the current identity/order contracts |
| Sparse-value gradients | structure/epoch, `value_plane`, value generation, logical-edge versus projection-local layout, accumulation type | gradient value planes, transpose projection, optimizer/accumulator operation contracts | Gradients can share one immutable relation without mutating structure or adding gradient state to every edge | Safely deferred; no optimizer semantics are frozen |
| fp16 and bf16 with fp32 accumulation | `numeric_type`, `value_numeric_policy`, operation-core `numeric_policy` separating storage, multiply, accumulation, scalar, bias, rounding, and saturation | candidates that advertise and validate exact supported tuples | Storage type is not confused with multiplication or accumulation type | Contract-ready; kernel support remains evidence-gated |
| Later fp8 or integer execution | versioned numeric-policy and candidate capability rejection; projection directory records storage/compute/accumulation types | versioned numeric enum additions, quantization schemas, architecture-specific candidates | No universal fp8 scale format or unsupported combination is frozen into launch records | Safely deferred until a real numerical contract and device candidate exist |
| Module-level quantization | `quantization_kind::per_module`, scale/offset types and pointers, group count, independent value generation | semantic-geometry module index, optional projection-local quantization section, measured candidate | Quantization belongs to a value plane/projection rather than relation identity | Contract-ready; module construction and kernels are deferred |
| CUDA Graph capture | externally supplied stream/workspace, prepared direct dispatch, graph-capture capability flag, graph-stable runtime lifetime and address check | separately versioned graph-stable binding/graph-exec owner | Persisted images contain no process addresses and ordinary prepared operations do not freeze launch pointers | Foundation-ready; capture implementation is deferred |
| Persistent CTA execution and device queues | prepared persistent state, launch-transient workspace, scheduling-summary section, CTA/architecture-specific projection vocabulary | scheduler-specific projection payload, queue state, graph-stable/persistent allocation | Scheduling policy is not embedded in biological identity or every operand | Safely deferred behind measured reuse and residency evidence |
| Nested warp/CTA/GPU/node partitions and boundary edges | partition identity on every axis, semantic geometry identity, independent device location, `device_fleet_view` planning seam | hierarchy/partition index and scheduling-summary sections, boundary-edge extension section, multi-device session/fleet owner | Nested topology stays cold or projection-local instead of expanding every hot axis or tile | Identity-ready; distributed execution is deferred |
| Volta, Ampere, Hopper, and Blackwell dispatch | device performance class is separate from biological identity; projection key and directory carry kind, variant, and architecture class | architecture-specific projection/candidate and measured planner evidence per build/device class | One persisted biological relation is not identified by GPU generation | Volta remains current baseline; later candidates are safely deferred |

The compatibility rule is additive: new optional projection schemas and
sections may be introduced without changing existing projection meanings.
Changes to identity semantics, fixed persistent directory records, or an
existing numeric enumeration require an explicit schema/interface version and
compatibility test.

## Fields required now

The following information is required in the current foundations because
omitting it would force a later semantic or persistence break:

- domain, exact order, semantic geometry, and partition identity per axis;
- immutable structure identity and structure epoch;
- mutable value generation, numeric policy, quantization description, and
  logical-edge or projection-local value layout;
- projection identity and projection catalog ownership;
- explicit output axis/order transition and forward/transpose value-map
  direction;
- device ordinal and residency on launch operands without making device class a
  biological identity;
- separately declared persistent state and transient workspace requirements;
- operation, projection, kernel, build, and device-performance identities in
  planning/dispatch records rather than pointer-derived identities.

The following information is deliberately not required in every hot record:
GPU/node topology, hierarchy depth, boundary-edge lists, graph-exec ownership,
queue heads, architecture-specific tile metadata, gradient optimizer state,
or fp8 scale encodings. Those belong in optional image sections, typed
projection payloads, prepared persistent state, session/fleet objects, or
launch bindings after activation.

## Hot-record budget

The host ABI size probe was compiled with C++17 against the current headers.
These are launch or prebound records, not per-edge or per-tile persistent
metadata:

| Record | Bytes | Frequency and budget decision |
| --- | ---: | --- |
| `axis_identity` | 32 | Per operand axis; already the minimum four 8-byte interned handles |
| `sequence_domain` | 40 | Per sequence chunk/view, not per base or event |
| `biological_operand_view` | 224 | Per launch operand; below its reviewed 256-byte ceiling |
| `relation_structure` | 96 | Per immutable relation binding |
| `value_plane` | 104 | Per mutable numerical plane |
| `value_binding` | 16 | Per launched value plane |
| `output_axis_contract` | 80 | Per prepared output axis, not per output element |
| `prepared_binding_contract` | 72 | Per prepared operation |
| `launch_bindings` | 232 | One aggregate per invocation |
| operation-core `numeric_policy` | 15 | Per prepared candidate/problem |
| operation-core `prepared_operation` | 256 | Per prepared strategy; direct-dispatch record |
| execution-image v2 header | 256 | One cold header per image |
| section directory entry | 64 | One cold record per section |
| projection directory entry | 64 | One cold record per projection |
| `prebound_projection_view_v1` | 128 | Per selected projection; kernel avoids parsing the image directory |

CE-ARCH-50 adds zero bytes to every record. Future partition, training,
architecture, or scheduler needs must first demonstrate why an optional section
or prepared side object is insufficient. Per-edge or per-tile growth requires a
measured end-to-end benefit and its own schema review.

## Transpose and backward

`execution_order.hh` already distinguishes forward and transpose value maps,
and execution image v2 has independent `forward_value_map` and
`transpose_value_map` sections plus transpose-capability flags. The operation
core can select a `transpose_or_backward` projection without redefining the
immutable relation. A backward operation may later add operation semantics and
candidate implementations, but it must reuse the same domain/order/structure
identity and explicitly declare its output order.

Activation requires parity against an independent reference, stale-map and
wrong-order rejection, forward/transpose value-position round trips, and
end-to-end accounting of projection construction and order transforms.

## Sparse-value gradients

Gradient storage is not placed in `relation_structure`. A sparse gradient can
be represented by a separate `value_plane` bound to the same structure/epoch,
with its own generation, numeric policy, quantization policy, and value layout.
Projection-local gradients require explicit reverse mapping to logical-edge
order or a declared projection-local consumer.

Deferred optimizer state, accumulation lifetime, atomic/deterministic policy,
and distributed reduction semantics belong to future operation contracts. They
must not be inferred from pointer equality or silently mutate the structure
epoch.

## Numerical and quantization evolution

Current records can express fp16 or bf16 storage with fp32 multiplication or
accumulation where an implementation advertises that exact combination.
Candidate capability filtering must continue rejecting all tuples that lack a
correct representation or implementation. Later fp8/integer support requires a
versioned numeric value and an explicit scale/rounding/saturation contract; it
does not justify placeholder enum values or universal scale pointers today.

Per-module quantization is already a value-plane granularity. The module index
is semantic-geometry or projection side data. Its bytes, scale-table bytes,
dequantization workspace, conversion traffic, and reuse break-even must be
reported separately before planner activation.

## CUDA Graph capture

The execution session separates graph-stable persistent allocation from
stream-ordered transient workspace, validates stable address ranges, and seals
preparation before launch. The operation core exposes a graph-capture
capability flag while ordinary launch bindings continue to supply pointers,
stream, scalars, and workspace externally.

A future captured instance must be a separately versioned owner of graph-stable
bindings and graph-exec lifetime. Capture must fail closed when a candidate
allocates, discovers descriptors, changes addresses, synchronizes, or queries
device facts during steady-state execution. Graph performance must include
capture/update cost and compare the complete repeated launch train; graph
support is not presumed beneficial for HBM-bound decomposition.

## Persistent CTA and device queues

Persistent scheduling uses typed prepared state and projection-local scheduling
summaries. Queue buffers and counters are plan-persistent, graph-stable, or
launch-transient according to their true lifetime. They are never embedded in
biological identity or the generic operand envelope.

Activation requires a real workload whose launch, synchronization, locality,
or module-skipping benefit pays for occupancy limits, queue traffic, fairness,
drain protocol, and teardown. Stop if a proposal creates a generic queue ABI or
new kernel family without benchmark evidence and a current consumer.

## Nested partitions and boundary edges

`axis_identity::partition` is required now because otherwise identical domains
on different ownership cuts are not interchangeable. Deeper warp, CTA, GPU,
and node nesting belongs in the execution image's optional
`hierarchy_partition_index`, `scheduling_summary`, or a versioned extension
section. Boundary edges likewise belong in a compact optional section or typed
projection payload.

The current runtime is intentionally single-device but exposes device facts and
a fleet view without copying fleet metadata into launch records. Multi-GPU
activation must define ownership, halo/boundary identity, peer/communication
capability, stream and handle ownership, failure semantics, and planner cost.
On the present V100 host, any later topology experiment must account for the
stronger 0-2 and 1-3 peer pairs and use serialized CUDA resource control; this
review creates no distributed task or lease.

## Volta, Ampere, Hopper, and Blackwell

Device performance class and runtime/kernel build identity are planning and
cache dimensions, never biological identities. Execution image projection
entries can carry an architecture class, and the operation registry can expose
architecture-specific candidates while preserving a conventional fallback.

V100 sm_70 remains the current native correctness/performance baseline. Ampere,
Hopper, and Blackwell candidates activate only when hardware and toolchain
evidence exist. New architecture candidates must state supported numeric
policy, projection, workspace, graph, determinism, and preparation contracts;
they do not redefine the semantic geometry or existing image sections.

## Data movement, memory, and resource accounting

Every activated future capability must separate host preparation, projection
construction, H2D, value remapping, dynamic packing, kernel, epilogue, order
transform, synchronization, communication, and D2H. Persistent structure,
plan preprocessing, graph-stable allocation, and launch-transient workspace are
reported independently. Optional image expansion and prebound hot-view bytes
are both reported; compressed inner bytes remain opaque to CellShard.

This validation required CPU compilation and focused host tests only. It
acquired no GPU, benchmark, profiler, sanitizer, dataset, or distributed
resource. Future CUDA validation uses the CUDA skill's controller, clean source
snapshots, serialized benchmark/profile leases, correctness before performance,
and device-class-specific evidence.

## Safely deferred

The following remain deferred without blocking the current ABI: concrete
transpose/backward kernels, gradient/optimizer operations, fp8 formats,
integer/quantized kernels, module discovery, graph-exec ownership, persistent
CTA kernels, device work queues, multi-GPU communication, multi-node failure
semantics, and Ampere/Hopper/Blackwell implementations.

Activation gates are: a current consumer and operation contract; exact scalar
or independent-reference parity; explicit identity/order/lifetime behavior;
complete data-movement and memory accounting; capability rejection and
conventional fallback; adversarial fixtures; and measured end-to-end benefit on
the relevant hardware/reuse regime. If those are absent, the feature remains a
planner capability or optional schema mechanism, not implementation work.

## Stop conditions and ownership seams

Stop and return to architecture review if a future change would:

- put pointer addresses, device class, or build identity into biological
  identity;
- require canonicalization inside an otherwise compatible operation;
- make mutable values or launch pointers part of immutable structure identity;
- mutate CPK1, execution-image v2, or an existing numeric meaning in place;
- add hierarchy, graph, queue, gradient, or architecture fields to every hot
  tile/operand without a measured consumer;
- make `DeviceMathContext` or another runtime island an integration ABI;
- require CellShard to interpret Cellerator projection sections.

Shared ABI headers, root CMake integration, projection schemas, public target
aliases, and integration documentation require serial ownership. Parallel
feature work stops at an interface checkpoint rather than editing those seams.

## Validation evidence

The compatibility review is accepted when all of these commands pass from the
CellStack root:

```text
cmake --build Cellerator/build-ce-arch -j 4 --target celleratorBiologicalAbiHostTest celleratorOperationCoreTest
Cellerator/build-ce-arch/celleratorBiologicalAbiHostTest
Cellerator/build-ce-arch/celleratorOperationCoreTest
c++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I Cellerator/components/CellPack/include -I Cellerator/include Cellerator/components/CellPack/tests/persistence/execution_image_v2_test.cc Cellerator/components/CellPack/src/persistence/execution_image_v2.cc -o Cellerator/build-ce-arch/cellPackExecutionImageV2Test
Cellerator/build-ce-arch/cellPackExecutionImageV2Test
/home/tumlinson/.agents/skills/cpp-context-compiler/scripts/ctxpp --root Cellerator lint include/Cellerator/execution include/Cellerator/runtime include/Cellerator/compute/math/operation_core components/CellPack/include/CellPack/persistence
```

Adversarial compatibility coverage retained from the foundation tests includes
equal shapes with unequal identities/orders, stale structure and value
generations, invalid output order, unsupported numeric combinations, unknown
required versus optional image entries, invalid projection references,
relocation/rebinding, and CPK1 compatibility loading. Later feature-specific
tests remain behind their activation gates; CE-ARCH-50 does not claim they
exist.

Handoff artifact: this matrix. CE-ARCH-60 may use it to audit final ABI and
documentation consistency. No implementation task is created by this review.
