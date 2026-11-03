# Native data structures for cellular geometry

Historical planning/review record. See [the realized implementation](IMPLEMENTATION.md) and [actual type map](implemented-type-map.csv) for the realized cutover and its evidence.

Status: concrete design proposal, 2026-11-01. No implementation or migration is
authorized or performed by this document. C++ declarations below are design
sketches, not compiled headers. Proposed symbols are not existing APIs unless
explicitly marked as reused.

Source inspected at `b6f1a812c369c9d60fa45eb3c3df1c12f2d5651e`,
including the current uncommitted typed Cell redesign. The reviewed
[scope](SCOPE.md) records the user's clean-API direction, explicit format
migration policy and required multi-device first milestone.
The governing [cellular geometry definition](../../cellular_geometry.md) is
protected. Its inspected SHA-256 is
`d37f504e5c1ab629334f4127dcaa60d008afe94e560d41aa936d8ba161124254`.
This plan does not amend, supersede, or grant permission to change it.

Companion artifacts:

- [Migration sequence and acceptance gates](MIGRATION.md).
- [Reviewed Cell integration and cleanup scope](SCOPE.md).
- [Source-to-destination type map](type-map.csv). Destination headers and symbols
  in that CSV are proposed, not claims that files have already moved.

## 1. Decision

Make reusable dependency knowledge independent of both a particular learned
model and a particular numerical representation. Bind that knowledge to the
actual model operations, then specialize their physical access and execution.
This is composition between independently useful contracts, not a mandatory
pipeline, model superclass, tensor facade, or compiler IR.

The new semantic home is `cellerator::geometry::dependency`. The explicit
subnamespace avoids confusing cellular dependency restrictions with the current
work-regrouping constraints in `geometry/admissibility.hh` or CellPack's physical
row/feature geometry. Existing objects keep their meanings during migration.

Specialist numerical formats have a coherent public home in
`cellerator::tensor`. General computational containers have a separate home in
`cellerator::containers`. Neither implements a replacement for ATen. Ordinary
dense/CSR tensors use existing implementations; native pointer-based vendor
interop remains available without an ATen owner.

The migration is a coordinated source cutover. Old names, forwarding headers
and ABI identities need not survive. Persisted encodings may change through
explicit versioned conversion; do not let old record layouts determine the
new native API. Preserve logical/numerical meaning, lifetimes and source-bound
evidence rather than obsolete wrappers. Update in-tree callers with the move;
external consumers are not a release gate for this work.

## 2. Public boundaries and dependencies

| Namespace | Owns | Does not own |
|---|---|---|
| `structure` | Persistent semantic IDs, domains, logical axes and occurrence identity | Physical slots, device ordinals, runtime registry generations |
| `geometry::dependency` | Applicability, possible participant patterns, declared influence/read/condition restrictions | Condition arithmetic, learned weights, current eligibility, allocation |
| `containers` | Generic segmented arrays, index sets, posting indexes, bitsets, bounded work storage | Biological exclusion, model assumptions, execution decisions |
| `tensor` | Concrete specialist value formats, their owners, typed views and format-specific encoding metadata | Universal arithmetic/autograd, scientific interpretation, scheduling |
| `layout` | Reusable order/permutation maps, physical projections, local indexing and gradient accumulation maps | Model-defined admissibility, current value generation |
| `memory` | Placement-aware borrowing and allocation ownership, suballocation and retention contracts | Numerical tensor semantics, model lifecycle |
| `ops` | Actual mechanisms/conditions, declared access/effects, derivative and omission contracts | A compulsory module registry or model superclass |
| `execution` | Prepared bindings/indexes, current work, tapes, generations, readiness and device realization | Authority to invent or remove model dependencies |

`nn`, `modeling`, `state`, and `Cell` compose these facilities. They remain
independently useful surfaces; a tensor or geometry pattern never requires a
`Cell`. The existing host-only structured-state semantics remain in `state`.
Gene vocabulary and biological annotation adapters remain in `biology`.

The current typed `Cell<Model, Instance, ModelHandle>` composes a model handle
and an instance value; its `model()` accessor exposes an ordinary callable.
The optional `CellViews` runtime bridge is separate. An instance can expose a
tensor view or a selection into pooled storage, but no tensor needs to inherit
Cell access. Prepared sources obtain semantic metadata, generations and
readiness from their actual owners. Cell calls and direct instance mutation do
not automatically publish revisions; equal row counts or borrowed identity
views do not establish prepared-source compatibility. See SCOPE section 2 for
the concrete integration tests and ownership boundaries.

Recommended dependency direction:

```text
structure <- geometry::dependency
containers <- geometry owners / tensor / layout / execution
memory <- native tensor owners / execution resource bindings
geometry + model operation access -> prepared execution bindings
tensor views + layout + bound values -> numerical kernels
```

These are dependency relationships, not required runtime stages. Host semantic
headers use C++20 where spans/concepts help; kernel-facing records remain small
CUDA-compatible PODs without host STL ownership. Optional host modules continue
to wrap canonical headers; NVCC is not required to parse modules.

## 3. Reuse the execution architecture

Execution already contains important parts of the proposed design. They should
be extended or adapted, not replaced by another tensor/runtime framework.

| Existing contract | Design consequence |
|---|---|
| [`logical_coverage_view_v1`](../../include/Cellerator/execution/joint_compiler/logical_coverage_v1.hh) | Coverage can be intervals, IDs, components, segments, unions or provider-defined. Reuse this vocabulary and adapters; its exact-coverage role is not a biological exclusion certificate. |
| [`hierarchical_index_space_view_v1`](../../include/Cellerator/execution/index_space/hierarchical_index_space_v1.hh) | Global logical size need not equal one contiguous allocation or one kernel-local index width. Retain compact local indexing with explicit recovery maps. |
| [`multi_atom_port_binding_v1` / `multi_extent_physical_binding_v1`](../../include/Cellerator/execution/object_binding/multi_extent_v1.hh) | One logical port can span several extents. Consume them directly when supported; assemble only when a consumer actually requires contiguity. |
| [`atom_requirement_v1`](../../include/Cellerator/execution/joint_compiler/atom_requirement_v1.hh) | Separate requested coverage/order/numeric/extent capabilities from addresses and actual device placement. |
| [`structural_atom_plane_binding_v1`](../../include/Cellerator/execution/atom_plane/structural_plane_binding_v1.hh) and [`relation_value_atom_plane_v1`](../../include/Cellerator/execution/atom_plane/relation_value_plane_v1.hh) | Immutable structure and mutable values have different owners and lifetimes. Equal shape or pointer equality cannot establish association. |
| [`projection_value_plane_v1`](../../include/Cellerator/execution/projection_value_plane/value_plane_v1.hh) | Keep independent value/gradient components and logical-edge mappings; changing a value generation does not rebuild structure. |
| [`prepared_binding_contract` / `launch_bindings`](../../include/Cellerator/execution/launch_bindings.hh) | Preparation excludes current launch pointers/stream. Preserve output effects, explicit workspace and generation checks. |
| [`prepared_atom_fragment_v1`](../../include/Cellerator/execution/atom_fragment/prepared_atom_fragment_v1.hh) | Bind useful local candidates to existing prepared programs; do not mandate a second executable graph. |
| [`external_atom_plane_descriptor_v1`](../../include/Cellerator/execution/atom_plane/external_plane_mapping_v1.hh) and [`atom_ready_lease_binding_v1`](../../include/Cellerator/execution/atom_plane/ready_lease_binding_v1.hh) | Borrow external storage and retain provider-defined readiness/leases explicitly. A view is not an owner. |

Those records are versioned, bounded contracts. For example, the current logical
coverage record requires exact, nonempty coverage and source/destination axes.
It must not be silently generalized to partial knowledge, empty support, or
arbitrary n-ary roles. New semantic selections adapt to it only when its
requirements are satisfied; otherwise introduce a separately versioned adapter.

The [native-value capability record](../../include/Cellerator/execution/native_value_instance/CAPABILITY.md)
provides a concrete shared-structure/private-values precedent, but its stated
single-device, widths, numeric and borrowing limits remain unchanged. Source
presence does not qualify general multi-device or arbitrary-width execution.

## 4. Logical identity and applicability

Move persistent semantic ID definitions to their canonical structure owner and
update in-tree users. Do not move all of `execution/identity.hh` into structure:
runtime handles, residency, performance class and registry generations remain
execution concerns. Split composite axes by meaning; serialized schemas retain
explicit decoding/conversion rather than fixing the public C++ struct layout.

Proposed host semantic declarations:

```cpp
namespace cellerator::structure {
// Rehome the existing persistent_identity<Tag> implementation here.
struct LogicalAxis {
    domain_id domain;
    order_id logical_order;
    std::uint64_t extent;
};
}

namespace cellerator::geometry::dependency {
enum class Knowledge : std::uint8_t {
    established_restriction, model_restriction, prior, optimization_hint
};
enum class Coverage : std::uint8_t {
    complete_envelope, partial_knowledge
};

struct Applicability {
    regime_id regime;
    coordinate_system_id coordinates;
    assumption_set_id assumptions;
    parameter_family_id parameter_family;
    structure::structure_epoch epoch;
};

enum class SelectionSpace : std::uint8_t {
    coordinates, occurrences, mechanisms, contributions, derivative_coordinates
};
struct SelectionRef {
    selection_id identity;
    structure::domain_id domain;
    SelectionSpace space;
};
struct SourceRef {
    source_id identity;
    structure::domain_id domain;
    representation_id representation;
};
struct ReadDependency {
    SourceRef source;
    SelectionRef selection;
};
struct ConditionReads {
    condition_id identity;
    std::span<const ReadDependency> inputs;
};
}
```

The lowercase ID names above are new typed persistent IDs in the proposed
dependency namespace, not aliases that accidentally make regimes, sources,
parameters and mechanisms interchangeable. `LogicalAxis` is a new semantic
record; it does not reinterpret existing `persistent_axis_identity.order`.
Explicit adapters bind logical order to physical order, geometry and partition.

Applicability describes a family in which a restriction holds. A parameter
update within that family does not necessarily invalidate it. Crossing an
assumption/family/regime boundary does. Scientific truth is supplied by the
model/consumer; validators establish record consistency and declared scope,
not the truth of a biological assertion.

The foundation may be shared across compatible models. Binding maps its sources
to model-specific fields, learned representations, histories or context
providers. Shared biological IDs do not imply identical latent coordinates,
weights or sufficient observations. Logical metadata is held in bulk and is
not replicated in every scalar or tile.

## 5. Possible dependencies without compulsory enumeration

Provide concrete pattern families rather than a universal stored graph:

```cpp
namespace cellerator::geometry::dependency {
struct EntityRef {
    structure::domain_id domain;
    structure::entity_id entity;
};
struct OccurrenceRef {
    structure::domain_id domain;
    structure::occurrence_id occurrence;
};
template<class Reference>
struct Participant {
    role_id role;
    Reference reference;
    std::uint64_t incidence_slot;
};
struct ExplicitIncidenceView; // Adapter over logical_support_view_v1.
struct JointGroups;           // Segmented ordered participant occurrences.
struct CartesianPattern;      // Role selections; no pair array required.
template<class... Patterns> struct UnionPattern;

template<class Pattern>
struct FoundationView {
    foundation_id identity;
    Applicability applicability;
    Coverage coverage;
    Knowledge authority;
    Pattern possible;
    std::span<const ConditionReads> condition_reads;
};
}
```

`ExplicitIncidenceView` retains ordered and repeated participants. The current
compact `(role, entity_id)` records can be reused with a separate role-to-domain
binding; changing their serialized layout is unnecessary. Participant role
domains are not inferred from position. `entity_id` and `occurrence_id` are
proposed distinct domain-local ID types, not physical indices. Repeated
incidences of one entity have distinct logical slots; actual event occurrences
retain their own occurrence IDs and an optional separate entity association.
Mixed-role patterns may use typed tuples or an explicit host reference tag;
homogeneous patterns need no per-element variant. Slots can remain implicit in
ordered arrays. These cold declarations lower to the existing compact records
where applicable.

Empty support, overlapping groups, feedback, multiple outputs and distinct
mechanisms sharing endpoints have explicit meanings. Input-free constants use
ordinary zero-input operations with declared outputs, not a zero-participant
edge passed to the v1 support validator, which rejects that case. A future
explicit zero-arity incidence encoding requires its own versioned adapter.
Union deduplication is by mechanism identity when
appropriate, never by equal endpoints alone.

Built-in patterns expose batched queries/traversal and operation-specific
lowering. Custom C++ pattern types participate through overloads/traits. A
pattern is not required to enumerate every candidate before preparation;
`CartesianPattern` can lower directly to a grouped dense operation. A coarse
runtime provider extension may dispatch one batch through an erased function
table, but no per-gene virtual call is required in the hot path. Unsupported
specializations use a compatible general operation or report unsupported scope.

The model still defines the actual operation family. Partial knowledge means
unknown dependencies cannot justify removing work from that family; it does
not authorize Cellerator to invent additional mechanisms. Omission outside a
complete model envelope is different from asserting natural impossibility.

Existing execution coverage is reused for concrete access requirements and
result association. Add pattern-query/lowering semantics and partial-knowledge
scope beside it; an opaque `provider_defined` payload alone does not implement
those semantics.

## 6. Operations, conditions and permission to omit work

Keep numerical implementations as ordinary typed C++ functions/callables. Reuse
the existing indexed-mechanism argument/output/evaluator/VJP contracts. The
model owns condition arithmetic and parameters; geometry declares the relevant
sources. No boolean-expression language is mandatory for learned conditions.

```cpp
namespace cellerator::ops {
enum class ConditionOutcome : std::uint8_t { excluded, eligible, unresolved };
enum class Omission : std::uint32_t {
    candidate = 1u << 0,
    input_read = 1u << 1,
    forward_contribution = 1u << 2,
    input_vjp = 1u << 3,
    condition_vjp = 1u << 4,
    parameter_vjp = 1u << 5,
    jvp = 1u << 6
};
struct DerivativeRequest {
    bool inputs, conditions, parameters, jvp;
    std::uint32_t order;
};
struct AccessContract {
    access_contract_id identity;
    std::uint64_t revision;
    operation_id operation;
    std::span<const geometry::dependency::ReadDependency> value_reads;
    std::span<const geometry::dependency::ConditionReads> conditions;
    std::span<const OutputAccess> writes; // Selection + existing output effect.
};
struct OmissionRule {
    rule_id identity;
    operation_id operation;
    access_contract_id access;
    std::uint64_t access_revision;
    geometry::dependency::condition_id condition;
    ConditionOutcome required_outcome;
    geometry::dependency::SelectionRef scope;
    OmissionMask permitted;
    std::uint32_t maximum_derivative_order; // 0: primal omissions only.
    geometry::dependency::Applicability applicability;
    NumericSemantics numeric_semantics;
};
}
```

`OmissionMask` is a bitmask of the declared `Omission` values; `OutputAccess`
reuses `execution::output_effect_contract` through an adapter. New IDs and
helper types in this sketch are declarations to add, not hidden existing APIs.
The rule's operation/access revision, triggering condition outcome and certified
derivative kinds/order must match the binding and request; any mismatch retains
work or reports an unsupported exact shortcut. The rule's scope identifies the
candidate, input, contribution or derivative slice affected. `SelectionSpace`
selects a checked descriptor: mechanism selections name a family and candidate
subset; contribution selections also identify output slots; derivative
selections additionally identify differentiated sources/roles and coordinates.
Coordinate and occurrence selections retain their own domain/order association.
The foundation/binding retains those descriptors; a selection ID is not a
permission to infer their meaning from a count. Concrete typed selections can
also be passed directly without a mandatory global registry.

A direct-dependency exclusion alone never implies a zero
whole output. No metadata rule is emitted for each pair if a family-level rule
suffices.

Apply omission independently to each requested derivative. A first-order VJP
rule does not certify higher-order derivatives. A realized zero learned gate
does not establish a zero derivative to gate parameters. The numerical contract
also matters: skipping `0 * NaN` changes IEEE results unless the operation's
declared semantics permit that shortcut. Exact exclusion, a model restriction,
and approximate sampling have separate execution modes and evidence.

Existing binary condition programs remain a useful implementation for supplied
predicate values. Their false result may suppress a whole block only where the
bound operation contract permits it. An unresolved condition is not exclusion.
Condition gradients may use a native derivative or existing ATen autograd;
the new geometry layer does not become an autodiff engine.

## 7. Specialist tensor ownership and physical views

Each format has a concrete view and a separate owner where ownership is useful.
Format-specific layout metadata stays with the format; reusable permutation,
projection and accumulation maps live in `layout`. There is no
`Tensor<AnyLayout, AnyBackend>` facade and no requirement to subclass ATen.

For a CellPack-like packed representation, the proposed split is:

```cpp
namespace cellerator::tensor::block_masked {
template<class Offset = std::uint32_t>
struct LayoutView {
    const Offset* tile_block_offsets;          // tiles + 1; descriptor offsets.
    const std::uint32_t* block_ids;            // tile_blocks entries.
    const std::uint32_t* participating_row_masks; // tile_blocks; 1 bit/local row.
    const Offset* block_row_entry_offsets;     // tile_blocks + 1; entry offsets.
    const std::uint32_t* feature_masks;        // row_block_entries; block-local bits.
    const Offset* row_block_value_offsets;     // row_block_entries + 1; value counts.
    const std::uint32_t* feature_block_offsets; // feature_blocks + 1.
    std::uint32_t tile_row_width;              // 1..32 for the retained grammar.
    std::uint64_t rows, features, tiles, stored_values;
    std::uint64_t feature_blocks, tile_blocks, row_block_entries;
};
template<class T, class Offset = std::uint32_t>
struct View {
    LayoutView<Offset> layout;
    T* values;
};
class Structure; // Immutable owner of maps/images and identity/order metadata.
template<class T>
class NativeTensor {
public:
    // Takes ownership of already allocated values; validates required capacity,
    // alignment and placement during construction. No implicit packing/copy.
    NativeTensor(std::shared_ptr<const Structure>, memory::cuda::DeviceBuffer);
    NativeTensor(NativeTensor&&) noexcept;
    NativeTensor& operator=(NativeTensor&&) noexcept;
    NativeTensor(const NativeTensor&) = delete;
    NativeTensor& operator=(const NativeTensor&) = delete;
    View<T> view() & noexcept;
    View<const T> view() const & noexcept;
    View<T> view() && = delete;
    View<const T> view() const && = delete;
    const Structure& structure() const & noexcept;
    const Structure& structure() const && = delete;
private:
    std::shared_ptr<const Structure> structure_;
    memory::cuda::DeviceBuffer values_;
};
}
```

This view is illustrative of the existing warp-tile grammar. Its eventual
offset widths and validation retain each format's actual count limits; the
sketch does not silently widen the frozen v1 format. Const views expose
`const T*`. Placement, axis association, generation and readiness bind once per
operation, instead of copying large identity records into every tile.
Tile `t` covers execution rows beginning at `t * tile_row_width`; row entries
follow set participating-row bits in increasing local-row order, and their
values follow set feature bits in increasing block-local order. Terminal
offsets are mandatory and offsets count the stated records/values, never bytes;
value addresses multiply by `sizeof(T)`. `Structure` retains feature and row
maps, including partition/global-row scope, for checked canonical recovery.
The v1 adapter aliases the existing maps/masks/offset arrays at matching widths;
its identity/count descriptors are copied only as metadata.

`Structure` owns or retains immutable index allocations. `NativeTensor` owns
one or a few bulk value allocations, not a tensor per module. Independent value
instances can share `Structure`. Borrowed views own nothing and remain valid
only while their owners, maps and relevant generations remain valid.
The structure owns/retains the immutable index allocations for a declared
placement. Its host-side identity/count descriptors and device `LayoutView`
are separate. Reusing the logical structure on another device requires a
validated replica/rebind, not dereferencing the first device's index pointers.
Move/destruction must not retire storage still used by an operation; execution
retains the needed owner/lease through completion. There is no automatic clone,
repack or copy in `view()`.

For allocation, move the existing explicit-device
`modeling::cuda::DeviceBuffer` and pinned owner implementation to `memory`.
Convert internal `runtime::device_buffer<T>` users to the canonical owner plus
explicit shared retention/borrowing where needed, then retire the redundant
public contract. Do not carry implicit current-device/synchronous helpers into
the new allocation contract. `memory::allocation` is a non-owning record, not a
replacement owner. Reuse session/workspace allocation where a session owns the
operation. Pooling, allocator choice and external owner retention remain
explicit; a new universal allocator backend is unnecessary.

An optional `interop/aten/` owner adapter retains existing `at::Tensor`s and
constructs the same native views when layout/dtype/device requirements match.
Matching views alias storage; packing into another encoding is an explicit copy
or transform. Raw pointer access does not automatically register derivatives,
mutation versions or cross-stream lifetime. ATen-backed and native owners have
the same format view, not necessarily the same owning C++ type.

Do not build a new ordinary dense/CSR/COO numerical container. Use `at::Tensor`
or existing caller-owned conventional arrays and NVIDIA descriptors. Retained
CSR launch/projection records may remain small borrowing adapters; that is not
an independent sparse tensor implementation. Remove redundant native matrix
owners after updating in-tree uses. Retain or migrate useful persisted inputs
through explicit codecs/conversions; an external consumer does not require a
permanent legacy C++ facade.

The initial concrete family set is intentionally small:

| Proposed format | Meaning and source |
|---|---|
| `tensor::block_masked::{LayoutView,View,Structure,NativeTensor}` | Cell-record/warp-tile grammar, decoupled from owning model objects and mutable value generations. |
| `tensor::quantized::{CsrView,BlockedEllView}` | Existing specialist low-bit affine values over sparse topology; not a new ordinary CSR/BSR implementation. |
| `tensor::state_codes::{ProfileView,View,NativeTensor}` | Existing per-feature codebooks and exact overrides, distinct from affine sparse codecs; replace the old model-state facade after its in-tree callers move. |
| `tensor::{blocked_ell,sliced_ell}::View` | Borrowing access to retained conventional formats where native consumers need it; vendor/ATen paths remain conventional alternatives. |
| `layout::projection::{FeatureMajorLayoutV1,TransposeMapV1,MmaHybridLayoutV1}` | Immutable physical maps/images whose current value and runtime associations remain execution bindings. |

Expose state-code headers under `tensor/state_codes/`, matching that concrete
namespace. Share generic offset/bitset helpers through `containers` where an
actual second user exists. Do not add placeholders for imagined representations.

## 8. Prepared bindings and live state

Extend existing execution records with a semantic association, rather than
placing streams or mutable activity inside the dependency foundation.

```cpp
namespace cellerator::execution {
namespace dependency {
struct DeviceExtentBinding; // Proposed checked device/generation/readiness record.
}
struct SourceBinding {
    geometry::dependency::SourceRef source;
    object_binding::multi_atom_port_binding_v1 logical;
};
struct SourceValues {
    geometry::dependency::SourceRef source;
    std::span<const dependency::DeviceExtentBinding> extents;
};
struct PreparedDependencyBinding {
    geometry::dependency::foundation_id foundation;
    structure::structure_epoch epoch;
    ops::AccessContract access;
    std::span<const SourceBinding> sources;
    PreparedConditionIndex condition_users;
    PreparedParticipantIndex participant_users;
    PreparedPatternAccess pattern_access;
    // Retains/borrows existing layout, atom requirement and prepared program.
};
struct EvaluationStamp {
    binding_incarnation_id binding;
    value_generation state, parameters, context, conditions, activity;
    time_generation temporal;
};
}
```

`SourceBinding` contains logical association and retained metadata only.
`SourceValues` supplies current placed extents/generations/readiness at invocation
and adapts to the existing launch/plane contracts; it is not retained as ordinary
launch-time state inside `PreparedDependencyBinding`. Persistent index/map
addresses have their own prepared-owner lifetime. An explicitly captured
stable-address path must pin its addresses through the existing graph/owner
contract. `EvaluationStamp` is a readable sketch: hot implementations may use
per-source revisions and compact owner stamps instead of copying every field
into every work item. Source, parameter, representation, history, explicit time
and stochastic inputs are declared where they affect computation.

New indexes map changed sources to conditions and mechanisms. The existing
reverse-posting algorithm supplies an explicit-incidence implementation. An
implicit pattern may use range/block invalidation without an expanded index.
Dirty work that is currently ineligible remains pending where later activation
requires recomputation. Shared inferred context has its own computation and
cache dependencies; its cost and invalidation cannot be hidden in a predicate.

Live activity/work selection and tapes are separate from reusable preparation.
A tape pins required forward associations and generations through backward;
mutating values does not retroactively change a saved evaluation. Reuse existing
publication/lease/close rules and native parameter owners. Caller-visible
statuses handle stale binding, insufficient capacity and unsupported derivative
scope before publishing outputs.

## 9. Lifecycle and distribution

| Change | Reuse | Reconsider/invalidate |
|---|---|---|
| Participant value write | Pattern and compatible layout | Affected mechanisms/results and declared condition readers |
| Conditioning source or inferred-context write | Possible pattern, compatible structure | Affected condition results, live work and dependent results |
| Parameter update inside declared family | Applicable structural envelope | Parameter-dependent conditions/results, derived value encodings |
| Regime, coordinate meaning, parameter-family assumptions | Unaffected independent foundations | Affected applicability, bindings and dependent preparation |
| Participant membership or possible connectivity | Unaffected snapshots retained by existing owners | Affected pattern indexes, physical maps and capacity estimates |
| Pure physical permutation/repacking | Logical model and dependency knowledge | Changed physical bindings/maps and readiness; explicitly move or alias values |
| Device placement or replica revision | Logical foundation | Physical binding, access readiness and affected communication |
| Derivative request/order changes | Applicable primal foundation | Derivative access, omission rules, tape/workspace and kernel eligibility |
| Time expiration/history advance | Unaffected timeless structure | Declared temporal/context readers even if no value write occurred |

One logical model or Cell may span devices. Reuse multi-extent ports and local
index spaces so this does not require an aggregate allocation or a Cell object
per device. Actual placement still needs per-device owners and valid bindings.
Multi-device forward/backward is required in the first implementation
milestone, with two CUDA devices in one host as its initial scope.

Consolidate physical extent binding with explicit device/placement and
readiness/lease association. The current `physical_extent_binding_v1` has a
residency kind but no device ordinal, so it alone does not authorize cross-device
access. Its internal API may change instead of accumulating legacy wrappers.
Separate prepared transfer requirements from current pointers, generations and
completion. Share, replicate or transfer only through supported runtime paths.
Reuse existing fleet/copy/reduction resources where their contracts suffice;
close missing dependency/fence semantics explicitly. The operation's reads,
conditioning, normalization, outputs and requested backward accesses all
contribute to the communication plan. A device-local head consumes declared
sources without owning biological identity or discarding nonlocal dependencies.
SCOPE section 4 defines the connected two-device example and acceptance.

## 10. Concrete use case and extension test

Use overlapping joint groups of 32 features, with two shared regulators per
condition and multiple groups contributing to the same output. The group sizes
are a test case, not a fixed architecture.

1. Bind one foundation to a native product mechanism and a different learned
   grouped computation. Explicitly validate that each model's admitted reads
   satisfy the foundation; do not assume equal shapes establish compatibility.
2. Lower implicit groups directly to batched dense work or a block-masked
   representation; keep an explicit-incidence reference implementation.
3. Evaluate a shared condition once per valid context generation and use its
   exact operation-specific omission rule before participant gathering where
   allowed. A generic data-dependent gate without that rule keeps necessary
   derivative work.
4. Keep producer/consumer values packed when orders agree. Canonicalize only
   at an observable boundary or when the complete operation benefits.
5. Permit two physical layouts and native/ATen-backed storage without changing
   the foundation or mechanism meaning. Test local and multi-extent binding.
6. Preserve overlap, repeated incidence, output accumulation, inferred context,
   hidden regulator gradients and finite-time indirect effects.

Preparation is optional: an ordinary dense callable remains a valid model
component. New format types, new pattern providers, new operations and new
hardware kernels are independent extension points. There is no requirement
that every extension register a global module, introduce a new IR node, or
materialize a full graph/Jacobian.

## 11. Additions required and implementation boundary

| Addition | Reuse first | Missing contract to supply |
|---|---|---|
| Dependency applicability and knowledge scope | Persistent IDs/epochs; existing logical support | Biological/model regime and completeness distinct from physical regrouping |
| Implicit/n-ary pattern providers | Explicit support; execution coverage and local index spaces | Direct traversal/lowering without enumerating all candidate pairs |
| Condition-source declarations and indexes | Condition programs; reverse postings; incremental reads | Stable source-to-condition mapping, inferred/temporal dependencies |
| Model binding and omission rules | Indexed mechanism access/effects/VJP; prepared launch validation | Read-satisfaction checks and separately scoped exact/approximate derivative omission |
| Concrete bulk tensor owners/views | CellPack grammar, codebooks, native buffers, external plane mapping | Cohesive public ownership/view facade per specialist format |
| Semantic-to-execution adapter | Atom requirements/planes/fragments/program v2 | Association of foundation/pattern/condition IDs with actual operations and live generations |
| Multi-device preparation/binding | Existing multi-extent ports, readiness and runtime resources | Canonical per-extent device association, explicit forward/backward transfers, completion/leases and gradient/update publication; required in the first milestone |

No implementation is included. Do not change kernels, source namespaces,
checkpoint owners, `Cell`, generated source maps, Todo authority, or protected
geometry text as part of approving or maintaining this plan. Implementation
requires a separately authorized bounded outcome.

KDA design references consumed: agent entrypoint and KernelBender's ownership,
lifetime/reuse and identity guidance. Inspected KDA revision:
`277c79b8cdd1d13c34a0dfd132b8bb504bcb5afc`; KernelBender revision:
`2e19a747a3be2abfb12243db809a3e85ae8b5d42`. These inform representation choices;
no kernel experiment, benchmark, GPU admission or performance claim follows.
