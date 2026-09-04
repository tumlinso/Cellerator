# JBC interface and hidden-assumption map

This receipt maps the twenty frozen JBC interfaces in
`planning/jbc-preledger-v1/02_INTERFACES_OWNERSHIP_AND_SOURCE_MAP.md` to the
source contracts that must be preserved during the Part One rehome. The frozen
document is historical design evidence; validation in the named source is the
authority for implemented behavior, while the Part One architecture is the
authority for destination ownership.

## Assumption legend

- **ID width** records persistent identity width, or the width of the scalar
  identity/count fields when an interface has no persistent identity of its own.
- **Pointer ownership** distinguishes non-owning ABI views from owning cold-side
  builders and runtime managers. Non-owning arrays remain live for the use of
  the view; native pointer bytes and process-local tokens are never persistent.
- **Generation model** lists the epochs whose freshness is validated rather
  than treating every mutation as a structure rebuild.
- **Exact coverage** distinguishes identity-bound certification from discovery
  or proposal evidence.
- **Allocator** describes the interface contract, not incidental test helpers.
- **Target** distinguishes portable compiler semantics from concrete runtime or
  transport implementation.
- **CellShard dependency** states the migration seam. `None` means no direct
  production-header dependency; `mirror` means fieldwise scalar translation;
  `rehome source` means useful CellShard implementation is the source to move;
  and `retained owner` means CellShard remains the implementation owner.

## Interface map

| ID | Interface and source namespace | ID width | Pointer ownership | Generation model | Exact coverage | Allocator assumption | Target assumption | CellShard dependency |
|---|---|---|---|---|---|---|---|---|
| JBC-I01 | Persistent identity; `cellerator::execution::joint_compiler` | 128-bit pair of nonzero `uint64_t` producer namespace and local identity | Pointer-free value; persisted fieldwise, never as native struct bytes | Identity is stable across process and generation changes | Identity primitive only; no coverage claim | Allocation-free | Portable compiler ABI; not a pointer, hash, or ordinal | mirror |
| JBC-I02 | Exact logical coverage and role; `cellerator::execution::joint_compiler` | 128-bit coverage, parent, domain, and order identities; 64-bit bounds/counts | Non-owning extent view with pointer/count consistency and caller lifetime | Structure epoch is explicit and nonzero | Certified exact role is explicit; extent bounds must be ordered, non-overlapping, and exactly bound to identity | Validation allocates nothing | Portable semantic geometry, independent of physical format | None |
| JBC-I03 | Atom requirement; `cellerator::execution::joint_compiler` | 128-bit requirement, atom, coverage, domain, order, and operation identities | Non-owning requirement value with no owned payload | Structure or mutable-value generation requirement is explicit | Requirement binds the exact-coverage identity; absence is invalid when required | Allocation-free | Portable compiler constraint | None |
| JBC-I04 | Atom affordance; `cellerator::execution::joint_compiler` | 128-bit affordance, atom, coverage, domain, order, operation, and projection identities | Non-owning value; external buffers are referenced, not adopted | Structure/value generation and validity are explicit | Affordance binds a certified exact-coverage identity | Allocation-free | Portable capability/cost semantics; backend choice remains external | None |
| JBC-I05 | Partial-result algebra; `cellerator::compute::decomposition` | 128-bit algebra and operation identities; 64-bit state bytes/alignment | Pointer-free descriptor; operation identities replace function pointers | State compatibility is expressed by identities/policy, not hidden process state | Algebra composes already certified contributions; it does not invent coverage | Allocation-free | Portable numerical/order/tree contract | None |
| JBC-I06 | Decomposition alternatives; `cellerator::compute::decomposition` | 128-bit decomposition, operation, coverage, algebra, domain, and order identities; 64-bit counts | Non-owning alternative/contribution arrays with explicit capacities | Structure/value/cost observations remain independently fresh | Contributions must cover the declared exact coverage without omission or overlap | Validation allocates nothing | Portable decomposition semantics; physical provider selected later | None |
| JBC-I07 | Fragment request and frontier; `cellerator::execution::joint_compiler` | 128-bit request, operation, coverage, order, and candidate identities; 64-bit counts | Request/result are caller-owned pointer-first views; result capacity is explicit and at most 64 candidates | Requested structure/value generations and evidence freshness are explicit | Request carries certified exact coverages; frontier candidates must remain coverage-bound | Validation/emission use caller capacity and allocate nothing | Portable discovery frontier, not a runtime launch object | None |
| JBC-I08 | Multi-extent external binding; `cellerator::execution::joint_compiler` | 128-bit binding/domain/order identities; 64-bit addresses, byte ranges, generation, and tokens | Non-owning extent array, maximum 1024; caller owns storage and device memory | All extents agree on nonzero value generation and order | Binding is a value-plane attachment, not a coverage certificate | Allocation-free; caller supplies extents and memory | Address space/device location are explicit; readiness/lease are process-local opaque tokens and cannot persist | None |
| JBC-I09 | Lowering resumption; `cellerator::execution::joint_compiler` | 128-bit request/resumption/operation identities; 64-bit frontier/count/capacity fields | Non-owning request/result/frontier storage with explicit capacity | Resumption binds the same compiler inputs and freshness identities | Resumed work may refine candidates but cannot weaken exact-coverage requirements | Allocation-free hot contract | Portable compiler state transition; no hidden thread or stream ownership | None |
| JBC-I10 | External global cost; `cellerator::planner::external_cost` | 128-bit candidate/build/device/evidence identities; 64-bit metric and generation fields | Pointer-light value/vector views; evidence storage remains caller-owned | Cost, evidence, build, and device freshness are independently identified | Cost ranks only correct, coverage-valid candidates | Validation allocates nothing | Backend-neutral measured cost including preparation and execution components | None |
| JBC-I11 | Execution export v2; `cellerator::profiling::joint_compiler` | 128-bit export/input/build/device/evidence identities; 64-bit counts and generations | Non-owning aggregate of coverage, decomposition, requirement, affordance, algebra, order, frontier, and stage arrays | Structure/value/cost/evidence generations and correctness/performance freshness are explicit | Export carries certified exact coverages and a correctness receipt | Validation allocates nothing; producer owns all arrays | Portable evidence/export ABI, not a serialized native memory image | None |
| JBC-I12 | Common atom envelope; `cellshard::compiler::atom` | Mirrored 128-bit persistent identities; 64-bit counts/generations | Pointer-first non-owning view with independently lifetimed subcomponents; optional builder owns vectors | Structure/value and subcomponent generations must agree | Parent and contribution coverage is validated exactly | View validation allocation-free; cold builder may report `allocation_failure` | Semantics are portable and belong in the Cellerator compiler | rehome source |
| JBC-I13 | Evidence atlas; `cellshard::compiler::evidence` | 128-bit workload/structure/atom identities; fixed 80-byte pointer-free record spine with 64-bit observation fields | Atlas view is non-owning; build/index APIs may own vectors | Observation/evidence generation and count are explicit | Records are proposal-only and explicitly are not exact-coverage certificates | Read validation allocation-free; cold build/index may allocate or fail | Portable discovery evidence, not storage or transport | rehome source |
| JBC-I14 | Composition and derivation DAG; `cellshard::compiler::composition` | 128-bit node/atom/coverage identities; 64-bit edge/count fields | Non-owning nodes/edges; caller supplies traversal workspace and capacities | Derivation identity/freshness is explicit at graph boundary | Roots and derived nodes retain coverage identity through edges | Validation uses caller workspace; maxima are 256 nodes and 1024 edges | Portable compiler composition grammar | rehome source |
| JBC-I15 | Basis manifest; `cellshard::compiler::basis` | 128-bit basis/workload/structure/atom identities and content digests; 64-bit ranges/counts | Non-owning atom table/range view | Workload and structure freshness are distinct; stale workload and stale structure are separate failures | Basis membership references certified atoms; manifest does not manufacture coverage | Validation allocation-free; construction may be cold-side | Portable compiler basis contract | rehome source |
| JBC-I16 | Persistent partial atom; `cellshard::compiler::partial` | 128-bit atom/parent/coverage/operation identities; 64-bit size/count fields | Pointer-first non-owning payload and contribution views | Structure, value, state, materialization, and cost generations are separate | Every partial contribution is bound to exact coverage and algebra | Validation allocation-free; materializer supplies storage | Portable compiler partial-result semantics | rehome source |
| JBC-I17 | Atom-store generation and physical instance; `cellshard::artifact::atom_store` | 128-bit store/artifact/content identities and digests; 64-bit generation, epoch, offset, and count fields | Pointer-free root/record metadata; CellShard owns durable bytes and mappings | Store generation and structure epoch are explicit; parent-root digest establishes lineage | Stores certified compiler artifacts but does not define their semantic coverage | CellShard storage layer owns allocation, persistence, and mapping | Concrete persistent storage/materialization contract | retained owner |
| JBC-I18 | Global operation provider and schedule; `cellshard::compiler::graph` | 64-bit node/provider/bounds plus 128-bit semantic/content identities where present | Recipe is a non-owning view; pointer parameters are supplied to validation and excluded from content digest | Structure/value/schedule identities and content digest bind the recipe | Schedule must select coverage-complete atoms/providers | Validation allocation-free; caller owns graph arrays | Compiler graph and schedule policy belong in Cellerator; launch realization remains runtime-side | rehome source |
| JBC-I19 | Distributed certificate; `cellshard::compiler::schedule` | Content digests plus 64-bit participant, atom, and contribution counts | Pointer-free certificate value; referenced schedules are external | Schedule/topology/evidence generations are bound by digest/freshness | Certificate attests complete participant/atom/contribution coverage | Allocation-free certificate validation | Portable compiler certification; concrete transport is separate | rehome source |
| JBC-I20 | Topology, transport, and residency; `cellshard::runtime::v2` | 64-bit device/NUMA/participant identifiers; residency lease uses slot, pin mask, and incarnation | Runtime manager owns atomics, mappings, leases, and transport buffers; compiler sees opaque handles only | Residency incarnation and lease generation are process-local and never persistent semantic identity | Runtime realizes an already certified schedule; it does not redefine coverage | Runtime manager may allocate, pin, stage, and synchronize explicitly | Concrete CUDA/NUMA/NCCL and host transport implementation | retained owner |

## Cross-interface hidden assumptions

1. Persistent semantic identities are 128-bit namespaced values. Equal lengths,
   pointer values, hashes, table ordinals, and process-local handles do not
   establish identity.
2. Stable ABI views are standard-layout, trivially copyable, pointer-light or
   pointer-free. A pointer/count pair is either both empty or both valid, and
   the caller retains storage for the full use of the view.
3. Persistence is field-oriented. Raw addresses, readiness tokens, lease tokens,
   traversal workspace, streams, and native struct padding are process-local.
4. Structure, value, state, materialization, cost, evidence, store, and residency
   generations have different meanings. A value update must not imply structure
   reconstruction.
5. Exact coverage is certified, identity-bound, and compositional. Evidence
   atlas records and discovery frontiers are proposals, never certificates.
6. Compiler validation is allocation-free. Allocation is restricted to
   caller-owned buffers, cold builders, persistent CellShard storage, or
   concrete runtime managers.
7. Cellerator's production JBC headers have no direct CellShard include. The
   seam is a fieldwise mirror/bridge today and becomes Cellerator-owned compiler
   contracts plus CellShard-owned concrete storage and runtime interfaces.

## Explicit contradictions with the frozen ownership map

The frozen map assigns JBC-I12 through JBC-I20 to CellShard. Part One changes
that ownership boundary without invalidating the useful implementations:

- **Contradiction — rehome:** JBC-I12, JBC-I13, JBC-I14, JBC-I15, JBC-I16,
  JBC-I18, and JBC-I19 are compiler semantics and therefore move to Cellerator.
  Their CellShard implementations are migration sources, not duplicate future
  authorities and not obsolete code to delete prematurely.
- **No contradiction — retain:** JBC-I17 remains CellShard-owned because it is
  concrete durable atom-store generation/materialization.
- **No contradiction — retain:** JBC-I20 remains CellShard-owned because it is
  concrete topology discovery, residency, transport, staging, and runtime
  realization. Cellerator consumes only the narrow opaque interface required to
  plan and certify execution.
- **No contradiction — preserve:** JBC-I01 through JBC-I11 were already placed
  under Cellerator namespaces by the frozen map and remain Cellerator-owned.

The resulting boundary is one-way at the semantic layer: Cellerator defines the
compiler artifacts and schedules; CellShard stores, transports, materializes,
and realizes them without rediscovering their biological meaning.
