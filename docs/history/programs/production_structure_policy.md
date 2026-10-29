# CE-PTR production structure policy

## Purpose

CE-PTR removes accidental generic ownership from the production Cellerator
core. It does not replace `std::vector<T>` with a generic Cellerator vector,
buffer, or pointer wrapper. Each migrated owner must instead name the
biological or execution structure, its identity, order, lifetime, bounds,
placement, and reuse contract.

This policy governs `include/Cellerator/` and `src/`. Tests, benchmarks,
historical compatibility code, examples, cold diagnostics, and framework ABI
adapters are not production-core evidence. Their standard-library use remains
permitted unless it leaks ownership into a core contract or measured hot path.
STL algorithms are not categorically prohibited.

## Required classification

Every production owner in the CE-PTR inventory records these fields:

| Field | Required meaning |
| --- | --- |
| subsystem | Canonical Cellerator owner and source path. |
| accidental representation | The generic owner or incidental structure present today. |
| semantic structure | The named biological, execution, relation, image, or workspace meaning. |
| lifetime | Build, structure epoch, prepared plan, launch, or cold boundary. |
| cardinality and bounds | Exact count source, fixed bound, queried requirement, or validated overflow path. |
| memory domain | Host, NUMA host, pinned host, device, managed, external, or caller/session workspace. |
| hot or cold | Whether the owner or operation participates in preparation or repeated execution. |
| migration disposition | Eliminate, image, table, relation, workspace, move, or cold keep. |
| allowlist rationale | Why the current occurrence may remain temporarily and which CE-PTR lane retires it. |

Shape alone never establishes semantic compatibility. Migrated structures must
preserve domain, order, geometry, partition, structure epoch, value generation,
stable row or feature identity, and provenance where the current contract
requires them.

## Replacement contract

- Durable immutable structures use validated pointer-free images with schema
  identity, explicit counts, relative offsets, alignment, and total byte size.
- Hot consumers use small typed non-owning views with explicit extents,
  placement, and validated lifetime; a generic view is plumbing, not the public
  semantic API.
- Ragged data uses named offset/member relations or sorted packed keys.
- Naturally bounded domains use named fixed-width tables with an explicit,
  separately prepared overflow representation where required.
- Temporary storage uses caller- or execution-session-owned workspaces whose
  exact requirements are queried before execution. Workspaces never grow in a
  prepared or sealed path.
- Mutable values do not own or reconstruct immutable structure. Pointer or
  stream rebinding does not force structural preparation.
- Allocation, transfer, synchronization, descriptor creation, and
  canonicalization remain explicit and measurable. Repeated prepared execution
  performs none of them unless its public contract names the operation.
- CellShard retains persistence, transport, staging, and distribution;
  BioPrep retains preprocessing policy and workflow orchestration;
  CelleraTorch remains an optional thin adapter. Native Cellerator does not
  depend on Torch.

No runtime benchmark tuner or continuously adapting optimizer is introduced.
Development measurements may select a static or preparation-time policy. V100
`sm_70` is the primary performance target while stable semantic interfaces stay
generation-conscious.

## Standard-library disposition

The permanent source gate treats the following owning families as controlled:
`std::vector`, `std::map`, `std::unordered_map`, `std::set`,
`std::unordered_set`, `std::priority_queue`, `std::shared_ptr`, `std::deque`,
and `std::list`.

New controlled owners are prohibited in the production roots. Existing debt is
an explicit path-and-family allowlist with a non-increasing occurrence ceiling
and a migration rationale in `scripts/check_no_inappropriate_core_stl.py`.
Removing or reducing debt passes; moving it, changing family, increasing its
count, or adding an unlisted production path fails. Comments and string
literals are removed before token matching so documentation does not create
false debt. The gate is intentionally lexical and conservative: code review
and the classified inventory remain authoritative for semantics.

New `std::shared_ptr` ownership is prohibited for core runtime and prepared
paths; session allocation handles, leases, and typed hot views replace it.
The final CE-PTR inventory retains one explicitly bounded legacy compatibility
owner in `runtime/device_buffer.cuh`; it is not a prepared-path precedent and
its blocking transfer helpers are not allocation-free execution. Node-based
maps and sets require an explicit benchmark exception against the named exact
table, packed-key, direct-index, or sort/compact alternative. No such permanent
exception is currently granted. Framework ABI requirements and cold external
objects remain boundary-only exceptions outside the enforced roots.

## Migration evidence

Each migration records semantic parity, determinism, construction and reuse
cost separately, allocation count and bytes, host and device high-water marks,
H2D/D2H/D2D bytes, synchronization and launch counts, and end-to-end latency.
GPU work additionally records kernel registers, spills/local-memory traffic,
occupancy, achieved memory throughput, and relevant warp stalls when tools
permit. Evidence identifies hardware/topology, driver, CUDA/compiler/library
versions, build mode and architecture, input shapes/distributions, dtype and
accumulation, warmups/repeats, included setup/transfer/output work, reuse,
correctness tolerance, and benchmark-mutex use.

Measured regressions are not accepted merely because the replacement looks
lower-level. Exceptions must be narrow, named, source-grounded, and recorded in
the inventory; a generic-container waiver is not a migration disposition.
