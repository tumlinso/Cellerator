# Future module boundaries

This inventory records which existing conventional contracts may participate in
future host-only modules after native CMake module scanning is available. It
does not define a second API and does not authorize `.ccm` consumers today.

## `cellerator.state`

The stable state vocabulary is already expressed by small, trivially copyable
headers under `include/Cellerator/execution/`:

- `identity.hh`: persistent and runtime identities, biological axes, structure
  epochs, value generations, and residency descriptors;
- `operands.hh`: neutral dense, bit-plane, event, segment, sparse-relation, and
  small-parameter views;
- `lifetimes.hh`: relation structure, value plane, structure requirements, and
  value binding;
- `biological_abi.hh`: the existing umbrella for identity, operand, and
  validation contracts.

A future state module may re-export the host-visible vocabulary represented by
those contracts. It must not expose identity-registry storage, allocation,
projection payloads, CUDA streams, planner keys, or vendor handles.

## `cellerator.execution`

The stable execution vocabulary begins with:

- `execution_order.hh`: declared order transitions and value-position maps;
- `launch_bindings.hh`: prepared-versus-launch lifetime separation, scalar
  bindings, stream descriptor, and caller-provided workspace;
- `execution_contract.hh`: the existing umbrella over order, bindings, and
  lifetimes.

`program.hh` is authoritative current implementation, but it is not ready to
be exported wholesale as a cross-project module. Its public request and result
types currently name internal operation-core, planner, projection, session,
and readiness types. The remap preserves that implementation and avoids
pretending the current dependency closure is a narrow external contract.

A future execution module therefore requires a separately authorized,
host-visible facade or opaque handle surface over the existing program. That is
an API decision, not a physical move, and is outside CE-REMAP.

## Explicit exclusions

Neither future module exports:

- CUDA kernels or device helper definitions;
- allocator, stream-pool, scratch, or vendor-library ownership;
- physical projection bytes;
- CellPack optimizer or geometry-compilation internals;
- planner implementation and empirical cache internals;
- CellShard persistence or delivery types beyond a sanctioned interop seam;
- Baseplane sequence-engine implementation;
- speculative GlassHelix scientific concepts.

Until the native scan proof passes, `.hh`, `.cc`, `.cu`, and `.cuh` remain the
only active implementation and interoperability contracts. CUDA targets do not
import modules.
