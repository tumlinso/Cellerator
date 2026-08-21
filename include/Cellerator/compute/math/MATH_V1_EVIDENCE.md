# CP-Math v1 evidence and retirement map

CE-ARCH-60 completed the transition from the experimental CP-Math source burst
to the supported Cellerator biological execution architecture.

## Supported execution graph

The supported graph is:

1. `Cellerator::biological_abi` for domain, order, geometry, partition,
   structure, value, operand, and launch identity;
2. `Cellerator::runtime` for the execution session, streams, library handles,
   and persistent/transient allocation;
3. `Cellerator::operation_core` for operation capabilities, preparation, direct
   dispatch, and launch binding validation;
4. `Cellerator::planner` for complete-workflow candidate selection;
5. `CellPack::semantic_geometry` and `CellPack::execution_image_v2` for stable
   organization and relocatable projection catalogs;
6. `Cellerator::sequence_integration` for native Baseplane operands and
   materialized or fused sequence-to-state execution.

The public sparse-layout C ABI in `include/Cellerator/abi.h` remains a separate
versioned compatibility surface. It was not repurposed as the common biological
ABI. CPK1 v1 bytes and canonical recovery maps remain unchanged. CellShard's
CPEXEC01 envelope remains unchanged and opaque.

## Retained v1 evidence

`cellerator_math_v1_evidence` is an `EXCLUDE_FROM_ALL` compatibility target. It
retains independent request validation, deterministic signature ideas,
logical referees, CSR/BELL lowering validation, native CPK1 tile adaptation,
and their focused tests. It is not linked by the operation core or planner.

`Cellerator::packed_dense_operand` preserves the measured device-resident
feature-order packing primitive and the existing background benchmark. Its v1
feature/order records remain compatibility metadata; new operations use the
biological ABI for semantic identity.

These sources preserve accepted evidence:

- `operation.hh`, `operation.cc`, and `operation_signature.cc`;
- `execution_plan.hh` as pointer-free v1 decision metadata only;
- `referee.hh` and `referee.cc`;
- `physical_csr.hh` and `physical_csr.cc`;
- `physical_bell.hh`, `physical_bell.cc`, and candidate validation;
- `native_tile_view.hh` and `native_tile_view.cc`;
- `packed_dense_operand.hh` and `packed_dense_operand.cu`.

## Retired implementation

The following concepts and implementations were removed because replacement
contracts and tests now exist:

- `DeviceMathContext` and its single mutable workspace;
- prepared-operation-owned streams and copied launch bindings;
- the virtual `SpMMBackend` hierarchy and global backend registry;
- the heuristic CP-Math planner;
- the old generic epilogue runtime;
- the unsafe experimental cuSPARSE Blocked-ELL prepared backend.

Their historical tests, commits, todo evidence, and benchmark records remain in
Git and todo-orchestrator history. A future cuSPARSE or Blocked-ELL candidate
must register through the operation core, use the execution session, prove
logical and padded capacities, and participate in end-to-end planning. It must
not resurrect the retired runtime island.
