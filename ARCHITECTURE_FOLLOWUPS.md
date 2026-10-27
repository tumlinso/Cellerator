# Architecture follow-ups

## Intended direction: ATen tensor substrate

Direction recorded on 2026-10-26: progressively adopt ATen (`at::Tensor`) for
much of Cellerator's standard tensor and memory representation, using PyTorch's
infrastructure for allocation, device management, dtype handling, views, and
interoperability. The aim is to reduce custom tensor infrastructure and make
Cellerator easier to maintain and extend as a high-performance C++/CUDA library.

ATen should provide the tensor substrate. Cellerator's specialized CUDA kernels,
custom memory layouts, and optimized execution paths remain available wherever
they offer meaningful advantages. Adopting ATen does not imply replacing those
implementations with standard PyTorch operations.

New tokenization facilities and Karyoplast model components should benefit from
this shared infrastructure, avoiding unnecessary custom tensor abstractions
and enabling straightforward PyTorch integration. Adoption should be progressive;
migrating existing representations is not a prerequisite for new functionality.

This records architectural intent, not a completed migration or an implementation
specification. Existing build and dependency behavior remains as implemented.
The timing, scope, and compatibility changes of individual migration steps
require separately scoped work. Recording this direction does not authorize
migration of the rest of Cellerator or begin tokenizer implementation planning.

## Resolved: CellShard-to-Cellerator runtime dependency

Cellerator is the primary compute and execution layer. The intended dependency
direction is `Cellerator -> CellShard`: Cellerator may consume CellShard storage,
layout, export, and device-view interfaces, but CellShard must not depend on
Cellerator compute, distributed-runtime, stream, scheduler, or collective types.

Resolved by the bounded post-remap completion pass. CellShard distributed APIs
now accept neutral borrowed device IDs and CUDA stream handles through a
CellShard-owned binding view. CellShard no longer includes Cellerator runtime or
distributed headers, names Cellerator runtime types, locates `Cellerator::dist`,
or links a Cellerator runtime target.

Cellerator retains execution topology, stream/resource ownership, NCCL policy,
and planner decisions. The Cellerator caller translates its local execution
context into the neutral CellShard binding at the interop call site. CellShard
retains storage, residency, transport, and generic shard assignment.

The remaining Cellerator compatibility-format sources embedded by CellShard do
not link or expose Cellerator runtime machinery; they are retained only for
historical matrix-format compatibility.
