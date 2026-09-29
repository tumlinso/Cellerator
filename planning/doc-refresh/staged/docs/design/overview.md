# Cellerator: compile the structure, not just the matrix

## The question

Which parts of a biological computation are genuinely new on each run? If relations, support patterns and useful order recur while values change, repeatedly recovering that organization is avoidable work. Cellerator investigates how much of it can be discovered, represented and compiled into a better execution plan.

The biological question precedes the layout choice. A sparse matrix is one projection of a relation; it need not be the only useful computational object. Support groups are computational findings, not automatically biological pathways. Structure earns its place through correctness, usable semantics and measured total cost.

## Three stages with different responsibilities

**Discovery/preparation** examines supplied systems or data for reusable structure. Its result must carry identity, admissibility and the evidence needed to justify reuse. It must not silently assume that an organization generalizes to every cell or workload.

**Compilation/realization** translates that structure and the requested operation into candidate execution objects. Semantic geometry preserves what is being computed; physical projections determine bytes, schedules and kernels for a target. Exact ownership of logical contributions matters more than whether a projection is CSR, packed tiles or an MMA/residual hybrid. Generated/assembled kernels and vendor candidates belong to the same costed decision process.

**Execution** binds current values, buffers and streams. Changing expression or learned parameters should not automatically reconstruct topology. Structural changes and value changes need distinct lifetimes and invalidation. Current implementations expose bounded routes; the general goal is not synonymous with one particular legacy execution image.

## Runtime adaptivity belongs in the design

Precompilation does not mean a single fixed path forever. The intended model discovers possible gates and reorganizations ahead of execution; changing cell state then selects among those prepared possibilities. Existing value-generation, activity or gate primitives may realize parts of this, but none alone proves the full state-adaptive compiler is complete. The [current snapshot](../status/current.md) states the delivered envelope.

A later supported mechanism from GlassHelix may add a variable, relation or gate to what Cellerator compiles. The compiler can specialize only what has actually become reusable: a variable's role, dependency structure, a measured value, an invariant or a justified update rule. It cannot constant-fold an unknown biological state simply because the model gave it a name. Such specialization should remain scoped and reversible.

## Map biology and hardware together

Repeated support can become grouping; grouping can become execution order and memory reuse; a suitable relation may admit a dense fragment plus exact residual work. The GPU's instruction/execution model is a design input, not a trophy. The same relation may need different physical realizations for width, precision, orientation or reuse. Preparation, packing, transfer, synchronization and output conversion belong in that comparison.

The later N=64 exact-cover success and the earlier PBMC-derived Tensor Core rejection are compatible observations. They concern different structures and candidates. A measured win licenses a regime-specific choice, not a repository-wide format doctrine.

## What this repository owns

General structured numerical semantics, compilation, execution geometry, candidate selection, prepared execution, mutation/lifetime and corresponding native operations belong here. CelleraTorch exposes supported routes to framework users without silently duplicating the numerical owner. Baseplane retains sequence semantics; GlassHelix retains scientific model/inference interpretation; CellShard retains persistence and physical delivery.

This cleanup does not finish all compiler language features, general training, all-GPU execution or automatic mechanism promotion. It makes existing paths and intended responsibilities understandable. Start at the [source map](../development/source-map.md). The existing Quarto [architecture](../architecture.qmd), [biological execution model](../biological_execution_model.qmd), and [core execution notes](../core_execution_cp_math.qmd) retain detailed contracts; this overview is the short reader entry, not a replacement for those subsystem references.
