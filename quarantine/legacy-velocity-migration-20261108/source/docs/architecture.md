# Architecture and design constraints

`velocity` is Cellerator's native portable reference path for RNA velocity.
It is an independently buildable C++20 component. The active implementation
is CPU-only and has no accelerator or ISA selection. Historical CUDA and
AVX2/F16C files remain in the parent checkout's quarantine and are not active
implementation evidence.

## Scientific scope

The implemented methods compute first-order neighborhood moments and
steady-state velocity, then second-order neighborhood moments and stochastic
velocity. Dynamical and Bayesian fitting, velocity graphs, backed orchestration,
Python runtime bindings, and accelerator execution are outside this component.
The earlier [Milestone 1, 2, and 3 records](validation.md) describe their own
source snapshots and acceptance exceptions; they do not qualify the current
caller-scheduled interface.

Numerical similarity to the declared scVelo comparison is the target. Bitwise
identity and exact result equality are not requirements. Different numerical
algorithms, solvers, reduction orders, and precision choices may be appropriate
when their scientific meaning and observed differences are recorded. The
[scVelo foundation audit](scvelo-foundation.md) distinguishes mathematical
reference behavior from its mutable AnnData workflow and implementation.

Stochastic fitting initializes from deterministic fits and their selected
genes. By default, refinement covers selected genes only; if no genes qualify,
the deterministic result remains. Refit-all is explicit and remains subject to
the enabled-gene filter. Fit-row masks apply consistently to initialization,
second-order scale estimation, and joint fitting; predictions cover all rows.
The per-gene objective weights `r1 = Mu - gamma*Ms - offset` and
`r2 = B - gamma*A - offset2` by fitted scales. The optional emitted
second-order residual is a different quantity from `r2`. Singular or
non-finite refinement retains the deterministic estimate and reports a
fallback status.

Scientific contracts specify equations, input and output ordering, graph and
moment conventions, units, masks, precision, and degenerate cases. Counts,
normalized values, smoothed moments, fit parameters, and velocities remain
distinguishable through direct names and descriptors rather than a large type
hierarchy.

## Reference operations and caller scheduling

The reference exposes free functions over prepared inputs, workspaces, plain
range/result descriptors, and caller-owned outputs. Cellerator owns block
scheduling, worker creation, synchronization, and aggregate resource policy.
There is no scheduler, task queue, global worker state, or internal locking in
this component. Each concurrent worker uses private workspace and local
callbacks/providers; writable output and scratch ranges are disjoint.

For caller-scheduled ranges, `validate_run()` establishes fixed value-content
preconditions once before workers start, subject to the `PreparedInput`
validation mode. Checked serial stage/convenience drivers perform their own
single scan during setup. Workspaces cache fit-row masks in worker scratch;
callers refresh them before scheduling when mask values changed between runs.
The caller keeps input structure, masks, selected gene order, original-source
values, and source-promotion availability fixed until all phases complete.
Preparation and acquisition remain cold boundaries; acquired
descriptors inside numerical work carry pointers, extents, strides, and type
information without owners or dynamic dispatch.

Stochastic refinement requires two global phases:

1. Compute initialization moments and fits over disjoint gene ranges.
2. Join initialization workers and resolve one global selection.
3. Compute second-order moments, scales, and stochastic fits over disjoint
   ranges using that immutable selection.
4. Join refinement workers before consuming shared results.

This ordering makes the global decision explicit and keeps each numerical
block independent. It does not prescribe a scheduler or thread count. The
serial workspace cap defaults to 64 MiB and includes worker numerical and
mask scratch plus requested sink-conversion and global-selection scratch. For
parallel blocks the caller budgets the aggregate of worker scratch, global
selection scratch, and source staging. Selective promotion checks a separate
64 MiB default cap against workspace scratch plus declared provider staging;
arbitrary provider allocations that are not declared cannot be measured by
this component.

Direct outputs are acquired once as `DenseOutput` and written in place. This
avoids the sink conversion path. A requested moment with matching computation
type is a zero-copy view of active workspace. Sink callbacks are synchronous;
borrowed blocks must be consumed before callback return.

## Storage and precision

W, S, and U can independently use FP16, FP32, or FP64 storage. FP16 enters as
raw binary16 bits through `NumericView` or `NumericEdit`; the small
`Cellerator::binary16` helper provides CPU conversion. AnnData++ integer arrays
are not treated as FP16, and this component adds no FP16 codec or persistent
file format. AnnData++ remains an independent dependency for supported arrays,
axes, and dataset interfaces.

`NumericView` is read-only and `NumericEdit` mutable. Borrowed factories do not
retain a data owner; retained factories keep the explicit shared owner alive.
Rank-one and rank-two metadata is inline. `bytes()` and `dense<T>()` return
plain acquired descriptors over that storage. Dense affine selection uses
`subview(axis,start,count,step)`. Irregular gene selection is represented by
`Input::genes` and prepared once into the selected gene order.

Computation uses FP32 or FP64. Automatic precision selects FP32 when any W/S/U
storage is FP16 and none is FP64, and FP64 otherwise. Explicit computation
precision wins. Output type is requested separately. The ordinary reference
path does not inspect originals or promote genes. Explicit selective-stage
composition may substitute requested FP32/FP64 source genes; availability is
fixed across initialization and refinement. The detailed ownership contract
is in the [precision guide](precision.md).

## Dependency and module boundary

Public modules are `velocity` and `velocity.precision`; the standalone CMake
target is `velocity::velocity`. AnnData++ is required for its core arrays,
axes, and dataset interface. `Cellerator::binary16` is a tiny CPU-only,
header-only conversion helper with an independent `CelleratorBinary16`
package; it does not bring in the full Cellerator runtime. The active target
has no CUDA toolkit discovery, accelerator package export, intrinsic headers,
or ISA backend.

Keep module interfaces small and implementation dependencies private. Treat
consumer import and rebuild costs as measurements, not assumptions. Direct
calculations, arrays, pointers, and small descriptors make ownership and
memory work inspectable. Abstractions earn their place by removing concrete
complexity or enforcing a useful contract. See the source-bound
[reference review](reference-review.md) for include and compactness decisions.
