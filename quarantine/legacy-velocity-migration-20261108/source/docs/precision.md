# Precision and selected originals

`import velocity.precision;` provides an optional sensitivity report and
explicit compositions that replace selected source-gene moments with original
FP32/FP64 values. The ordinary `reference` calls do not acquire an original
source, screen genes, or retry at higher precision. The active implementation
is CPU-only.

Storage and computation precision are separate. W, S, and U storage can
independently be FP16, FP32, or FP64. FP16 is supplied as raw `std::uint16_t`
binary16 bits through `NumericView`/`NumericEdit`; Cellerator's small,
header-only CPU helper converts values during numerical traversal. No full
widened shadow is required by the view contract. AnnData++ integer arrays are
not FP16 storage, and these views do not define a file format. Computation is
FP32 or FP64. `ComputePrecision::automatic` selects FP32 when any input is
FP16 and none is FP64; otherwise it selects FP64. An explicit precision
request overrides that rule.

## Advisory sensitivity

`sensitivity_gene()` starts from an existing gene fit and its two moment
columns. Caller scratch holds at least one score per row and must not alias
the observations. The report checks how changing active trimming cutoffs
changes status, selection, R², and prediction residuals. It does not estimate
an error bound and never requests promotion automatically. Thresholds are
configurable through `SensitivityOptions`.

`SensitivityWorkspace<T>` is an optional scratch owner. It checks the bound
stage workspace and caller-declared promotion staging against the supplied
numeric budget, then retains only the rows-element score array. Report values
remain caller-owned. If the caller supplies score storage directly, that
storage is part of the caller's run budget.

## Original-value providers

`OriginalSource` is a small caller-owned callback descriptor. The callback is
invoked only for explicitly requested source genes and returns either a dense
N-by-1 spliced/unspliced `NumericView` pair or `false`. Both columns must be
FP32/FP64; FP16 originals are unavailable for promotion. A provider that
materializes sparse or backed data must sum duplicate expression coordinates,
declare its staging bytes, and keep returned values and owners alive through
consumption. Worker callbacks and providers are worker-local; velocity adds no
locking or scheduling layer.

`OriginalArraySource` retains dense array views and supplies affine
single-column subviews without copying. Borrowed source bytes still need an
external owner. Keep the source object alive while its callback is in use.

The source-gene mask in `PromotionOptions` is immutable for the whole run.
Availability is also fixed: if a requested original cannot be acquired during
initialization, that gene retains its baseline storage through every later
phase. In a stochastic run, a previously promoted source must still be
available with a valid FP32/FP64 pair during refinement or reapplication
throws. The caller must keep original values stable across phases; the
implementation does not compare them to detect mutation. W is never promoted.

## Stage composition and resource accounting

The caller-scheduled path applies `precision::promote()` to each worker's
moment block before fitting and supplies disjoint
`PromotionBuffers` status ranges. A stochastic run promotes initialization
blocks, joins and freezes availability with the global selection, then uses
`precision::reapply()` on each refinement block. It does not acquire newly
available originals in the second phase. The caller supplies worker-local
provider contexts and independently accounts their staging. The source-gene
mask is indexed by source gene; status is indexed by prepared-gene occurrence
and includes repeated selected occurrences.

The `SelectiveStages`/`SelectiveStages32` and
`SelectiveStochasticStages`/`SelectiveStochasticStages32` aliases remain
optional serial compositions around reference stages. Their template classes
are `BasicSelectiveStages<T>` and `BasicSelectiveStochasticStages<T>`. They do
not replace or intercept the independent reference path.

The serial workspace cap defaults to 64 MiB and covers numerical worker and
mask scratch, requested sink conversion, and low-signal selection scratch.
Promotion separately checks `PromotionOptions::workspace_bytes` (64 MiB by
default) against workspace scratch plus declared staging for requested genes.
Undeclared provider allocations are outside that check. With caller-scheduled
workers, budget the aggregate of each worker's scratch, the one global
selection buffer if required, and each provider's staging. Provision disjoint
output/scratch ranges and worker-local callbacks. See the [API contract](api.md)
for the two-phase stochastic join order.

The original CUDA promotion/sensitivity implementation and its support remain
in the parent checkout's quarantine. They are historical provenance, not an
active backend or evidence for this CPU API.
