# Shared regulatory support with context activity

The selected construction is a composition of a prepared Cellerator relation
with Torch source and target activity gates. Its purpose is to learn shared
influences across contexts while retaining stable source, target and logical
edge identities. This is a modeling restriction, not evidence that the learned
influences are causal molecular regulation.

For a declared logical edge set `E`, each edge `e` has source `src(e)`, target
`dst(e)` and its own trainable value `w[e]`. The model computes

```text
q[b,i] = s[b,i] * x[b,i]
v[b,j] = sum over e with dst(e)=j: w[e] * q[b,src(e)]
y[b,j] = a[b,j] * v[b,j]
```

Repeated source/target pairs remain distinct logical edges; their contributions
add. A zero value or activity does not remove an edge from differentiation.
Torch owns activity generation, multiplication, loss and Adam orchestration.
Cellerator owns prepared relation execution, coefficient storage and the native
handle. The optional `cellerator.torch` adapter wraps that same handle and
publishes guarded updates.
The wrapper uses existing one-input product mechanisms and explicit additive
output contributions. It introduces no new native planner or numerical kernel.

The biological hypothesis is that one potential influence support and shared
edge values are useful across the selected contexts, with context variation
captured by source and target activities. GlassHelix must choose and justify
that hypothesis, its observation model and its typed representation. Gene or
anchored module coordinates can support a biological interpretation; arbitrary
latent coordinates need an effective latent interpretation. Activities are not
chromatin measurements merely because they are named activities. An RNA-only
experiment cannot use ATAC-derived support as if it were RNA-only evidence.

The exact computational identity is `W_b = diag(a_b) W diag(s_b)`. It permits
shared support preparation and avoids persistently storing `B * |E|` separate
edge values. Exact algebra establishes equality of this expression and its
materialized implementation; it does not establish scientific validity or a
speed advantage. The shared storage formulas are `4|E|` bytes for FP32 values
and `4B(n+m)` for FP32 activities. Explicit instance edges require `4B|E|`
bytes, in addition to the activities from which they are computed. These
formulas exclude identities, optimizer state, saved activations and prepared
execution buffers. For small batches or very sparse supports, the activity
arrays and preparation work can outweigh the saved instance edges.

For output cotangent `lambda`, let `u = a * lambda`. The derivatives are

```text
dx[b] = s[b] * (W.T u[b])
ds[b] = x[b] * (W.T u[b])
da[b] = lambda[b] * v[b]
dw[e] = sum over b: u[b,dst(e)] * q[b,src(e)]
```

Torch chains these through learned activities. Shared parameter uses accumulate
before one update. Duplicate logical edges have separate parameter gradients.
This composition is exercised in FP32; it makes no new mixed-precision claim.

## A restriction that can fail

Two-sided diagonal gates cannot represent arbitrary changing relations. For
four nonzero entries of a two-by-two effective relation, diagonal scaling
preserves the cross ratio

```text
W_b[0,0] * W_b[1,1] / (W_b[0,1] * W_b[1,0]).
```

Two contexts with relations `[[1,1],[1,1]]` and `[[1,1],[1,2]]` have cross
ratios one and two. They cannot both arise by diagonal scaling of one fixed
`W`. A fixed support can also omit a changing interaction. Scaling can exchange
magnitude among values and activities, so learned factors are not automatically
identifiable. A scientific assessment needs a matched reference and, where
feasible, a less restricted rival.

## Complete-cost comparison

[The lifecycle benchmark](../../bench/learning/shared_support_lifecycle.py)
compares the native composition with an explicit Torch per-instance-edge
`index_add` expression. Both use identical seeded support, initial values,
inputs, activities, target, mean-square loss and single-tensor Adam options.
Before timing, it checks forward outputs, input/activity/coefficient gradients
and one optimizer update. This is implementation equality, not a biological
experiment.

The script requires an assigned GPU lease and explicit execution:

```sh
python bench/learning/shared_support_lifecycle.py --run-cuda \
  --output /absolute/path/to/shared-support-lifecycle.json
```

Install the `cellerator` distribution with its optional Torch integration enabled
before running this benchmark; it discovers the packaged native modules through
Python distribution metadata and does not use a development-tree library path.

The output includes a batch-sharing regime (`B=32,n=64,m=32,E=256`) and a
small counter regime (`B=1,n=64,m=32,E=16`). It records fresh program setup,
identity packing, initial coefficient transfer (separate for the Torch reference
and included in native preparation), support upload where separately
observable, optimizer construction, input/target transfer, forward, loss,
backward and optimizer/publication costs. Native support upload is included
in program preparation because its public API does not expose that boundary.
Times use synchronized wall-clock stage boundaries, two warmups and the median
of five repetitions by default. Complete iteration wall time includes Python
and synchronization overhead. Fresh preparation is measured after CUDA context
and library loading; it is not process startup time. Checkpoint reconstruction
is not measured.

Both paths pack the common declared semantic support. The Torch reference
materializes per-instance edges during forward rather than caching a stale
instance relation. Each timed iteration transfers CPU data and targets, and
both optimizers update the same shared edge parameterization. This comparison
does not measure a model with independent trainable per-instance weights.

Torch allocator peaks are recorded, but raw native CUDA allocations are outside
those counters. The native handle exposes no allocation-byte query, so native
reserved bytes are explicitly unknown. Synchronized `cudaMemGetInfo` observations
before and after setup and at iteration boundaries also record global device
usage deltas, including native allocations and Torch pools. Those observations
can include other processes and miss transient peaks; they are not per-owner
bytes and must not be added to Torch allocator counters. The benchmark includes source hashes,
loaded library hash, GPU/framework versions, configurations and individual
samples. It reports `evaluated_not_promoted`; performance or storage promotion
requires reviewing those complete costs and any missing memory evidence. The
script never runs a GPU workload by default.
