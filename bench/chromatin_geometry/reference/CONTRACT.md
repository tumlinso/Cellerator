# Independent numerical and workload references

These references are private to the chromatin-conditioned CUDA study. All
views point to host data; they make no CUDA calls and are not installed API.

The ordered-product oracle evaluates supplied `q=b`, STE `q=p*b`, and
thresholded teacher gates using the group's ordered argument slots. It does
not deduplicate repeated participants. The VJP for each slot multiplies every
other slot, so repeated coordinates and zeros retain their derivatives.
Output incidences may overlap; shared theta gradients reduce across groups.
Supplied and hardened-teacher gates stop task gate gradients: `dQ` and
`dscore` remain zero while mechanism gradients run on primal members. A
hard-excluded forward skips the group's participant and parameter reads. STE
backward alone visits its complete declared derivative membership, including
dormant groups, to compute `dQ` and `dscore=dQ*p*(1-p)/T`. Dormant groups
receive no primal theta or participant gradient. The STE validator is analytic
and never uses finite differences.

GroupedLinear uses one row-major weight matrix per declared parameter. Each
group's ordered participant list supplies its input columns; local output
incidences map its output rows into canonical coordinates. FP16 inputs and
weights are decoded, each local dot accumulates in FP32, and local F rounds to
FP16 before gate multiplication and overlapping output reduction. Backward
rounds local dF, dX, and batch-reduced dW to the selected dtype. Canonical
overlap and shared-weight reductions use the already rounded local values and
FP32. Supplied and hardened-teacher `dQ` remain zero; GroupedLinear evaluates
its gate VJP only for STE. FP32 is an independent no-staging alternative.

The frozen grouped-linear absolute bound follows the existing
`grouped_linear_smoke.cu` per-dot formula:

```text
u = epsilon(float32) / 2
k = 2 * dot_length * u
bound_dot = (2*k/(1-k)) * sum(abs(products))
            + (FP16 ? one_fp16_ulp(expected) : 0)
```

Canonical FP32 overlap bounds propagate each local dot bound and add the same
gamma-style bound for the reduction terms. `LinearValidationBounds` returns
absolute bounds for canonical y/dX/dW/dQ/dscore and local dX/dW. The
`within_absolute_bound` predicate applies those bounds without a relative
tolerance.

The three-input teacher uses row `4*a+2*b+c`. Its independent Bernoulli row
loss is stable binary cross entropy, with `dlogit=sigmoid(logit)-target`.
`teacher_row_mean_reference` reproduces independent table-row updates for
minibatches: it averages labels within each observed row, leaves empty rows
unchanged, and preserves the per-logit learning rate across rows. Hardening
uses probability `>=0.5`; the emitted table is checked for all eight rows.

`workload_manifest()` freezes a bounded named matrix covering seven- and
2,304-node DAGs, coherent and diverse edge-root maps, sparse and dense input
activity, low and high neighborhood/output overlap, frozen and every-step
updates, reuse horizons 1/16/256, and input-change fractions 0/1/10/100%.
Every fixture uses one shared `ConditionDAG` bank. Historical cases have
2,176 condition-edge root bindings; the bounded complete cases have 64.
The seven-node case is four feature leaves, `all(0,1)`, `any(2,3)`, and
`all(0,1,2)`. Its historical root map uses the 1/16 prefix and then roots
1/2/3/5. The 2,304-node case is 128 leaves and 2,176 distinct contiguous
window conjunction roots, all one level above the leaves. Its historical map
binds every root once. Two complete-consumer cases bind 2,176 numerical
groups one-to-one to the historical roots; bounded 64-edge complete cases
bind each group to one declared edge and expose the selected root ancestor
closure for a demand-pruned matched control. Separately labeled 64-group
producer-only cases retain the full historical edge map and must not be
treated as end-to-end consumer cases. Fixtures report the full bank size,
unique demanded roots, and reachable nodes separately. `ConditionDAG` nodes
are topologically ordered and use contiguous child ranges matching native
`NodeV1`.

The requested change fraction applies to the actual leaf-input prefix in the
binary accessibility observations `[cell, feature]`; it does not prescribe
the resulting gate-root change fraction. The fixture reevaluates the actual
shared DAG on before/after input planes and records every bound root decision
and one exact shared-bank dirty-node closure per cell. Distal
neighborhood memberships use explicit feature IDs in the upper half of the
feature axis. These are controlled execution fixtures, not discovered
regulatory biology.
