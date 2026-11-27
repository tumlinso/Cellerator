# Working on Cellerator neighbors

Read the [cellular geometry definition](../../cellular_geometry.md) alongside
the [repository work contract](../../AGENTS.md). The [component README](README.md)
describes the existing routed search. The ideas below are directions to explore.

## What makes one cell a neighbor of another?

For a given question, what evidence can cell *j* lend to focal cell *i*?
Sparse measurements of different cells can reveal complementary pieces of a
shared program. Two cells can be good neighbors for one output and poor
neighbors for another. Similar expression today can also conceal different
directions of change. A neighbor relation can be directed, depend on state,
and reach across physical tiles. Each focal cell keeps its own observations
and output alongside any context it borrows.

One way to discover such a relation starts with sparse counts and competing
matching rules. Hide B in the focal cell before computing depth, retrieving
candidates, or scoring matches. Give rules based on A, C, and their combination
the same broad source pool, including cells outside an initial PCA neighborhood.
Use the sources' B measurements to predict the hidden count, sampling targets
regardless of whether the count is zero. If a rule also improves prediction
of independently hidden D and E, that suggests a shared program. Compare with
focal-only prediction and with matched, shuffled sources to learn what choosing
those particular cells contributed. A cell may have little to gain from
borrowing for a particular program.

## What smoothing does to learned dynamics

For one program, let $P_{ij}$ be nonnegative source weights summing to one
for each focal cell. The pooled view $h_i=\sum_j P_{ij}x_j$ can expose
features rarely detected together in one measurement. Keeping $x_i$
alongside $h_i$ also lets a model learn from the focal cell's departure
from its neighborhood. A sparse count that breaks local agreement may be the
first evidence of a branch.

For a nonlinear dynamics model $f$, evaluating the dynamics at the pooled
state $f(\sum_j P_{ij}x_j)$ generally differs from pooling the predicted changes
$\sum_j P_{ij}f(x_j)$. Consider two equally weighted cells whose projected
velocities point in opposite directions. Their mean velocity is zero even
though both are moving. Fitting that mean alone can hide both futures. Cells
on opposite sides of a possible separatrix may therefore be especially useful
to **compare**, even where averaging them would erase the distinction.

The weights can move too. Along smooth trajectories with differentiable
weights, $\dot h_i=\sum_j P_{ij}\dot x_j+\sum_j \dot P_{ij}x_j$.
The second term is change caused by reweighting the neighborhood. A hard
neighbor swap can instead produce a jump in the pooled view. When interpreting
motion, specify whether the relation was held fixed or recomputed.

Ask locally what prediction loses when two neighborhoods are merged. A
well-sampled basin may support broad pooling. A narrow transition may warrant
fine resolution across its boundary and broad pooling along it. The target
determines which claim can be tested. Held-out counts assess reconstruction.
Time-resolved or lineage observations can test whether pooling preserves
different futures. When learning both the relation and the dynamics, judge
predictions against observations held out before smoothing. A smoother can
make its own targets easier to fit by erasing a rare branch.

## Machinery worth exploring

Several rules can share one candidate list and one gather of reference
features. Score the rules together, normalize each over its candidate cells,
and apply their weights to a block of target genes. The resulting rule-by-gene
tile tests which rules transfer. Reuse scores across targets when the same
conditioning inputs were visible. Keep cell and feature identities through
packing so physical reordering alone preserves the calculation. Use
caller-owned storage and keep reusable numerical contractions with their ops.
Rule families, search strategy, and tile shape are experimental choices.

One possible downstream use is to generate a species map from relationships
that recur across datasets, including the contexts in which they change.
Neighbor discovery can begin from an arbitrary lossless packing.
