# Counted flat all-of condition probe

This isolated CPU experiment checks a narrow alternative to evaluating dirty
condition roots: for flat all-of rules, keep a per-cell satisfied-participant
count and update it through predicate-to-participant postings. It does not
modify Cellerator production code or cover nested all-of/any-of/not DAGs.

## Methods

The executable compares three complete update paths over the same deterministic
rules, initial predicate values, and batches of toggles:

1. **Full scan:** apply toggles to each changed cell, then reevaluate every rule
   for that cell.
2. **Dirty recompute:** build reverse predicate postings, deduplicate the rules
   reached by each changed cell's toggles, and reevaluate each dirty rule from
   its ordered participants.
3. **Counted:** build the same postings, update the satisfied count for every
   participant occurrence reached by a toggle, deduplicate touched rules, then
   refresh their eligibility bits.

Postings retain repeated participant occurrences. A repeated predicate can
therefore increment or decrement the same rule counter more than once; rules
with positive and negative requirements are also represented exactly. Empty
all-of rules start eligible and remain so because no predicate can dirty them.
Updates are batches: all toggles for a cell are applied before dirty rules or
eligibility bits are finalized.

`--suite` first checks both incremental paths against an independent scalar
rule evaluator. Focused cases cover overlap, repeated participants, negative
participants, empty and all-true rules, contradictory requirements, and
multiple toggles of one predicate within a batch. A deterministic generated
case adds 100 batches with repeated and overlapping references. The timed
fixtures then compare the three strategies on fixed shapes.

## Reproduction

```sh
cmake -S bench/sparse_support/experiments/counted_conditions \
  -B /tmp/counted-condition-probe-build
cmake --build /tmp/counted-condition-probe-build -j2
/tmp/counted-condition-probe-build/counted_condition_probe --suite \
  > bench/sparse_support/experiments/counted_conditions/evidence/results-20261027.jsonl
# Repeat the suite into a separate artifact to compare run-to-run variation.
/tmp/counted-condition-probe-build/counted_condition_probe --suite \
  > bench/sparse_support/experiments/counted_conditions/evidence/results-repeat-20261027.jsonl
```

The captured run used GNU C++ 13.3.0 with `-O3` on an Intel Xeon E5-2698 v4
2.20 GHz host. It was CPU-only, single-threaded, and not pinned to a CPU. Each
update timing is the median of five trials. The first five fixtures process 600
batches per trial; the dense-toggle fixture processes 240. Each preparation
timing is the median of five engine constructions. Figures are microbenchmark
evidence for these fixtures, not a production-path claim.
The repository HEAD was `fc47b98d501c8c54890ad64e79bcbdea58399ecf`; exact
source and evidence file hashes are retained in `evidence/sha256.txt`.

## Results

Times below are median microseconds per batch. Estimated break-even is the
extra counted preparation time over dirty recompute divided by steady-state
time saved per batch. A fixture amortizes that preparation only when its
configured reuse count reaches the estimated break-even.

| Fixture | Full scan | Dirty recompute | Counted | Counted prep (dirty prep) | Extra persistent payload | Estimated break-even |
|---|---:|---:|---:|---:|---:|---:|
| Sparse incidence | 55.756 | 0.565 | 0.360 | 1.352 ms (0.584 ms) | 256 KiB | 3,745 batches |
| Shared, long rules, 95% participant match | 67.409 | 5.283 | 1.359 | 1.109 ms (0.574 ms) | 128 KiB | 136 batches |
| Shared, long rules, 50% participant match | 41.685 | 3.783 | 2.101 | 0.711 ms (0.275 ms) | 128 KiB | 259 batches |
| Shared, long rules, six toggles per cell | 52.474 | 30.732 | 8.898 | 1.018 ms (0.473 ms) | 128 KiB | 25 batches |
| Shared, short rules (arity 2) | 19.117 | 0.350 | 0.227 | 0.232 ms (0.133 ms) | 128 KiB | 807 batches |
| Shared, long rules, dense 64-toggle batches | 10.439 | 39.510 | 98.749 | 0.331 ms (0.185 ms) | 64 KiB | No crossover |

Counted had lower steady-state update time in the first five shapes. At 600
batches, preparation leaves the sparse-incidence and short-rule cases net
slower; the long, shared-rule cases repay it. Their counted break-even falls
between roughly 25 and 259 updates, depending on toggle count and participant
match rate. The high-pass batched fixture shows the largest gain: dirty
recompute checked about 3.29 million participants across the run, while the
count path maintained about 922 thousand participant counters.

The dense-toggle fixture is the counterexample: each of two cells receives 64
toggles over 64 predicates in one batch. Counted takes 123.4 microseconds per
batch, versus 40.2 microseconds for dirty recompute and 11.6 microseconds for
scanning every rule in each changed cell. The count path visits about 16.4k
participant postings per batch; dirty recompute deduplicates those to 1,024
rules and checks about 2.1k participants. Full scan avoids the posting walk
altogether. Counted is therefore not a universal replacement, even for flat
all-of rules.

The separate repeat run preserved every row's strategy ordering. Dense-case
update times varied with the unpinned CPU (counted 92.5–98.7 microseconds,
dirty 37.8–39.5, full scan 10.4–10.6); the gap and conclusion persisted.

Counted stores one 32-bit counter for every cell-rule pair. Its extra persistent
payload is therefore exactly `4 * cell_count * rule_count` bytes in these
fixtures, before allocator and container overhead. The rule and posting tables
are shared in concept between the two indexed methods; the counted method also
initializes every cell-rule counter during preparation. The JSONL file records
the operation counts, preparation time, persistent payload sizes, and shared
toggle-batch payload for each result. The derived break-even ignores changes in
allocation policy, downstream bitmap/workset costs, cache state, and state
lifecycle costs.

## Interpretation boundary

This supports only a possible specialization when rules are flat all-of, few
predicates change per update, shared postings reach long rules, and the
prepared counters will be reused long enough to repay both initialization and
`4 * cells * rules` bytes of extra state. A dirty predicate that reaches many
long rules is the strongest measured case. Dense toggles favor recomputing a
deduplicated dirty set or scanning the changed cells. Short rules or few reused
updates also favor recomputation after preparation is counted.

The existing condition program supports nested all-of/any-of/not nodes, so
this prototype cannot replace its general evaluator. A production version
would need an explicit flat-rule eligibility contract, counter-width bounds,
and a sound reset/invalidation rule whenever the cell predicate state or
structure epoch changes. None of those integration costs is represented by
these timings. No GPU, CUDA, biological interpretation, or real workload
trace was used.
