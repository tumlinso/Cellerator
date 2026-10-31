# V100 changed-source frontier validation and optimization

## Contract and baseline

For device-resident CSR reverse incidence arrays, a list of changed source
ordinals must produce the exact set of incident interaction ordinals. Output
order is unspecified. Duplicate changed sources, repeated incidences and
overlapping rows must not duplicate outputs. Invalid selected rows, source
ordinals, or interaction ordinals set a status flag; insufficient output
capacity reports the exact union cardinality, and every failure sets
`written == 0`. Invalid content does not suppress writes already in flight,
so flagged storage must not be consumed. Valid pointers, capacities and an
explicit stream are checked synchronously. Eligibility is a later caller-owned
decision.

The current implementation clears one claim word per interaction, traverses
each selected posting row with one warp, uses a device-wide CAS to deduplicate,
and atomically claims output positions. The focused smoke test is the baseline;
an independent `std::set` oracle checks returned worklists. Costs include
`O(interaction_count)` claim initialization, selected-row reads, atomics,
output writes, and host/device setup. No prior frontier timing or GPU result is
recorded.

## Risks and candidate directions

- Validate duplicates, many short rows, a very long row, empty dimensions,
  31/32/33 element tails, malformed selected row boundaries, invalid posting
  ordinals, mixed valid/error work, and exact/undersized/zero output capacity.
- Candidate A keeps the current warp-per-source path. Candidate B compares
  against a device gather followed by Thrust sort/unique; include its temporary
  output allocation or reuse and full comparison costs. Candidate B may help
  only in regimes where atomic contention is high.
- Consider changing the production kernel only after the admitted V100 results
  show a repeatable, material advantage across representative sparse and
  dense-degree shapes, with identical oracle output and acceptable complete
  costs. Retain the existing path when evidence does not select one globally.
- GPU job admission, UUID selection, execution, sanitizer, and timing belong to
  the root and Device Scheduler. A visible device or a compiled binary is not
  launch authority.

## Validation and promotion

Compile only `sm_70`. Run the smoke matrix against an independent CPU union
oracle, then Compute Sanitizer memcheck and racecheck on an admitted GPU. Keep
correctness separate from timing. A performance packet should vary source
count, interaction count, selected-source count, and posting-degree skew; warm
both methods, repeat stable resident inputs, and report scratch clearing,
temporary buffers, output capacity, launches, allocations, and host setup.
Optimize only a confirmed bottleneck, rerun the full oracle and sanitizer
checks, and retain a measured winner over the existing method. No GPU execution
or performance goal has been asserted yet.
