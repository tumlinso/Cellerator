# V100 regulatory gate validation draft

Target: Tesla V100, SM70, CUDA 12.9. Test the word-mask rule gate and the
experimental 32-instance bitplane gate. No performance promotion is implied by
this validation.

## Contracts and references

For each `(instance, rule)` occurrence, eligibility equals the accessibility
bit AND all required inputs active AND all forbidden inputs inactive. The
independent reference is a scalar loop over the original rule and boolean state
vectors, not the packed preparation code. For bitplanes, evaluate the same
predicate for each of 32 cells per `(rule, instance-word)`, then compare every
valid cell with that scalar result. Invalid tail cells must be zero; duplicate
work items must yield duplicate outputs.

Cover multiple feature and rule words, a partial 32-cell tail, sparse and empty
rules, optional all-open accessibility, invalid occurrence and rule contents,
insufficient capacity, empty work, and duplicate work. Keep standard occurrence
compaction and bitplane output as separate APIs and inspect status before using
any output.

## Baseline and candidates

The current standard gate loops over active 64-bit mask words per occurrence;
the experimental bitplane kernel intersects 32 instances in one 32-bit word
per rule/instance-word. Preserve the current standard gate as the baseline.
Compare identical predicates over the same cell/rule matrix. Report resident
evaluation separately from feature-major conversion, allocation, and any result
expansion to cell/rule occurrences. Repeat for sparse worklists and small versus
large batches. The bitplane path is useful only when reuse of the converted
feature planes pays for conversion and its caller can consume packed outputs.

## Commands and promotion

Configure a private build with CUDA 12.9, C++17, `CMAKE_CUDA_ARCHITECTURES=70`,
and `CELLERATOR_BUILD_FRAMEWORK_CUDA_SMOKE=ON`; build only
`cellerator_framework_cuda_regulatory_gates_smoke`. Root owns Scheduler
admission and execution. Run correctness before Compute Sanitizer memcheck and
racecheck. Candidate performance requires repeated admitted measurements on a
quiescent V100 against the same predicates, input density, and output
requirements. Retain only changes that pass the independent scalar oracle and
improve complete cost in a declared regime; otherwise preserve the existing
implementation and report the boundary.
