# Optional sparse application policies

This change exposes two caller-selected policies while preserving the existing
defaults. It does not select a policy from measured density and makes no
general workload or biological claim.

## Logical edge values

`logical_edge_values(policy="workset_union", primal=..., derivative=...,
cells=...)` gathers values only for the unique logical ordinals in P∪D and
returns the existing canonical logical E-vector. Entries outside the union are
zero. The scatter remains differentiable to the shared entity/scorer parameter
blocks. The native relation operation remains responsible for checking the
workset ordinal/identity pairing. Values are derived from live parameter blocks
on each call; no value survives a parameter publication. The default `all`
path remains available and is used by both application forwards unless the
caller opts in with `edge_value_policy="workset_union"`.

## Karyoplast relation input

`KaryoplastEventSlice.forward(..., relation_input="current_event")` retains
the current-event relation input. Opting into `eligible_history` uses the
provider's `work[4]` eligible source occurrences, summed by entity while
preserving repetitions. This choice does not alter event transitions, gate
context, or provider windowing. Requesting history without `work[4]` raises a
clear error. The selected policy is stored in `application_state` and therefore
travels with the existing checkpoint state; a progressed stream rejects a
policy change.

Continue an opted-in history stream, including after checkpoint restore, with
`relation_input="eligible_history"` explicitly. Omitting it requests the
existing current-event default. A legacy checkpoint without a policy key
retains its original current-event meaning.

## Validation

`tests/sparse_support/test_optional_policies.py` includes CPU reference checks
for full versus indexed values and shared-parameter VJPs, derivative-only
candidates, duplicate ordinal normalization, empty unions, repeated history
occurrences, and policy locking. Its native-owner integration checks are gated
by `CELLERATOR_RUN_CUDA_TESTS=1` for the scheduler-admitted CUDA regression.
The local run used the frozen v13 package overlay for imports and passed all
six CPU reference tests; two CUDA tests were skipped because this host session
did not have an admitted GPU run. No real-data testing was performed, and no
performance result is claimed here. The subsequent
[v14 admitted integration](../v14-optional-policies/REPORT.md) passed 83 tests
and two subtests, including the three native-owner CUDA policy tests.
