# Active projection v1

`active_projection_v1` has two exact realizations of the same primal support:
the persistent byte/bit mask route and a compact logical-index map.  The map is
formed in increasing logical-edge order, so it is an explicit identity map and
not a new topology.  Its caller-owned preparation storage makes activity scan,
mapping, transfer and execution costs observable.

`projection_identity_v1` records immutable structure ID/epoch separately from
the shared, cohort, or per-instance activity generation.  A value or activity
generation can refresh a map without representing a structure rebuild; a new
structure epoch requires a replacement map.  A map never authorizes access to
a different epoch.

Primal and parameter-response maps are distinct.  A false predicate excludes
primal evaluation, including an inactive nonfinite value.  A zero coefficient
in `w*x` still retains the parameter response `x` when derivative support says
it is relevant.  Approximate dropping is disabled by default.  When enabled it
requires `bounded`, `empirical`, or `unassessed` provenance; only `bounded`
claims a supplied absolute-error bound.  Dropping is a numerical admission
choice, never a causal-absence assertion.

The baseline authority is FP32.  This support layer does not quantize values;
an FP16 projection belongs only to a consuming route that declares its own
rounding and response policy.
