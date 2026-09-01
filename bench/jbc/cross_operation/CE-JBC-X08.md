# Cross-operation projection-family disposition

`cross_operation_pareto_artifact_v1` is the lane's integration artifact. It
admits at most 64 independently certified, complete-cost measurements over one
exact support family and operation set. Its Pareto metrics are amortized
end-to-end latency, persistent bytes, transient bytes, and launch count.

Promotion is conservative: a generalized family is promoted only when every
non-dominated candidate is generalized. A specialized-only frontier retains
the specialized family; a mixed frontier retains measured plurality. This
artifact defines the decision procedure and contains no hardware performance
claim. Production evidence must supply the hardware, toolchain, workload,
warmup/repeat, correctness, and complete-cost identities carried by each
candidate measurement.
