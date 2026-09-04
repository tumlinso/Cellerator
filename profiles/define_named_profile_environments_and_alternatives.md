# Named profile environments v1

A profile environment has one stable identity and a bounded table of named
semantic states. Each state has independent state, name, evidence, and optional
branch-condition identities plus a non-negative prior weight. Baseline,
activated, perturbed, and unknown are flags rather than distinct program IRs.

Aliases map additional stable symbolic names to existing state identities. An
explicit default selects one state. Validation rejects duplicate identities or
names, dangling and duplicate aliases, missing defaults, and invalid weights.
The environment view owns no storage and carries no source path or duplicated
program representation.
