# GPU1 attempt 01

Scheduler job `b0cc669c-569d-47a4-b7c1-553e9426b012` reached the runner but
exited before correctness comparisons or timing with
`compiled predicate-edge posting fanout mismatch`. The runner had omitted the
prepared raw edge-posting count on each ChangeSet. The fixture's independently
stored fanout is now checked while that count is accumulated. This attempt is
retained for provenance only and contributes no performance evidence.

The scheduler receipt records terminal failure and verified cleanup.
