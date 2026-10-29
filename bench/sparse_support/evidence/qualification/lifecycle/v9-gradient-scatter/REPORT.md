# Sparse support v9 lifecycle and training regression

## Result

The frozen v9 Python suite passed **26/26 tests** on the scheduler-admitted GPU3 job. The suite included the repaired derivative gate-gradient scatter, the independent full-vector Torch VJP comparison, repeated gate-index accumulation across two cells, lifecycle/checkpoint regressions, native indexing bindings, and synthetic-driver regressions. No tests were skipped.

## Execution evidence

- Scheduler job: `7df35f8b-2bf1-4878-8bcd-efcd45932fae`; attempt: `4534f1d7-2761-4462-8ea0-abe54205ed4f`.
- Result: `succeeded`; `cleanup_verified=true`.
- GPU: `GPU-6c1cac7f-a360-0aef-ba98-2828bfd1db1a` (exclusive, 6 GiB requested).
- Preconditions: 60-second quiescence configured; scheduler recorded three consecutive idle samples at 0% utilization before execution.
- Duration: pytest reported 4.59 seconds.
- Frozen package identity: `be20f601882e918cdd6c74ace0843234a1b4a5c1fd74e2036cfb476222f86fa2`; `_torch` extension SHA256: `93dd66d4559e455d2fda7be4bd95b79d2ca9d460e7cd309d2978706a4ab958d2`.
- New regression source SHA256: `9077cbd808a258eb04085c3ce346a4e88f04dd5564c03451091b33d70c3a1eac`.

The regression checks the complete score-gradient vector against a Torch surrogate VJP. It also repeats derivative gate indices across two cells and compares the accumulated full-vector gradient against Torch autograd. This validates the high-level binding scatter behavior for the covered shape and dtype.

## Limits

This is a focused correctness suite, not broad model-learning or performance evidence. It does not establish correctness for all workset sizes, sparse patterns, stream topologies, precisions, or optimizer trajectories. The 50-epoch learning claim requires the separate post-fix run owned by the training agent.
