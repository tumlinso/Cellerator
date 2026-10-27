# CE-SIG1-B01: Measure and tune sparse execution regimes

Using finite scheduler-admitted V100 jobs and KDA's experiment workflow, compare complete candidate rescans, incremental postings/intersections/bitmap gating, sparse SIMT, and packed batched forward/backward. Vary batch size, gate density, degree skew, changed-state fraction, window size and support reuse one factor at a time. Measure preparation, proposal/scoring, gating, work construction, packing, forward, derivative-only backward, optimizer publication, support revision, complete inference/training, peak memory and compilation break-even. Use paired CUDA-event measurements on frozen inputs with warmup and repeated samples; retain source/binary identities and profiler diagnostics.

Acceptance: promote a backend only in regimes with repeatable complete-cost benefit at declared numerical and learning quality. Retain measured fallbacks and negative results. No GPU work begins without admitted resource authorization.
