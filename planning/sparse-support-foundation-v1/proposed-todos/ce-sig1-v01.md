# CE-SIG1-V01: Validate correctness, learning, and lifecycle

Build independent CPU and Torch references for indexes, windows, conditions, support identity, packed forward/backward, recurrent latent state and declared straight-through estimator. Use finite differences only for smooth components and validate pruning gradients against the declared estimator. Exercise stream ordering, live tape rejection, optimizer identity, checkpoint continuation, promotion/revocation, sanitizer coverage and held-out learning on both synthetic fixtures.

Acceptance: all correctness and learning criteria in the scope are evidenced; report discovery coverage, pruning quality, numerical error and learning controls separately. Do not substitute a reference-only or skipped-device run for compiled execution acceptance.
