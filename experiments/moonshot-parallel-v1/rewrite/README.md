# Supplied linear rewrite and live residual capacity

Run the required CPU check with the installed Torch environment:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 /home/tumlinson/Software/venvs/cellerator-ml2-py313-cu126/bin/python -B experiments/moonshot-parallel-v1/rewrite/check_rewrite.py
```

`SnapshotController` owns a defensively cloned state, linear law, readout, ordered
coordinate names and incarnations, epochs, and physical replica maps. A tape
acquired through the controller builds a real autograd graph. It yields state,
law and readout gradients, and retains its publication interlock until backward
or explicit release. A rejected backward retains the tape. Direct tape
construction is rejected. Publication and restore require drained tapes and the
current ordered coordinate identities and epoch.

A supplied invertible map `q = T s` publishes `T A T^-1`, `R T^-1`, and `T s`.
A supplied quotient checks `P E = I`, `A E = E (P A E)`, and the current state's
membership in the invariant manifold before publishing `P A E`, `R E`, and `P s`.
Equality is checked with explicit float32/float64 numerical tolerances. Physical
projections rebuild from the new declaration. Replicas gather coherent owner
values and scatter adjoints with a sum, including repeated indices.

`ResidualRegrowth` adds `V tanh(U x)`. Seeded U is nonzero and V starts at zero.
The first loss gives outgoing V a gradient; incoming U becomes trainable after
an external optimizer step. Recycling zero-output slots preserves the current
function. Discarding a live slot requires explicit caller authorization.
Recycling validates the affected incoming and outgoing moment storage before
changing any coefficient. The caller drains tapes before an in-place recycle.

`refactor_training_transform` and `refactor_training_quotient` stage fresh owners
for the entire declared model. They migrate residual incoming coefficients as
`U T^-1` and `U E`, reset their incompatible Adam moments, and retain outgoing V
coefficients and moments. They return `(controller, branch, optimizer)` for the
caller to install after success; existing owners stay usable if staging fails.
These operations require the source controller's tapes to be drained. The
checkpoint records state, law, readout, coordinate incarnations, replica maps,
epochs, residual coefficients/seed, and external Adam state. Restore creates
fresh owners and validates shapes, dtypes, initialized moment completeness,
nonnegative second moments and hyperparameters. The supported checkpoint has a
single Adam parameter group in declared U,V order; other orderings are rejected.

The checks cover outputs and state/parameter gradients, supplied map witnesses,
off-manifold rejection, publication failure preserving owners, physical replica
adjoints, live regrowth and loss reduction, partial moment resets, serialized
restore, and the next optimizer step matching uninterrupted execution. The
source-bound capability receipt is generated only after all three suites pass.

This is a same-thread CPU experiment with external synchronization. Adam's
scalar step survives coordinate resets, so bias correction age is retained.
The quotient preserves the supplied invariant domain and its corresponding
responses. General nonlinear discovery, biological equivalence, native model
publication, native concurrent execution and CUDA execution are unqualified.
