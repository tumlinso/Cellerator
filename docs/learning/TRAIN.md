# CE-ML2-TRAIN implementation contract

Approved scope: a native prepared product mechanism, `r[b,m] = k[p(m)] *
product_j x[b,i(m,j)]`, with declared additive output assembly. Biological
domain/order/geometry/partition identities are checked independently of shape.
Repeated argument slots and shared coefficient identities retain multiplicity.
Preparation owns forward and reverse incidence; value changes do not rebuild it.

Native Cellerator owns numerical kernels, FP32 canonical coefficients, an FP16
derived plane, saved execution operands, read tickets and guarded publication.
The existing execution session owns allocations; prepared_program_v2 dispatches
stages. The optional `cellerator.torch` adapter owns dispatcher registration, first-order
autograd, modules, stock optimizer integration and framework checkpoint handling.
Cellerator owns the native mechanism and parameter state. The old N16 route is retained.

Mixed precision evaluates stored half operands with FP32 arithmetic. Backward
uses those stored values with an identity straight-through surrogate for
quantization and zero-safe product derivatives. Coefficient gradients stay
FP32; input gradients accumulate in FP32 and cast only to the original input
dtype. A reference uses `v + (v.half().float() - v).detach()` rather than allowing
an intermediate half gradient to round the master gradient. This is not the
mathematical derivative of discrete rounding. One backward is supported
per forward. Outstanding tapes prevent updates; abandoned tapes release safely.
Completion events order storage reuse and writes across streams.

The supported mutation boundary runs stock Adam exactly once after preflight,
then refreshes the half plane and publishes a new value generation. A failed
partial update poisons state; restoration requires a whole model/optimizer
checkpoint. Detectable unsanctioned tensor mutations are rejected; raw pointer
or `.data` bypass is outside the guarantee. Scaler skips do not publish a new
generation. Checkpoints contain logical declarations/IDs and optimizer state,
never pointers or device caches.

Qualification includes independent references, full gradients, irregular
arities, zeros, identity failures, accumulation, shared use, stream ordering,
stale tensors, replay rejection, installed C++/Python composition, save/reload
and next-step parity, sanitizer checks and whole-lifecycle timing. No dataset
is needed. No speedup is presumed.

Execution note: the exclusive CE-ML2-L-EXEC claim uses the registered Cellerator
checkout. The user clarified that configured Codex subagents perform delegated
implementation and review; local workers are reserved for Project Control
observers. Root retains architecture, integration and task acceptance. No
GlassHelix implementation or CE-ML2-BIO work is authorized by this task.
