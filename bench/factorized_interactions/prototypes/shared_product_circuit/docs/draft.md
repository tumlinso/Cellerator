# Shared-product circuit: bounded prototype

User-authorized continuation, 2026-11-03. Deliver one source-bound measured
comparison, an independent interpretation, then stop for discussion. This
does not reopen the previous factorized or chromatin campaigns.

## Fixed law and experiment

Eight mechanisms, one target, channels D=32; measured C=1024, correctness-only
C=64. Mechanism m reads [0,1,2+m], alpha=m%4, bias=m%3, context terms
[(0,m%4),(1,(m+1)%4)]. One repeated-role fixture replaces tail 0 with input 0.
Inputs x[10,D,C], z[2,C], alpha[4,D], bias[3], gamma[4] are runtime FP16.
Output y[1,D,C], cotangent and all five gradients are FP32. All arithmetic
and reductions are FP32; no fast-math or divisions by participant values.

Candidate: p=a*b; S=sum_m(gate_m*alpha_m*c_m); y=p*S. The matched unfactored
control evaluates and sums the eight complete products. Both share ownership,
gate/slope routine, launches, scratch layouts and compensated reduction helpers.
Allow ordinary compiler common-subexpression elimination in both. Recompute S
in backward; keep the existing two-plane gate/slope tape design. Explicit
owners reduce tied alpha/bias/gamma and context gradients; no input atomics.
For the fixed FP16 input storage, a*b is exactly representable in FP32. Both
arms explicitly compute that common product once: the unfactored arm keeps it
inside every complete contribution, while the circuit pulls it outside the
output sum and common-input adjoint sums. Shared tail/parameter/gate partials
use the same common product. This initial source-level choice avoids making
the compensated helper obstruct a conventional common-subexpression control.
The existing native direct request is a descriptive third arm, with actual
backend recorded (possibly stable), not the causal control for factorization.

## Ownership and private interface

Principal owns CUDA arms, native runner and CPU-only private build helper.
Fresh verifier owns fixtures, validation/measurement driver and its tests.
The designated operator alone owns Autobench registration and GPU submissions.
Root owns this contract, initial interpretation, final report and acceptance.
All additions stay in this prototype directory. Existing oracle, wire format,
public APIs, packages, bindings, defaults and paused definitions remain intact.
Peers communicate directly about interfaces. Preserve concurrent/unknown work.

Runner uses existing CFINT001/CFOUT001 conventions, with candidate selection
unfactored|circuit|native_direct and explicit unsupported-case rejection.
Principal and verifier settle any CLI extensions directly before use, retaining
identical measured work and independent validation. CPU-only build outputs
carry source/dependency/toolchain/binary identities; no unscheduled CUDA work.

## Validation and measurement

Reuse tests/factorized_interactions/reference.py unchanged. Acceptance for FP32
output/VJPs: abs(error) <= 2e-6 + 2e-5*sum_abs_reference_contributions, evaluated
from storage-quantized values. Freeze fixtures before device work: two seeded
value sets, zeros, signed cancellation, saturated gates, repeated tail role,
overwrite/accumulate/repeated VJPs and changed parameters without recompilation.
Check actual candidate identity and all six returned tensors. Targeted memcheck,
racecheck, initcheck and synccheck precede timing. Scope claims to this prototype.
Parameter-refresh checks use changed FP16 parameter values with the same frozen
executable/manifest/topology in separate invocations; they do not claim an
in-process prepared-owner mutation/lifetime test.

Ten warmups per arm; seven paired blocks of 30 complete forward/VJP evaluations
at C=1024. Synchronize at each block end. Alternate circuit/control order and
rotate native's position. Primary metric is uninstrumented resident full
forward+five-VJP host time per evaluation. Diagnostic forward/backward event
measurements are separate; report preparation/upload and memory separately.
Retain raw samples, case/source/binary identities, actual backend, UUID and
original finished execution/cleanup receipts. No optimizer/cold-step claim.

Use one admitted V100 through Autobench/Scheduler, explicit UUID and clean timing
preconditions. CPU-heavy builds/analysis stop for timing. At most THREE admitted
GPU attempts, each at most 180 seconds: validation, measurement, one corrective
retry. Shared worker scratch stays below 128 MiB. Only one correctness/plumbing
repair pass is allowed; no performance-driven revisions. Stop at the first
trustworthy result, or at the bound with preserved failure evidence.

## Deliverable and review

Root first records timings/errors/assumptions and its interpretation. A fresh
reviewer receives fixed raw evidence before root's verdict, challenges the
interpretation, then exchanges conclusions. Return the measured result or
bounded failure and any disagreement to the user. Do not automatically enlarge
the motif, change shapes, tune kernels, package the prototype, or start another
experiment. No commit/push is requested.

Guidance: installed CUDA skill, host-execution rules, KDA agent entrypoint,
basic-flow prompt and experiment engine. Use existing lightweight candidate
evaluation support where applicable; do not mutate an old qualified collector's
identity or build another experiment platform for this single comparison.
