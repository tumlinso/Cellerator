# NF1 developmental native seam, version 1

The public contract is
`Cellerator/compute/operation/native_foundation_contract.hh`, namespace
`cellerator::compute::operation::nf1`. It composes operation-core v2 numerical
and determinism policy, execution identity and effect contracts, and existing
prepared-program callbacks. It does not define another session, allocator,
tensor container, code generator or backend catalog.

| Capability | Contract and owner | Acceptance still required |
| --- | --- | --- |
| Multi-operand operation | `operation_contract`, `operand_signature`, `output_signature`; operation-core contract owner | Real nonlinear n-ary host/CUDA providers; ordered inputs, shared/repeated incidence and output assembly tests |
| Independent instances | `prepared_identity`, `generation_stamp`, `instance_binding`; existing prepared/value-plane owners | Two independent values/state bindings share immutable topology, with real lifetime/readiness and stale-generation tests |
| Differential actions | `derivative_request`, `primal_record`; differential operation owner | Supported JVP/VJP/second directions against independent FP64 oracle, including width33 and stale/nonfinite/branch controls |
| Support activity | `support_contract`, `invalidation_for`; existing support/projection owner | Persistent versus compact-active real executions, zero-primal response witness, exact epoch change, independently bounded dropping |
| Custom blocks | `compiled_block`, `bind_compiled_stage`; existing `prepared_program_v2` owner | Grouped provider registration, runtime views and CUDA execution when advertised; unsupported derivative returns failure |
| Consumer build | proposed `Cellerator::native_foundation`; B-lane fragments and M-lane root integration | Source-linked minimal host/CUDA targets, installed/exported native consumer, real nonempty named CTests; no current target claim |

Definition IDs refer to code/semantics; structure IDs and epochs refer to
immutable support; value-instance IDs and generations refer to independently
bound state. Explicit parameter tie groups convey caller intent; equal pointers,
values or code never infer it. Instance descriptors carry no pointers and do
not own data. Actual runtime launches remain subject to the existing residency,
capacity, structure and alias checks, including overlap checks required by the
owning runtime. A semantic descriptor match is not permission to dereference
unvalidated memory.

Every stage reads one declared snapshot. Alias permissions do not relax this
rule. Output roles identify destinations; `assembly_owner` identifies their
unique writer in this operation, and the existing output-effect policy states
initialization/accumulation. Duplicate destination roles are rejected. Stages
and their declared dependencies use the existing prepared-program graph.

Primal records bind state, parameters, forcing, context, activity and branch
versions. Zero optional stamps mean the corresponding dependency is absent;
a provider declaring complete dependencies must not omit an input it reads.
Unknown dependency/effect declarations disable fusion and memoization.
`permits_result_reuse` additionally requires the full matching valid primal
record and definition, not merely a shared prepared pointer. Values/masks and
structure epochs invalidate different preparation products.

Derivative capability bits must match actual callbacks. Mathematical derivatives
at stored values are distinct from differentiating rounding; through-rounding
and unsupported nonsmooth requests reject. Direction and response axes/scales,
and differentiated object (vector field, discrete step, observation or actual
rollout), are explicit. A null derivative callback is never numerical zero.
A numerical zero contribution cannot delete response support without proof;
predicate exclusion and multiplication by zero remain distinct for NaN/Inf.

Precompiled blocks supply a provider-defined typed payload through the existing
`launch_binding_v2` and one callback per prepared group. Five host contract test
programs validate this seam. One test executes a real consumer-defined nonlinear
3-input/2-output callback at width33 through `execute_prepared_program_v2` and
checks double-precision expressions. This is a bounded host callback witness,
not the full native foundation, a CUDA provider, a JIT or automatic differentiation.

M10 publishes the developmental interface receipt after native integration.
The C06 source manifest provides source/interface hashes for that publication;
it is not producer checkpoint authority. NARY, DIFFERENTIAL and SUPPORT drafts
must bind this concrete header rather than absent future implementation paths.
Consumer-build publication describes the required target and root test hook;
the target itself remains B-lane implementation work. Every interface revision
must update its hash, affected consumers and qualification; provisional v1
names do not authorize silently breaking a consumed receipt.
