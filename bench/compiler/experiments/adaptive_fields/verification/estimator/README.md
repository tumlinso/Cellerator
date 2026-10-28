# E5 estimator acceptance oracle

## Scope and verdict

This directory contains an **independent CPU mathematical oracle**, not
acceptance of a production field plan. The small oracle validates the declared
STE score VJP, repeated gate-index accumulation, dormant-candidate selection
semantics, and reverse equations for a two-step tanh recurrence. The smoke
replay independently composes the exported six-step event trace in CPU Torch.
It supports native numerical fidelity for the operation's shared-matrix
contract, but it also exposes an unresolved mismatch between that operation
contract and the Python parameter-owner layout.

Run:

```sh
python bench/compiler/experiments/adaptive_fields/verification/estimator/verify_estimator_oracle.py
python bench/compiler/experiments/adaptive_fields/verification/estimator/replay_e5_smoke.py \
  bench/compiler/experiments/adaptive_fields/gpu/evidence/raw/e5_seed20261027_current_event_all.npz
```

Observed on CPU with PyTorch: gate VJP passed at FP64 (`rtol=atol=1e-14`);
two-step recurrent Torch autograd and manually reversed VJPs passed at FP64
(`rtol=atol=1e-13`). The command reports the frozen numerical outputs. It makes
no CUDA calls.

The exported replay uses artifact SHA-256
`39d8b2702750862991efaf7f8aeb0a05af19289db2bbe6b6147f593a4b6752ce`. A CPU
Torch composition using the native operation's declared shared transition
produced maximum absolute errors of `5.27e-4` across per-step latent outputs,
`3.05e-7` on the final relation output, `2.39e-7` on loss, and `3.13e-8` on the
flat owner gradient. The small per-step transition drift accumulates from
FP16 round trips and CUDA-vs-CPU arithmetic; all values are recorded by
`replay_e5_smoke.py`. Native direct and experiment-local stage paths also agree
in the original artifact, but share the same native primitives.

## Estimator contract

For score `s_i`, temperature `T`, probability `p_i = sigmoid(s_i/T)`, and
hard admission `a_i = [p_i >= threshold]`, the existing STE is
`q_i = p_i + stop_gradient(p_i*a_i - p_i)`. Its forward value is `p_i*a_i`;
its declared derivative is `dq_i/ds_i = p_i*(1-p_i)/T`, including for a
closed gate. If derivative work items `j` map to gate candidate `g_j`, the
expected score gradient is

```text
dL/ds_i = p_i*(1-p_i)/T * sum(j where g_j == i, dL/dq_j)
```

The sum is a scatter-add: repeated indices across cells must accumulate. A
dormant candidate omitted from the sampled derivative workset gets zero for
that step; a selected dormant candidate receives its surrogate VJP. This is
exact parity with the declared sampled estimator, not a claim that the sampling
policy is unbiased. No finite difference through the hard threshold is valid
for checking this surrogate derivative.

The script exercises two admitted and two dormant candidates, repeats the two
admitted indices and one selected dormant index across two cells, and leaves a
second dormant index out. It independently accumulates the upstream terms and
applies the analytic sigmoid derivative, then compares that result with Torch
autograd through the explicit STE expression.

## Existing implementation and evidence

At source snapshot `c63f591d04ef74b7566937d161c80936dcc4cf05`:

- [`learned_gates`](../../../../../../python/cellerator/torch/sparse_support.py)
  implements the thresholded forward value and explicit STE. Workset selection
  builds separate primal and derivative worksets, with admitted candidates and
  bounded dormant exploration.
- [`test_cuda_training.py`](../../../../../../tests/sparse_support/test_cuda_training.py)
  compares native sparse relation output, input/weight/score VJPs with an
  independent Torch composition; it verifies a selected dormant gate gets a
  score VJP, an omitted dormant gate does not, repeated gate indices accumulate,
  and finite-differences are not used. A separate check compares one latent
  transition's forward and VJP against Torch composition.
- [`test_cuda_lifecycle_extra.py`](../../../../../../tests/sparse_support/test_cuda_lifecycle_extra.py)
  checks the flat canonical parameter owner survives guarded updates,
  parameter-reader tapes constrain support publication, and a checkpoint after
  promotion/revocation restores owner values, support identity/epoch, optimizer
  state, and subsequent Adam continuation.
- [`test_native_owner_protocol.py`](../../../../../../tests/sparse_support/test_native_owner_protocol.py)
  supplies CPU fake-owner checks that guarded publication preserves one owner
  leaf and checkpoints restore optimizer/window state. It is lifecycle protocol
  coverage, not the real CUDA sparse owner.
- [`guarded_step`](../../../../../../python/cellerator/torch/mechanism.py) requires
  every native owner in the optimizer, checks owner identity/version and
  preflight, rejects non-finite gradients before mutation, performs one
  optimizer step, and publishes each changed owner once. Checkpoint load
  validates native-owner identity/metadata and prepares state before mutation.

The raw trace exposes a model contract conflict: the authoritative C++ operation
declaration calls the transition shared ([`training.hh`](../../../../../../include/Cellerator/compute/operation/sparse_support/training.hh:277)),
and its CUDA forward/VJP indexes a single `A`, `B`, and `b`
([`training.cu`](../../../../../../src/compute/operation/sparse_support/training.cu:421)).
The Python binding nevertheless requires 3D per-mechanism matrix tensors
([`module.cc`](../../../../../../bindings/sparse_support/module.cc:706)), and the
Python owner allocates `transition_A/B` as `[mechanisms,width,width]` and bias
as `[mechanisms,width]` ([`sparse_support.py`](../../../../../../python/cellerator/torch/sparse_support.py:64)). The accepted extra slices
are not read or differentiated by this kernel. The replay follows the explicit
native shared-matrix contract and uses the first slice; that verifies native
operation fidelity but cannot establish the intended semantics of all
trainable transition parameters. No production change is in this experiment's
scope.

## Coverage boundary and open cases

The repository already has unusually strong isolated E5 pieces. The smoke is
one seed and one `current_event`/`all`-values fixture; the raw artifact records
no repeated derivative gate index, no support publication or checkpoint
continuation, and no independent Torch oracle was part of the GPU job. The
current evidence does **not** by itself prove:

- A multi-step ranked/event recurrent training sequence matches a direct Torch
  composition under both current-event and eligible-history inputs.
- Field-visible staging preserves exact primal/derivative identities,
  per-step tape dependencies, and explicit estimator semantics end to end.
- Full-support and indexed-union value policies produce identical gradients
  across those complete recurrent fixtures.
- A structural publication between pinned windows preserves the complete
  recurrent/optimizer/checkpoint trace in the proposed field-plan execution.

The standalone two-step check does not use the native owner. The CPU replay
does use width-64 FP16 input/storage conversions and FP32 accumulation, but its
semantics remain bounded by the shared-versus-per-mechanism owner mismatch
above. Further E5 acceptance needs that contract resolved, then must cover all
three deterministic seeds, both relation-input policies, full-support and
indexed-union values, exact repeated gate-index accumulation, and a support
publication/checkpoint continuation under the scheduled GPU run.
