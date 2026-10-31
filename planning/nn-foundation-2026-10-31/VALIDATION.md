# NN foundation validation, 2026-10-31

The first foundation implements a general information-bound `Cell`, explicit
`DynamicalCell`/`DynamicalModel` compatibility, feature encoding, dense/packed/CSR
relation attention, shared auxiliary views, and native grouped-linear autograd.
The models remain ordinary callable or Torch module compositions. This is a
framework milestone, not scientific validation of a cellular model.

## CPU and model integration

The combined framework suite ran **244 tests: 236 passed, 8 skipped**, with no
failures or errors. The skipped cases were six CUDA cases exercised separately
on admitted hardware, one explicitly opt-in two-process Gloo check, and one
optional native modeling component unavailable in this CPU package.
[`cpu-validation.json`](cpu-validation.json) records the exact test identities,
source hashes, Python/Torch versions, and compatible native CPU extension hash.

The new NN tests cover hidden-value masking, measured zero, identity-preserving
reordering, dense/CSR agreement, repeated participants, group isolation, explicit
cross-cell support through caller-defined groups, empty neighborhoods, causal
prediction, shifted event labels, and leading/consecutive empty histories.
They also cover shared readout evaluation, frozen and joint gradients, ordinary
parameter ownership, generic/live cells, and a native CPU Product2 composition.
The complete existing framework suite exercises the migrated dynamics,
simulation, population, inference, inheritance, and trajectory-training users.

Reproduction using a compatible installed CPU extension without replacing it:

```sh
python scripts/validation/nn_foundation_cpu.py \
  --native-package /path/to/installed/cellerator \
  --require-native --pattern 'test*.py' --output /tmp/nn-cpu-validation.json
```

The runner explicitly loads source Python with the supplied compiled-extension
search path and masks CUDA devices. It never launches GPU work.

## Native CUDA training

The [grouped-linear report](grouped_linear/REPORT.md) binds admitted V100/SM70
execution, source and extension hashes, commands, receipts, and cleanup.
The numerical job and a separate memcheck job each passed all **13 tests with
zero skips**. Compute Sanitizer reported **zero errors**. Both completed jobs
verified resource cleanup.

Checks cover FP16/FP32 forward and input/weight VJPs, FP32 master weights with
FP16 computation, empty/ragged groups, noncontiguous inputs, non-default streams,
shared weights, ordinary optimizer updates, saved-parameter mutation rejection,
and deliberate rejection of native higher-order differentiation. Existing
cuBLAS numerical kernels and C++ bindings were reused unchanged.

## Runnable examples and packaging

The [training example](../../examples/nn_foundation/train.py) uses ordinary CPU
Adam loops and synthetic data. The independent run recorded these rounded
losses in [`example-training.json`](example-training.json):

| Objective | Initial | Final |
|---|---:|---:|
| Masked expression MSE | 0.048773 | 0.000660 |
| Next-event cross-entropy | 0.819076 | 0.115528 |
| ATAC BCE, frozen base | 0.660246 | 0.295442 |
| ATAC BCE, joint base | 0.660246 | 0.000007 |

A CPU wheel built successfully and was extracted into a temporary isolated
installation. Smoke checks passed for generic Cell binding, both NN facades,
views and their gradients, grouped-linear reference training, and native CPU
Product2 gradients. Root import remained Torch-free. All 75 packaged Python
files matched current source bytes. The existing Python-surface compatibility
suite passed both tests against the isolated installation.
[`wheel-validation.json`](wheel-validation.json) records its artifact identity.

## Limits

These checks establish bounded software behavior and memory-access correctness.
They do not establish biological validity, a model's identifiability, a speedup,
distributed GPU training, compiler integration, or automatic mixed-precision
registration. Native CUDA autograd is first-order only. Canonical native
parameter owners retain their existing guarded-update contracts; the new
operator uses ordinary Torch-owned parameters.

Preexisting dirty CUDA implementation/test files and missing submodule work
were preserved. Validation used explicit source hashes rather than presenting
the uncommitted implementation as a new Git revision. No installed package,
CUDA numerical kernel, or C++ binding was changed by this milestone.
