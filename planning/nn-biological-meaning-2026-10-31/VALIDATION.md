# Biological NN integration qualification, 2026-10-31

This additive pass connects biological participants, their declared relationships,
conditioning, requested observations, and supervision to the existing NN
foundation. Raw tensor APIs, generic cells, explicit dynamics, native owners,
vocabularies, and numerical kernels remain available. It is a Python-facing
modeling milestone; first-class idiomatic C++ model authoring remains an
architectural target rather than an established equivalent surface.

## CPU integration and independent review

The combined framework suite ran **328 tests: 314 passed and 14 skipped**, with
no failures or errors. The skipped cases comprise six new admitted-only CUDA
checks, six earlier grouped-linear CUDA checks, one opt-in two-process Gloo
check, and one unavailable optional native modeling component. The earlier
grouped-linear CUDA qualification remains separately source-bound; the six new
CUDA checks ran on admitted hardware as described below.

[`cpu-validation.json`](cpu-validation.json) records test identities, source
hashes, Python 3.13.11 / Torch 2.7.0+cu126, and the compatible installed CPU
extension used without replacing it. Reproduce from a checkout:

```sh
python scripts/validation/nn_foundation_cpu.py \
  --native-package /path/to/installed/cellerator \
  --require-native --pattern 'test*.py' --output /tmp/biological-cpu.json
```

Coverage includes native vocabulary resolution and noncontiguous token IDs,
reordered features, sparse adaptation, repeated occurrences, optional existing
axes, prepared relation reuse and stale guards, selected/full readout agreement,
conditioning and gradients, window/physical-cell alignment, explicit loss units,
empty objectives, manifests, overlapping shared programs, and legacy dynamics.
The new registered gate conditioner is covered on CPU, including broadcasts,
parameter/context gradients, and bypass under `context=None`.

Independent assessment found four concrete gaps: contradictory packed offsets,
broadcast-compatible target shapes, undeclared standalone readout bindings, and
mismatched readout/view bindings. All four were repaired with regression tests.
The reviewer reproduced rejection of all four original counterexamples and ran
the 33 query, supervision, and manifest tests successfully. No duplicate test
method names remained in those suites.

## Admitted V100 compositions

The corrected exclusive V100/SM70 job passed **6/6 tests, zero skips**, with
matching source hashes and verified scheduler cleanup. The
[GPU report](gpu/REPORT.md) records exact numerical contracts, tolerances,
artifact identity, scheduler receipts, and the failed first fixture attempt.
The successful job was `5f310ca8-9a45-4bae-9bb3-833e81c293dd` on
`GPU-d745da9c-7649-8334-41f1-483afa6f3206`.

Checks cover FP16/FP32 prepared-attention forward and VJPs against an independent
Torch reference, empty/duplicate relations, explicit transfer/reuse/stale guards,
bound feature and selected-output ordering, actual query conditioning,
frozen/joint gradients, overlapping program gradients, and composition with the
existing native grouped-linear operation. Prepared execution and selected
readouts ran under scalar-read guards. No numerical kernel or binding rebuild,
profiler, benchmark, or new sanitizer campaign was performed.

## Learning examples and isolated wheel

All ten deterministic synthetic objectives in
[`example-training.json`](example-training.json) decreased, including bound
expression/event learning and frozen/joint ATAC BCE readouts over event histories.
These exercise ordinary optimizers and explicit supervision; they establish no
biological validity or mechanism recovery. Losses are rounded in the record.

A CPU wheel built with the existing optional-dependency configuration. Its
isolated extracted installation passed Torch-free root imports, all fifteen new
NN facade routes, selected biological queries and gradients, prepared program
composition, explicit gate conditioning, native CPU Product2 gradients,
grouped-linear reference gradients, and both existing Python-surface tests.
All **81 packaged Python files** matched source bytes. No installed package was
changed. [`wheel-validation.json`](wheel-validation.json) records artifact hashes;
[`validate_wheel.py`](validate_wheel.py) reproduces the extraction and checks:

```sh
python -m pip wheel . --no-deps --no-build-isolation --wheel-dir /tmp/cellerator-wheel
python planning/nn-biological-meaning-2026-10-31/validate_wheel.py \
  --wheel /tmp/cellerator-wheel/cellerator-0.1.0-cp313-cp313-linux_x86_64.whl \
  --output /tmp/biological-wheel-validation.json
```

All **37 documentation pages** rendered successfully in a temporary staged
book. One included scope link was repaired and its page rerendered without
warnings. [`docs-validation.json`](docs-validation.json) records source hashes,
commands' successful results, and the warning/resolution. Rendering executes no
example or GPU code and publishes no site.

The source-map generator's temporary-repository lifecycle checks passed;
[`source-map-lifecycle.json`](source-map-lifecycle.json) records their results.
Final generated routing evidence remains in the canonical
[`validation.json`](../../docs/development/source_map/validation.json), with its
normal source/index/worktree staleness checks; it is not numerical acceptance.
The final live-tree validator could not hold a stable snapshot while concurrent
native NN sources and build logs changed and new files arrived. Its actual
nonpassing result is preserved in
[`source-map-checkpoint.json`](source-map-checkpoint.json); a source-map refresh
after that separate workstream becomes quiescent remains necessary. No staleness
check was weakened. The qualified Python/GPU source hashes still matched at the
final check recorded in
[`qualification-source-check.json`](qualification-source-check.json).

## Preservation and limits

[`preservation.json`](preservation.json) records byte comparison of all **2,042**
preexisting files under `src`, `include`, `bindings`, `state`, and `events` at
the qualification checkpoint; all matched. This pass changed none of those
files. Concurrent native NN development subsequently added and changed native
files while documentation and inventory checks finished. Those edits are
preserved and included in the user-requested checkpoints, but are not qualified
by this pass's Python-facing test results. Preexisting unrelated CUDA work and
submodule removals were also preserved.

Manifests record declarations beside ordinary weights, not executable model
reconstruction or guarantees of selector equivalence. No histories, latent
activations, or conditioning values are serialized by default. Existing native
owner restrictions remain specialized, and ordinary neural parameters retain
ordinary optimizer semantics. Unsupported `GatedInteractor` context now raises;
its no-context behavior is preserved.

This qualification is bounded by the recorded cases and environment. It does
not establish biological validity, scientific identifiability, speedup,
distributed scheduling/training, model-Hub integration, comprehensive native
checkpoint qualification, or C++/Python authoring parity. Existing native
checkpoint implementations were not modified; optional native builds and prior
qualification retain their own boundaries.
