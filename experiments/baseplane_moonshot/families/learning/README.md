# Small learning and specialization providers

Cellerator owns these source-independent CPU numerical experiments. Baseplane
owns sequence features, exact motifs, provenance, boundary interpretation and
scientific questions. This is a standalone C++17 header, with no production
training/runtime dependency.

Link `Cellerator::moonshot_learning` from the experimental provider build and
include `<learning.hpp>`. Its namespace is `ce_moon::learning`. Alternatively,
add this family directory as an explicit include path. The provider root is a
caller-supplied CMake path; no sibling repository is inferred.

`fit_logistic(rows, targets, n, d, weights, options)` takes row-major `n*d`
finite double features and `n` targets in `[0,1]`. The caller owns `d+1` output
coefficients: features first, bias last. Fitting starts at zero, uses deterministic
full-batch gradients, and writes the model only on success. Prediction takes one
feature row and those coefficients; no targets participate in prediction.
Private host scratch costs `O(n+d)`, with no device stream or allocation.

`FitOptions` defaults are `epochs=600`, `learning_rate=0.25`, `l2=0.001`,
`budget_weight=0`, `positive_budget=0.5`. The objective is mean binary cross
entropy plus `0.5*l2*sum(feature_weights^2)` and
`budget_weight*max(0,mean(predicted_probability)-positive_budget)^2`.
Bias is unregularized. The compute penalty bounds expected activation pressure;
it does not enforce a discrete boundary count. Inputs need caller-controlled
scaling; fixed-step training is a bounded experiment, with no convergence
promise. `logistic_objective` exposes the full gradient for inspection.
`FitReport` records initial/final objectives, final mean probability and epochs.

The primary experiment differences are retained:

- E29 fits eight relaxed Bernoulli gates supplied by the caller, preserving the
  teacher's weight version separately from the hard circuit version. LUT row
  index is `4*a+2*b+c`; `harden` thresholds at probability `>=0.5` and
  `emit_lop3` emits the native instruction source with a constant immediate.
  The teacher's continuous confidence is lost in the hard circuit. The native
  instruction source is authored; this family does not compile or launch CUDA.
- E30 fits individual state-to-bank logistic scorers and picks the highest score
  with stable lowest-index tie handling. Selection carries state and weight
  versions. Caller bank identities and exact predicate meanings stay external.
  Independent sigmoid bank scores are routing heuristics, not a normalized
  posterior.
- E31 groups exact three-input Boolean expressions by exhaustive truth table and
  extracts a lower instruction-count expression from a bounded rewrite space.
  Domain tags must match at joins. At most four saturation rounds are allowed;
  the result is exact for Boolean inputs, and optimality outside the searched
  space is not claimed. Instruction count is an estimate, without SASS, register
  pressure or memory movement qualification. Strict floating reassociation is
  denied; approximate mode explicitly permits its numerical loss.
- E32 specializes a teacher into an immutable circuit guarded by weight, query,
  state and domain versions. Any mismatch runs the current unspecialized teacher.
  The caller must advance weight versions when modifying parameters and domain
  versions when changing coordinate/validity semantics. Guards are applicability
  checks; they do not certify biological facts or sequence-specific kernel reuse.

The shared logistic scorer additionally supports E14 boundary learning and
caller-defined scores for E27, E37, E39 and E40. Those sequence experiments,
budgets, alternative parses and adaptation rules belong to their Baseplane
consumers. No biological dataset or throughput benchmark was run here.

The dedicated host test checks full-objective finite differences including the
rate penalty, held-out numeric rows, deterministic fitting, extreme logits,
invalid inputs, exhaustive truth-table hardening/rewrite equivalence, fitted
bank selection and four independent stale-guard cases. Reproduce from the
Cellerator experimental module:

```sh
cmake -S experiments/baseplane_moonshot -B /tmp/ce-moon-learning-build -DCMAKE_BUILD_TYPE=Release
python /tmp/moonshot_build_slot.py cmake --build /tmp/ce-moon-learning-build --target ce_moon_learning_test -j1
ctest --test-dir /tmp/ce-moon-learning-build -R ce_moon_learning_test --output-on-failure -V
```

The build-slot wrapper is a campaign resource coordinator, external to this
library. See `receipt.json` for the recorded source digest and observed output.
