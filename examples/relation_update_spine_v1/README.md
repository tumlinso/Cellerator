# Regulatory relation learning: RU1 synthetic demo

This consumer links the native relation core and the bounded source-to-action
compiler adapter. It contains no private CUDA arithmetic. See the authoritative
[execution model](../../docs/biological_execution_model.qmd) and the measured
[RU1 demo receipt](../../docs/relation_update_spine_v1/demo.json).

Native descriptors and independently parsed closures produce equivalent value-owned
recipes. Ten compiler actions reach the real forward, transpose, gradient, update,
publication and read-lease entrypoints. One prepared topology and one gradient
preparation serve both caller delta and gradient-step updates, at generations
1, 2 and 3. The later update style does not change the prepared gradient math.
Unknown update kinds still fail before submission.

## Run the independent fixture today

```sh
c++ -std=c++20 -O2 -Wall -Wextra -Werror -ffp-contract=off \
  -DCELLERATOR_RU1_REFERENCE_ONLY=1 regulatory_learning.cc -o /tmp/ru1-reference
/tmp/ru1-reference
```

Or configure this directory independently with CMake. The output explicitly says no Cellerator or GPU was run. It checks all finite half roundtrips, tie/overflow cases, forward/transpose adjoint, continuous-model edge finite differences and the bounded mixed-precision update fixture.

## Run the native demo

Build from the integrated repository, without `CELLERATOR_RU1_REFERENCE_ONLY`, then run `ceRelationUpdateDemo --device 0 --route hybrid` and `--route sparse` using the existing project GPU lease. The binary requires an actual sm70 device and fails rather than skipping when it is unavailable.

The synthetic model has a 16x16 dense regulatory module, six irregular residual links, one empty output row, one isolated source and 16 state conditions. One topology serves native/compiler-origin forward, transpose and edge VJP, caller delta and gradient-step updates, generations 1/2/3, a true external-stream read lease and stale-generation rejection. Example-side loss and observation transfers keep the narrative inspectable; they are not a proposed hot-loop loss implementation or a benchmark.

`planning/relation-update-spine-v1/05_VALIDATION_AND_DEMO.md` defines the stronger integrated acceptance. A loss decrease on synthetic data is not a scientific result or a general performance claim.
