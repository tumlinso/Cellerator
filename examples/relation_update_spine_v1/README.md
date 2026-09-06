# Regulatory relation learning: RU1 prospective demo

This program is written now against the **proposed post-epic** Cellerator API. It is not evidence that the new implementation already exists. The planning package contains matching declaration sketches; never add those sketches or the syntax-test shims to production include paths.

The ordinary build must link real `cellerator_relation_update_core` and `cellerator_relation_update_compiler` targets established by the integration tasks, using the repaired sm70 WMMA path and exact sparse residual. An implementing agent may revise API spelling coherently, but may not replace actual computation with a demo-local kernel or reference fallback. The CMake names are proposed in-tree integration targets, not an installed SDK claim.

## Run the independent fixture today

```sh
c++ -std=c++20 -O2 -Wall -Wextra -Werror -ffp-contract=off \
  -DCELLERATOR_RU1_REFERENCE_ONLY=1 regulatory_learning.cc -o /tmp/ru1-reference
/tmp/ru1-reference
```

Or configure this directory independently with CMake. The output explicitly says no Cellerator or GPU was run. It checks all finite half roundtrips, tie/overflow cases, forward/transpose adjoint, continuous-model edge finite differences and the bounded mixed-precision update fixture.

## Run after implementation

Build from the integrated repository, without `CELLERATOR_RU1_REFERENCE_ONLY`, then run `ceRelationUpdateDemo --device 0 --route hybrid` and `--route sparse` using the existing project GPU lease. The binary requires an actual sm70 device and fails rather than skipping when it is unavailable.

The synthetic model has a 16x16 dense regulatory module, six irregular residual links, one empty output row, one isolated source and 16 state conditions. One topology serves native/compiler-origin forward, transpose and edge VJP, caller delta and gradient-step updates, generations 1/2/3, a true external-stream read lease and stale-generation rejection. Example-side loss and observation transfers keep the narrative inspectable; they are not a proposed hot-loop loss implementation or a benchmark.

`planning/relation-update-spine-v1/05_VALIDATION_AND_DEMO.md` defines the stronger integrated acceptance. A loss decrease on synthetic data is not a scientific result or a general performance claim.
