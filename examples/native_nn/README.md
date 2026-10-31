# Native C++ NN reference models

These small examples show one way to compose native Cellerator NN types from
ordinary C++ structs. They use explicit parameter groups, typed results, and
the standard LibTorch optimizer/autograd facilities. The synthetic fixtures
exercise API behavior; they are not biological evidence or a prescribed
scientific architecture.

`reference_models.hh` defines two models:

- `ExpressionReferenceModel` aligns noncontiguous canonical feature IDs to
  source columns, supplies their separate model rows to `FeatureEncoder`, adds
  a `RelationAttention` update as an explicit residual, and exposes expression
  plus pooled features. Its masked MSE uses only selected targets.
- `EventHistoryReferenceModel` maps repeated event occurrences to model rows,
  then builds packed teacher-forcing inputs as `[START, e0, ..., e(n-2)]` with
  targets `[e0, ..., e(n-1)]`. `OccurrenceCoordinates` keeps each occurrence
  and group order explicit; a prepared within-history relation is filtered to
  causal prefixes before attention. Empty histories contain a START context
  representation and contribute no next-event target or loss.

Both models expose a target-bound `FeatureReadout` for an auxiliary ATAC query.
The readout consumes pooled base features. `nn::predict` can evaluate the base
once for multiple query requests. Freezing the base parameter group leaves
autograd connected to inputs; making the group trainable enables joint updates.

`train.cc` runs bounded deterministic CPU training for the masked expression
objective, next-event objective, and frozen-base/joint ATAC readouts for both
models. Its printed losses describe only those synthetic fixtures. The native
model regression test checks identity alignment, hidden-value invariance,
masked-target behavior, query sharing, frozen-input gradients, shifted event
labels, causal leakage, group reordering, repeated occurrences, and empty
histories.

Build and run the native demo with the optional Torch NN component:

```sh
cmake -S . -B /tmp/cellerator-native-nn \
  -DCELLERATOR_ENABLE_CUDA=OFF \
  -DCELLERATOR_ENABLE_NN=ON \
  -DCELLERATOR_BUILD_NN_TRAIN=ON
cmake --build /tmp/cellerator-native-nn --target native_nn_train native_nn_models_test
/tmp/cellerator-native-nn/native_nn_train
ctest --test-dir /tmp/cellerator-native-nn -R native_nn_models_test --output-on-failure
```

The example has no Python dependency and runs on CPU; CUDA is not required.
