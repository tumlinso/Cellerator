# CMake package order repair

## Generated import order

The full package now imports target exports in dependency order:

1. PartOne and CUDA memory.
2. The shared factorized base (`support`, `layout`).
3. Native foundation and NativeData host targets.
4. Factorized interaction and NativeData CUDA targets.
5. Framework CUDA and optional parameter-transfer targets.
6. NN targets when requested.

Framework and NN package-config snippets are appended only after their exported
target dependencies. The component wrapper uses the narrow three-export route
for an exact factorized-interaction request and for requests against a
component-only prefix. It marks unsupported optional components false and
fails required unsupported components.

## Checks

- Reconfigured the native-only, Torch-enabled and Python-enabled build trees
  after the change. All three configure/generate steps passed.
- Rebuilt the native CUDA runner, 200-update C++ example, CUDA host-contract
  test and host preparation/pattern tests. CTest passed 3/3. No GPU workload was
  executed locally.
- Rebuilt `cellerator_nn_cuda` and `cellerator_nn_python` after the config
  change. Both passed; existing non-fatal PyTorch NVRTC/NVTX3 discovery
  warnings remain.
- Installed `FactorizedInteractionCUDA` to
  `/tmp/cellerator-factorized-interaction-component-prefix-v3`. The external
  CXX-only consumer configured and compiled/linked. It requests the required
  factorized component plus optional `nn`, confirms that `nn` is unavailable,
  repeats the exact component lookup, and verifies the canonical memory target
  resolves to the same archive. The executable was not run because CUDA runtime
  registration/runtime probing is outside the local CPU-only lane.
- Independent configure fixtures passed fresh full-first, NN-first,
  narrow-first, and repeated mixed load sequences. The actual narrow v3 prefix
  accepts optional NN and rejects required NN. Reviewer evidence is in
  `package-review-full-first.log`, `package-review-nn-first.log`,
  `package-review-narrow-first.log`,
  `package-review-optional-narrow-real.log`,
  `package-review-missing-required-real.log`, and
  `package-review-identities.json`.

The full-prefix ordering fixtures use placeholder archives and a stub Torch
package, so they establish CMake export parsing and target identity only; they
do not establish linkability. The actual v3 narrow consumer is the compile/link
evidence. A full installed SDK is still unavailable because its broad build
hits the unrelated `math/matrix/contracts.hh` error recorded in
`integration-status.md`.

## Snapshot identity

- Generated wrapper:
  `/tmp/cellerator-factorized-interaction-python-build/CelleratorConfig.cmake`
  SHA-256 `22defd1b1e99391dbd3136c1b837e52604092525a5a46f91c060a2498fa5c7b7`.
- Generated full config:
  `/tmp/cellerator-factorized-interaction-python-build/CelleratorFullConfig.cmake`
  SHA-256 `b37e35cca44883a8b324bf89fcdcd0c815cd87159cfc9af14cb02b23d09452a2`.
- Installed narrow wrapper:
  `/tmp/cellerator-factorized-interaction-component-prefix-v3/lib/cmake/Cellerator/CelleratorConfig.cmake`
  SHA-256 matches the generated wrapper above.
- Source and export hashes are recorded in `cmake-package-snapshot.json`.
