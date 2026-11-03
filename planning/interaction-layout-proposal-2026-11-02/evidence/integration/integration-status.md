# Native component integration evidence

## Build configuration

- CMake 3.28.3, CUDA Toolkit/NVCC 12.9.86, GCC 12.4 host compiler, target `sm_70`.
- Native component uses `Cellerator::layout`, the canonical `Cellerator::memory_cuda`, and CUDA runtime. The core target does not link Torch.
- No GPU kernel or CUDA runtime workload was executed locally. CUDA targets were compiled and linked only.

## Native target and host checks

Command:

```sh
cmake --build /tmp/cellerator-factorized-interaction-build \
  --target cellerator_factorized_interaction_runner \
           cellerator_factorized_interaction_training \
           cellerator_factorized_interaction_cuda_contract_test \
           cellerator_factorized_interaction_preparation_test \
           cellerator_factorized_interaction_pattern_test -j2
ctest --test-dir /tmp/cellerator-factorized-interaction-build \
  --output-on-failure -R '^factorized_interaction_'
```

Result: build passed; preparation, ExplicitPattern bridge, and CUDA host-contract tests passed (3/3). The C++ training example compiles with its 200-update loop and target 0.25; it was not run locally. An independent CPU replay at 200 and 10,000 steps stayed finite and inside the initial FP16 parameter domain; see `training-example-replay.md`.

## Installed component consumer

The package now offers an additive `FactorizedInteractionCUDA` install component. It installs the focused public headers and the exact base (`support`, `layout`), memory, and executor archive/export sets. The full Cellerator config imports those same target sets, avoiding duplicate imported-target definitions.

Commands:

```sh
cmake --build /tmp/cellerator-factorized-interaction-install-build \
  --target cellerator_support cellerator_layout cellerator_memory_cuda \
           cellerator_factorized_interaction_cuda -j2
cmake --install /tmp/cellerator-factorized-interaction-install-build \
  --prefix /tmp/cellerator-factorized-interaction-component-prefix-v3 \
  --component FactorizedInteractionCUDA
cmake -S tests/factorized_interactions/installed_consumer \
  -B /tmp/cellerator-factorized-interaction-installed-consumer-v5 \
  -DCMAKE_PREFIX_PATH=/tmp/cellerator-factorized-interaction-component-prefix-v3 \
  -DCUDAToolkit_ROOT=/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9
cmake --build /tmp/cellerator-factorized-interaction-installed-consumer-v5 \
  --target factorized_interaction_installed_consumer -j2
```

Result: component install succeeded; the isolated consumer configured and compiled/linked. It requests the factorized component with optional `nn`, verifies the component is found and `nn` is unavailable, then repeats the exact factorized request and checks that `Cellerator::memory_cuda` resolves to the same installed archive both times. A separate configure check confirms a required unavailable `nn` component is rejected. The consumer binary is intentionally not run: it links a CUDA translation unit, so its startup may perform CUDA fat-binary registration despite `main` only calling host preparation and workspace sizing.

The package export inventory confirms each of `Cellerator::support`, `Cellerator::layout`, `Cellerator::memory_cuda`, and `Cellerator::factorized_interaction_cuda` is defined in exactly one export set; see `package-export-identity.txt`. A full SDK prefix was not produced: the default all-target build reaches an unrelated compile failure in `include/Cellerator/math/matrix/contracts.hh` while building `cellerator_op_matrix_patch` (`ex::structure_id`, `ex::structure_epoch`, `ex::same_identity`, and `ex::valid_identity` are unavailable). I did not modify that unrelated target. Independent configure fixtures against the generated full config now pass bare, NN-first, narrow-first, and repeated mixed loads, preserving the single canonical `memory_cuda` target identity. Those fixtures use placeholder archives and a stub Torch package, so they establish package export parsing, not linking. A full installed `native_data`/`nn` consumer has not been linked; see `package-order-repair.md` and `cmake-package-snapshot.json` for current source/config identities and review evidence.

## Optional Torch adapter

Both optional build lanes passed after the export split:

- `cmake --build /tmp/cellerator-factorized-interaction-torch-build --target cellerator_nn_cuda -j2`
- `cmake --build /tmp/cellerator-factorized-interaction-python-build --target cellerator_nn_python -j2`

The Python extension output is `_nn_native.cpython-313-x86_64-linux-gnu.so`. Runtime autograd/version/lifetime tests and the repeated training example run are delegated to the Oracle's Scheduler-admitted GPU qualification. These builds emitted non-fatal PyTorch CMake warnings about NVRTC hash discovery and NVTX3.

## Durable identities and logs

- `source-sha256.json`: compiler-reported repository dependency closure plus public contracts, integration files, runner, tests, and example.
- `artifact-sha256.json`: runner, training example, native archive, Torch archive, Python extension, and installed component archives.
- `compiler-dependency-closure.txt`: source/header paths collected from target `.o.d` files.
- `native-build-ctest.log`, `installed-component-consumer.log`, `torch-adapter-build.log`, and `python-extension-build.log`: exact command outputs and exit codes.
- `package-order-repair.md`: dependency order, current package-load review, optional-component behavior and evidence boundary.
- `training-example-replay.md`, `cpu_training_replay.py`, and `cpu-training-replay.log`: example target rationale, CPU-only trajectory replay, and source/binary hashes.
