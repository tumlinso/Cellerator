# Single-cell MMA prototypes

These isolated experiments expose three SM70 routes: nonlinear 16×16 matrix patches, four independent 8×4 by 4×8 contractions per warp, and a two-input product with its directional derivative. They consume caller-owned state and parameters; the modules specify admission, alias and generation policies. DIFF owns later patch/quad backward adapters.

CPU algebra checks:

```sh
python3 -B experiments/moonshot-parallel-v1/mma/check_mma.py --record
```

The aggregate checks use a scalar stored-half patch oracle, product directional finite differences including zero/repeated inputs, and the quad register-coordinate mapping checker. These establish reference mathematics and mapping coverage. They do not execute device kernels.

Build all modules outside the repository with CUDA 12.9:

```sh
cmake -S experiments/moonshot-parallel-v1/mma -B /tmp/moonshot-mma-aggregate-build -DMOONSHOT_CUDA=ON -DCMAKE_CUDA_COMPILER=/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/bin/nvcc -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/moonshot-mma-aggregate-build -j2
python3 -B experiments/moonshot-parallel-v1/mma/check_mma.py --build-dir /tmp/moonshot-mma-aggregate-build --record
```

The final command runs only compiled host admission checks. `MOONSHOT_CUDA=OFF` supports the product CPU oracle target plus the Python aggregate CTest; CUDA compilation is necessary for the metadata tests of prepared GPU APIs.

For a root controller with assigned CUDA resources, the tiny GPU correctness command is:

```sh
python3 -B experiments/moonshot-parallel-v1/mma/check_mma.py --build-dir /tmp/moonshot-mma-aggregate-build --gpu --record
```

It launches `patch16/patch16_smoke`, `quad/moonshot_quad_test`, and `product/moon_product_smoke --gpu`. Exit77 or any other nonzero result fails the aggregate rather than becoming validation. Module CTest registration includes a GPU test; use the aggregate host command while resources are unavailable.

Precision: patch inputs and nonlinear intermediate use FP16 storage with FP32 products/output; quad uses FP16 operands and FP32 accumulation/output; product and JVP use FP32 with contraction disabled. Preserve same-stored-value references and make quantization surrogates explicit when DIFF adds derivatives.

`capability.json` and `results/*.json` report separate CPU reference, compiled admission and device stages, source hashes, unsupported routes and `timing: not_measured`. Evidence does not establish a CelleraTorch/native-core adapter or scientific interpretation. A capability result applies to its recorded source hashes; changing a module requires its affected checks again.

A targeted repair check adds `--module product` (or `patch16` / `quad`) to the controller command. Each module writes stdout, stderr, exit code, source hashes and binary hash immediately to `results/gpu-<module>.json`; a later failure preserves prior successful outputs. Aggregate and per-module records remain separate.

All three SM70 GPU smoke checks passed in root controller evidence `9c227568-ac4c-4049-859d-5d3860cda5fb`. The successful controller transcript and lease snapshot are preserved in `results/controller-success*`; historical product failure logs remain separate.

Required gate: `python3 -B experiments/moonshot-parallel-v1/mma/verify_gpu_evidence.py`. It verifies the latest complete run, all numerical source hashes, fixture coverage, controller transcript hashes and the assigned resource receipt, then checks host algebra/mapping. It launches no GPU work. Changes to numerical source invalidate the evidence. Temporary negative controls rejected changed source and transcript.

Build provenance requires a fresh object directory. Run `python3 -B build_mma.py
--build-dir results/gpu-bin` from this directory before controller-owned GPU
execution. The script refuses an existing build directory, records exact source
and build inputs, compiler configuration and executable hashes, and the checker
rejects changed inputs or replaced executables before execution. The manifest
hash appears in the controller transcript and the pure evidence gate checks it.
Earlier evidence without this binding is retained as `historical-unbound-*`; it
cannot qualify the current source. `test_provenance.py` exercises rejection of
stale build inputs, swapped executables and missing manifests without a GPU.
