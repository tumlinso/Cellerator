# Single-cell MMA prototypes

These isolated experiments expose three SM70 routes: nonlinear 16×16 matrix patches, four independent 8×4 by 4×8 contractions per warp, and a two-input product with its directional derivative. They consume caller-owned state and parameters; the modules specify admission, alias and generation policies. DIFF owns later patch/quad backward adapters.

CPU algebra checks:

```sh
python3 -B experiments/moonshot-parallel-v1/mma/check_mma.py --record
```

The aggregate checks use a scalar stored-half patch oracle, product directional finite differences including zero/repeated inputs, and the quad register-coordinate mapping checker. These establish reference mathematics and mapping coverage. They do not execute device kernels.

Build all modules outside the repository with CUDA 12.9 and a fresh object
directory. `build_mma.py` creates the directory, binds the exact inputs and
resulting binaries in a manifest, and saves that manifest under `results/` for
the later controller-owned GPU run:

```sh
MMA_BUILD_DIR="/tmp/cellerator-mma-$(date -u +%Y%m%dT%H%M%SZ)-$$"
python3 -B experiments/moonshot-parallel-v1/mma/build_mma.py --build-dir "$MMA_BUILD_DIR"
python3 -B experiments/moonshot-parallel-v1/mma/check_mma.py --build-dir "$MMA_BUILD_DIR" --record
```

The final command runs only compiled host admission checks. `MOONSHOT_CUDA=OFF` supports the product CPU oracle target plus the Python aggregate CTest; CUDA compilation is necessary for the metadata tests of prepared GPU APIs.

The target argument vector for root-controller GPU qualification is:

```sh
python3 -B experiments/moonshot-parallel-v1/mma/check_mma.py --build-dir "$MMA_BUILD_DIR" --gpu --record
```

It launches `patch16/patch16_smoke`, `quad/moonshot_quad_test`, and `product/moon_product_smoke --gpu`. Exit77 or any other nonzero result fails the aggregate rather than becoming validation. Module CTest registration includes a GPU test; use the aggregate host command while resources are unavailable.

Precision: patch inputs and nonlinear intermediate use FP16 storage with FP32 products/output; quad uses FP16 operands and FP32 accumulation/output; product and JVP use FP32 with contraction disabled. Preserve same-stored-value references and make quantization surrogates explicit when DIFF adds derivatives.

`capability.json` and `results/*.json` report separate CPU reference, compiled admission and device stages, source hashes, unsupported routes and `timing: not_measured`. `device_modules` and the forward status report per-module numerical smoke only; they do not establish controller admission or full acceptance. The full acceptance gate is `verify_gpu_evidence.py`, which checks the controller receipt as well as source, build, module and transcript bindings. Evidence does not establish a CelleraTorch/native-core adapter or scientific interpretation. A capability result applies to its recorded source hashes; changing a module requires its affected checks again.

A targeted repair check adds `--module product` (or `patch16` / `quad`) to the controller command. Each module writes stdout, stderr, exit code, source hashes and binary hash immediately to `results/gpu-<module>.json`; a later failure preserves prior successful outputs. Aggregate and per-module records remain separate.

The current successful controller evidence ID, transcript, lease and bound build manifest are recorded in `results/controller-success.json` and the associated `results/controller-success*` files. This current pointer is updated when a newer qualified run replaces it. Prior source-bound runs are preserved in dated `results/historical-*` directories; historical unbound evidence and product failure logs remain separate.

Required gate: `python3 -B experiments/moonshot-parallel-v1/mma/verify_gpu_evidence.py`. It verifies the latest complete run, all numerical source hashes, fixture coverage, controller transcript hashes, the assigned resource receipt and the controller's binary receipt for the exact build manifest, then checks host algebra/mapping. It launches no GPU work. Changes to numerical source invalidate the evidence. Temporary negative controls rejected changed source and transcript.

The controller-owned GPU command uses the same fresh directory:
`python3 -B experiments/moonshot-parallel-v1/mma/check_mma.py --build-dir
"$MMA_BUILD_DIR" --gpu --record`. Put that argument vector in the CUDA
controller foreground spec; do not run it directly on the shared host. The
build helper refuses an existing object directory, records exact source and
build inputs, compiler configuration and executable hashes, and the checker
rejects changed inputs or replaced executables before execution. The manifest
hash appears in the controller transcript. Set the spec's `binary_paths` to
`experiments/moonshot-parallel-v1/mma/results/build-manifest.json` and preserve
the controller's binary receipt with the transcript and lease evidence; the
pure evidence gate checks that receipt and the module executable hashes.
Earlier evidence without this binding is retained as `historical-unbound-*`; it
cannot qualify the current source. `test_provenance.py` exercises rejection of
stale build inputs, false or missing fresh-object attestations, swapped
executables and missing manifests without a GPU.
