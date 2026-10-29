# Biological modeling framework expansion, 2026-10-29

This is a source-bound delivery record for the user-authorized implementation
push. It is not Todo authority, a scientific validation report, or a performance
qualification. Existing dirty tokenization and other work was preserved. No
commit, push, GPU workload, or environment repair was performed.

## What the framework now expresses

- Shared models with private, arbitrary cellular state; external state references;
  observations, experiments, trajectories, molecular events and lineage.
- Sparse relationships and joint interactor computations, state-dependent gates,
  native changed-dependency worklists and cached incremental recomputation.
- Sparse/ragged storage, stable identities through compaction, shared gene-aware
  quantization profiles, per-state exact overrides, and an optional ggml bridge.
- Discrete, continuous and stochastic dynamics; sparse stoichiometric reaction
  networks; explicit inheritance; conservative cell/environment material exchange.
- Independent cell execution with routed context, suspension/resumption, explicit
  lifecycle effects, exact parameter revisions, host backing, topology-aware cost
  plans, and callback-driven asynchronous execution.
- Learned sparse experts and overlapping modules; differentiable collective
  dispatch; bounded trajectory training with private ragged states and a shared
  optimizer owner. Inference includes particles and bounded event-history search.

The C++ headers are under `include/Cellerator/modeling` and
`include/Cellerator/execution/conditional`. The optional CUDA backend contains
compact dispatch/combine and backward, packed-state quantization, event memory,
changed-source frontiers, CSR attention and backward, cuBLAS segmented linear
forward/backward, fused packed embeddings, sparse reaction-event application,
and regulatory gates. The separate parameter-transfer library supplies real
asynchronous host-to-device transfers with exact-revision leases. These
operations do not require a dense biological adjacency matrix or one independent
parameter copy per cell.

KernelBender-informed experimental alternatives include independent-cell
bitplanes (32 cellular instances per word) for shared Boolean rules, and
warp-owned fused attention for neighborhoods of at most 32 edges. They are
explicit alternatives; existing paths remain available and no speedup is claimed.

Start with [the framework guide](../../docs/biological_framework.qmd),
[the examples](../../examples/framework/README.md),
[distributed execution](../../docs/distributed_execution.qmd), and
[the CUDA guide](../../docs/framework_cuda.qmd).

## Integrated checks

The source-package unittest discovery completed **188 tests, one skipped**, with
no failures before the final device additions. The skip is the opt-in distributed
multi-process test; its owner separately ran the two-process CPU Gloo checks.
Focused native tests and installed consumers have independent owner receipts.
Final combined build, staged package and native CTest results follow.

### Final integration receipts

- The root build compiled and linked all five optional Python modules, the host
  modeling/conditional libraries, twelve CUDA translation units across the
  numerical library and parameter-transfer bridge, and eleven CUDA smoke
  executables. Final attention and regulatory candidates were rebuilt afterward.
- All **18 targeted native CTest entries passed**, including the Python
  engine-to-native dispatch bridge, exact parameter revisions, scheduler examples,
  incremental computation, inheritance, reactions and conservative exchange.
- Python-component installation succeeded at
  `/tmp/cellerator-framework-stage-20261029`; source Python was copied there for
  staging rather than installed into the user's environment. All five native
  modules imported, and a Product2 compatibility computation passed.
- Against that staged package, **197 Python tests ran: 196 passed and one was
  skipped** (188 framework tests, six host-only CUDA binding tests, three ggml
  tests). Test directories were invoked separately because their quantization
  test filenames collide under one combined unittest discovery process.
- Staged `cellular_ai.py`, `trajectory_training.py`, and
  `scheduled_native_experts.py` ran successfully. The last includes planning,
  residency, signed/repeated routes, native execution and updates of original
  Cell owners. The training example checked shared gradients and private states.
- The final fused-attention host fixture and regulatory `--host-only` fixture
  passed. The latter covers cross-word gates and independent-cell bitplane tails.
  They do not execute device kernels. `git diff --check` passed.

The root did not perform a wheel build or the full legacy repository test suite.
Native installed-consumer evidence is from the focused owners' recorded builds;
the root's installation check is the Python component staging described above.

The root CMake build is `/tmp/cellerator-framework-20261029`, configured with:

```sh
cmake -S . -B /tmp/cellerator-framework-20261029 \
  -DCELLERATOR_ENABLE_CUDA=OFF \
  -DCELLERATOR_ENABLE_PYTHON=ON \
  -DCELLERATOR_BUILD_PRODUCT2=ON \
  -DCELLERATOR_BUILD_TESTS=ON \
  -DCELLERATOR_ENABLE_HARDWARE_PROBE=OFF \
  -DCELLERATOR_ENABLE_GGML=ON \
  -DFETCHCONTENT_SOURCE_DIR_GGML=/tmp/cellerator-ggml-upstream \
  -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF \
  -DCELLERATOR_ENABLE_FRAMEWORK_CUDA=ON \
  -DCELLERATOR_ENABLE_FRAMEWORK_CUDA_PYTHON=ON \
  -DCELLERATOR_ENABLE_FRAMEWORK_PARAMETER_TRANSFER=ON \
  -DCELLERATOR_BUILD_FRAMEWORK_CUDA_SMOKE=ON \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DPython_EXECUTABLE=/home/tumlinson/Software/miniconda3/bin/python \
  -Dpybind11_DIR=/home/tumlinson/Software/miniconda3/lib/python3.13/site-packages/pybind11/share/cmake/pybind11
```

`CELLERATOR_ENABLE_CUDA=OFF` bypasses the older large CUDA/compiler build; the
independent framework CUDA option enables the new numerical target. The optional
ggml checkout is pinned at `facdc44a67da17df4433adcc468b424f48e9ded2`.

The CUDA owner records exact source fingerprints, installation checks and
limitations in [its validation record](cuda/VALIDATION.md). **New device kernels
have been compiled and linked for SM70, not executed or numerically qualified on
a GPU.** Host oracle execution is explicitly distinguished from device execution.

## Remaining engineering

GPU numerical checks, sanitizer passes, performance measurements, broader
distributed training, transport/resource integration, biological validation and
API consolidation are deferred. Placement costs and biological gates are supplied
by the caller. New CUDA gradients are explicit native APIs; a complete autograd
binding is not implied by a backward function. Native attention, reaction and
other specialized kernels have distinct scope from their Python reference APIs.
No universal quantization fidelity or automatic optimization claim is made.

The workflow front door returned `workflow_task_already_active`; existing claims
were left intact. The user explicitly requested continuing implementation and
deferring workflow/environment reconciliation. Repository source maps and broader
legacy integration were not regenerated. Quarto rendering was not completed;
the incremental owner encountered a cache-database error and stopped investigating.
