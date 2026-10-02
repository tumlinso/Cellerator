# Experimental numerical providers

This isolated C++17 module supplies four numerical research providers for Baseplane's sequence experiments. Baseplane supplies exact sequence, validity, strand, source support and biological interpretation. Cellerator owns the numerical effects, tensor arithmetic, fitting and mechanisms from inception. These providers do not depend on completion of ML2 or introduce a production runtime.

| Provider | Target and header | Actual operations | Baseplane consumers |
| --- | --- | --- | --- |
| [Effects](families/effects/README.md), CE-MOON-020 | `Cellerator::moonshot_effects`, `<ce_moon/effects.hpp>` | Monomial/block affine maps, quadratic jets, sparse checkpoints, weighted state algebras and precision intervals | `effects`; weighted `automata` |
| [Tensor](families/tensor/README.md), CE-MOON-030 | `Cellerator::moonshot_tensor`, `<ce_moon/tensor.hpp>` | Object-feature transforms, directed QK scores with bounded emission, binary path counts, sampled state responses | `tensor` |
| [Learning](families/learning/README.md), CE-MOON-040 | `Cellerator::moonshot_learning`, `<learning.hpp>` | Deterministic logistic fitting, relaxed Boolean gates and hardening, fitted bank selection, bounded Boolean rewrites and version-guarded specialization | `compiler`, `hierarchy`, `hypotheses` |
| [Mechanisms](families/mechanisms/README.md), CE-MOON-050 | `Cellerator::moonshot_mechanisms`, `<ce_moon/mechanisms.hpp>` | Port condensation/reconstruction, coarse corrections and fitted step size, sparse factor joins/aggregates, affine counterfactual DAGs | `ports` |

The common target `Cellerator::moonshot` exports `<ce_moon/reference.hpp>`: `Dfa32`, `CountedDfa32`, `MonomialAffine<N>`, guarded lifting/residual trees and finite relations. Composition visits the left effect then the right. DFA states are in `[0,32)`; counted composition rejects unsigned overflow. Affine maps validate permutations and use binary64 arithmetic. Residual trees take finite nonempty scalar vectors, and `above` certifies only the declared scalar maximum query. Floating arithmetic remains approximate.

Provider inputs have numerical axes and explicit weights/state. Source identifiers and factor posting keys are opaque caller tokens. Tensor host tiles have 256 row-major FP32 entries with active axes in `[1,16]`; their products accumulate in double and zero padding. Learning accepts finite row-major double features and caller-owned model coefficients; fit writes the model only on success. Mechanism matrices use row-major double values with checked dimensions and axis bounds. Returned host vectors own their storage; bounded tuple emission reports required/written/overflow counts. These contracts supply no biological hierarchy, relation or causal meaning.

## Build and consume

From the Cellerator repository root:

```sh
cmake -S experiments/baseplane_moonshot -B /tmp/ce-moon-host \
  -DCE_MOON_ENABLE_CUDA=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
python3 /tmp/moonshot_build_slot.py cmake --build /tmp/ce-moon-host -j1
ctest --test-dir /tmp/ce-moon-host --output-on-failure
```

The build-slot wrapper is external campaign resource coordination. Outside this campaign, `cmake --build /tmp/ce-moon-host -j1` reproduces the single-job build.

Baseplane consumers explicitly select the provider directory; no sibling path is inferred:

```sh
cmake -S /explicit/path/to/Baseplane/experiments/moonshot -B /tmp/bp-moon-host \
  -DCE_MOON_SOURCE_DIR=/explicit/path/to/Cellerator/experiments/baseplane_moonshot
python3 /tmp/moonshot_build_slot.py cmake --build /tmp/bp-moon-host -j1
ctest --test-dir /tmp/bp-moon-host --output-on-failure
```

An enclosing CMake build can supply `Cellerator::moonshot` and the family targets directly. Consumers link their required target and include its header. Family-local `CMakeLists.txt` files are discovered independently; shared numerical headers and the root build file retain one integration owner.

## Optional CUDA

`<ce_moon/volta.cuh>` supplies representative DFA, counted-DFA, monomial, lifting, WMMA, threshold, DP4A, butterfly and texture kernels. `<ce_moon/tensor.cuh>` supplies asynchronous tiny-tile wrappers. Caller-owned nonaliasing buffers, capacities, explicit streams, legal warp participation and prevalidated domains remain required. Tensor CUDA accepts FP16-representable active inputs with FP32 accumulation; binary path counts on this one tile have a narrower exactness claim than general weighted multiplication. Mechanism CUDA contains small numerical representatives; the shared-world optimization is a host path. The learning family's emitted LOP3 source has no family CUDA executor.

```sh
cmake -S experiments/baseplane_moonshot -B /tmp/ce-moon-cuda \
  -DCE_MOON_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/bin/nvcc
python3 /tmp/moonshot_build_slot.py cmake --build /tmp/ce-moon-cuda -j1
ctest --test-dir /tmp/ce-moon-cuda --output-on-failure
```

This compiles for sm_70 and links two independent CUDA translation units containing the shared header. It does not launch the tensor GPU probe. GPU execution requires controller-assigned resources and a separately recorded command; no automatic benchmark is present.

## Evidence and limits

All four providers have authored, compiled, run and compared host fixtures. Historical family receipts retain their original source identity and coverage. Integration added guards for finite extreme interpolation, matrix shape/index overflow and nonfinite emitted factor scores, alongside reproductions. The two repaired family host tests pass; [repair receipts](families/tensor/integration-repair.json) and [mechanisms repair receipt](families/mechanisms/integration-repair.json) record exact commands and hashes. Earlier foundation review also guarded lifting/residual arithmetic and independent CUDA-header linkage.

The integrated source at `b72bfa3af4782303bc8630ff355d308b54f55ede` compiled all enabled sm_70 targets with CUDA 12.9 and passed 6/6 host tests. Comparator commit `f6425a78ad728212ba3d24a40cfa1630d2ad18b8` then rebuilt the tensor CUDA harness and passed a host-only self-check rejecting nonfinite actual/expected values. The controller reran that exact binary under the native CUDA foreground interlock: E21-E24 GPU comparisons, directed IDs and capacity overflow passed. Effects/mechanisms CUDA representatives remain compile-only; learning remains host-only. No benchmark metric or timing claim was recorded. [Integration evidence](../../planning/baseplane_moonshot_bootstrap/results/cellerator/integration.md) records exact source, build logs and the final GPU boundary. Historical family receipts remain unchanged.

Quadratic jets truncate higher-order response, interpolation approximates unsampled responses, pruning can change downstream answers, fixed-step fitting has no convergence guarantee, and Boolean rewrite optimality covers only its bounded search space. Version guards require the caller to advance versions correctly. The eight-logit teacher fixture establishes neither held-out biological generalization nor a trained genome model. No performance, broad training, public API qualification or biological efficacy follows from these fixtures.
