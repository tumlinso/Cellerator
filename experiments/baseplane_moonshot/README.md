# Experimental numerical providers

This independent C++17 module adopts the reusable numerical seeds from the
moonshot bootstrap. It does not depend on completion of ML2 or modify the
production runtime. Baseplane owns sequence interpretation and source support.

```sh
cmake -S experiments/baseplane_moonshot -B /tmp/ce-moon-host -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build /tmp/ce-moon-host -j 2
ctest --test-dir /tmp/ce-moon-host --output-on-failure
```

Consumers explicitly provide the provider directory as `CE_MOON_SOURCE_DIR` and
call `add_subdirectory` if `Cellerator::moonshot` is absent. Link that target to
include `<ce_moon/reference.hpp>`. No sibling path is inferred. It exports
`Dfa32`, `CountedDfa32`, `MonomialAffine<N>`, `lift`, `unlift`, `ResidualTree` and
`Relation16`. Composition visits the left effect then the right effect. DFA
states are in `[0,32)`; counted composition rejects unsigned overflow. Affine
operators use validated permutations and double precision. Residual tree inputs
are finite, nonempty scalar vectors; `above` certifies only scalar maximum
queries. Lifting rejects nonfinite inputs and intermediate overflow, including
overflow while building or reconstructing a residual tree. Floating operations
are approximate.

`<ce_moon/volta.cuh>` uses namespace `ce_moon_cuda` for DFA, counted DFA,
monomial composition, lifting, WMMA, relation thresholding, DP4A, butterfly and
texture response kernels. Caller-owned nonaliasing buffers, capacities,
validated state domains and streams remain explicit. Its comments define
layout and precision. These functions do not provide a generalized launch API.
`CE_MOON_ENABLE_CUDA=ON` enables compilation for sm_70; it does not launch a GPU.
It also links two independent CUDA translation units that include the provider
header. Kernel definitions use internal linkage for independent consumers.
Set `CMAKE_CUDA_COMPILER` to a toolkit that supports sm_70 when needed.

Each future family owns `families/<name>/CMakeLists.txt`, sources and tests.
The build discovers these directories, so effects, tensor, learning and
mechanisms workers can add targets independently and link the provider target.

`python/learn_lut.py` retains the small synthetic eight-logit learning and
hardening fixture. Agreement with its Boolean teacher is not held-out
generalization or biological validation. No timing, GPU correctness or
scientific usefulness is established by a successful host build.
