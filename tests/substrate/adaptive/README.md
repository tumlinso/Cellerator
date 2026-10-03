# Native adaptive host slice

Run the focused gate with an actual installed host SDK:

```sh
python3 -B tests/substrate/adaptive/run_native.py --sdk /tmp/ce-is1-sdk-a
```

The consumer compiles the scoped adaptive source and links the installed native
structured-state library. Linear and quadratic callbacks are test providers;
production frontier providers belong to the mathematics owner after MERGE-B.
The callback receives the last transmitted state, delta and prior output.
The ledger conservatively invalidates on structure, parameters (including actual
weights), activity, slot incarnation and complete declared program/query/world
context. Value generation may advance without transmission. Callers must declare
all relevant context and advance program identity when replacing callbacks.
The current-output discrepancy is measured by an additional full evaluation;
this slice makes no speedup claim. Threshold zero still uses floating arithmetic.

Supplied linear maps are checked by inverse, dynamics intertwining and readout
identities over the complete matrices, plus mapped state initialization. The
`exact_linear` label denotes these algebraic identities verified within the
caller-supplied finite absolute tolerance; it does not mean bitwise floating
equality. A supplied projection/reduction must carry `approximate_linear`; the
report exposes inverse, dynamics and readout residuals and current readout
discrepancy. These are local discrepancies, not trajectory error bounds.

Publication stages before replacement and blocks while external native tapes
hold metadata leases. The caller serializes all access and keeps the publication
owner alive until every lease is released. This adds no differentiation engine,
concurrent publication protocol or CUDA stream ownership. Changed bases advance
slot incarnation; ordered coordinates and generations remain native. Optimizer
moments reset unless an explicit mapping preserves coordinate/incarnation, state,
basis row, and the unchanged law/readout. Recycled slots selectively reset.
Arbitrary nonlinear rewrites, recurrent residual insertion and general optimizer
basis migration are unsupported.

Integration owner SDK recipe (shared build files are outside this leaf's scope):

```cmake
add_library(cellerator_substrate_adaptive STATIC src/math/adaptive/reuse.cc)
target_compile_features(cellerator_substrate_adaptive PUBLIC cxx_std_20)
target_include_directories(cellerator_substrate_adaptive PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(cellerator_substrate_adaptive PUBLIC cellerator_substrate_structured_state)
```

Include this target in the existing SDK export/install target set and install
`include/Cellerator/math/adaptive/reuse.hh` alongside the existing public headers.
After installation, consumers include that header, link the adaptive target, and
supply their mathematics owner callbacks. No registry extension is required.
