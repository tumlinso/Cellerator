# Semantic Spine v1 deferred obligations

The completed witness is deliberately narrow: native C++ and the existing
parser/Sema/IR source slice share a canonical relation descriptor, and real
Volta forward/transpose execution reuses a prepared pair. The architecture
remains governed by [the architecture spine](../architecture.qmd) and the
[epic contract](../../planning/semantic-spine-v1/02_SEMANTIC_AND_NATIVE_CONTRACT.md).

The following remain unimplemented by this epic:

- Full `.cell` source-to-device executable compilation, implicit `.cell` mode,
  imports/module loading, cross-translation-unit optimization, and driver/JIT work.
- General dtype/width support, affine accumulation, duplicate-edge physical
  candidates, broader nonfinite/arithmetic restrictions, and a general planner
  portfolio. The native witness uses N1, f16 relation values and f32 arithmetic;
  valid but unsupported requests must remain explicit rejections.
- General contraction GPU integration, chain/hierarchy/gradient execution,
  composition optimization, exchange execution, and publication-effect scheduling.
  Support-dot/channel and classification tests establish semantics, not these
  accelerator implementations.
- Multi-device or multi-stream prepared-pair ownership, broader training,
  optimizer steps, new geometry search, and new MMA/WMMA/PTX kernels.
- Public API stabilization, installed SDK packaging, full standard-library
  execution, and whole-program optimization.
- Performance promotion. The matched direct-provider comparison characterizes
  adapter cost on the recorded V100 fixture; it establishes no universal speedup,
  planner supremacy, scientific dataset result, or cross-hardware claim.

The meaningful `.ceh` umbrella was preserved byte-for-byte as `.cell`. That
conversion is not a module implementation. Existing providers and useful legacy
algorithms remain available under the dispositions recorded in
[source_disposition.json](source_disposition.json). No follow-on epic is started.
