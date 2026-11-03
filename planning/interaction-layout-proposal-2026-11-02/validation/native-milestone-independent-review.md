# Independent native milestone review

Review scope: the experimental trainable native factorized-interaction component,
its C++ example, focused package install, and existing source-bound device
correctness and sanitizer evidence. This is a milestone assessment, not a
performance acceptance or a comprehensive proof for every possible input.

## Assessment

No material blocker to the bounded native correctness and usability milestone
was identified in this review. Packet remains the current default route and is
experimental; callers can explicitly request direct. Neither a performance
winner nor a production packet-promotion regime has been established. The
performance campaign is paused, with no accepted timings or speedup claim.

The native interface expresses ordered and repeated participant occurrences,
independently indexed parameter banks, explicit numerical context, overlapping
outputs, FP16/FP32 storage selections and FP32 cotangents/gradients. It exposes
separate structural ownership, allocation-free asynchronous calls, explicit
workspace and per-invocation tape lifetimes. The reviewed C++ example runs
forward, all five VJPs and repeated parameter updates without a Torch dependency.
The public header and example were inspected; this was not a line-by-line audit
of every CUDA kernel or a rerun of the independent numerical oracle.

## Identity and checks

- Recomputed hashes for all 84 current files in
  `../evidence/integration/source-sha256.json`; all matched. This includes the
  example, native execution sources, installed consumer, adapter and relevant
  CMake sources. The corresponding current native, example, adapter and build
  files also matched the filtered qualified source manifest for closure
  `1f8e34ce89594fc3d8fb3d63c46600091c24f3016d6a9ddfd9dc3630eb08543c`.
- The later native candidate runner is
  `cc1b11414e9e30daaf9ab169f12d558a81340b62a3ef15d0bf9e66b94567b0af`,
  baseline runner is
  `f388b997e513f927072fad314a4e2cabce6d38a55b9017df143bf5ed3a4d8275`,
  and native archive is
  `73f4f987db9dc4480d0efef371bc358c24b64ad6be3b472412936289fd0a574c`.
  These are the identities bound by the independent nine-report qualification
  bundle acceptance, not a claim that the earlier Torch/example job used the
  same binaries.
- Re-ran `ctest --test-dir /tmp/cellerator-factorized-interaction-build
  --output-on-failure -R '^factorized_interaction_'`: 3/3 host-only structural,
  pattern-binding and CUDA host-contract tests passed. No GPU job was run in
  this review. Protected `cellular_geometry.md` SHA256 remains
  `d37f504e5c1ab629334f4127dcaa60d008afe94e560d41aa936d8ba161124254`.
- Inspected `../evidence/integration/integration-status.md` and the package
  identity record: the focused installed component configured and linked a
  separate CXX consumer without Torch. The recorded package source identities
  match current files. The consumer was not executed. Full SDK install/link is
  outside this evidence because an unrelated existing matrix-patch target fails;
  mixed full-SDK fixture checks are configure-only with placeholders.
- Inspected `native-correctness-v15.suite.json` (SHA256
  `403372eb78a8a031d94821ef901f76b876e2fa8ed82f741e832c80b662c1b29f`):
  it records 29/29 bounded CUDA forward/VJP cases and a 200-update example,
  output `0.202392578125` to `0.230103` toward `0.25`. Its source closure is
  `9afe1e5a6e119f5f1565ac02c2971cdf5ad4117453f0aa9f604f69371bb582a6`;
  the same job records prepared Torch first-order gradient, mutation, reuse and
  cross-stream checks. This first suite covered five of eight storage triples.
- Reviewed the subsequent independent qualification acceptance
  `qualification-bundle-independent-acceptance.json` (SHA256
  `592d36531610a41d9ced736d71201352a1cd4d2dd88b3a05f0ea4d5cb00db7e7`),
  including 32/32 dtype/mode checks, 12 auxiliary cases for each packet and
  strongest-direct request, and 14 cases for each of memcheck, racecheck,
  initcheck and synccheck. That review checked the nine original finished
  Scheduler/Autobench receipts, matching V100 UUID, successful exit and cleanup,
  source/binary identity and relocated bundle reconstruction. Numerical checks
  cover the frozen corpus, not arbitrary values or shapes.

The source hashes and host tests above were executed for this review. Device,
Torch and package execution claims rely on the attributable prior reports and
are not represented as new device or installed-consumer runs. Performance,
whole-model speedup, biological validity and higher-order gradients are outside
this milestone.
