# Shared-support composition evidence

The installed CelleraTorch package passed 9 tests with no failures, errors or skips. The tests cover the declared shared-support expression and its CUDA gradients, logical permutations, repeated support, optimizer publication and checkpoint identity. This is engineering qualification on fixtures; it does not establish a biological fit.

The root ran both manifest commands under CUDA controller evidence `8452c23e-942e-4085-8602-ae9b4b3562a7`. `manifest.json` records the exact source, installed package, wheel and native library hashes before execution; `run.json` confirms unchanged inputs after execution. The native library matches the earlier CE-ML2-TRAIN qualification. New composition execution was tested here.

The benchmark compares identical support and coefficients in a native shared-support composition and a materialized Torch expression. Both paths include input and target uploads, forward, loss, backward, Adam updates and coefficient publication. Cold setup includes identity packing, initial coefficient upload, preparation and optimizer creation. Each row is the median of five measured iterations after two warmups; these are synchronized wall times.

| Fixture | Path | Complete setup (ms) | Complete iteration (ms) |
|---|---|---:|---:|
| shared_batch | native_composition | 9.681241 | 1.816046 |
| shared_batch | torch_materialized | 0.855325 | 2.955643 |
| small_counter_regime | native_composition | 2.101673 | 1.870941 |
| small_counter_regime | torch_materialized | 0.593357 | 2.865728 |

The results retain both the shared-batch fixture and a small counter regime. They are measurements, not a promotion or general performance victory. Full stage samples, numerical agreement and array storage formulas are in `benchmark.json`.

Torch peak allocation counters exclude raw native allocations. Global free-memory observations also include native allocations and Torch pools, can include other processes, and miss within-stage transients. They cannot establish per-owner native reserved bytes or be added to Torch counters. Native support upload is included in preparation where the public API cannot separate it. Checkpoint rebuild cost was not measured.

The pure verification command is:

```sh
/home/tumlinson/Software/venvs/cellerator-ml2-py313-cu126/bin/python docs/learning/verify_bio_evidence.py verify
```

The verifier checks current source/install/library hashes, the original command manifest, positive pytest counts with no skips or failures, successful controller identity, complete-cost samples and medians, and retained evidence hashes. Preparation and verification launch no GPU work.
