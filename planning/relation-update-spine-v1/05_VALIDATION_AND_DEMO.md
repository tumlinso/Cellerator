# Acceptance, demonstration and performance evidence

## Three separate kinds of evidence

**Package evidence now:** generated views agree, dependencies plus lane queues form a DAG, unordered file ownership does not collide, manifests cover delivered files, bootstrap interlocks fail closed, and the CPU reference-only fixture compiles/runs. A syntax-only compile against declaration sketches verifies C++ call structure, not a linked engine.

**Implementation evidence later:** the new actual Cellerator core and compiler functions compile/link, focused tests execute, and the normal demo reaches real sm70 kernels. Planning declarations and local stubs may never enter its production include/link path.

**Performance evidence later:** repeatable phase-complete sparse/hybrid measurements under matched numerical policies. No speedup or CUDA correctness is claimed by package construction.

## The example

`examples/relation_update_spine_v1/regulatory_learning.cc` is the normal future C++ consumer. `reference.hh` contains example-only independent logical math and a reproducible synthetic fixture; no GPU implementation is hidden there. The CUDA branch requires the proposed installed-in-tree core headers after the epic. `CELLERATOR_RU1_REFERENCE_ONLY` compiles a separate fixture checker that says explicitly it did not run Cellerator or a GPU.

The fixture has 20 source regulators, 19 destination readouts and 16 independent state conditions. Destinations 0..15 and sources 0..15 form a full local regulatory module. Six links outside that module form an irregular residual; destination 18 and source 19 are empty/isolated. This is a synthetic model-construction witness, not a claim about an inferred regulatory network or biological predictive accuracy.

Normal execution constructs native semantics and independently parses an embedded compiler-origin closure. It compares them, prepares one relation, loads generation 1 and checks both forward/transpose origins against an independent reference. Edge gradients are evaluated from native and compiler descriptors in physical order. A caller delta advances generation 2; an explicit gradient-step advances generation 3. A cross-stream const read lease is acquired/returned, a stale-generation call is rejected, and subsequent forward uses the same topology. Canonical export is performed for assertions, not as an internal update step.

Half-policy conformance tests must also include deliberately non-half inputs for which full-f32 and rounded-operand results differ by substantially more than the numerical tolerance. A test whose two policies fall inside the same error interval cannot establish that rounding was honored. Large/cancellation-heavy tests should derive error bounds from reduction length and sums of absolute products instead of borrowing the tiny fixture threshold.

The example loss is `0.5 * sum((RX - target)^2) / 16`; its cotangent is `(RX-target)/16`. It generates a modest synthetic target from another weight vector on the same support, with a small deterministic perturbation to exercise non-half-representable cotangents. X values are exactly half-representable. The mixed VJP is compared against half-rounded X/cotangent and its error versus the full-f32 VJP is reported separately. This prevents a small loss decrease from being mistaken for exact differentiation through quantization.

## Hard numerical acceptance

Forward/transpose full-f32 behavior is compared with double reference sums on the stored f16 weights. Use explicit, recorded absolute/relative or conditioning-aware error bounds; the initial fixture tolerances are 2e-6 + 2e-6*abs(reference) for the bounded dense outputs and mixed VJP, and one half rounding interval around a correctly rounded update when f32 accumulation differences can straddle a rounding boundary. Exact rounding unit tests use exact known inputs and bitwise expected half results. Loosening tolerances requires an explained error analysis and measured error distribution, not just making a failed test pass.

Finite differences use a double continuous model without perturbation-time half rounding. Check both weight VJP and input adjoint separately. Include non-square axes, shuffled edges, identity high bits, empty support/rows, isolated sources, mixed signs, duplicate rejection, overflow-safe capacities, alias offsets, stale generations and same-pointer new input versions. Nonfinite propagation is tested independently of finite-value relative tolerances, including zero alpha with NaN/infinite gradients under the specified FMA policy.

WMMA acceptance must exercise an actual dense tile plus real sparse residual. Verify per-edge output and actual branch/launch attribution. Test direct pointer/stride/capacity rejection, valid packed layouts and zero/partial tiles. A forced hybrid request that silently uses sparse does not pass.

## Test registration and hard gates

The integrator must register exactly the future CTest names in `machine/acceptance_matrix.json`. `scripts/run_gate.py` first enumerates CTest's JSON inventory, rejects missing tests and skip properties, runs the exact required names, then parses JUnit results to require the expected successful testcase set. The GPU binaries must internally require sm70; missing GPU exits nonzero, not CTest skip. No unvalidated receipt can replace this execution.

I05 and I08 use the existing project hardware lease. Required Compute Sanitizer memcheck covers each GPU acceptance binary and the normal demo. Racecheck/synccheck cover shared-memory WMMA and event-lifecycle fixtures where supported; tool limitations are documented and block the relevant gate rather than silently passing it. Evidence includes source HEAD plus dirty paths, build/tool identity, GPU UUID, command/exit, numerical profile, fixture identity and raw output hashes. Post-pruning code must be rerun, not certified using pre-pruning binaries.

## Benchmark design

Use dense-heavy, irregular sparse and exact mixed supports, with independent input/weight resets for fair repetitions. Compare forced sparse and hybrid under the same half-rounded policy, plus a separately labelled full-f32 path. Report warm-up, repetitions, median/spread and preparation amortization at reuse 1, 16, 128 and a workload-justified longer horizon. Separate weight refresh, dense operand packing, score production/extraction, residual, update, publication and observation. Report peak memory and logical-export counts. The demo's timing is not a promotion benchmark.

Do not autotune by repeatedly updating application-owned live weights. A nonmutating measurement setup or disposable clone is required. A negative WMMA result remains valuable evidence and must inform the local default; it does not permit omitting its required correct executable path.
