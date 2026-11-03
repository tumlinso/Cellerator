# Independent condition and consumer review

Initial source review on 2026-11-02 against the approved study contract in
`bench/chromatin_geometry/docs/draft.md`. This review did not run any GPU work.
The observations below identify required follow-up, not a performance result or
acceptance of the full study.

## Findings

1. **The grouped-linear validation bound does not implement the required
   operator-derived numerical contract.** `backends/consumer_tests.cu:140`
   applies a generic `2e-3 * (1 + abs(expected))` bound to dW and local
   dX/dW for both FP16 and FP32. For example, an FP32 zero reference permits
   an absolute error of 0.002 even when every dot-product term is zero; the
   existing operator's bound is zero in that case. The source's independent
   `reference::grouped_linear_allowed_error` uses dot length, the sum of
   absolute products, and FP16 ULP, but this consumer check does not call it.
   Derive and freeze compound bounds across local rounding and overlap
   reductions, then validate every output and VJP against them.

2. **The two-buffer candidate is not a double-buffered staging pipeline.**
   `backends/consumers.cu:369` fills all demanded tile inputs, then reaches a
   single barrier at `:394` before doing numerical work. With two buffers,
   `:376` places forward-demanded and derivative-only features in different
   locations; it does not alternate buffers to overlap staging one tile or
   chunk with consumption of another. The implemented split may be measured
   on its own merits, but cannot establish the planned one-versus-two-buffer
   pipelining ablation. Implement the requested comparison or report it as
   missing, with its actual staging semantics.

3. **All-zero-arity product topologies are rejected.**
   `backends/consumers.cu:446` requires a non-null arguments pointer whenever
   both batch and group count are nonzero, regardless of `argument_count`.
   A valid set of zero-arity product groups has no argument allocation and
   uses the empty product of one. The test at `consumer_tests.cu:112` includes
   one zero-arity group mixed with groups having arguments, and does not cover
   an entirely empty arguments array. Make pointer checks depend on their
   corresponding extents and test this case.

4. **The required poisoned-excluded-forward case is not present in the
   reviewed consumer tests.** The test probes zero and repeated arguments,
   arity 32/33, tail cells, and stale revisions, but does not place poison in
   participant values reachable only through excluded forward groups. Add a
   controlled case with no active consumer of the poisoned values and check
   correct forward results for every family. Source inspection should also
   substantiate the claimed absence of reads: a discarded poisoned value
   alone does not prove that no memory read occurred.

## Integration and evidence boundaries

The reviewed condition fixture's STE setting exposes Boolean-DAG decisions and
an all-candidate derivative plane but no probability buffer. It is an isolated
condition fixture, not a complete STE producer-to-consumer route. The
integrator is wiring the native STE chain separately; the combined test must
establish the declared score, probability, threshold, derivative and update
contract. No conclusion is drawn here about that in-progress integration.

The ordered-product code inspected preserves argument slot order and computes
derivatives by omitting each slot, so the source approach accommodates repeated
participants and zeros. The cohort construction sorts exact primal/derivative
signatures and preserves a cell-order mapping. These observations are source
inspection, not numerical, race, or GPU-performance qualification. In
particular, the planned paired controls, actual HBM and shared-memory behavior,
all sanitizer modes, and end-to-end full-versus-update costs require their
scheduled evidence.

The condition fixture was edited concurrently during review. The initial
versions inspected were SHA256 `388dc27711a3ae7b505c8fdf69be43b0dad37b3024f83dd330cb9c5b0a8dbc2e`
for `conditions.hh` and `ec0bc9f60514a21804d0b373a764a382d085bf0d917da251bb4674c0f4a1c701`
for `conditions.cu`; their later hashes were respectively
`830d8ea27dfa4d4fb04f0332a7f5a6280eee7df99b516d9b899fbf345fcf2866` and
`a81c41458cd22301135697a957db1d6437eeed084408908d36a7037d9bba0659`.
The condition integration note applies to the initial inspected version and
must be rechecked against the final artifact.

Other inspected source identities, all under `bench/chromatin_geometry`:

| File | SHA256 |
| --- | --- |
| `backends/consumers.cu` | `2d894d362e317e2531610568fc28b3cdb6684b3d5a6869e357f7b07d7e433a91` |
| `backends/consumer_fixture.hh` | `d5ff3ecf437f38200966bc1dd5585682ba2aea9932a97b57db5f9c63438eb9ca` |
| `backends/consumer_fixture.cc` | `1300d15cfb90d0bac791302e03fc43085dc7665692b53a3b0c26f25eb0ec5933` |
| `backends/consumer_tests.cu` | `ba3003383549116eb67b8fd80da6465b22403f47bf5143dc3dad894938a0cbcf` |

Findings require rechecking after source changes. No new host or device test was
run in this bounded review.
