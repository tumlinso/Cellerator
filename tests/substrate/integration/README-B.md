# Native core B composition

Run from the repository root:

```sh
python3 -B tests/substrate/integration/check_b.py --build-dir /tmp/ce-is1-core-b --sdk-prefix /tmp/ce-is1-sdk-b
```

The gate configures the selectable substrate component graph, builds the host
owners, installs a sibling development prefix, and compiles/runs one consumer
against that prefix. Each invocation configures again to refresh source revision
metadata. This is development consumption evidence, not package distribution or
relocation qualification. No GPU or leaf suite is run here.

Components are `packing_strategies` (packing_strategy), `frontier` (effects and
native_numeric), `response` (prepared_host), and `adaptive` (structured_state).
The existing effects/mechanism numerical owners remain dependencies. The host
response dependency requires CUDA development headers, but no CUDA runtime
execution or CUDA component is selected by this gate.

Callable composition seams:

- `adaptive::bind_polynomial` returns evaluator/delta callbacks for a
  `delta_ledger`. It calls the frontier owner for the forward and exact quadratic
  delta, including the quadratic remainder. Coefficients and current generation
  metadata are borrowed and must outlive both callbacks. The caller serializes
  calls and includes coefficient values, dependencies and whole-world hypotheses
  in the ledger context. Last transmitted state supplies the delta base.
- `frontier::make_polynomial_block` binds the same native FP64 polynomial owner
  to DIFF's `response::bind_custom_rule` and the existing prepared program runner.
  `polynomial_binding` is the typed launch payload. Contract spans, block, primal
  matrices, tape and live generations remain caller-owned through synchronous
  execution. Only square FP64 forward/JVP/VJP with nearest-even propagate policy
  and overwrite output are supported. Second directions, device streams and
  other policies are rejected. Matrices allocate; allocation-free memoization is
  not declared. Native owner failures map to the runner's invalid_argument.
- `requested_scaled_tanh` consumes canonical coefficient slots and selects
  requested output/state/parameter responses. The consumer checks JVP/VJP
  duality and a changed parameter generation rejects the saved primal.
- `publication` borrows no external tensor owner. Its lease prevents rewriting
  native snapshot metadata until tapes have drained; caller serializes access
  and keeps the publication alive through lease release.

The consumer starts from a strategy-produced logical relation result, feeds it
into the frontier and ledger, binds frontier JVP/VJP as custom stages, then feeds
frontier values/tangents into requested response. It checks stale generations
preserve the tangent destination and a pinned publication rejects rewriting.
Exact refers to the declared algebraic polynomial identity, not bitwise equality
or biological equivalence. The accepted leaf evidence remains separate.
