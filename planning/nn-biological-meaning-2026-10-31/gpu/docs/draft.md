# Biological-meaning NN GPU integration qualification draft

## Objective and boundary

Qualify the new biological-binding and query compositions on one admitted Tesla V100 (`sm_70`) using focused forward/backward correctness checks. This is a correctness smoke, not a benchmark, kernel campaign, or scientific validation. No numerical CUDA kernel or production code is changed here. All new qualification code and evidence stay under this directory, except the opt-in test entrypoint `tests/framework/test_biological_gpu.py`.

The current neural components are ordinary Torch modules. The GPU checks should establish that the metadata/preparation APIs direct the intended biological occurrences and that the resulting differentiable Torch computation behaves consistently with an independent CPU/reference route. Structural preparation is explicitly host-side; device structures are moved explicitly and should be reused without structural GPU-to-host checks in each forward.

## Baseline and references

- Existing `RelationAttention` raw CSR and dense implementations are the numerical baseline for `PreparedRelation` on identical support.
- Existing Torch autograd and explicit loss/alignment behavior are the oracle for feature readout and `InteractionPrograms` compositions. Where there is no raw equivalent, compare GPU results and vector-Jacobian products with a same-weight CPU module initialized from identical state.
- The existing grouped-linear native adapter, if importable from the task's isolated extension directory, is optional supporting evidence only. It is not required to accept this biological-composition smoke and must not be rebuilt or installed globally.
- The job runner asserts explicit admission by `CELLERATOR_TEST_CUDA=1`; tests skip when ordinary discovery lacks that admission marker. Test discovery must never request or initialize a GPU by default.

## Workload and contracts

Use compact deterministic fixtures: packed source/target occurrences with repeated identities; CSR with a duplicate edge, an empty target neighborhood, and overlapping source participation; small widths divisible by head count; 2-3 samples/windows with feature subsets; and multiple learned programs with overlapping membership. The structural feature and target orders, offsets, axes, and structural generation are fixed independently of learned values.

Check:

1. `FeatureBinding` maps named identities to declared model indices, while selected readout outputs preserve requested identity ordering. Repeated occurrences remain separate positions.
2. `PreparedRelation` on CUDA matches raw CSR and dense equivalent support in output, input VJP, and parameter VJP. Exercise FP32 and FP16 if those operations accept both. Record input/weight/output storage types and any accumulation type explicitly; attention may use the module's ordinary Torch precision behavior.
3. Value-only changes reuse one prepared relation and yield results based on updated values. An altered occurrence order or structural generation fails before any silent remapping. Include empty neighborhoods and duplicate/overlap support.
4. A biological query selecting a feature subset matches selection from the full declared readout, and gradients reach selected inputs/head parameters. Explicit conditioning changes result and has a gradient path; frozen-base and joint modes update only the intended parameters.
5. `InteractionPrograms` handles overlapping memberships and context-conditioned program queries on GPU. Gradients from features and multiple members reach shared program parameters; empty support remains zero where specified.
6. When practical, run one ordinary differentiable surrounding-layer composition with the already-built native grouped-linear extension. This is ancillary and uses the existing kernel unchanged.
7. On the prepared path, no repeated structural `.item()` validation or implicit structural device-to-host transfer occurs per forward; verify by source and/or a targeted instrumentation counter where practical. No profiler is used.

## Numerical acceptance

For FP32, compare CUDA output and VJPs with CPU/reference at `rtol=3e-4, atol=3e-5` or tighter where stable. For FP16, use represented FP16 values as reference inputs and compare outputs/VJPs with `rtol=8e-3, atol=6e-3`, with documented reason if an operation needs a wider bound. Do not silently widen thresholds to pass. The primary criterion is matching the specified semantics; randomized dropout is disabled.

## Execution and exact validation

1. Author this draft before qualification implementation.
2. Add an opt-in `unittest` test file and isolated admitted-job runner. CPU references and syntax can be checked without CUDA allocation.
3. Fix test/source artifact identity and record SHA-256 hashes before submission.
4. Submit one finite, exclusive UUID-selected V100 job via Device Scheduler using the existing Cellerator client route; no direct/unmanaged CUDA fallback. GPU memory is an admission estimate. Correctness does not need a quiescence precondition.
5. Run the isolated GPU test pattern in that job; wait, collect logs/evidence, verify cleanup, and do not assume resource release until cleanup is confirmed.
6. Record scheduler job/attempt/GPU UUID, runtime/toolchain, source and native extension hashes, test totals, precision, and cleanup state in `REPORT.md` and structured evidence. No timing or biological-validity claim is made.

## Risks and limits

- APIs are still being implemented in parallel; tests must match the delivered contract without expanding production scope.
- The current Device Scheduler route and V100 admission may be unavailable. If blocked, report the failed admission and finish independent CPU-reference/syntax checks only.
- FP16 attention and gradient arithmetic can differ slightly by device/library implementation; tolerances remain explicit and tied to these small smoke shapes.
- A successful smoke only qualifies the exercised paths and shapes. It does not qualify exhaustive inputs, scaling, training convergence, performance, all module compositions, or scientific validity.
