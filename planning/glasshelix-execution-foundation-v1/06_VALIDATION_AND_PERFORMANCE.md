# Validation and performance requirements

## Four distinct evidence levels

1. Offline package checks verify catalog consistency, dependency/queue/cross graphs, scope conflicts, references and hashes.
2. Declaration-only syntax checks verify the prospective example's source against guarded planning declarations.
3. Independent reference execution checks fixture mathematics without GlassHelix or Cellerator.
4. Post-epic native/CPU/CUDA tests establish actual implemented capability.

Only level 4 can satisfy native execution and GPU gates. The artifact's `evidence/local_validation.json` records exactly which levels ran during construction.

## Exact test inventories

`machine/acceptance_matrix.json` maps every task to its conditions and qualification. Native command gates use an exact CTest inventory, not a loose regular expression. The runner rejects absent tests, disabled tests, skip return codes, skip expressions and skipped/not-run JUnit cases. A test that executes nothing is not a pass. Baseline and pure contract tasks use governance records and are not represented as GPU tests.

Use independent formulas and independently constructed index maps. Cover empty/tail behavior, irregular n-ary arities, role order, shared definitions/independent values, stale generations, unsupported differentiation, masked NaNs, zero primal/nonzero parameter response, exact a+b null direction and explicitly nonclosed reductions.

Require dot-product consistency of paired JVP/VJP, finite-difference controls on appropriate smooth fixtures and explicit differentiation of the implemented step. Do not mistake derivative-through-real-arithmetic for derivative-through-rounded-storage. Include a forward-only custom operation whose derivative request is rejected.

## Failure safety

Preflight all launch bindings before the first enqueue. Test a late invalid binding with output sentinels and launch counters. After an actual partial CUDA submission failure, poison affected instances and preserve cleanup ownership. Readiness publication means ordered completion dependency, not synchronous completion. Check current-state, parameter, support and derivative generations independently.

## Hardware evidence

GPU groups require the actual installed resource-lease verifier and a shared host lock path configured identically across both repositories. A tasks discover the verifier; no invented command is shipped. GPU tests must assert required device/provider behavior internally. External `nvidia-smi` inventory is supplementary, not proof that kernels executed.

Run compute-sanitizer qualification as actual named tests/wrappers with nonzero error exit and zero-error summaries. Run intentional context-fault cases in separate processes and label them; do not count an expected failure as a sanitizer-clean pass. Do not skip because a V100 is missing. Record it as blocked and continue available host tasks.

## Performance

Collect cold preparation, resident execution and amortized lifetime separately. Include input packs, support selection/compaction, numerical stages, generation publication, transfers and required observation. Report exclusions. Separate profiling traces from uninstrumented timing. Use fixed source/build identities, warmups, multiple repeated samples and uncontaminated GPU access.

Sweep widths including 1,16,33 and larger sizes; interaction count; arity; degree skew; active fraction; support churn; reuse horizon; candidate sharing; and response-direction counts. Real scenarios are correctness-controlled synthetic toolbox workloads, not a biological benchmark.

A faster lower-precision or pruned calculation is an explicit error/performance tradeoff, not an exact implementation speedup. Preserve a correct baseline and measured non-promotion. Graph capture is optional and bounded; mutable/replay generation correctness cannot be bypassed for launch savings.

## No fabricated fixtures

The delivered demo is not permitted to become a dedicated hard-coded backend. Inspectable composition and custom blocks must use general admitted numerical machinery. The Cellerator standalone consumer proves that its additions are reusable beyond the GlassHelix example. Reference formulas can compare results but must never service a purported CUDA call.
