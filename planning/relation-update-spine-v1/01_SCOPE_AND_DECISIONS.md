# Relation Update Spine v1: scope and decisions

## Result to build, not work already implemented

One biological relation, prepared once, shall support N16 forward propagation, transpose application, a support-restricted edge-value vector-Jacobian product (VJP), a caller delta or a simple gradient step, real asynchronous publication and subsequent reuse. Native C++ and the bounded compiler bridge shall describe the same mathematics and use the same core implementation. The existing N1 spine remains a regression contract.

This package contains 34 implementation/validation leaves and one coordinating epic. That count emerged from independent correctness and ownership boundaries, not from a quota. There are eight lane records including the coordinator, but no requirement to run eight agents simultaneously. The work is one small mutable computation family, not a general training compiler.

**Authorization now:** construct this package and a prospective demonstration. **Not authorized now:** apply Todo, activate or switch runs, claim tasks, dispatch implementation agents, create worktrees, alter project source/APIs or implement the described kernels. Files are delivered as a repository-relative overlay, not silently written into the remote workspace.

## Governing inputs

The comprehensive review is preserved byte-for-byte in `basis/architecture-review.html`. Its Sections 7.4, 7.5, 10, 11, 13 and 19 provide the architectural basis. Semantic Spine v1 already resolved basic apply/transpose semantic convergence, scalar versus edge-channel contraction meaning and primitive/composition classification. Do not reopen those as unsolved foundation tasks.

The user's later decisions refine that review: compatibility is not a priority; useful old code belongs in core; WMMA repair and an actually executable N16 hybrid edge-gradient route belong in this epic; event-backed publication is required; learning is permitted when biology enables meaningful novelty or a genuine performance advantage. Generic ML work that a mature library can perform equally well without losing those advantages need not be rebuilt. This epic is not a ceiling on the finished library's ML capability.

## Explicit choices

| Question | Decision and reason |
|---|---|
| Computational center | Extend `prepared_relation_pair`; no parallel prepared-training abstraction. |
| Width | New N16 witness and retained N1. Width representation remains general, executable support is explicit. |
| Values | One authoritative f16 physical weight plane; f32 gradient and update arithmetic. No hidden master weights. |
| Dense operands | Keep existing f32 forward/transpose inputs. Edge VJP has explicit full-f32 and half-rounded operand policies, described in 02. |
| Updates | Dense-on-existing-support delta add and SGD-style gradient-step composition use one mutation engine. No new support edges, general sparse conflict updates or optimizer framework. |
| Internal order | Default to the current forward physical value order, shared by transpose references and gradient/update. Measure alternatives rather than treating that order as a permanent law. |
| Publication | Real ready event after update, and a matching consumer-done edge before next in-place mutation. |
| Concurrency | One owner stream and serialized host calls; one outstanding external const read lease for this bounded version. Sequential reads may use different same-device streams. |
| WMMA | Fix direct rectangular legality and provide aligned packing, exact support extraction and sparse residual. No assumption of universal performance superiority. |
| Source origin | A bounded embedded source composition is parsed and semantically lowered independently of native descriptor construction. Both execute through core. |
| Demonstration | Synthetic regulatory network with a dense local module and irregular links, not asserted to be a real inferred biological network. |
| Hardware | Actual sm70 acceptance, with an installed CUDA 12.x toolchain verified at execution. No Ampere permission is implied. |
| Graphs | Existing N1 read-only replay retained; full mutable capture explicitly rejected/deferred. |
| Legacy | Extract good N16 kernels, preserve remaining useful epilogue code and consumers, retire redundant on-path ownership only after replacement tests. |

## Boundaries

No full `.cell` translation unit pipeline, implicit mode, imports, cross-TU optimization, broad IR editing, general planner changes, geometry-census redesign, sparse-exchange optimization, SDK stabilization, standard-library campaign, JIT, multi-device execution, arbitrary precision/width portfolio or generic dense autograd. No topology mutation, dynamic-support learning, master weights, momentum, Adam, scheduler, bias/ReLU/RMS training expansion or dense-loss library.

A local prepared sparse-versus-hybrid choice and explicit force option are needed to expose the implementations. They are not authorization to integrate or redesign the graph-wide planner. Likewise, occupied support grouping for this bounded gradient cover is not the general rectangle-census scalability project.

## Evidence-based freedom

The declarations in `contracts/` are a concrete design baseline, not ABI stabilization. The foundation task may improve names, split internal data or alter mechanics when it preserves the agreed mathematical and lifetime contracts. Changes must keep the demo, task dependencies, acceptance and documentation coherent. No future implementing agent should preserve an expensive abstraction solely because a planning sketch included it.

The experiment succeeds by proving the full shared semantic and mutable execution path. It need not prove that hybrid is fastest on every fixture. It must contain a correct, reachable WMMA path, exact residual/fallback and honest measurements. A result showing that sparse wins some or all tested lifetimes must be preserved, not hidden by excluding packing or transfer costs.
