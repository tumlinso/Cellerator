# 1. Scope and design decisions

## Fixed purpose, limited commitment

Cellerator's goal is to make biological mathematics and computation as fast as practically possible by exploiting biological meaning, stable structure, identities, representative data and target hardware. This epic improves the trustworthiness of the path carrying that information into execution. It is not a miniature version of the entire review roadmap.

The comprehensive review is the authoritative design basis, qualified by the user's subsequent decisions. Current source is implementation evidence, not an immutable specification. JBC and CCP are mechanical planning precedents, not reasons to keep their old architectural choices.

The allocation is **29 atomic tasks, one root epic, seven lanes**. The small size is achieved by limiting computation and integration, not by concealing many unbounded features inside a single task. There is no promised elapsed completion time. A material prerequisite failure should be surfaced rather than expanding this into a new compiler campaign.

## Concrete choices

| Choice | Decision for this epic | Why |
|---|---|---|
| Executable family | Relation apply and transpose | Exercises direction, exact support, axis identity, values and reusable preparation |
| Required hardware witness | Real sm70 device; existing kernels | Makes semantic unification cross the accelerator boundary without a new kernel project |
| Required width | N1, with multiple independently bound state vectors | Existing builtin forward/transpose routes share this regime; wider kernels are not a prerequisite |
| Required numerical tuple | f16 relation storage, f32 input/multiply/accumulation/output | Matches existing narrow candidate contracts; not a global Cellerator precision policy |
| Numerical checks | Explicit absolute+relative tolerance on matching stored inputs | Avoids requiring CPU/GPU bitwise equality while preserving exact structure and type policies |
| Native exposure | Provisional prepared forward/transpose pair | Enough for a real consumer-shaped program without an SDK overhaul |
| Compiler exposure | Real parser/Sema/IR-origin descriptor and common validator | Tests semantic authority without promising a functioning full `.cell` driver |
| Runtime lifetime | One device and one caller stream per pair | Avoids a multi-reader/multi-stream ownership redesign in the first epic |
| Support changes | Topology and order fixed for a prepared pair; value generation changes | Demonstrates the most useful existing reuse boundary |
| Physical selection | A truthful existing narrow portfolio; no new planner search | Proves actual reachability rather than optimization research |
| Compatibility | No promise to preserve bad interfaces; useful code preserved | There is no external consumer to justify maintaining semantic mistakes |
| `.ceh` | Remove redundant content or convert meaningful content to `.cell` | Avoids a permanent duplicate Cellerator header format |
| Cross-unit semantics | Record future commitment only | Modules, implicit `.cell` dialect and whole-program optimization remain outside the executable milestone |

N1 is an execution-regime limit, not a semantic cardinality limit. Keep 64-bit logical extents/counts and reject unimplemented local-index ranges before narrowing. Similarly, a mathematically valid update or duplicate-edge relation may be unsupported by the selected physical candidate. That distinction must remain visible.

## What to preserve

Retain the useful persistent identity, structure epoch, logical/physical order, existing packing/projection constructors, forward/transpose value-position mapping, numerical type separation and stream/readiness work. Avoid creating new generic storage containers, a second runtime session, or another parallel scalar interpreter.

Preserve the source language's useful low-level philosophy and familiar relation expression. Do not use this task as an opportunity to rewrite the language grammar. Native C++ is a first-class user, not a compiler-embedding workaround.

## Source changes which are justified

A small canonical relation descriptor and validator; checked legacy/native adapters; bounded changes to existing relation IR lowering; a real source-to-descriptor test bridge; narrowly scoped contraction/classification repairs; the provisional prepared-pair implementation; meaningful tests; minimal opt-in build integration; and content-preserving `.ceh` conversion.

The integrated source should have one authority for relation meaning. Legacy structures that remain at provider boundaries are transport/adaptation records, not competing definitions of the mathematics. A removed contract needs an explicit disposition and regression evidence, not an assertion that a newer version must be better.

## Explicit exclusions

No full `.cell` executable compilation, implicit-dialect parser rollout, module/import loader, cross-TU optimizer, installed SDK packaging, public API freeze, new MMA/WMMA/PTX strategy, general gradient/training program, contraction provider portfolio, exchange optimization, geometry-search optimization, learned planner, JIT, multi-device runtime, full standard library or general repository cleanup.

The `.ceh` decision is a deliberate narrow exception to “no cleanup”: preserve and relocate its content, not redesign the library. Historical text mentioning `.ceh` need not be erased. Ordinary `.hh` files remain appropriate for native C++.

## Hard completion boundary

Finish only when both origins reach one canonical semantic contract; real forward and transpose accelerator execution is correct on independent fixtures; structure survives two value generations; stale/incompatible requests fail before output mutation; the contraction and classification counterexamples are addressed; no active `.ceh` remains; and the delivered example passes.

Then stop. Do not automatically start the next epic, retire unrelated worktrees, merge all branches, activate another campaign or claim that full language support is complete.
