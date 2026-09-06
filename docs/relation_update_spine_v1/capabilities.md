# Relation Update Spine v1 capabilities

One native prepared relation owns immutable FMP1/CTP1 topology and one mutable physical f16 value plane. Native C++ and independently parsed/lowered relation closures consume the same calculus and execute through the same core.

| Surface | Accepted bound |
|---|---|
| Forward/transpose | N1 retained; packed N16 f32 dense operands, stored f16 weights, f32 arithmetic |
| Edge VJP | N16 full-f32 operands or explicit RNE f16-rounded operands with f32 accumulation; physical edge order |
| Value updates | Caller delta and explicit gradient-step/FMA; f16 storage, no f32 master plane; exact next generation |
| Compiler origin | Existing narrow parser/Sema/IR closure, independently compared effects; executable forward, transpose, VJP, update/publication and observation actions |
| Geometry | Cold exact bounded rectangular cover plus exact sparse residual; direct physical projection consumption |
| WMMA | Actual sm70 m16n16k16 path, legal aligned packed operands, score extraction and residual; forcing cannot bypass legality or numerical policy |
| Dispatch | Automatic sparse; explicit legal hybrid preserved. No measured hybrid crossover on I06 fixtures |
| Lifetime | One owner stream/device and at most one external const read lease; ready event after writes, reader-done join before next writer; exact identity/generation tickets |
| Reuse | One topology preparation; changing values/operand versions and both update styles reuse it; bounded scratch and events allocated cold |
| Capture | Retained N1 read-only graph replay; new mutable/update/publication/read-lease capture rejected before side effects |
| Retained training | Bias/ReLU/RMS, backward epilogue and fused sparse/bias updates remain with live native/Torch consumers; N16 transpose arithmetic now shared |

Validation includes all 25 RU1 tests, retained native training/program/concurrency tests, the Torch autograd adapter, the 12 mandatory acceptance cases, and the final 14 Compute Sanitizer runs. Missing sm70 is failure, never a skip. The final command receipt pins source, binaries, exact commands, raw summaries, numerical policy, GPU UUID, hardware lease and benchmark mutex.

The mixed synthetic regulatory demo executes independent native and compiler origins, both update styles, three generations and three returned read leases with one preparation. Both forced routes pass independent references. Loss decrease is a fixture witness, not evidence of biological predictive quality.

I06 measured 321 unprofiled samples from 51 processes plus eligibility controls and separate kernel/API traces. At 1,024 updates, dense half-policy resident wall was 47.483 us/step sparse versus 54.642 hybrid; mixed was 47.394 versus 57.373. Setup-inclusive accounted lifetimes also favored sparse. See [performance analysis](evidence/performance_analysis.md) for scope, exclusions and distributions. No universal speedup, complete application latency or process-wide peak-memory claim is made.

See [native contracts](native_runtime.md), [compiler boundary](compiler_boundary.md), [readiness](readiness.md), [source disposition](source_disposition.json), [final evidence](evidence/final_commands.json), and [deferred work](deferred_work.md). The authoritative architecture remains the documentation spine; this page records the bounded implementation outcome.
