# Sparse support, indexing, and gating foundation v1

## Objective

Build a reusable native Cellerator foundation for discovering and executing sparse interactions, with two trainable application slices: event-stream embeddings for Karyoplast-style workloads and ranked-gene embeddings for Geneformer-style workloads. The first physical consumer extends existing CE-GEO packing, while the logical support, conditions, latent state, and application semantics remain independent of that representation.

## Agreed boundaries

- Preserve logical entity and relation identities across packing, support revisions, promotion, and revocation. Keep immutable structure epochs distinct from mutable parameter and state generations.
- Provide shared topology with per-cell activity; reverse dependency postings; sorted membership intersections; temporal occurrence lists; multi-participant conditions; variable-sized work sets with explicit capacity and overflow reporting.
- Keep native C++ contracts and reference behavior central. Numerical latent-state operations stay under the existing operation layer. Torch integration borrows native structures and canonical parameter owners; ordinary native builds stay Torch-free.
- Implement overlapping learned latent mechanisms with the initial transition `z' = tanh(Az + Bu + b)`, shared parameters across cells, indexed gather/transition/VJP, and no assumed chromatin state or named biological mechanisms.
- Gate conditions can use observed features, latent state, and temporal predicates. Keep forward and derivative work sets distinct; derivative sampling for dormant candidates is bounded, explicit estimator behavior.
- Train with soft gates, then fine-tune the pruned forward using the declared straight-through estimator. Preserve a float32 reference and use FP16 storage/multiplication with FP32 accumulation, gradients, and master parameters where appropriate. No BF16 or TF32 dependency.
- Reuse current geometry/packing mechanisms as the initial physical backend. Add prepared logical-to-physical maps, per-cell gates, forward/backward, gate/scorer gradients, and recurrent backward. Keep irregular SIMT. Do not confuse a support delta with the MMA residual area.
- Support both learn-then-compile and continued adaptation: stable compiled base plus sparse delta, frozen tapes/windows, safe revision publication, and explicit cache invalidation. Checkpoints restore logical state first and rebuild/rebind physical projections.
- Event slices preserve positive occurrence order, repetitions, and identity. Windows filter candidate edges; expiration does not reverse earlier latent updates. Ranked slices preserve gene identity and rank independently; rank is not time.
- Candidate proposal is bounded and replaceable. Existing postings, recent co-activation and seeded exploration propose support; pair/triple participant proposals are sampled, not exhaustively enumerated. Report exploration coverage and misses separately from exact eligibility retrieval.
- Compare complete preparation/training/inference costs for rescans, postings/intersections/bitmaps, SIMT, and packed execution. Width 64 is the initial packed anchor; no backend is assumed universally best.

## Explicit non-goals

No full biological model or biological validation, universal packer, continuous-time law, exhaustive new learned similarity index, or claim that bounded support exploration is complete. No compiled-forward claim of ordinary Torch autograd: custom backward integration, canonical parameter ownership, tape lifetime, optimizer behavior, and compiler boundary must be qualified. No GPU launches without scheduler admission and finite, retained evidence.

## Delivery sequence

1. Contracts, CPU references and exact indexes; runnable uncompiled event and ranked trainable slices.
2. Packed batched forward/backward with Torch parity.
3. Learn, prune, fine-tune and compile lifecycle.
4. Bounded adaptation, support revision, checkpoint continuation and safe projection replacement.
5. Correctness/learning acceptance, then bounded V100 complete-cost comparisons and tuning.

Task ingestion or plan preview is not implementation activation. Execution, subagent dispatch, worktree creation and scheduler jobs remain governed by Project Control and Device Scheduler.
