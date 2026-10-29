# Sparse support, indexing, and gating implementation contract

User-approved implementation, 2026-10-27. Native logical support and learned conditions remain independent of physical packing. Two synthetic trainable consumers: ordered event embeddings with occurrence/count/time windows, and rank-aware gene embeddings. No biological validation or implicit time-dependent law.

## Authority and ownership

The user explicitly authorized this implementation. The installed Project Control next_task front door returned workflow_task_already_active; its preserved selecting record refers to CE-TOKENIZATION-ATEN, not this work, and has no selected task/attempt. The user subsequently confirmed tokenization was finished and authorized invalidation. The stale pre-task cache selection was atomically invalidated using the installed runtime API, without modifying Todo authority. Plan registration preview still returns native_operation_rejected; native task claim/completion remain unavailable and are not asserted. Additive source work is bounded by the approved request and this contract. GPU execution requires separate finite Device Scheduler admission.

## Native interfaces

Logical IDs, immutable structure epochs, mutable state and parameter generations are distinct. Reverse/dependency postings, sorted intersections, temporal occurrence indexes and bounded seeded discovery feed capacity-checked work views. Condition DAGs share predicates and produce sparse changed work or cross-cell bitmaps. Primal and surrogate-derivative sets are explicit. Participants retain role, order and multiplicity.

A single canonical native FP32 parameter owner exposes one stable Torch leaf, with existing guarded Adam events. Parameters are shared entity/mechanism/scorer/transition parameters; new support creates no per-edge optimizer parameters. Freeze support/projection for each live tape. Publish disjoint stable base plus adaptation delta between windows; never confuse adaptation delta with the packed MMA residual.

## Numerical and packed contract

Initial relation: Y[b,d,k] = sum_e W[e] q[b,e] X[b,s,k]. Width 64 anchors actual CE-GEO projection/value-map reuse. Add per-cell gates, batched forward and explicit VJP dX,dW,dq, direct physical/logical maps, and sparse SIMT fallback. Latent update z_next=tanh(Az+Bu+b) has native selected gather, transition and VJP. Half storage/multiply, FP32 accumulation/gradients/master parameters; explicit FP32 reference. No BF16/TF32 dependency.

p=sigmoid(score/T), q=p*(p>=threshold). Soft warmup precedes actual pruned forward fine-tuning. STE q_train=p+stop_gradient(q-p); gate derivative follows p on the bounded derivative set, other gradients use actual q. Derivative-only work is timed and reported. A zero gate is not structural deletion.

## Validation and evaluation

Independent CPU/Torch reference checks for capacity/empty/repeated/expiry/multi-participant/generation cases. Explicit surrogate VJP oracle; finite differences for smooth components only. Recurrent gradients, real packed forward/backward, tape safety, optimizer identity, and checkpoint continuation. Synthetic hidden rules stay validator-only. Compare heldout learned loss against soft and shuffled support controls, report discovery and pruning quality separately.

Native host tests under tests/sparse_support; Python trainable qualification under tests/sparse_support; benchmark/profiling artifacts under bench/sparse_support/evidence. CMake uses additive optional native and Torch component targets. Canonical source remains authoritative; existing ctxpp index is stale/degraded, use focused canonical inspection without rewriting generated indexes.

Complete measurements include prepare/propose/index/gate/worklist/pack/forward/backward/optimizer/revision and peak memory; report amortization at short/long reuse. Compare full rescan, incremental postings, bitmaps, SIMT, and gated packed. CUDA events establish speed; Nsight Systems/Compute explain dominant kernels. No universal winning backend or speedup is presumed. GPU jobs are finite/exclusive UUID requests with quiescence, explicit CUDA12.9 SM70 tools and NCU2025.2. Preserve source/binary hashes, tool versions, seeds, numerical gates, raw timings and scheduler cleanup receipts. KDA experiment engine handles repeated candidate comparisons.
