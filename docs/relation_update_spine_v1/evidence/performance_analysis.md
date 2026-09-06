# V100 lifetime comparison

No hybrid crossover was found. Automatic dispatch remains sparse; the forced legal hybrid path remains implemented and tested. This is a bounded synthetic N16 result, not a universal WMMA or biological application claim.

## Matched half-policy results

Medians in microseconds per update. Each resident sample includes forward, transpose, gradient, in-place f16 gradient-step and ready publication. Seven samples after two warmups; disposable weights reset outside each timed lifetime.

| Support | Updates | Sparse resident wall | Hybrid resident wall | Sparse accounted lifetime | Hybrid accounted lifetime |
|---|---:|---:|---:|---:|---:|
| dense | 1 | 59.515 | 68.263 | 9616.744 | 14245.018 |
| dense | 16 | 48.285 | 55.445 | 640.309 | 945.129 |
| dense | 128 | 47.575 | 54.757 | 121.469 | 165.922 |
| dense | 1024 | 47.483 | 54.642 | 56.806 | 68.350 |
| mixed | 1 | 59.051 | 69.777 | 9478.530 | 16477.268 |
| mixed | 16 | 48.149 | 58.205 | 638.342 | 1116.161 |
| mixed | 128 | 47.469 | 57.464 | 120.695 | 187.178 |
| mixed | 1024 | 47.394 | 57.373 | 56.564 | 73.647 |

## Scope and accounting

GPU 0: Tesla V100-SXM2-16GB, sm_70, UUID GPU-21131915-1488-23af-38dd-1743ae1f5cc8, PCI 02:00.0; driver 580.173.02, CUDA/nvcc 12.9.86, g++-12 CUDA host compiler, Release. One GPU, no inter-GPU communication. Exact build cache, tool outputs, GPU inventory, lease, benchmark-mutex logs, commands and hashes are embedded in performance.json. Both controllers completed successfully and released the reservation/mutex.

Main fixtures have 4,099 sources and 4,101 destinations: dense 65,536 edges, mixed 65,542 edges (only six residual edges), irregular 6,150 edges. Tiny fixtures use 19 sources and 21 destinations and 256/262 edges. Logical edges are shuffled; source/operand fingerprints and all sample distributions are in JSON. Dense operands are deterministic non-half f32; matched sparse/hybrid gradients both round them to f16 and accumulate in f32. Forward/transpose use stored f16 weights with f32 arithmetic. Full-f32 gradients are a separately labelled semantic control.

Accounted lifetime charges cold topology and gradient preparation, initial H2D, per-sample weight reset, resident wall interval, and post-lifetime lease/D2D/D2H observation, amortized over the update count. Caller allocations, fixture/reference construction, validation downloads, teardown and cold temporary-memory high-water marks are excluded. CUDA event intervals include submission gaps; they are not pure kernel sums. Cold preparation is one observation per process, even though repeated in each sample row. Three extra independent processes per mixed route confirm the preparation difference.

Every measured sample passed the independent gradient tolerance and exact stored-f16 recurrence. There were 51 processes and 321 accepted samples, plus two expected failing eligibility controls. I05 supplies independent forward/transpose, lifecycle and sanitizer acceptance. This comparison measures no predictive learning quality.

## Attribution (separate Nsight Systems traces)

Nsight Systems 2025.3.1.90 captured one 128-step lifetime after warmup. These perturbed traces are excluded from promotion medians. The following totals are divided by 128 steps; pack rows sum all pack launches.

| Trace | Kernel | Launches per step | GPU microseconds per step |
|---|---|---:|---:|
| mixed_sparse | feature_major_small_n_kernel | 1 | 22.828 |
| mixed_sparse | transpose_backward_n16_kernel | 1 | 10.784 |
| mixed_sparse | pack_kernel | 2 | 4.771 |
| mixed_sparse | thread_per_edge_kernel | 1 | 4.260 |
| mixed_sparse | update_kernel | 1 | 2.038 |
| mixed_hybrid | feature_major_small_n_kernel | 1 | 22.615 |
| mixed_hybrid | transpose_backward_n16_kernel | 1 | 10.770 |
| mixed_hybrid | pack_kernel | 4 | 9.814 |
| mixed_hybrid | extract_kernel | 1 | 2.592 |
| mixed_hybrid | rectangular_mma_kernel | 1 | 2.392 |
| mixed_hybrid | update_kernel | 1 | 2.135 |
| mixed_hybrid | thread_per_edge_kernel | 1 | 2.027 |
| dense_hybrid | feature_major_small_n_kernel | 1 | 23.090 |
| dense_hybrid | transpose_backward_n16_kernel | 1 | 10.609 |
| dense_hybrid | pack_kernel | 4 | 9.611 |
| dense_hybrid | extract_kernel | 1 | 2.587 |
| dense_hybrid | rectangular_mma_kernel | 1 | 2.296 |
| dense_hybrid | update_kernel | 1 | 2.060 |

The hybrid mixed route executes ten kernels per step versus six for half sparse. Actual WMMA, extraction and residual are present. Hybrid packs four times per step versus two for sparse. The extra packing/extraction/residual launch costs outweigh faster rectangular score production here. Forward and transpose dominate kernel time; likely limiters are sparse index/weight traffic and 16-column reuse, with launch overhead dominant for tiny cases. No HBM-bandwidth, cache-efficiency or occupancy measurement was made, so this is attribution and a limiter hypothesis, not a roofline claim.

mixed_sparse: 128 readiness cudaEventRecord calls average 0.708 microseconds of profiled host API time; boundary timing records are excluded. No device allocation, event creation, memory copy or device-wide synchronization appears in the captured hot interval. The one event synchronization is the explicit lifetime observation boundary.
mixed_hybrid: 128 readiness cudaEventRecord calls average 0.853 microseconds of profiled host API time; boundary timing records are excluded. No device allocation, event creation, memory copy or device-wide synchronization appears in the captured hot interval. The one event synchronization is the explicit lifetime observation boundary.
dense_hybrid: 128 readiness cudaEventRecord calls average 0.695 microseconds of profiled host API time; boundary timing records are excluded. No device allocation, event creation, memory copy or device-wide synchronization appears in the captured hot interval. The one event synchronization is the explicit lifetime observation boundary.

## Controls and decision

Irregular forced hybrid fails with no legal nonempty rectangular cover; full-f32 forced hybrid fails because that semantic profile cannot use WMMA. Full-f32 sparse and N1 forward-only controls pass; N1 is not an N16 training speed comparison. At 128 updates tiny dense half sparse/hybrid take 28.006/34.642 us per step, tiny mixed 30.149/39.606. Fixed operand versions do not skip packing and show no cache benefit.

At 128 updates, dense half resident memory accounts for 3,867,712 persistent + 262,400 scratch bytes on sparse, versus 4,166,000 + 786,688 on hybrid; both also own 1,573,888 caller device bytes. JSON includes exact counts for every fixture. These are accounted live allocations, not a measured cold-preparation peak. Internal logical exports are zero in the timed loop; one physical lease observation follows each lifetime. No hidden canonical export is charged as an internal step.

At 1,024 updates, hybrid resident wall is about 15% higher on dense and 21% higher on mixed; accounted lifetime remains higher. Three fresh mixed runs have sparse gradient preparation 0.350–0.368 ms versus hybrid 7.443–7.559 ms. No break-even was observed over the measured horizons. Preserve conservative automatic sparse policy and legal forcing; do not promote hybrid or start an unrequested optimization campaign. Broader real biological supports, other GPUs, arbitrary N, pack caching and process-wide peak profiling remain unmeasured.
