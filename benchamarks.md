# CellShard V100 Benchmark Notes

Date: 2026-04-06

System:
- 4x Tesla V100 16GB, `sm_70`
- Fast NVLink pairs: `GPU0 <-> GPU2`, `GPU1 <-> GPU3`
- CUDA toolchain: `/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9`
- Nsight Systems: `/opt/nvidia/nsight-systems/2024.6.2/bin/nsys`

Benchmark binary:
- `/tmp/cellshard_v100_profile`
- GPU-only loops only: no packfile I/O in the measured regions

Large benchmark config:
- `--parts 64`
- `--rows-per-part 65536`
- `--cols 65536`
- `--avg-nnz-row 256`
- `--shards 16`
- `--build-repeats 4`
- `--upload-repeats 2`
- `--bucket-repeats 4`
- `--bucket-count 128`

## What Was Measured

1. `device_coo_to_compressed`
- Single-GPU device-only COO -> compressed conversion.
- Inputs are copied once, then only the device conversion loop is timed.

2. `multi_gpu_hot_upload`
- 4-GPU upload of host-resident pinned compressed shards.
- Measures stage/release behavior without disk fetch.

3. `multi_gpu_bucket_rebuild`
- 4-GPU resident shard operator path using `build_bucketed_shard_major_view`.
- First-iteration workspace allocation/warmup is excluded from the timed loop.

## Key Results

Steady-state large run after the latest optimization pass:
- `device_coo_to_compressed`: `235.308 ms`, about `5.312 GiB/s` approximate device throughput.
- `multi_gpu_hot_upload`: `575.744 ms` for `12.031 GiB`, about `20.897 GiB/s` aggregate.
- `multi_gpu_bucket_rebuild`: `28.752 ms` for `1,073,741,824 nnz/pass`, about `149.4B nnz/s` approximate.

Earlier large GPU-only run before the latest allocator-cache and bucket-warmup pass:
- `multi_gpu_hot_upload`: `668.686 ms`, about `17.992 GiB/s`.
- `multi_gpu_bucket_rebuild`: `49.721 ms`.

Measured improvement from the latest pass:
- Hot upload improved by about `16%`.
- Bucket rebuild steady-state wall time improved by about `42%`.

## What The Profiling Showed

Primary profile outputs:
- `/tmp/profile_out/nsys/cellshard_gpu_large_v1`
- `/tmp/profile_out/nsys/cellshard_gpu_large_v2`

Post-optimization NVTX timings from `cellshard_gpu_large_v2`:
- `device_coo_to_compressed_loop`: `206.435 ms`
- `multi_gpu_hot_upload_loop`: `574.253 ms`
- `multi_gpu_bucket_loop`: `36.848 ms`
- `generate_parts`: about `17.2 s`

Interpretation:
- Synthetic data generation is still expensive, but it is outside the hot GPU loops.
- The measured GPU loops are now dominated by actual CUDA work, not packfile fetch.

### Upload Path

Before the allocator cache:
- `cudaMalloc` and `cudaFree` were visible inside the hot upload loop.
- Upload loop runtime API cost included about `94.056 ms` in `cudaMalloc` and `17.536 ms` in `cudaFree`.

After the allocator cache:
- Hot upload loop runtime APIs were mostly:
  - `cudaStreamSynchronize_v3020`: `568.729 ms`
  - `cudaMemcpyAsync_v3020`: `11.861 ms`
- `cudaMalloc` and `cudaFree` dropped out of the timed upload loop.

Memcpy view of the upload loop after the cache:
- `12.031 GiB` H2D over `160` copies
- Summed copy-engine time: `2239.43 ms`

Interpretation:
- Upload is now mainly PCIe / copy-engine limited, not allocator-limited.
- The remaining improvement headroom is mostly about reducing or hiding H2D traffic.

### Bucket Rebuild Path

Before the bucket warmup/direct pointer path:
- First-iteration `cudaMallocHost` and `cudaMalloc` noise still appeared in the timed loop.

After the warmup pass:
- Timed bucket loop runtime APIs were mostly:
  - `cudaStreamSynchronize_v3020`: `99.857 ms`
  - `cudaLaunchKernel_v7000`: `23.285 ms`
  - `cudaMemcpyAsync_v3020`: `0.752 ms`
- No meaningful allocator noise remained in the hot bucket loop.

Post-optimization hot kernels:
- `reorder_shard_major_segments<__half>`: `125.211 ms` total
- `sortI32_by_key_merge_core`: `112.928 ms` total
- `sortI32_by_key_local_core`: `102.145 ms` total

Interpretation:
- The bucket path is now dominated by a real GPU kernel: `reorder_shard_major_segments`.
- The new `gather_compressed_part_pointers` fast path worked, but its cost is tiny compared with the rebuild kernel.

### COO -> Compressed Build Path

Hot kernels in the build loop:
- `sortI32_by_key_merge_core`
- `sortI32_by_key_local_core`
- `gather_core<int,256>`
- `gather_half_by_permutation`

Interpretation:
- This path is still fundamentally sort-dominated.
- The current cuSPARSE-based sorted path is reasonable on V100; further gains here likely need either a custom radix path or better reuse/capture of the repeated sort pipeline.

## What Was Learned

- Removing disk from the benchmark was necessary. The old cold-stage profile hid the actual GPU behavior.
- Pinned host residency matters. Without it, hot shard upload underutilizes the copy engines.
- A packed compressed-shard upload is the right shape for CellShard. One allocation per shard is materially better than one allocation per part.
- A small device allocation cache is worth it on this V100 setup. It removed allocator churn from the hot upload loop.
- The bucket path needed warmup to measure steady-state behavior correctly.
- The operator-side direct descriptor gather worked: it removed pointless host pointer staging, but that was never the main bottleneck once the rest of the path was cleaned up.
- After the cleanup, upload is mostly transfer-bound and bucket rebuild is mostly kernel-bound.

## Remaining Headroom

1. Upload path:
- Keep more shard data resident on GPU for longer.
- Reduce H2D copy count further by building or caching shard-wide host slabs instead of separate host part payloads.
- Explore GPUDirect Storage / `cuFile` only if disk re-enters the target workload.

2. Host registration:
- `cudaHostRegister` still dominates the whole-program API summary because host part pinning is done per generated matrix.
- If these host shards are reused heavily, build them directly in pinned memory or keep a reusable pinned host pool.

3. Bucket rebuild kernel:
- `reorder_shard_major_segments` is the main remaining GPU kernel target.
- Likely next ideas: row-binning by `nnz`, vectorized copy for longer rows, or a two-path kernel for light vs heavy rows.

4. COO -> compressed build:
- Repeated build loops may benefit from CUDA Graph capture.
- If this path becomes more important than the bucket path, test a custom CUB radix pipeline against the current cuSPARSE sort path.

5. Format/layout strategy:
- If downstream work repeatedly alternates row-wise and column-wise phases, keeping both CSR and CSC views may still outperform repeated transpose/rebuild work.
