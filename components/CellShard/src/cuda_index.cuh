#pragma once

#include <cuda_runtime.h>

#include "types.cuh"

namespace cellshard::cuda_index {

__host__ __device__ __forceinline__ types::u32 add_u32(types::u32 lhs, types::u32 rhs) {
    return lhs + rhs;
}

__host__ __device__ __forceinline__ types::u32 sub_u32(types::u32 lhs, types::u32 rhs) {
    return lhs - rhs;
}

__host__ __device__ __forceinline__ types::u32 shr_u32(types::u32 value, unsigned int shift) {
    return value >> shift;
}

__device__ __forceinline__ types::u32 global_tid_1d() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ __forceinline__ types::u32 global_stride_1d() {
    return gridDim.x * blockDim.x;
}

__device__ __forceinline__ types::u32 segment_tid_2d() {
    return blockIdx.x * blockDim.y + threadIdx.y;
}

__device__ __forceinline__ types::u32 segment_stride_2d() {
    return gridDim.x * blockDim.y;
}

__device__ __forceinline__ types::u32 atomic_add_u32(types::u32 *addr, types::u32 value) {
    return atomicAdd(addr, value);
}

} // namespace cellshard::cuda_index
