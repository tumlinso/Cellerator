#include <cstdint>

#ifndef CELLERATOR_ENABLE_PROFILING_MARKERS
#define CELLERATOR_ENABLE_PROFILING_MARKERS 0
#endif

namespace
{
struct static_marker
{
    std::uint64_t correlation_id;
    char const* name;
};

#if CELLERATOR_ENABLE_PROFILING_MARKERS
[[maybe_unused]] __device__ __constant__ static_marker relation_marker{
    1101, "ce_exop_relation_hybrid_mma"};
[[maybe_unused]] __device__ __constant__ static_marker segment_marker{
    1601, "ce_exop_segment_softmax_max"};
#endif
}

extern "C" __global__ void ce_exop_relation_hybrid_mma(float const* input, float* output)
{
    auto const index = blockIdx.x * blockDim.x + threadIdx.x;
    output[index] = input[index];
}

extern "C" __global__ void ce_exop_segment_softmax_max(float const* input, float* output)
{
    auto const index = blockIdx.x * blockDim.x + threadIdx.x;
    output[index] = input[index] > 0.0f ? input[index] : 0.0f;
}
