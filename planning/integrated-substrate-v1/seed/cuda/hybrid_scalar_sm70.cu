// N02 seed: one warp per pair of 32-state scalar-affine effects.
// Authored, not compiled or run in the bootstrap authoring environment.
// Caller/prepare contract: grid and block are one-dimensional (y=z=1);
// blockDim.x is a positive multiple of 32; all maps are
// validated 0..31; buffers contain pair_count*32 entries and do not alias outputs.
// Arrays are pair-major. Visit left, then right. No independently executing
// subwarp, implicit warp synchrony, allocation or stream selection is used here.
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
#error "This seed is intended for SM70 or newer"
#endif
extern "C" __global__ void is1_compose_hybrid_scalar32(
    const std::uint32_t* __restrict__ left_t,
    const float* __restrict__ left_a, const float* __restrict__ left_b,
    const std::uint32_t* __restrict__ right_t,
    const float* __restrict__ right_a, const float* __restrict__ right_b,
    std::uint32_t* __restrict__ out_t,
    float* __restrict__ out_a, float* __restrict__ out_b,
    std::size_t pair_count) {
    const unsigned lane=threadIdx.x & 31u;
    const std::size_t pair=static_cast<std::size_t>(blockIdx.x)*(blockDim.x/32u)+(threadIdx.x/32u);
    if(pair>=pair_count)return; // uniform across this entire warp
    const std::size_t i=pair*32u+lane;
    const unsigned entry=left_t[i];
    const unsigned rt=right_t[i];
    const float ra=right_a[i], rb=right_b[i];
    constexpr unsigned full=0xffffffffu;
    const unsigned selected_t=__shfl_sync(full,rt,entry);
    const float selected_a=__shfl_sync(full,ra,entry);
    const float selected_b=__shfl_sync(full,rb,entry);
    out_t[i]=selected_t;
    out_a[i]=selected_a*left_a[i];
    out_b[i]=fmaf(selected_a,left_b[i],selected_b);
}
