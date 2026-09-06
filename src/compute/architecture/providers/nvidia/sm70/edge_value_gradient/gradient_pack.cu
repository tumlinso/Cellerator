#include <Cellerator/compute/architecture/providers/nvidia/sm70/edge_value_gradient/gradient_pack.cuh>
#include <algorithm>
#include <limits>
namespace cellerator::compute::architecture::providers::nvidia::sm70::edge_value_gradient {
namespace {
__global__ void pack_kernel(pack_request r) {
    const std::uint64_t count=std::uint64_t(r.output_rows)*r.output_stride;
    for(std::uint64_t i=std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
        i<count;i+=std::uint64_t(gridDim.x)*blockDim.x) {
        const auto row=std::uint32_t(i/r.output_stride), k=std::uint32_t(i%r.output_stride);
        const auto source=r.gather_ids?r.gather_ids[row]:row;
        r.output[i]=(k<16u&&source<r.input_rows)
            ?__float2half_rn(r.input[std::uint64_t(source)*r.input_stride+k])
            :__float2half_rn(0.0f);
    }
}
}
contract::status_v1 enqueue_gradient_pack(const pack_request &r) noexcept {
    const auto input_elements=std::uint64_t(r.input_rows)*r.input_stride;
    const auto output_elements=std::uint64_t(r.output_rows)*r.output_stride;
    if(!r.half_rounded || r.input_stride<16u || r.output_stride<16u
        || r.output_stride%16u || r.input_capacity<input_elements
        || r.output_capacity<output_elements
        || (!r.gather_ids && r.output_rows>r.input_rows)
        || (r.gather_ids && r.gather_capacity<r.output_rows))
        return contract::status_v1::invalid_argument;
    if(!r.output_rows)return contract::status_v1::success;
    const auto a=reinterpret_cast<std::uintptr_t>(r.input),b=reinterpret_cast<std::uintptr_t>(r.output);
    const auto maximum=std::numeric_limits<std::uintptr_t>::max();
    if(!r.input||!r.output||a%alignof(float)||b%32u
        || input_elements>maximum/sizeof(float)||output_elements>maximum/sizeof(__half)
        || a>maximum-input_elements*sizeof(float)||b>maximum-output_elements*sizeof(__half))
        return contract::status_v1::invalid_argument;
    if(a<b+output_elements*sizeof(__half)&&b<a+input_elements*sizeof(float))
        return contract::status_v1::invalid_argument;
    if (r.gather_ids) {
        const auto ids=reinterpret_cast<std::uintptr_t>(r.gather_ids);
        const auto bytes=std::uint64_t(r.output_rows)*sizeof(std::uint32_t);
        if (ids%alignof(std::uint32_t) || ids>maximum-bytes
            || (ids<b+output_elements*sizeof(__half) && b<ids+bytes))
            return contract::status_v1::invalid_argument;
    }
    const auto blocks=unsigned(std::min<std::uint64_t>((output_elements+255)/256,65535));
    pack_kernel<<<blocks,256,0,r.stream>>>(r);
    return cudaGetLastError()==cudaSuccess?contract::status_v1::success:contract::status_v1::cuda_failure;
}
}
