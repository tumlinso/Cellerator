#include <Cellerator/compute/architecture/providers/nvidia/sm70/edge_value_gradient/relation_gradient.cuh>
#include <Cellerator/compute/architecture/providers/nvidia/sm70/edge_value_gradient/gradient_pack.cuh>
namespace cellerator::compute::architecture::providers::nvidia::sm70::edge_value_gradient {
namespace {
// Dominant expected cost is scattered dense operand traffic; each thread reuses
// its compact edge reference across 16 f32 FMAs without intermediate conversion.
__global__ void full_f32_kernel(contract::support_view_v1 support,
    const float* source, const float* cotangent, float* output) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= support.local_edge_count) return;
    auto edge = support.edges[index];
    float sum = 0;
    #pragma unroll
    for (unsigned k = 0; k < 16; ++k)
        sum = fmaf(source[std::size_t(edge.source_local)*16+k],
            cotangent[std::size_t(edge.destination_local)*16+k],sum);
    output[index] = sum;
}
}
contract::status_v1 enqueue_relation_gradient(const relation_gradient_request& r) noexcept {
    using contract::status_v1;
    const auto nx = std::uint64_t(r.support.source_count)*16;
    const auto ny = std::uint64_t(r.support.destination_count)*16;
    if (r.source_capacity<nx || r.cotangent_capacity<ny ||
        r.output_capacity<r.support.local_edge_count ||
        (nx && !r.source) || (ny && !r.cotangent)) return status_v1::invalid_argument;
    if (!r.support.local_edge_count) return status_v1::success;
    if (!r.support.edges || !r.output || !nx || !ny) return status_v1::invalid_argument;
    if (r.half_rounded) {
        if (!r.source_scratch || !r.cotangent_scratch ||
            r.source_scratch_capacity<nx || r.cotangent_scratch_capacity<ny)
            return status_v1::invalid_argument;
        auto a=enqueue_gradient_pack({r.source,r.support.source_count,16,r.source_capacity,
            r.source_scratch,r.support.source_count,16,r.source_scratch_capacity,
            nullptr,0,true,r.stream});
        if(a!=status_v1::success) return a;
        auto b=enqueue_gradient_pack({r.cotangent,r.support.destination_count,16,r.cotangent_capacity,
            r.cotangent_scratch,r.support.destination_count,16,r.cotangent_scratch_capacity,
            nullptr,0,true,r.stream});
        if(b!=status_v1::success) return b;
        contract::launch_request_v1 launch{};
        launch.support=r.support;
        launch.dense={r.source_scratch,r.cotangent_scratch,16};
        launch.output_order=contract::output_order_v1::projection_native;
        launch.output=r.output;launch.stream=r.stream;
        return contract::enqueue_sparse_v1(launch);
    }
    full_f32_kernel<<<(r.support.local_edge_count+127)/128,128,0,r.stream>>>(
        r.support,r.source,r.cotangent,r.output);
    return cudaPeekAtLastError()==cudaSuccess?status_v1::success:status_v1::cuda_failure;
}
}
