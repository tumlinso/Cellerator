#include <Cellerator/compiler/substrate/guarded_scaled_tanh.hh>
#include <cuda_runtime.h>
#include <limits>

namespace cellerator::compiler::substrate {
namespace {
__global__ void multiply(device_scaled_tanh_binding b) {
    auto i = std::size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i < b.extent) b.intermediate[i] = __fmul_rn(b.state[i], b.parameters[i]);
}
__global__ void activate(device_scaled_tanh_binding b) {
    auto i = std::size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i < b.extent) b.output[i] = tanhf(b.intermediate[i]);
}
__global__ void fused(device_scaled_tanh_binding b) {
    auto i = std::size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i < b.extent) {
        float z = __fmul_rn(b.state[i], b.parameters[i]);
        b.intermediate[i] = z;
        b.output[i] = tanhf(z);
    }
}
bool overlap(const void* a, const void* b, std::size_t bytes) {
    auto x = reinterpret_cast<std::uintptr_t>(a), y = reinterpret_cast<std::uintptr_t>(b);
    const auto max = std::numeric_limits<std::uintptr_t>::max();
    return bytes > max-x || bytes > max-y || (x < y+bytes && y < x+bytes);
}
}
cudaError_t launch_scaled_tanh(realization r, device_scaled_tanh_binding b, cudaStream_t stream) noexcept {
    if (r != realization::direct && r != realization::fused_scaled_tanh) return cudaErrorInvalidValue;
    if (!b.extent) return cudaSuccess;
    if (!b.state || !b.parameters || !b.intermediate || !b.output
        || b.extent > std::numeric_limits<std::size_t>::max()/sizeof(float)
        || (b.extent-1)/256 >= 2147483647u) return cudaErrorInvalidValue;
    auto bytes = b.extent*sizeof(float);
    if (overlap(b.state,b.intermediate,bytes) || overlap(b.parameters,b.intermediate,bytes)
        || overlap(b.state,b.output,bytes) || overlap(b.parameters,b.output,bytes)
        || overlap(b.intermediate,b.output,bytes)) return cudaErrorInvalidValue;
    auto blocks = static_cast<unsigned>((b.extent-1)/256+1);
    if (r == realization::fused_scaled_tanh) {
        fused<<<blocks,256,0,stream>>>(b);
        return cudaGetLastError();
    }
    multiply<<<blocks,256,0,stream>>>(b);
    auto status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    activate<<<blocks,256,0,stream>>>(b);
    return cudaGetLastError();
}
cudaError_t launch_guarded_scaled_tanh(const scaled_tanh_plan& plan, specialization_guard live,
        device_scaled_tanh_binding b, cudaStream_t stream, realization* selected) noexcept {
    // Different arithmetic policy requires preparation, not silent fallback.
    if (live.policy != numerical_policy::nearest_ieee_f32 || live.extent != b.extent)
        return cudaErrorInvalidValue;
    auto route = plan.choose(live);
    if (selected) *selected = route;
    return launch_scaled_tanh(route,b,stream);
}
} // namespace cellerator::compiler::substrate
