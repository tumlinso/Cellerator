#include <Cellerator/modeling/cuda/dispatch.hh>

#include <algorithm>
#include <climits>
#include <limits>

namespace cellerator::modeling::cuda {
namespace {

constexpr int kThreads = 256;
constexpr int kReductionThreads = 128;
constexpr std::int64_t kMaxGridBlocks = 65535;

bool product(std::int64_t rows, std::int64_t columns, std::size_t& out) noexcept {
    if (rows < 0 || columns < 0) return false;
    const auto r = static_cast<std::uint64_t>(rows);
    const auto c = static_cast<std::uint64_t>(columns);
    if (c != 0 && r > std::numeric_limits<std::size_t>::max() / c) return false;
    const auto elements = r * c;
    // Kernel row offsets are represented by signed 64-bit plan indices.
    if (elements > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) return false;
    out = static_cast<std::size_t>(elements);
    return true;
}

bool valid_plan(DispatchPlanView plan) noexcept {
    if (plan.selection_count < 0 || plan.origin_count < 0) return false;
    if (plan.selection_count > 0 && plan.origin_count == 0) return false;
    if (plan.origin_count > 0 && plan.origin_offsets == nullptr) return false;
    if (plan.selection_count > 0 &&
        (plan.selection_origin == nullptr || plan.origin_dispatch == nullptr)) return false;
    // origin_dispatch contains one dispatch ordinal per selected occurrence.
    return true;
}

bool valid_capacity(const void* pointer, std::size_t capacity,
                    std::size_t required) noexcept {
    return required == 0 || (pointer != nullptr && capacity >= required);
}

int blocks_for(std::size_t elements) noexcept {
    const auto blocks = elements / kThreads + (elements % kThreads != 0);
    return static_cast<int>(std::min<std::size_t>(
        std::max<std::size_t>(1, blocks), static_cast<std::size_t>(kMaxGridBlocks)));
}

template <class T> __device__ float read_float(T value);
template <> __device__ float read_float(float value) { return value; }
template <> __device__ float read_float(__half value) { return __half2float(value); }
template <class T> __device__ T write_float(float value);
template <> __device__ float write_float(float value) { return value; }
template <> __device__ __half write_float(float value) { return __float2half_rn(value); }

template <class T>
__global__ void gather_kernel(const T* input, T* output,
                              const std::int64_t* origin_for_dispatch,
                              std::int64_t selections, std::int64_t features,
                              std::size_t elements) {
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < elements;
         index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const auto dispatch = static_cast<std::int64_t>(index / static_cast<std::size_t>(features));
        const auto feature = static_cast<std::int64_t>(index % static_cast<std::size_t>(features));
        const auto origin = origin_for_dispatch[dispatch];
        // Index bounds are guaranteed by the owning immutable host plan.
        output[index] = input[static_cast<std::size_t>(origin) *
                                  static_cast<std::size_t>(features) +
                              static_cast<std::size_t>(feature)];
    }
}

template <class T>
__global__ void combine_kernel(const T* expert_output, T* origin_output,
                               const std::int64_t* origin_offsets,
                               const std::int64_t* origin_dispatch,
                               const float* weights,
                               std::int64_t origins, std::int64_t features,
                               std::size_t elements) {
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < elements;
         index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const auto origin = static_cast<std::int64_t>(index / static_cast<std::size_t>(features));
        const auto feature = static_cast<std::int64_t>(index % static_cast<std::size_t>(features));
        float sum = 0.0f;
        const auto begin = origin_offsets[origin];
        const auto end = origin_offsets[origin + 1];
        // The plan preserves logical selection order, so this reduction is
        // deterministic and does not contend on duplicate destination rows.
        for (auto position = begin; position < end; ++position) {
            const auto dispatch = origin_dispatch[position];
            const float weight = weights == nullptr ? 1.0f : weights[dispatch];
            sum += weight * read_float(expert_output[
                static_cast<std::size_t>(dispatch * features + feature)]);
        }
        origin_output[index] = write_float<T>(sum);
    }
}

template <class T>
__global__ void grad_expert_kernel(const T* grad_origin, T* grad_expert,
                                   const std::int64_t* origin_for_dispatch,
                                   const float* weights,
                                   std::int64_t selections, std::int64_t features,
                                   std::size_t elements) {
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < elements;
         index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const auto dispatch = static_cast<std::int64_t>(index / static_cast<std::size_t>(features));
        const auto feature = static_cast<std::int64_t>(index % static_cast<std::size_t>(features));
        const auto origin = origin_for_dispatch[dispatch];
        const float weight = weights == nullptr ? 1.0f : weights[dispatch];
        grad_expert[index] = write_float<T>(weight * read_float(grad_origin[
            static_cast<std::size_t>(origin) * static_cast<std::size_t>(features) +
            static_cast<std::size_t>(feature)]));
    }
}

template <class T>
__global__ void grad_weight_kernel(const T* grad_origin, const T* expert_output,
                                   float* grad_weight,
                                   const std::int64_t* origin_for_dispatch,
                                   std::int64_t selections,
                                   std::int64_t features) {
    __shared__ float partial[kReductionThreads];
    for (std::int64_t dispatch = blockIdx.x; dispatch < selections; dispatch += gridDim.x) {
        float sum = 0.0f;
        const auto origin = origin_for_dispatch[dispatch];
        for (std::int64_t feature = threadIdx.x; feature < features;
             feature += blockDim.x) {
            const auto index = static_cast<std::size_t>(dispatch) *
                                   static_cast<std::size_t>(features) +
                               static_cast<std::size_t>(feature);
            const auto origin_index = static_cast<std::size_t>(origin) *
                                          static_cast<std::size_t>(features) +
                                      static_cast<std::size_t>(feature);
            sum += read_float(grad_origin[origin_index]) * read_float(expert_output[index]);
        }
        partial[threadIdx.x] = sum;
        __syncthreads();
        for (int stride = kReductionThreads / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0) grad_weight[dispatch] = partial[0];
        __syncthreads();
    }
}

template <class T>
__global__ void grad_input_kernel(const T* grad_gathered, T* grad_input,
                                  const std::int64_t* origin_offsets,
                                  const std::int64_t* origin_dispatch,
                                  std::int64_t origins, std::int64_t features,
                                  std::size_t elements) {
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < elements;
         index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const auto origin = static_cast<std::int64_t>(index / static_cast<std::size_t>(features));
        const auto feature = static_cast<std::int64_t>(index % static_cast<std::size_t>(features));
        float sum = 0.0f;
        for (auto position = origin_offsets[origin]; position < origin_offsets[origin + 1]; ++position) {
            const auto dispatch = origin_dispatch[position];
            sum += read_float(grad_gathered[
                static_cast<std::size_t>(dispatch) * static_cast<std::size_t>(features) +
                static_cast<std::size_t>(feature)]);
        }
        grad_input[index] = write_float<T>(sum);
    }
}

template <class T>
cudaError_t gather(const T* input, std::size_t input_capacity,
                   T* output, std::size_t output_capacity,
                   DispatchPlanView plan, std::int64_t features,
                   cudaStream_t stream) noexcept {
    if (!valid_plan(plan) || features < 0) return cudaErrorInvalidValue;
    std::size_t input_required = 0, output_required = 0;
    if (!product(plan.origin_count, features, input_required) ||
        !product(plan.selection_count, features, output_required)) return cudaErrorInvalidValue;
    if (!valid_capacity(input, input_capacity, input_required) ||
        !valid_capacity(output, output_capacity, output_required)) return cudaErrorInvalidValue;
    if (output_required == 0) return cudaSuccess;
    if (features == 0) return cudaSuccess;
    gather_kernel<<<blocks_for(output_required), kThreads, 0, stream>>>(
        input, output, plan.selection_origin, plan.selection_count, features, output_required);
    return cudaGetLastError();
}

template <class T>
cudaError_t combine(const T* input, std::size_t input_capacity,
                    T* output, std::size_t output_capacity,
                    DispatchPlanView plan, std::int64_t features,
                    cudaStream_t stream) noexcept {
    if (!valid_plan(plan) || features < 0) return cudaErrorInvalidValue;
    std::size_t input_required = 0, output_required = 0;
    if (!product(plan.selection_count, features, input_required) ||
        !product(plan.origin_count, features, output_required)) return cudaErrorInvalidValue;
    if (!valid_capacity(input, input_capacity, input_required) ||
        !valid_capacity(output, output_capacity, output_required)) return cudaErrorInvalidValue;
    if (output_required == 0) return cudaSuccess;
    if (features == 0) return cudaSuccess;
    combine_kernel<<<blocks_for(output_required), kThreads, 0, stream>>>(
        input, output, plan.origin_offsets, plan.origin_dispatch, plan.dispatch_weight,
        plan.origin_count, features, output_required);
    return cudaGetLastError();
}

template <class T>
cudaError_t combine_backward(const T* grad_origin, std::size_t grad_origin_capacity,
                             const T* expert_output, std::size_t expert_capacity,
                             T* grad_expert, std::size_t grad_expert_capacity,
                             float* grad_weight, std::size_t grad_weight_capacity,
                             DispatchPlanView plan, std::int64_t features,
                             cudaStream_t stream) noexcept {
    if (!valid_plan(plan) || features < 0) return cudaErrorInvalidValue;
    std::size_t origin_required = 0, expert_required = 0;
    if (!product(plan.origin_count, features, origin_required) ||
        !product(plan.selection_count, features, expert_required)) return cudaErrorInvalidValue;
    if (!valid_capacity(grad_origin, grad_origin_capacity, origin_required) ||
        !valid_capacity(grad_expert, grad_expert_capacity, expert_required) ||
        (grad_weight != nullptr &&
         !valid_capacity(expert_output, expert_capacity, expert_required)) ||
        (grad_weight != nullptr && grad_weight_capacity < static_cast<std::size_t>(plan.selection_count)))
        return cudaErrorInvalidValue;
    if (expert_required == 0) {
        // A zero feature extent still has one weight gradient per selection.
        // Its reduction is the empty sum, so initialize every requested
        // derivative instead of returning with caller storage untouched.
        if (grad_weight != nullptr && plan.selection_count > 0) {
            grad_weight_kernel<<<blocks_for(static_cast<std::size_t>(plan.selection_count)),
                                 kReductionThreads, 0, stream>>>(
                grad_origin, expert_output, grad_weight, plan.selection_origin,
                plan.selection_count, features);
            return cudaGetLastError();
        }
        return cudaSuccess;
    }
    grad_expert_kernel<<<blocks_for(expert_required), kThreads, 0, stream>>>(
        grad_origin, grad_expert, plan.selection_origin, plan.dispatch_weight,
        plan.selection_count, features, expert_required);
    auto error = cudaGetLastError();
    if (error != cudaSuccess || grad_weight == nullptr) return error;
    // One 128-thread reduction CTA computes one routing-weight gradient.
    grad_weight_kernel<<<blocks_for(static_cast<std::size_t>(plan.selection_count)),
                         kReductionThreads, 0, stream>>>(
        grad_origin, expert_output, grad_weight, plan.selection_origin,
        plan.selection_count, features);
    return cudaGetLastError();
}

template <class T>
cudaError_t gather_backward(const T* grad_gathered, std::size_t gathered_capacity,
                            T* grad_input, std::size_t input_capacity,
                            DispatchPlanView plan, std::int64_t features,
                            cudaStream_t stream) noexcept {
    if (!valid_plan(plan) || features < 0) return cudaErrorInvalidValue;
    std::size_t gathered_required = 0, input_required = 0;
    if (!product(plan.selection_count, features, gathered_required) ||
        !product(plan.origin_count, features, input_required)) return cudaErrorInvalidValue;
    if (!valid_capacity(grad_gathered, gathered_capacity, gathered_required) ||
        !valid_capacity(grad_input, input_capacity, input_required)) return cudaErrorInvalidValue;
    if (input_required == 0) return cudaSuccess;
    if (features == 0) return cudaSuccess;
    grad_input_kernel<<<blocks_for(input_required), kThreads, 0, stream>>>(
        grad_gathered, grad_input, plan.origin_offsets, plan.origin_dispatch,
        plan.origin_count, features, input_required);
    return cudaGetLastError();
}

}  // namespace

cudaError_t gather_fp16(const __half* i, std::size_t ic, __half* o, std::size_t oc,
                        DispatchPlanView p, std::int64_t f, cudaStream_t s) noexcept {
    return gather(i, ic, o, oc, p, f, s);
}
cudaError_t gather_fp32(const float* i, std::size_t ic, float* o, std::size_t oc,
                        DispatchPlanView p, std::int64_t f, cudaStream_t s) noexcept {
    return gather(i, ic, o, oc, p, f, s);
}
cudaError_t combine_fp16(const __half* i, std::size_t ic, __half* o, std::size_t oc,
                         DispatchPlanView p, std::int64_t f, cudaStream_t s) noexcept {
    return combine(i, ic, o, oc, p, f, s);
}
cudaError_t combine_fp32(const float* i, std::size_t ic, float* o, std::size_t oc,
                         DispatchPlanView p, std::int64_t f, cudaStream_t s) noexcept {
    return combine(i, ic, o, oc, p, f, s);
}
cudaError_t combine_backward_fp16(const __half* go, std::size_t goc,
                                  const __half* eo, std::size_t eoc,
                                  __half* ge, std::size_t gec,
                                  float* gw, std::size_t gwc,
                                  DispatchPlanView p, std::int64_t f,
                                  cudaStream_t s) noexcept {
    return combine_backward(go, goc, eo, eoc, ge, gec, gw, gwc, p, f, s);
}
cudaError_t combine_backward_fp32(const float* go, std::size_t goc,
                                  const float* eo, std::size_t eoc,
                                  float* ge, std::size_t gec,
                                  float* gw, std::size_t gwc,
                                  DispatchPlanView p, std::int64_t f,
                                  cudaStream_t s) noexcept {
    return combine_backward(go, goc, eo, eoc, ge, gec, gw, gwc, p, f, s);
}
cudaError_t gather_backward_fp16(const __half* gg, std::size_t ggc,
                                 __half* gi, std::size_t gic,
                                 DispatchPlanView p, std::int64_t f,
                                 cudaStream_t s) noexcept {
    return gather_backward(gg, ggc, gi, gic, p, f, s);
}
cudaError_t gather_backward_fp32(const float* gg, std::size_t ggc,
                                 float* gi, std::size_t gic,
                                 DispatchPlanView p, std::int64_t f,
                                 cudaStream_t s) noexcept {
    return gather_backward(gg, ggc, gi, gic, p, f, s);
}

}  // namespace cellerator::modeling::cuda
