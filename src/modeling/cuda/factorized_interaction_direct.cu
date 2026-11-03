#include <Cellerator/modeling/cuda/factorized_interaction.hh>
#include "factorized_interaction_numerics.cuh"
#include "factorized_interaction_stable.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace cellerator::modeling::cuda {
namespace {

using layout::FactorizedInteractionRecord;
using layout::FactorizedInteractionTermRecord;
using layout::FactorizedInteractionView;

constexpr std::uint64_t kCellLanes = 32;
constexpr std::uint32_t kMaxWarps = 4;
constexpr unsigned kFullWarp = 0xffffffffu;
constexpr unsigned kMaxGridX = 65535;

bool checked_bytes(std::uint64_t count, std::size_t item_size,
                   std::size_t& out) noexcept {
    if (item_size && count > std::numeric_limits<std::size_t>::max() / item_size)
        return false;
    out = static_cast<std::size_t>(count) * item_size;
    return true;
}

bool checked_product(std::uint64_t a, std::uint64_t b,
                     std::uint64_t& out) noexcept {
    if (b && a > std::numeric_limits<std::uint64_t>::max() / b) return false;
    out = a * b;
    return true;
}

std::uint64_t ceil_div(std::uint64_t value, std::uint64_t divisor) noexcept {
    return value / divisor + (value % divisor != 0);
}

bool valid_dtype(memory::cuda::DType type) noexcept {
    return type == memory::cuda::DType::float16 ||
           type == memory::cuda::DType::float32;
}

std::size_t dtype_bytes(memory::cuda::DType type) noexcept {
    return type == memory::cuda::DType::float16 ? sizeof(__half) : sizeof(float);
}

bool valid_plan_header(const FactorizedInteractionView& plan) noexcept {
    if (plan.schema != layout::factorized_interaction_schema) return false;
    if (plan.mechanism_count > std::numeric_limits<std::size_t>::max() /
                                   sizeof(FactorizedInteractionRecord) ||
        plan.term_count > std::numeric_limits<std::size_t>::max() /
                              sizeof(FactorizedInteractionTermRecord)) return false;
    return true;
}

struct Counts {
    std::uint64_t input_values = 0;
    std::uint64_t context_values = 0;
    std::uint64_t alpha_values = 0;
    std::uint64_t output_values = 0;
    std::uint64_t gate_values = 0;
    std::uint64_t tape_values = 0;
    std::uint64_t forward_tasks = 0;
    std::uint64_t backward_tasks = 0;
    std::uint64_t forward_scratch_values = 0;
};

bool counts_for(const FactorizedInteractionView& plan,
                FactorizedInteractionProblem problem, Counts& out) noexcept {
    if (!valid_plan_header(plan)) return false;
    const auto cells = problem.cells;
    const auto channels = problem.channels;
    if (!checked_product(plan.extents.inputs, channels, out.input_values) ||
        !checked_product(out.input_values, cells, out.input_values) ||
        !checked_product(plan.extents.contexts, cells, out.context_values) ||
        !checked_product(plan.extents.alpha_rows, channels, out.alpha_values) ||
        !checked_product(plan.extents.outputs, channels, out.output_values) ||
        !checked_product(out.output_values, cells, out.output_values) ||
        !checked_product(plan.extents.outputs, channels, out.forward_scratch_values) ||
        !checked_product(out.forward_scratch_values, cells, out.forward_scratch_values))
        return false;

    // A tape entry is only produced for a numerically active invocation.
    if (plan.mechanism_count && cells && channels &&
        !checked_product(plan.mechanism_count, cells, out.gate_values)) return false;
    if (!checked_product(out.gate_values, 2, out.tape_values)) return false;

    if (plan.mechanism_count && cells && channels) {
        const auto cell_tiles = ceil_div(cells, kCellLanes);
        const auto warps = std::min<std::uint64_t>(channels, kMaxWarps);
        const auto channel_tiles = ceil_div(channels, warps);
        std::uint64_t tasks = 0;
        if (!checked_product(plan.mechanism_count, cell_tiles, tasks) ||
            !checked_product(tasks, channel_tiles, out.forward_tasks) ||
            !checked_product(plan.mechanism_count, cell_tiles, out.backward_tasks))
            return false;
    }

    std::size_t ignored = 0;
    return checked_bytes(out.input_values, sizeof(float), ignored) &&
           checked_bytes(out.context_values, sizeof(float), ignored) &&
           checked_bytes(out.alpha_values, sizeof(float), ignored) &&
           checked_bytes(plan.extents.biases, sizeof(float), ignored) &&
           checked_bytes(plan.extents.gammas, sizeof(float), ignored) &&
           checked_bytes(out.output_values, sizeof(float), ignored) &&
           checked_bytes(out.tape_values, sizeof(float), ignored) &&
           checked_bytes(out.gate_values, sizeof(float), ignored) &&
           checked_bytes(out.forward_scratch_values, sizeof(float), ignored) &&
           out.forward_tasks <= static_cast<std::uint64_t>(
               std::numeric_limits<std::int64_t>::max()) &&
           out.backward_tasks <= static_cast<std::uint64_t>(
               std::numeric_limits<std::int64_t>::max());
}

bool view_has(memory::cuda::ConstDeviceView view, std::size_t bytes) noexcept {
    return bytes == 0 || (view.data && view.bytes >= bytes);
}
bool view_has(memory::cuda::DeviceView view, std::size_t bytes) noexcept {
    return bytes == 0 || (view.data && view.bytes >= bytes);
}
bool aligned_for(const void* pointer, std::size_t alignment) noexcept {
    return !pointer || reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0;
}
bool aligned_view(memory::cuda::ConstDeviceView view,
                  std::size_t alignment) noexcept {
    return view.bytes == 0 || aligned_for(view.data, alignment);
}
bool aligned_view(memory::cuda::DeviceView view,
                  std::size_t alignment) noexcept {
    return view.bytes == 0 || aligned_for(view.data, alignment);
}

bool bind_device(int& device, memory::cuda::ConstDeviceView view,
                 std::size_t required_bytes) noexcept {
    if (!view_has(view, required_bytes)) return false;
    if (!view.bytes) return required_bytes == 0;
    if (!view.data) return false;
    if (view.device < 0) return false;
    if (device < 0) device = view.device;
    return device == view.device;
}
bool bind_device(int& device, memory::cuda::DeviceView view,
                 std::size_t required_bytes) noexcept {
    if (!view_has(view, required_bytes)) return false;
    if (!view.bytes) return required_bytes == 0;
    if (!view.data) return false;
    if (view.device < 0) return false;
    if (device < 0) device = view.device;
    return device == view.device;
}

cudaError_t current_device_matches(int device) noexcept {
    if (device < 0) return cudaSuccess;
    int current = -1;
    const auto status = cudaGetDevice(&current);
    if (status != cudaSuccess) return status;
    return current == device ? cudaSuccess : cudaErrorInvalidDevice;
}

struct Region {
    std::uintptr_t begin = 0;
    std::uintptr_t end = 0;
    bool present = false;
};

bool make_region(const void* pointer, std::size_t bytes, Region& region) noexcept {
    if (!bytes) return true;
    if (!pointer) return false;
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin) return false;
    region = {begin, begin + bytes, true};
    return true;
}

bool regions_disjoint(const Region* writable, std::size_t writable_count,
                      const Region* readonly, std::size_t readonly_count) noexcept {
    for (std::size_t i = 0; i < writable_count; ++i) {
        if (!writable[i].present) continue;
        for (std::size_t j = i + 1; j < writable_count; ++j) {
            if (writable[j].present && writable[i].begin < writable[j].end &&
                writable[j].begin < writable[i].end) return false;
        }
        for (std::size_t j = 0; j < readonly_count; ++j) {
            if (readonly[j].present && writable[i].begin < readonly[j].end &&
                readonly[j].begin < writable[i].end) return false;
        }
    }
    return true;
}

bool stable_plan_pointers(const FactorizedInteractionView& plan) noexcept {
    return plan.mechanisms && (!plan.term_count || plan.terms) &&
        plan.output_packets.offsets && plan.output_packets.entries &&
        plan.local_output_mechanisms.offsets && plan.local_output_mechanisms.entries &&
        plan.packet_mechanisms && plan.input_packets.offsets && plan.input_packets.entries &&
        plan.local_input_roles.offsets && plan.local_input_roles.entries &&
        plan.alpha_mechanisms.offsets && plan.alpha_mechanisms.entries &&
        plan.bias_mechanisms.offsets && plan.bias_mechanisms.entries &&
        (!plan.extents.contexts || plan.context_terms.offsets) &&
        (!plan.extents.gammas || plan.gamma_terms.offsets) &&
        (!plan.term_count || (plan.context_terms.entries && plan.gamma_terms.entries));
}

template<class T>
__device__ __forceinline__ float load_value(const T* values,
                                            std::uint64_t index) {
    return static_cast<float>(values[index]);
}
template<>
__device__ __forceinline__ float load_value<__half>(const __half* values,
                                                    std::uint64_t index) {
    return __half2float(values[index]);
}

template<class T>
__device__ __forceinline__ void store_value(T* values, std::uint64_t index,
                                            float value) {
    values[index] = static_cast<T>(value);
}
template<>
__device__ __forceinline__ void store_value<__half>(__half* values,
                                                    std::uint64_t index,
                                                    float value) {
    values[index] = __float2half_rn(value);
}

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(kFullWarp, value, offset);
    return value;
}

template<class Activation, class Parameter>
__device__ __forceinline__ stable_detail::SigmoidAndSlope gate_for(
    const FactorizedInteractionRecord& mechanism,
    const FactorizedInteractionTermRecord* terms,
    const Activation* z, const Parameter* bias, const Parameter* gamma,
    std::uint64_t cells, std::uint64_t cell) {
    stable_detail::Accurate value(load_value(bias, mechanism.bias));
    for (std::uint32_t i = 0; i < mechanism.term_count; ++i) {
        const auto term = terms[mechanism.term_begin + i];
        const auto z_value = load_value(z, term.context * cells + cell);
        value.add_product(load_value(gamma, term.gamma), z_value);
    }
    return stable_detail::sigmoid_and_slope(value.value());
}

template<class Activation, class Parameter>
__global__ void direct_forward_contributions(
    const FactorizedInteractionRecord* mechanisms,
    const FactorizedInteractionTermRecord* terms,
    std::uint64_t mechanism_count, std::uint64_t cells,
    std::uint64_t channels, std::uint64_t cell_tiles,
    std::uint64_t channel_tiles, std::uint32_t warps,
    const Activation* x, const Activation* z,
    const Parameter* alpha, const Parameter* bias, const Parameter* gamma,
    float* output_scratch, float* saved_gates, float* saved_slopes,
    std::uint64_t task_count) {
    __shared__ float gate_tile[kCellLanes];

    for (std::uint64_t task = blockIdx.x; task < task_count;
         task += gridDim.x) {
        const auto cell_tile = task % cell_tiles;
        const auto group = task / cell_tiles;
        const auto channel_tile = group % channel_tiles;
        const auto mechanism_id = group / channel_tiles;
        if (mechanism_id >= mechanism_count) continue;

        const auto mechanism = mechanisms[mechanism_id];
        const auto warp = static_cast<std::uint32_t>(threadIdx.x >> 5);
        const auto lane = static_cast<std::uint32_t>(threadIdx.x & 31);
        const auto cell = cell_tile * kCellLanes + lane;
        const auto channel = channel_tile * warps + warp;

        if (warp == 0) {
            const auto gate_result = cell < cells
                ? gate_for(mechanism, terms, z, bias, gamma, cells, cell)
                : stable_detail::SigmoidAndSlope{0.0f, 0.0f};
            gate_tile[lane] = gate_result.gate;
            if (channel_tile == 0 && cell < cells) {
                const auto gate_index = mechanism_id * cells + cell;
                saved_gates[gate_index] = gate_result.gate;
                saved_slopes[gate_index] = gate_result.slope;
            }
        }
        __syncthreads();

        if (cell < cells && channel < channels) {
            const auto x_base = channel * cells + cell;
            const auto x0 = load_value(x, mechanism.inputs[0] * channels * cells + x_base);
            const auto x1 = load_value(x, mechanism.inputs[1] * channels * cells + x_base);
            float product = x0 * x1;
            if (mechanism.arity >= 3)
                product *= load_value(x, mechanism.inputs[2] * channels * cells + x_base);
            if (mechanism.arity >= 4)
                product *= load_value(x, mechanism.inputs[3] * channels * cells + x_base);
            const auto a = load_value(alpha, mechanism.alpha * channels + channel);
            const auto contribution = gate_tile[lane] * a * product;
            const auto output_index = mechanism.target * channels * cells + x_base;
            atomicAdd(output_scratch + output_index, contribution);
        }
        __syncthreads();
    }
}

template<class Output>
__global__ void direct_output_assign(const float* scratch, Output* output,
                                     std::uint64_t value_count) {
    for (std::uint64_t i = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < value_count; i += std::uint64_t(gridDim.x) * blockDim.x)
        store_value(output, i, scratch[i]);
}

template<class Activation, class Parameter>
__global__ void direct_backward_contributions(
    const FactorizedInteractionRecord* mechanisms,
    const FactorizedInteractionTermRecord* terms,
    std::uint64_t mechanism_count, std::uint64_t cells,
    std::uint64_t channels, std::uint64_t cell_tiles,
    std::uint32_t warps, const Activation* x, const Activation* z,
    const Parameter* alpha, const Parameter* gamma,
    const float* output_gradient, const float* saved_gates,
    const float* gate_adjoint,
    float* grad_x, float* grad_z, float* grad_alpha,
    float* grad_bias, float* grad_gamma, std::uint64_t task_count) {
    for (std::uint64_t task = blockIdx.x; task < task_count;
         task += gridDim.x) {
        const auto cell_tile = task % cell_tiles;
        const auto mechanism_id = task / cell_tiles;
        if (mechanism_id >= mechanism_count) continue;

        const auto mechanism = mechanisms[mechanism_id];
        const auto warp = static_cast<std::uint32_t>(threadIdx.x >> 5);
        const auto lane = static_cast<std::uint32_t>(threadIdx.x & 31);
        const auto cell = cell_tile * kCellLanes + lane;
        for (std::uint64_t channel = warp; channel < channels; channel += warps) {
            float product = 0.0f;
            float x0 = 0.0f, x1 = 0.0f, x2 = 0.0f, x3 = 0.0f;
            float upstream = 0.0f;
            float a = 0.0f;
            float gate = 0.0f;
            if (cell < cells) {
                const auto x_base = channel * cells + cell;
                const auto target_index = mechanism.target * channels * cells + x_base;
                upstream = output_gradient[target_index];
                gate = saved_gates[mechanism_id * cells + cell];
                a = load_value(alpha, mechanism.alpha * channels + channel);
                const auto input_base = channels * cells;
                x0 = load_value(x, mechanism.inputs[0] * input_base + x_base);
                x1 = load_value(x, mechanism.inputs[1] * input_base + x_base);
                product = x0 * x1;
                if (mechanism.arity >= 3) {
                    x2 = load_value(x, mechanism.inputs[2] * input_base + x_base);
                    product *= x2;
                }
                if (mechanism.arity >= 4) {
                    x3 = load_value(x, mechanism.inputs[3] * input_base + x_base);
                    product *= x3;
                }

                const auto participant_factor = upstream * gate * a;
                const auto dx_base = participant_factor;
                atomicAdd(grad_x + mechanism.inputs[0] * input_base + x_base,
                         dx_base * x1 * (mechanism.arity >= 3 ? x2 : 1.0f) *
                         (mechanism.arity >= 4 ? x3 : 1.0f));
                atomicAdd(grad_x + mechanism.inputs[1] * input_base + x_base,
                         dx_base * x0 * (mechanism.arity >= 3 ? x2 : 1.0f) *
                         (mechanism.arity >= 4 ? x3 : 1.0f));
                if (mechanism.arity >= 3) {
                    atomicAdd(grad_x + mechanism.inputs[2] * input_base + x_base,
                             dx_base * x0 * x1 *
                             (mechanism.arity >= 4 ? x3 : 1.0f));
                }
                if (mechanism.arity >= 4) {
                    atomicAdd(grad_x + mechanism.inputs[3] * input_base + x_base,
                             dx_base * x0 * x1 * x2);
                }
            }

            // Alpha is shared across cells. Reduce a channel's contribution
            // within this warp before issuing one canonical-parameter atomic.
            float alpha_partial = cell < cells ? upstream * gate * product : 0.0f;
            alpha_partial = warp_sum(alpha_partial);
            if (lane == 0)
                atomicAdd(grad_alpha + mechanism.alpha * channels + channel,
                         alpha_partial);
        }

        if (warp == 0) {
            const auto ds = cell < cells
                ? gate_adjoint[mechanism_id * cells + cell]
                : 0.0f;
            for (std::uint32_t i = 0; i < mechanism.term_count; ++i) {
                const auto term = terms[mechanism.term_begin + i];
                float context_value = 0.0f;
                if (cell < cells) {
                    context_value = load_value(z, term.context * cells + cell);
                    const auto gamma_value = load_value(gamma, term.gamma);
                    atomicAdd(grad_z + term.context * cells + cell,
                             ds * gamma_value);
                }
                const auto gamma_partial = warp_sum(ds * context_value);
                if (lane == 0) atomicAdd(grad_gamma + term.gamma, gamma_partial);
            }
            const auto bias_partial = warp_sum(ds);
            if (lane == 0) atomicAdd(grad_bias + mechanism.bias, bias_partial);
        }
        __syncthreads();
    }
}

template<class Activation, class Parameter, class Output>
cudaError_t launch_forward(const FactorizedInteractionView& plan,
                           FactorizedInteractionProblem problem,
                           const FactorizedInteractionValues& values,
                           memory::cuda::DeviceView output,
                           memory::cuda::DeviceView tape,
                           memory::cuda::DeviceView scratch,
                           std::size_t scratch_bytes,
                           const Counts& counts, cudaStream_t stream) noexcept {
    const auto output_scratch = static_cast<float*>(scratch.data);
    if (scratch_bytes) {
        auto status = cudaMemsetAsync(output_scratch, 0, scratch_bytes, stream);
        if (status != cudaSuccess) return status;
    }

    if (counts.forward_tasks) {
        const auto warps = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(problem.channels, kMaxWarps));
        const auto cell_tiles = ceil_div(problem.cells, kCellLanes);
        const auto channel_tiles = ceil_div(problem.channels, warps);
        const auto blocks = static_cast<unsigned>(
            std::min<std::uint64_t>(counts.forward_tasks, kMaxGridX));
        direct_forward_contributions<<<blocks, warps * 32, 0, stream>>>(
            plan.mechanisms, plan.terms, plan.mechanism_count,
            problem.cells, problem.channels, cell_tiles, channel_tiles, warps,
            static_cast<const Activation*>(values.x.data),
            static_cast<const Activation*>(values.z.data),
            static_cast<const Parameter*>(values.alpha.data),
            static_cast<const Parameter*>(values.bias.data),
            static_cast<const Parameter*>(values.gamma.data),
            output_scratch, static_cast<float*>(tape.data),
            static_cast<float*>(tape.data) + counts.gate_values,
            counts.forward_tasks);
        auto status = cudaGetLastError();
        if (status != cudaSuccess) return status;
    }

    if (counts.output_values) {
        constexpr int threads = 256;
        const auto needed = counts.output_values / threads +
                            (counts.output_values % threads != 0);
        const auto blocks = static_cast<unsigned>(std::min<std::uint64_t>(needed, kMaxGridX));
        direct_output_assign<<<blocks, threads, 0, stream>>>(
            output_scratch, static_cast<Output*>(output.data), counts.output_values);
        return cudaGetLastError();
    }
    return cudaSuccess;
}

template<class Activation, class Parameter>
cudaError_t launch_backward(const FactorizedInteractionView& plan,
                            FactorizedInteractionProblem problem,
                            FactorizedInteractionPrecision precision,
                            const FactorizedInteractionValues& values,
                            memory::cuda::ConstDeviceView output_gradient,
                            memory::cuda::ConstDeviceView tape,
                            const FactorizedInteractionGradients& gradients,
                            memory::cuda::DeviceView scratch,
                            FactorizedInteractionGradientMode mode,
                            const Counts& counts, cudaStream_t stream) noexcept {
    const std::size_t gx_bytes = static_cast<std::size_t>(counts.input_values) * sizeof(float);
    const std::size_t gz_bytes = static_cast<std::size_t>(counts.context_values) * sizeof(float);
    const std::size_t ga_bytes = static_cast<std::size_t>(counts.alpha_values) * sizeof(float);
    const std::size_t gb_bytes = static_cast<std::size_t>(plan.extents.biases) * sizeof(float);
    std::size_t gg_bytes = 0;
    if (!checked_bytes(plan.extents.gammas, sizeof(float), gg_bytes))
        return cudaErrorInvalidValue;

    if (mode == FactorizedInteractionGradientMode::overwrite) {
        const auto zero = [stream](memory::cuda::DeviceView v, std::size_t bytes) {
            return bytes ? cudaMemsetAsync(v.data, 0, bytes, stream) : cudaSuccess;
        };
        auto status = zero(gradients.x, gx_bytes); if (status != cudaSuccess) return status;
        status = zero(gradients.z, gz_bytes); if (status != cudaSuccess) return status;
        status = zero(gradients.alpha, ga_bytes); if (status != cudaSuccess) return status;
        status = zero(gradients.bias, gb_bytes); if (status != cudaSuccess) return status;
        status = zero(gradients.gamma, gg_bytes); if (status != cudaSuccess) return status;
    }

    if (!counts.backward_tasks) return cudaSuccess;

    const auto* gates = static_cast<const float*>(tape.data);
    const auto* slopes = gates + counts.gate_values;
    auto* gate_adjoint = static_cast<float*>(scratch.data);
    auto status = stable_detail::launch_gate_adjoint(
        plan, problem, precision, values,
        static_cast<const float*>(output_gradient.data), gates, slopes,
        gate_adjoint, stream);
    if (status != cudaSuccess) return status;

    const auto warps = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(problem.channels, kMaxWarps));
    const auto cell_tiles = ceil_div(problem.cells, kCellLanes);
    const auto blocks = static_cast<unsigned>(
        std::min<std::uint64_t>(counts.backward_tasks, kMaxGridX));
    direct_backward_contributions<<<blocks, warps * 32, 0, stream>>>(
        plan.mechanisms, plan.terms, plan.mechanism_count,
        problem.cells, problem.channels, cell_tiles, warps,
        static_cast<const Activation*>(values.x.data),
        static_cast<const Activation*>(values.z.data),
        static_cast<const Parameter*>(values.alpha.data),
        static_cast<const Parameter*>(values.gamma.data),
        static_cast<const float*>(output_gradient.data),
        gates, gate_adjoint,
        static_cast<float*>(gradients.x.data), static_cast<float*>(gradients.z.data),
        static_cast<float*>(gradients.alpha.data), static_cast<float*>(gradients.bias.data),
        static_cast<float*>(gradients.gamma.data), counts.backward_tasks);
    return cudaGetLastError();
}

template<class Activation, class Parameter>
cudaError_t dispatch_forward_output(
    const FactorizedInteractionView& plan, FactorizedInteractionProblem problem,
    const FactorizedInteractionValues& values, FactorizedInteractionPrecision precision,
    memory::cuda::DeviceView output, memory::cuda::DeviceView tape,
    memory::cuda::DeviceView scratch, std::size_t scratch_bytes,
    const Counts& counts, cudaStream_t stream) noexcept {
    if (precision.output == memory::cuda::DType::float16)
        return launch_forward<Activation, Parameter, __half>(plan, problem, values,
            output, tape, scratch, scratch_bytes, counts, stream);
    return launch_forward<Activation, Parameter, float>(plan, problem, values,
        output, tape, scratch, scratch_bytes, counts, stream);
}

bool forward_views_valid(const FactorizedInteractionView& plan,
                         FactorizedInteractionProblem problem,
                         FactorizedInteractionPrecision precision,
                         const FactorizedInteractionValues& values,
                         memory::cuda::DeviceView output,
                         memory::cuda::DeviceView tape,
                         memory::cuda::DeviceView scratch,
                         const Counts& counts, bool stable_backend,
                         int& device) noexcept {
    std::size_t x_bytes = 0, z_bytes = 0, alpha_bytes = 0, bias_bytes = 0,
                gamma_bytes = 0, output_bytes = 0, tape_bytes = 0, scratch_bytes = 0;
    if (!checked_bytes(counts.input_values, dtype_bytes(precision.activations), x_bytes) ||
        !checked_bytes(counts.context_values, dtype_bytes(precision.activations), z_bytes) ||
        !checked_bytes(counts.alpha_values, dtype_bytes(precision.parameters), alpha_bytes) ||
        !checked_bytes(plan.extents.biases, dtype_bytes(precision.parameters), bias_bytes) ||
        !checked_bytes(plan.extents.gammas, dtype_bytes(precision.parameters), gamma_bytes) ||
        !checked_bytes(counts.output_values, dtype_bytes(precision.output), output_bytes) ||
        !checked_bytes(counts.tape_values, sizeof(float), tape_bytes) ||
        !checked_bytes(stable_backend ? 0 : counts.forward_scratch_values,
                       sizeof(float), scratch_bytes))
        return false;
    if (!aligned_view(values.x, dtype_bytes(precision.activations)) ||
        !aligned_view(values.z, dtype_bytes(precision.activations)) ||
        !aligned_view(values.alpha, dtype_bytes(precision.parameters)) ||
        !aligned_view(values.bias, dtype_bytes(precision.parameters)) ||
        !aligned_view(values.gamma, dtype_bytes(precision.parameters)) ||
        !aligned_view(output, dtype_bytes(precision.output)) ||
        !aligned_view(tape, alignof(float)) ||
        !aligned_view(scratch, alignof(float))) return false;

    const bool active = plan.mechanism_count && problem.cells && problem.channels;
    if (active && (!plan.mechanisms ||
                   (plan.term_count && !plan.terms) ||
                   !aligned_for(plan.mechanisms, alignof(FactorizedInteractionRecord)) ||
                   (plan.term_count && !aligned_for(plan.terms, alignof(FactorizedInteractionTermRecord))) ||
                   !plan.extents.inputs || !plan.extents.outputs ||
                   !plan.extents.alpha_rows || !plan.extents.biases)) return false;
    int d = -1;
    if (!bind_device(d, values.x, active ? x_bytes : 0) ||
        !bind_device(d, values.z, active ? z_bytes : 0) ||
        !bind_device(d, values.alpha, active ? alpha_bytes : 0) ||
        !bind_device(d, values.bias, active ? bias_bytes : 0) ||
        !bind_device(d, values.gamma, active ? gamma_bytes : 0) ||
        !bind_device(d, output, output_bytes) ||
        !bind_device(d, tape, tape_bytes) ||
        !bind_device(d, scratch, scratch_bytes)) return false;

    std::size_t mechanism_bytes = 0, term_bytes = 0;
    if (active && (!checked_bytes(plan.mechanism_count, sizeof(FactorizedInteractionRecord), mechanism_bytes) ||
                   !checked_bytes(plan.term_count, sizeof(FactorizedInteractionTermRecord), term_bytes)))
        return false;
    Region writable[3], readonly[7];
    if (!make_region(output.data, output.bytes, writable[0]) ||
        !make_region(tape.data, tape.bytes, writable[1]) ||
        !make_region(scratch.data, scratch.bytes, writable[2]) ||
        !make_region(values.x.data, values.x.bytes, readonly[0]) ||
        !make_region(values.z.data, values.z.bytes, readonly[1]) ||
        !make_region(values.alpha.data, values.alpha.bytes, readonly[2]) ||
        !make_region(values.bias.data, values.bias.bytes, readonly[3]) ||
        !make_region(values.gamma.data, values.gamma.bytes, readonly[4]) ||
        !make_region(active ? plan.mechanisms : nullptr, mechanism_bytes, readonly[5]) ||
        !make_region(active ? plan.terms : nullptr, term_bytes, readonly[6]) ||
        !regions_disjoint(writable, 3, readonly, 7)) return false;
    device = d;
    return true;
}

bool backward_views_valid(const FactorizedInteractionView& plan,
                          FactorizedInteractionProblem problem,
                          FactorizedInteractionPrecision precision,
                          const FactorizedInteractionValues& values,
                          memory::cuda::ConstDeviceView output_gradient,
                          memory::cuda::ConstDeviceView tape,
                          const FactorizedInteractionGradients& gradients,
                          memory::cuda::DeviceView scratch,
                          const Counts& counts, int& device) noexcept {
    std::size_t x_bytes = 0, z_bytes = 0, alpha_bytes = 0, bias_bytes = 0,
                gamma_bytes = 0, output_bytes = 0, tape_bytes = 0,
                scratch_bytes = 0;
    if (!checked_bytes(counts.input_values, dtype_bytes(precision.activations), x_bytes) ||
        !checked_bytes(counts.context_values, dtype_bytes(precision.activations), z_bytes) ||
        !checked_bytes(counts.alpha_values, dtype_bytes(precision.parameters), alpha_bytes) ||
        !checked_bytes(plan.extents.biases, dtype_bytes(precision.parameters), bias_bytes) ||
        !checked_bytes(plan.extents.gammas, dtype_bytes(precision.parameters), gamma_bytes) ||
        !checked_bytes(counts.output_values, sizeof(float), output_bytes) ||
        !checked_bytes(counts.tape_values, sizeof(float), tape_bytes) ||
        !checked_bytes(counts.gate_values, sizeof(float), scratch_bytes)) return false;
    (void)output_bytes;
    (void)bias_bytes;
    (void)gamma_bytes;

    const bool active = plan.mechanism_count && problem.cells && problem.channels;
    if (active && (!plan.mechanisms ||
                   (plan.term_count && !plan.terms) ||
                   !aligned_for(plan.mechanisms, alignof(FactorizedInteractionRecord)) ||
                   (plan.term_count && !aligned_for(plan.terms, alignof(FactorizedInteractionTermRecord))) ||
                   !plan.extents.inputs || !plan.extents.outputs ||
                   !plan.extents.alpha_rows || !plan.extents.biases)) return false;

    std::size_t gx_bytes = 0, gz_bytes = 0, ga_bytes = 0, gb_bytes = 0,
                gg_bytes = 0, dy_bytes = 0;
    if (!checked_bytes(counts.input_values, sizeof(float), gx_bytes) ||
        !checked_bytes(counts.context_values, sizeof(float), gz_bytes) ||
        !checked_bytes(counts.alpha_values, sizeof(float), ga_bytes) ||
        !checked_bytes(plan.extents.biases, sizeof(float), gb_bytes) ||
        !checked_bytes(plan.extents.gammas, sizeof(float), gg_bytes) ||
        !checked_bytes(counts.output_values, sizeof(float), dy_bytes)) return false;

    int d = -1;
    if (!bind_device(d, values.x, active ? x_bytes : 0) ||
        !bind_device(d, values.z, active ? z_bytes : 0) ||
        !bind_device(d, values.alpha, active ? alpha_bytes : 0) ||
        !bind_device(d, values.gamma, active ? gamma_bytes : 0) ||
        !bind_device(d, output_gradient, active ? dy_bytes : 0) ||
        !bind_device(d, tape, active ? tape_bytes : 0) ||
        !bind_device(d, gradients.x, gx_bytes) ||
        !bind_device(d, gradients.z, gz_bytes) ||
        !bind_device(d, gradients.alpha, ga_bytes) ||
        !bind_device(d, gradients.bias, gb_bytes) ||
        !bind_device(d, gradients.gamma, gg_bytes) ||
        !bind_device(d, values.bias, active ? bias_bytes : 0) ||
        !bind_device(d, scratch, scratch_bytes)) return false;
    if (!aligned_view(values.x, dtype_bytes(precision.activations)) ||
        !aligned_view(values.z, dtype_bytes(precision.activations)) ||
        !aligned_view(values.alpha, dtype_bytes(precision.parameters)) ||
        !aligned_view(values.bias, dtype_bytes(precision.parameters)) ||
        !aligned_view(values.gamma, dtype_bytes(precision.parameters)) ||
        !aligned_view(output_gradient, alignof(float)) ||
        !aligned_view(gradients.x, alignof(float)) ||
        !aligned_view(gradients.z, alignof(float)) ||
        !aligned_view(gradients.alpha, alignof(float)) ||
        !aligned_view(gradients.bias, alignof(float)) ||
        !aligned_view(gradients.gamma, alignof(float)) ||
        !aligned_view(tape, alignof(float)) ||
        !aligned_view(scratch, alignof(float))) return false;

    std::size_t mechanism_bytes = 0, term_bytes = 0;
    if (active && (!checked_bytes(plan.mechanism_count, sizeof(FactorizedInteractionRecord), mechanism_bytes) ||
                   !checked_bytes(plan.term_count, sizeof(FactorizedInteractionTermRecord), term_bytes)))
        return false;
    Region writable[6], readonly[9];
    if (!make_region(gradients.x.data, gradients.x.bytes, writable[0]) ||
        !make_region(gradients.z.data, gradients.z.bytes, writable[1]) ||
        !make_region(gradients.alpha.data, gradients.alpha.bytes, writable[2]) ||
        !make_region(gradients.bias.data, gradients.bias.bytes, writable[3]) ||
        !make_region(gradients.gamma.data, gradients.gamma.bytes, writable[4]) ||
        !make_region(scratch.data, scratch.bytes, writable[5]) ||
        !make_region(values.x.data, values.x.bytes, readonly[0]) ||
        !make_region(values.z.data, values.z.bytes, readonly[1]) ||
        !make_region(values.alpha.data, values.alpha.bytes, readonly[2]) ||
        !make_region(values.bias.data, values.bias.bytes, readonly[3]) ||
        !make_region(values.gamma.data, values.gamma.bytes, readonly[4]) ||
        !make_region(output_gradient.data, output_gradient.bytes, readonly[5]) ||
        !make_region(tape.data, tape.bytes, readonly[6]) ||
        !make_region(active ? plan.mechanisms : nullptr, mechanism_bytes, readonly[7]) ||
        !make_region(active ? plan.terms : nullptr, term_bytes, readonly[8]) ||
        !regions_disjoint(writable, 6, readonly, 9)) return false;
    device = d;
    return true;
}

} // namespace

cudaError_t factorized_interaction_direct_workspace(
    const FactorizedInteractionView& plan, FactorizedInteractionProblem problem,
    FactorizedInteractionWorkspace* result) noexcept {
    if (!result) return cudaErrorInvalidValue;
    Counts counts;
    if (!counts_for(plan, problem, counts)) return cudaErrorInvalidValue;
    FactorizedInteractionBackend backend;
    if (factorized_interaction_backend(plan, problem,
            FactorizedInteractionRoute::direct, &backend) != cudaSuccess)
        return cudaErrorInvalidValue;
    std::size_t tape_bytes = 0, forward_bytes = 0, backward_bytes = 0;
    if (!checked_bytes(counts.tape_values, sizeof(float), tape_bytes) ||
        !checked_bytes(backend == FactorizedInteractionBackend::stable
                           ? 0 : counts.forward_scratch_values,
                       sizeof(float), forward_bytes) ||
        !checked_bytes(counts.gate_values, sizeof(float), backward_bytes))
        return cudaErrorInvalidValue;
    *result = {tape_bytes, forward_bytes, backward_bytes};
    return cudaSuccess;
}

cudaError_t factorized_interaction_direct_forward(
    const FactorizedInteractionView& plan, FactorizedInteractionProblem problem,
    FactorizedInteractionPrecision precision, FactorizedInteractionValues values,
    memory::cuda::DeviceView output, memory::cuda::DeviceView tape,
    memory::cuda::DeviceView scratch, cudaStream_t stream) noexcept {
    if (!valid_dtype(precision.activations) || !valid_dtype(precision.parameters) ||
        !valid_dtype(precision.output)) return cudaErrorInvalidValue;
    Counts counts;
    if (!counts_for(plan, problem, counts)) return cudaErrorInvalidValue;
    FactorizedInteractionBackend backend;
    if (factorized_interaction_backend(plan, problem,
            FactorizedInteractionRoute::direct, &backend) != cudaSuccess)
        return cudaErrorInvalidValue;
    const bool active = plan.mechanism_count && problem.cells && problem.channels;
    const bool stable_backend = backend == FactorizedInteractionBackend::stable;
    if (active && stable_backend && !stable_plan_pointers(plan))
        return cudaErrorInvalidValue;
    int device = -1;
    if (!forward_views_valid(plan, problem, precision, values, output, tape,
                             scratch, counts, stable_backend, device))
        return cudaErrorInvalidValue;
    auto status = current_device_matches(device);
    if (status != cudaSuccess) return status;

    std::size_t scratch_bytes = 0;
    std::size_t output_bytes = 0;
    if (!checked_bytes(stable_backend ? 0 : counts.forward_scratch_values,
                       sizeof(float), scratch_bytes) ||
        !checked_bytes(counts.output_values, dtype_bytes(precision.output), output_bytes))
        return cudaErrorInvalidValue;

    if (stable_backend) {
        if (!active)
            return output_bytes ? cudaMemsetAsync(output.data, 0, output_bytes, stream)
                                : cudaSuccess;
        auto* gates = static_cast<float*>(tape.data);
        return stable_detail::launch_forward(plan, problem, precision, values,
            output, gates, gates + counts.gate_values, stream);
    }

    if (precision.activations == memory::cuda::DType::float16) {
        if (precision.parameters == memory::cuda::DType::float16)
            return dispatch_forward_output<__half, __half>(plan, problem, values,
                precision, output, tape, scratch, scratch_bytes, counts, stream);
        return dispatch_forward_output<__half, float>(plan, problem, values,
            precision, output, tape, scratch, scratch_bytes, counts, stream);
    } else if (precision.parameters == memory::cuda::DType::float16) {
        return dispatch_forward_output<float, __half>(plan, problem, values,
            precision, output, tape, scratch, scratch_bytes, counts, stream);
    }
    return dispatch_forward_output<float, float>(plan, problem, values,
        precision, output, tape, scratch, scratch_bytes, counts, stream);
}

cudaError_t factorized_interaction_direct_backward(
    const FactorizedInteractionView& plan, FactorizedInteractionProblem problem,
    FactorizedInteractionPrecision precision, FactorizedInteractionValues values,
    memory::cuda::ConstDeviceView output_gradient,
    memory::cuda::ConstDeviceView tape,
    FactorizedInteractionGradients gradients, memory::cuda::DeviceView scratch,
    FactorizedInteractionGradientMode mode, cudaStream_t stream) noexcept {
    if (!valid_dtype(precision.activations) || !valid_dtype(precision.parameters) ||
        !valid_dtype(precision.output) ||
        (mode != FactorizedInteractionGradientMode::overwrite &&
         mode != FactorizedInteractionGradientMode::accumulate))
        return cudaErrorInvalidValue;
    Counts counts;
    if (!counts_for(plan, problem, counts)) return cudaErrorInvalidValue;
    FactorizedInteractionBackend backend;
    if (factorized_interaction_backend(plan, problem,
            FactorizedInteractionRoute::direct, &backend) != cudaSuccess)
        return cudaErrorInvalidValue;
    const bool active = plan.mechanism_count && problem.cells && problem.channels;
    const bool stable_backend = backend == FactorizedInteractionBackend::stable;
    if (active && stable_backend && !stable_plan_pointers(plan))
        return cudaErrorInvalidValue;
    int device = -1;
    if (!backward_views_valid(plan, problem, precision, values, output_gradient,
                              tape, gradients, scratch, counts, device))
        return cudaErrorInvalidValue;
    auto status = current_device_matches(device);
    if (status != cudaSuccess) return status;

    if (stable_backend && active) {
        const auto* gates = static_cast<const float*>(tape.data);
        return stable_detail::launch_backward(plan, problem, precision, values,
            static_cast<const float*>(output_gradient.data), gates,
            gates + counts.gate_values, gradients,
            static_cast<float*>(scratch.data), mode, stream);
    }

    if (precision.activations == memory::cuda::DType::float16) {
        if (precision.parameters == memory::cuda::DType::float16)
            return launch_backward<__half, __half>(plan, problem, precision, values,
                output_gradient, tape, gradients, scratch, mode, counts, stream);
        return launch_backward<__half, float>(plan, problem, precision, values,
            output_gradient, tape, gradients, scratch, mode, counts, stream);
    } else if (precision.parameters == memory::cuda::DType::float16) {
        return launch_backward<float, __half>(plan, problem, precision, values,
            output_gradient, tape, gradients, scratch, mode, counts, stream);
    }
    return launch_backward<float, float>(plan, problem, precision, values,
        output_gradient, tape, gradients, scratch, mode, counts, stream);
}

} // namespace cellerator::modeling::cuda
