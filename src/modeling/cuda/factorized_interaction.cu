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
using View = layout::FactorizedInteractionView;
using Precision = FactorizedInteractionPrecision;
using Problem = FactorizedInteractionProblem;
using Values = FactorizedInteractionValues;
using Gradients = FactorizedInteractionGradients;
namespace mc = memory::cuda;
constexpr unsigned kThreads = 128;
constexpr unsigned kGridLimit = 65535;
constexpr unsigned kWarpMask = 0xffffffffu;

bool multiply(std::uint64_t a, std::uint64_t b, std::uint64_t& r) noexcept {
    if (b && a > std::numeric_limits<std::uint64_t>::max() / b) return false;
    r = a * b;
    return true;
}
bool bytes_for(std::uint64_t n, std::size_t width, std::size_t& r) noexcept {
    if (n > std::numeric_limits<std::size_t>::max() / width) return false;
    r = static_cast<std::size_t>(n) * width;
    return true;
}
std::uint64_t divide_up(std::uint64_t n, std::uint64_t d) noexcept {
    return n / d + (n % d != 0);
}
unsigned linear_grid(std::uint64_t n) noexcept {
    return static_cast<unsigned>(std::min<std::uint64_t>(
        kGridLimit, std::max<std::uint64_t>(1, divide_up(n, kThreads))));
}
unsigned task_grid(std::uint64_t n) noexcept {
    return static_cast<unsigned>(std::min<std::uint64_t>(kGridLimit, n));
}
bool dtype_ok(mc::DType t) noexcept {
    return t == mc::DType::float16 || t == mc::DType::float32;
}
std::size_t width(mc::DType t) noexcept {
    return t == mc::DType::float16 ? sizeof(__half) : sizeof(float);
}

struct Counts {
    std::uint64_t x = 0, z = 0, alpha = 0, output = 0, gates = 0, partials = 0;
    std::uint64_t gate_tasks = 0, forward_tasks = 0, backward_tasks = 0;
    std::size_t tape_bytes = 0, forward_bytes = 0, backward_bytes = 0;
    bool active = false;
};

bool counts(const View& p, Problem s, Counts& n) noexcept {
    if (p.schema != layout::factorized_interaction_schema || p.reserved ||
        p.max_packet_inputs > layout::factorized_interaction_max_packet_inputs ||
        p.max_packet_contexts > layout::factorized_interaction_max_packet_contexts ||
        p.max_packet_mechanisms > layout::factorized_interaction_max_packet_mechanisms ||
        p.max_packet_outputs > layout::factorized_interaction_max_packet_mechanisms)
        return false;
    if (!multiply(p.extents.inputs, s.channels, n.x) || !multiply(n.x, s.cells, n.x) ||
        !multiply(p.extents.contexts, s.cells, n.z) ||
        !multiply(p.extents.alpha_rows, s.channels, n.alpha) ||
        !multiply(p.extents.outputs, s.channels, n.output) ||
        !multiply(n.output, s.cells, n.output) ||
        !multiply(p.packet_output_count, s.channels, n.partials) ||
        !multiply(n.partials, s.cells, n.partials)) return false;
    n.active = p.mechanism_count && s.cells && s.channels;
    if (n.active) {
        if (!p.packet_count || !p.max_packet_inputs || !p.max_packet_mechanisms ||
            !multiply(p.mechanism_count, s.cells, n.gates) ||
            !multiply(p.packet_count, divide_up(s.cells, 32), n.gate_tasks)) return false;
        const auto forward_cells = s.channels == 1 ? 32u : 8u;
        const auto forward_channels = s.channels == 1 ? 1u : 16u;
        const auto backward_cells = s.channels == 1 ? 32u : 4u;
        const auto backward_channels = s.channels == 1 ? 1u : 8u;
        if (!multiply(p.packet_count, divide_up(s.cells, forward_cells), n.forward_tasks) ||
            !multiply(n.forward_tasks, divide_up(s.channels, forward_channels), n.forward_tasks) ||
            !multiply(p.packet_count, divide_up(s.cells, backward_cells), n.backward_tasks) ||
            !multiply(n.backward_tasks, divide_up(s.channels, backward_channels), n.backward_tasks))
            return false;
    }
    std::size_t ignored = 0;
    std::uint64_t tape_values = 0;
    if (!multiply(n.gates, 2, tape_values)) return false;
    return bytes_for(n.x, 4, ignored) && bytes_for(n.z, 4, ignored) &&
        bytes_for(n.alpha, 4, ignored) && bytes_for(n.output, 4, ignored) &&
        bytes_for(p.extents.biases, 4, ignored) && bytes_for(p.extents.gammas, 4, ignored) &&
        bytes_for(tape_values, 4, n.tape_bytes) && bytes_for(n.partials, 4, n.forward_bytes) &&
        bytes_for(n.gates, 4, n.backward_bytes) &&
        n.output <= static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) &&
        n.gate_tasks <= static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) &&
        n.forward_tasks <= static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) &&
        n.backward_tasks <= static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
}

bool plan_pointers(const View& p) noexcept {
    return p.extents.inputs && p.extents.outputs && p.extents.alpha_rows && p.extents.biases &&
        p.mechanisms && p.packets && p.packet_mechanisms && p.packet_inputs &&
        p.packet_outputs && p.role_refs && p.local_output_mechanisms.offsets &&
        p.local_output_mechanisms.entries && p.output_packets.offsets &&
        p.output_packets.entries && (!p.term_count || (p.terms && p.context_refs)) &&
        (!p.packet_context_count || p.packet_contexts);
}

bool stable_plan_pointers(const View& p) noexcept {
    return plan_pointers(p) && p.input_packets.offsets && p.input_packets.entries &&
        p.local_input_roles.offsets && p.local_input_roles.entries &&
        p.alpha_mechanisms.offsets && p.alpha_mechanisms.entries &&
        p.bias_mechanisms.offsets && p.bias_mechanisms.entries &&
        (!p.extents.contexts || p.context_terms.offsets) &&
        (!p.extents.gammas || p.gamma_terms.offsets) &&
        (!p.term_count || (p.context_terms.entries && p.gamma_terms.entries));
}

struct Buffer {
    mc::ConstDeviceView view;
    std::size_t used = 0, alignment = 1;
};
bool bind(Buffer b, int& device) noexcept {
    if (!b.used) return true;
    const auto address = reinterpret_cast<std::uintptr_t>(b.view.data);
    if (!address || b.view.bytes < b.used || address % b.alignment || b.view.device < 0 ||
        b.used > std::numeric_limits<std::uintptr_t>::max() - address) return false;
    if (device < 0) device = b.view.device;
    return b.view.device == device;
}
bool overlap(Buffer a, Buffer b) noexcept {
    if (!a.used || !b.used) return false;
    const auto x = reinterpret_cast<std::uintptr_t>(a.view.data);
    const auto y = reinterpret_cast<std::uintptr_t>(b.view.data);
    return x <= y ? y - x < a.used : x - y < b.used;
}
bool bind_buffers(const Buffer* reads, unsigned nr, const Buffer* writes,
                  unsigned nw, int& device) noexcept {
    for (unsigned i = 0; i < nr; ++i) if (!bind(reads[i], device)) return false;
    for (unsigned i = 0; i < nw; ++i) {
        if (!bind(writes[i], device)) return false;
        for (unsigned j = 0; j < nr; ++j) if (overlap(writes[i], reads[j])) return false;
        for (unsigned j = 0; j < i; ++j) if (overlap(writes[i], writes[j])) return false;
    }
    return true;
}
cudaError_t current_device(int device) noexcept {
    if (device < 0) return cudaSuccess;
    int actual = -1;
    const auto status = cudaGetDevice(&actual);
    return status != cudaSuccess ? status : actual == device ? cudaSuccess : cudaErrorInvalidDevice;
}
bool primal_buffers(const View& p, const Counts& n, Values v, Precision precision,
                    Buffer (&out)[5]) noexcept {
    const auto a = width(precision.activations), w = width(precision.parameters);
    const std::uint64_t elements[]{n.x, n.z, n.alpha, p.extents.biases, p.extents.gammas};
    const mc::ConstDeviceView views[]{v.x, v.z, v.alpha, v.bias, v.gamma};
    for (unsigned i = 0; i < 5; ++i) {
        const auto alignment = i < 2 ? a : w;
        std::size_t size = 0;
        if (n.active && !bytes_for(elements[i], alignment, size)) return false;
        out[i] = {views[i], size, alignment};
    }
    return true;
}
cudaError_t clear(mc::DeviceView v, std::size_t bytes, cudaStream_t stream) noexcept {
    return bytes ? cudaMemsetAsync(v.data, 0, bytes, stream) : cudaSuccess;
}

template<class T> __device__ __forceinline__ float load(const T* x, std::uint64_t i) {
    return static_cast<float>(x[i]);
}
template<> __device__ __forceinline__ float load<__half>(const __half* x, std::uint64_t i) {
    return __half2float(x[i]);
}
template<class T> __device__ __forceinline__ void store(T* x, std::uint64_t i, float v) {
    x[i] = static_cast<T>(v);
}
template<> __device__ __forceinline__ void store<__half>(__half* x, std::uint64_t i, float v) {
    x[i] = __float2half_rn(v);
}
__device__ __forceinline__ float warp_sum(float x) {
    for (int delta = 16; delta; delta >>= 1) x += __shfl_down_sync(kWarpMask, x, delta);
    return x;
}

template<class Activation, class Parameter>
__global__ void packet_gates(View p, Problem s, const Activation* z,
    const Parameter* bias, const Parameter* gamma, float* gates, float* slopes,
    std::uint64_t tasks) {
    extern __shared__ float context[];
    const auto cell_tiles = (s.cells + 31) / 32;
    for (std::uint64_t task = blockIdx.x; task < tasks; task += gridDim.x) {
        const auto packet = p.packets[task / cell_tiles];
        const auto cell_begin = (task % cell_tiles) * 32;
        for (std::uint64_t i = threadIdx.x; i < std::uint64_t(packet.context_count) * 32; i += blockDim.x) {
            const auto cell = cell_begin + i % 32;
            context[i] = cell < s.cells ? load(z, p.packet_contexts[packet.context_begin + i / 32] * s.cells + cell) : 0.0f;
        }
        __syncthreads();
        for (std::uint64_t i = threadIdx.x; i < std::uint64_t(packet.mechanism_count) * 32; i += blockDim.x) {
            const auto cell = cell_begin + i % 32;
            if (cell < s.cells) {
                const auto m = p.packet_mechanisms[packet.mechanism_begin + i / 32];
                const auto record = p.mechanisms[m];
                stable_detail::Accurate a(load(bias, record.bias));
                for (std::uint32_t k = 0; k < record.term_count; ++k) {
                    const auto term_id = record.term_begin + k;
                    const auto term = p.terms[term_id];
                    const float value = packet.direct_context
                        ? load(z, term.context * s.cells + cell)
                        : context[std::uint64_t(p.context_refs[term_id]) * 32 + i % 32];
                    a.add_product(load(gamma, term.gamma), value);
                }
                const auto result = stable_detail::sigmoid_and_slope(a.value());
                gates[m * s.cells + cell] = result.gate;
                slopes[m * s.cells + cell] = result.slope;
            }
        }
        __syncthreads();
    }
}

template<class Activation, class Parameter, unsigned Cells, unsigned Channels>
__global__ void packet_forward(View p, Problem s, const Activation* x,
    const Parameter* alpha, const float* gates, float* partials, std::uint64_t tasks) {
    extern __shared__ __align__(16) unsigned char storage[];
    constexpr unsigned Tile = Cells * Channels;
    const auto cell_tiles = (s.cells + Cells - 1) / Cells;
    const auto channel_tiles = (s.channels + Channels - 1) / Channels;
    for (std::uint64_t task = blockIdx.x; task < tasks; task += gridDim.x) {
        const auto cell_begin = (task % cell_tiles) * Cells;
        const auto channel_begin = ((task / cell_tiles) % channel_tiles) * Channels;
        const auto packet = p.packets[task / cell_tiles / channel_tiles];
        auto* inputs = reinterpret_cast<Activation*>(storage);
        const auto input_bytes = std::size_t(packet.input_count) * Tile * sizeof(Activation);
        auto* contributions = reinterpret_cast<float*>(storage + ((input_bytes + 3) & ~std::size_t(3)));
        for (std::uint64_t i = threadIdx.x; i < std::uint64_t(packet.input_count) * Tile; i += blockDim.x) {
            const auto cell = cell_begin + i % Cells;
            const auto channel = channel_begin + (i / Cells) % Channels;
            const auto occurrence = p.packet_inputs[packet.input_begin + i / Tile];
            if (cell < s.cells && channel < s.channels)
                inputs[i] = x[(occurrence * s.channels + channel) * s.cells + cell];
            else store(inputs, i, 0.0f);
        }
        __syncthreads();
        for (std::uint64_t i = threadIdx.x; i < std::uint64_t(packet.mechanism_count) * Tile; i += blockDim.x) {
            const auto cell = cell_begin + i % Cells;
            const auto channel = channel_begin + (i / Cells) % Channels;
            const auto slot = i / Tile;
            float value = 0.0f;
            if (cell < s.cells && channel < s.channels) {
                const auto m = p.packet_mechanisms[packet.mechanism_begin + slot];
                const auto record = p.mechanisms[m];
                float product = 1.0f;
                #pragma unroll
                for (unsigned r = 0; r < 4; ++r) if (r < record.arity) {
                    const auto local = p.role_refs[4 * packet.mechanism_begin + r * packet.mechanism_count + slot];
                    product *= load(inputs, std::uint64_t(local) * Tile + i % Tile);
                }
                value = (gates[m * s.cells + cell] * load(alpha, record.alpha * s.channels + channel)) * product;
            }
            contributions[i] = value;
        }
        __syncthreads();
        for (std::uint64_t i = threadIdx.x; i < std::uint64_t(packet.output_count) * Tile; i += blockDim.x) {
            const auto cell = cell_begin + i % Cells;
            const auto channel = channel_begin + (i / Cells) % Channels;
            if (cell < s.cells && channel < s.channels) {
                const auto output = packet.output_begin + i / Tile;
                float total = 0.0f;
                for (auto k = p.local_output_mechanisms.offsets[output]; k < p.local_output_mechanisms.offsets[output + 1]; ++k) {
                    const auto slot = p.local_output_mechanisms.entries[k] - packet.mechanism_begin;
                    total += contributions[slot * Tile + i % Tile];
                }
                partials[(output * s.channels + channel) * s.cells + cell] = total;
            }
        }
        __syncthreads();
    }
}

template<class Output>
__global__ void reduce_outputs(View p, Problem s, const float* partials,
    Output* output, std::uint64_t count) {
    for (std::uint64_t i = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += std::uint64_t(gridDim.x) * blockDim.x) {
        const auto target = i / s.cells / s.channels;
        const auto tail = i % (s.channels * s.cells);
        float sum = 0.0f;
        for (auto k = p.output_packets.offsets[target]; k < p.output_packets.offsets[target + 1]; ++k)
            sum += partials[p.output_packets.entries[k] * s.channels * s.cells + tail];
        store(output, i, sum);
    }
}

template<class Activation, class Parameter, unsigned Cells, unsigned Channels>
__global__ void packet_backward(View p, Problem s, const Activation* x,
    const Parameter* alpha, const float* gates, const float* dy,
    float* dx, float* da, std::uint64_t tasks) {
    extern __shared__ __align__(16) unsigned char storage[];
    constexpr unsigned Tile = Cells * Channels;
    const auto cell_tiles = (s.cells + Cells - 1) / Cells;
    const auto channel_tiles = (s.channels + Channels - 1) / Channels;
    for (std::uint64_t task = blockIdx.x; task < tasks; task += gridDim.x) {
        const auto cell_begin = (task % cell_tiles) * Cells;
        const auto channel_begin = ((task / cell_tiles) % channel_tiles) * Channels;
        const auto packet = p.packets[task / cell_tiles / channel_tiles];
        auto* inputs = reinterpret_cast<Activation*>(storage);
        const auto input_bytes = std::size_t(packet.input_count) * Tile * sizeof(Activation);
        auto* input_gradient = reinterpret_cast<float*>(storage + ((input_bytes + 3) & ~std::size_t(3)));
        auto* alpha_gradient = input_gradient + std::uint64_t(packet.input_count) * Tile;
        for (std::uint64_t i = threadIdx.x; i < std::uint64_t(packet.input_count) * Tile; i += blockDim.x) {
            const auto cell = cell_begin + i % Cells;
            const auto channel = channel_begin + (i / Cells) % Channels;
            const auto occurrence = p.packet_inputs[packet.input_begin + i / Tile];
            if (cell < s.cells && channel < s.channels)
                inputs[i] = x[(occurrence * s.channels + channel) * s.cells + cell];
            else store(inputs, i, 0.0f);
            input_gradient[i] = 0.0f;
        }
        for (std::uint64_t i = threadIdx.x; i < std::uint64_t(packet.mechanism_count) * Channels; i += blockDim.x)
            alpha_gradient[i] = 0.0f;
        __syncthreads();
        for (std::uint64_t i = threadIdx.x; i < std::uint64_t(packet.mechanism_count) * Tile; i += blockDim.x) {
            const auto cell_local = i % Cells;
            const auto channel_local = (i / Cells) % Channels;
            const auto cell = cell_begin + cell_local;
            const auto channel = channel_begin + channel_local;
            const auto slot = i / Tile;
            if (cell < s.cells && channel < s.channels) {
                const auto m = p.packet_mechanisms[packet.mechanism_begin + slot];
                const auto record = p.mechanisms[m];
                float values[4];
                unsigned refs[4];
                float product = 1.0f;
                #pragma unroll
                for (unsigned r = 0; r < 4; ++r) {
                    refs[r] = r < record.arity ? p.role_refs[4 * packet.mechanism_begin + r * packet.mechanism_count + slot] : 0;
                    values[r] = r < record.arity ? load(inputs, std::uint64_t(refs[r]) * Tile + i % Tile) : 1.0f;
                    product *= values[r];
                }
                const float upstream = dy[(record.target * s.channels + channel) * s.cells + cell];
                const float a = load(alpha, record.alpha * s.channels + channel);
                const float g = gates[m * s.cells + cell];
                const float scale = (upstream * g) * a;
                #pragma unroll
                for (unsigned r = 0; r < 4; ++r) if (r < record.arity) {
                    float other = 1.0f;
                    #pragma unroll
                    for (unsigned k = 0; k < 4; ++k) if (k != r) other *= values[k];
                    atomicAdd(input_gradient + std::uint64_t(refs[r]) * Tile + i % Tile, scale * other);
                }
                atomicAdd(alpha_gradient + slot * Channels + channel_local, (upstream * g) * product);
            }
        }
        __syncthreads();
        for (std::uint64_t i = threadIdx.x; i < std::uint64_t(packet.input_count) * Tile; i += blockDim.x) {
            const auto cell = cell_begin + i % Cells;
            const auto channel = channel_begin + (i / Cells) % Channels;
            if (cell < s.cells && channel < s.channels) {
                const auto occurrence = p.packet_inputs[packet.input_begin + i / Tile];
                atomicAdd(dx + (occurrence * s.channels + channel) * s.cells + cell, input_gradient[i]);
            }
        }
        for (std::uint64_t i = threadIdx.x; i < std::uint64_t(packet.mechanism_count) * Channels; i += blockDim.x) {
            const auto channel = channel_begin + i % Channels;
            if (channel < s.channels) {
                const auto m = p.packet_mechanisms[packet.mechanism_begin + i / Channels];
                atomicAdd(da + p.mechanisms[m].alpha * s.channels + channel, alpha_gradient[i]);
            }
        }
        __syncthreads();
    }
}

template<class Activation, class Parameter>
__global__ void gate_backward(View p, Problem s, const Activation* z,
    const Parameter* gamma, const float* gate_adjoint,
    float* dz, float* db, float* dgamma, std::uint64_t warp_tasks) {
    const auto lane = threadIdx.x & 31;
    const auto warp = threadIdx.x / 32;
    const auto warps = blockDim.x / 32;
    const auto cell_tiles = (s.cells + 31) / 32;
    for (std::uint64_t task = std::uint64_t(blockIdx.x) * warps + warp;
         task < warp_tasks; task += std::uint64_t(gridDim.x) * warps) {
        const auto m = task / cell_tiles;
        const auto cell = (task % cell_tiles) * 32 + lane;
        const auto record = p.mechanisms[m];
        float value = 0.0f;
        if (cell < s.cells) value = gate_adjoint[m * s.cells + cell];
        const float bias_sum = warp_sum(value);
        if (lane == 0) atomicAdd(db + record.bias, bias_sum);
        for (std::uint32_t k = 0; k < record.term_count; ++k) {
            const auto term = p.terms[record.term_begin + k];
            float gamma_value = 0.0f;
            if (cell < s.cells) {
                gamma_value = value * load(z, term.context * s.cells + cell);
                atomicAdd(dz + term.context * s.cells + cell, value * load(gamma, term.gamma));
            }
            const float gamma_sum = warp_sum(gamma_value);
            if (lane == 0) atomicAdd(dgamma + term.gamma, gamma_sum);
        }
    }
}

template<class Activation, unsigned Cells, unsigned Channels>
std::size_t forward_shared(const View& p) noexcept {
    const auto inputs = std::size_t(p.max_packet_inputs) * Cells * Channels * sizeof(Activation);
    return ((inputs + 3) & ~std::size_t(3)) + std::size_t(p.max_packet_mechanisms) * Cells * Channels * sizeof(float);
}
template<class Activation, unsigned Cells, unsigned Channels>
std::size_t backward_shared(const View& p) noexcept {
    const auto inputs = std::size_t(p.max_packet_inputs) * Cells * Channels * sizeof(Activation);
    return ((inputs + 3) & ~std::size_t(3)) +
        (std::size_t(p.max_packet_inputs) * Cells * Channels +
         std::size_t(p.max_packet_mechanisms) * Channels) * sizeof(float);
}

template<class Activation, class Parameter>
cudaError_t launch_forward(const View& p, Problem s, Values v, const Counts& n,
    float* gates, float* slopes, float* partials, cudaStream_t stream) noexcept {
    const auto* x = static_cast<const Activation*>(v.x.data);
    const auto* z = static_cast<const Activation*>(v.z.data);
    const auto* alpha = static_cast<const Parameter*>(v.alpha.data);
    packet_gates<<<task_grid(n.gate_tasks), kThreads,
        std::size_t(p.max_packet_contexts) * 32 * sizeof(float), stream>>>(
        p, s, z, static_cast<const Parameter*>(v.bias.data),
        static_cast<const Parameter*>(v.gamma.data), gates, slopes, n.gate_tasks);
    auto status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    if (s.channels == 1)
        packet_forward<Activation, Parameter, 32, 1><<<task_grid(n.forward_tasks), kThreads,
            forward_shared<Activation, 32, 1>(p), stream>>>(p, s, x, alpha, gates, partials, n.forward_tasks);
    else
        packet_forward<Activation, Parameter, 8, 16><<<task_grid(n.forward_tasks), kThreads,
            forward_shared<Activation, 8, 16>(p), stream>>>(p, s, x, alpha, gates, partials, n.forward_tasks);
    return cudaGetLastError();
}

template<class Activation, class Parameter>
cudaError_t launch_backward(const View& p, Problem s, Values v, const Counts& n,
    Precision precision, const float* gates, const float* slopes, const float* dy,
    Gradients gradients, float* gate_adjoint,
    cudaStream_t stream) noexcept {
    auto status = stable_detail::launch_gate_adjoint(
        p, s, precision, v, dy, gates, slopes, gate_adjoint, stream);
    if (status != cudaSuccess) return status;
    const auto* x = static_cast<const Activation*>(v.x.data);
    const auto* alpha = static_cast<const Parameter*>(v.alpha.data);
    auto* dx = static_cast<float*>(gradients.x.data);
    auto* da = static_cast<float*>(gradients.alpha.data);
    if (s.channels == 1)
        packet_backward<Activation, Parameter, 32, 1><<<task_grid(n.backward_tasks), kThreads,
            backward_shared<Activation, 32, 1>(p), stream>>>(p, s, x, alpha, gates, dy, dx, da, n.backward_tasks);
    else
        packet_backward<Activation, Parameter, 4, 8><<<task_grid(n.backward_tasks), kThreads,
            backward_shared<Activation, 4, 8>(p), stream>>>(p, s, x, alpha, gates, dy, dx, da, n.backward_tasks);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    std::uint64_t tasks = 0;
    if (!multiply(p.mechanism_count, divide_up(s.cells, 32), tasks)) return cudaErrorInvalidValue;
    gate_backward<Activation, Parameter><<<task_grid(divide_up(tasks, kThreads / 32)), kThreads, 0, stream>>>(
        p, s, static_cast<const Activation*>(v.z.data), static_cast<const Parameter*>(v.gamma.data),
        gate_adjoint, static_cast<float*>(gradients.z.data),
        static_cast<float*>(gradients.bias.data), static_cast<float*>(gradients.gamma.data), tasks);
    return cudaGetLastError();
}

} // namespace

cudaError_t factorized_interaction_backend(const View& p, Problem s,
    FactorizedInteractionRoute route, FactorizedInteractionBackend* result) noexcept {
    if (!result || p.schema != layout::factorized_interaction_schema || p.reserved ||
        (route != FactorizedInteractionRoute::packet && route != FactorizedInteractionRoute::direct))
        return cudaErrorInvalidValue;
    // gamma_63 < 4e-6 for FP32; the 64-contribution cap leaves room inside
    // the frozen 2e-5 sum-absolute-contribution bound for local reductions and
    // bounded-arity arithmetic. The same conservative validity rule is used
    // by both requested routes, including the packet's four-cell alpha fold.
    constexpr std::uint64_t limit = 64;
    const auto over = [](std::uint64_t n, std::uint64_t repeat) {
        return n && repeat > limit / n;
    };
    const bool active = p.mechanism_count && s.cells && s.channels;
    const bool stable = active && (p.max_mechanism_terms > limit ||
        p.max_output_mechanisms > limit || p.max_input_roles > limit ||
        p.max_context_terms > limit || over(p.max_alpha_mechanisms, divide_up(s.cells, 4)) ||
        over(p.max_bias_mechanisms, divide_up(s.cells, 32)) ||
        over(p.max_gamma_terms, divide_up(s.cells, 32)));
    *result = stable ? FactorizedInteractionBackend::stable
        : route == FactorizedInteractionRoute::packet ? FactorizedInteractionBackend::packet
                                                    : FactorizedInteractionBackend::direct;
    return cudaSuccess;
}

cudaError_t factorized_interaction_workspace(const View& p, Problem s,
    FactorizedInteractionRoute route, FactorizedInteractionWorkspace* result) noexcept {
    if (route == FactorizedInteractionRoute::direct)
        return factorized_interaction_direct_workspace(p, s, result);
    if (route != FactorizedInteractionRoute::packet || !result) return cudaErrorInvalidValue;
    Counts n;
    if (!counts(p, s, n)) return cudaErrorInvalidValue;
    FactorizedInteractionBackend backend;
    const auto status = factorized_interaction_backend(p, s, route, &backend);
    if (status != cudaSuccess) return status;
    *result = {n.tape_bytes, backend == FactorizedInteractionBackend::stable ? 0 : n.forward_bytes, n.backward_bytes};
    return cudaSuccess;
}

cudaError_t factorized_interaction_forward(const View& p, Problem s,
    Precision precision, Values v, mc::DeviceView output, mc::DeviceView tape,
    mc::DeviceView scratch, cudaStream_t stream, FactorizedInteractionRoute route) noexcept {
    if (route == FactorizedInteractionRoute::direct)
        return factorized_interaction_direct_forward(p, s, precision, v, output, tape, scratch, stream);
    if (route != FactorizedInteractionRoute::packet || !dtype_ok(precision.activations) ||
        !dtype_ok(precision.parameters) || !dtype_ok(precision.output)) return cudaErrorInvalidValue;
    Counts n;
    if (!counts(p, s, n) || (n.active && !plan_pointers(p))) return cudaErrorInvalidValue;
    FactorizedInteractionBackend backend;
    auto selection = factorized_interaction_backend(p, s, route, &backend);
    if (selection != cudaSuccess ||
        (n.active && backend == FactorizedInteractionBackend::stable && !stable_plan_pointers(p)))
        return cudaErrorInvalidValue;
    if (backend == FactorizedInteractionBackend::stable) n.forward_bytes = 0;
    Buffer reads[5];
    std::size_t output_bytes = 0;
    if (!primal_buffers(p, n, v, precision, reads) || !bytes_for(n.output, width(precision.output), output_bytes))
        return cudaErrorInvalidValue;
    const Buffer writes[]{{output, output_bytes, width(precision.output)},
                          {tape, n.tape_bytes, 4}, {scratch, n.forward_bytes, 4}};
    int device = -1;
    if (!bind_buffers(reads, 5, writes, 3, device)) return cudaErrorInvalidValue;
    auto status = current_device(device);
    if (status != cudaSuccess) return status;
    if (!n.active) return clear(output, output_bytes, stream);
    auto* gates = static_cast<float*>(tape.data);
    auto* slopes = gates + n.gates;
    if (backend == FactorizedInteractionBackend::stable)
        return stable_detail::launch_forward(p, s, precision, v, output, gates, slopes, stream);
    auto* partials = static_cast<float*>(scratch.data);
    if (precision.activations == mc::DType::float16) {
        status = precision.parameters == mc::DType::float16
            ? launch_forward<__half, __half>(p, s, v, n, gates, slopes, partials, stream)
            : launch_forward<__half, float>(p, s, v, n, gates, slopes, partials, stream);
    } else {
        status = precision.parameters == mc::DType::float16
            ? launch_forward<float, __half>(p, s, v, n, gates, slopes, partials, stream)
            : launch_forward<float, float>(p, s, v, n, gates, slopes, partials, stream);
    }
    if (status != cudaSuccess) return status;
    if (precision.output == mc::DType::float16)
        reduce_outputs<<<linear_grid(n.output), kThreads, 0, stream>>>(p, s, partials, static_cast<__half*>(output.data), n.output);
    else
        reduce_outputs<<<linear_grid(n.output), kThreads, 0, stream>>>(p, s, partials, static_cast<float*>(output.data), n.output);
    return cudaGetLastError();
}

cudaError_t factorized_interaction_backward(const View& p, Problem s,
    Precision precision, Values v, mc::ConstDeviceView dy, mc::ConstDeviceView tape,
    Gradients gradients, mc::DeviceView scratch, FactorizedInteractionGradientMode mode,
    cudaStream_t stream, FactorizedInteractionRoute route) noexcept {
    if (route == FactorizedInteractionRoute::direct)
        return factorized_interaction_direct_backward(p, s, precision, v, dy, tape, gradients, scratch, mode, stream);
    if (route != FactorizedInteractionRoute::packet || !dtype_ok(precision.activations) ||
        !dtype_ok(precision.parameters) || !dtype_ok(precision.output) ||
        (mode != FactorizedInteractionGradientMode::overwrite && mode != FactorizedInteractionGradientMode::accumulate))
        return cudaErrorInvalidValue;
    Counts n;
    if (!counts(p, s, n) || (n.active && !plan_pointers(p))) return cudaErrorInvalidValue;
    FactorizedInteractionBackend backend;
    auto selection = factorized_interaction_backend(p, s, route, &backend);
    if (selection != cudaSuccess ||
        (n.active && backend == FactorizedInteractionBackend::stable && !stable_plan_pointers(p)))
        return cudaErrorInvalidValue;
    Buffer reads[7];
    Buffer primals[5];
    if (!primal_buffers(p, n, v, precision, primals)) return cudaErrorInvalidValue;
    for (unsigned i = 0; i < 5; ++i) reads[i] = primals[i];
    std::size_t dy_bytes = 0;
    if (n.active && !bytes_for(n.output, 4, dy_bytes)) return cudaErrorInvalidValue;
    reads[5] = {dy, dy_bytes, 4};
    reads[6] = {tape, n.tape_bytes, 4};
    const mc::DeviceView destination[]{gradients.x, gradients.z, gradients.alpha, gradients.bias, gradients.gamma};
    const std::uint64_t elements[]{n.x, n.z, n.alpha, p.extents.biases, p.extents.gammas};
    Buffer writes[6];
    for (unsigned i = 0; i < 5; ++i) {
        std::size_t size = 0;
        if (!bytes_for(elements[i], 4, size)) return cudaErrorInvalidValue;
        writes[i] = {destination[i], size, 4};
    }
    writes[5] = {scratch, n.backward_bytes, 4};
    int device = -1;
    if (!bind_buffers(reads, 7, writes, 6, device)) return cudaErrorInvalidValue;
    auto status = current_device(device);
    if (status != cudaSuccess) return status;
    if (n.active && backend == FactorizedInteractionBackend::stable) {
        const auto* gates = static_cast<const float*>(tape.data);
        return stable_detail::launch_backward(p, s, precision, v, static_cast<const float*>(dy.data),
            gates, gates + n.gates, gradients, static_cast<float*>(scratch.data), mode, stream);
    }
    if (mode == FactorizedInteractionGradientMode::overwrite) {
        for (unsigned i = 0; i < 5; ++i) {
            status = clear(destination[i], writes[i].used, stream);
            if (status != cudaSuccess) return status;
        }
    }
    if (!n.active) return cudaSuccess;
    const auto* gates = static_cast<const float*>(tape.data);
    const auto* slopes = gates + n.gates;
    const auto* upstream = static_cast<const float*>(dy.data);
    auto* gate_adjoint = static_cast<float*>(scratch.data);
    if (precision.activations == mc::DType::float16) {
        return precision.parameters == mc::DType::float16
            ? launch_backward<__half, __half>(p, s, v, n, precision, gates, slopes, upstream, gradients, gate_adjoint, stream)
            : launch_backward<__half, float>(p, s, v, n, precision, gates, slopes, upstream, gradients, gate_adjoint, stream);
    }
    return precision.parameters == mc::DType::float16
        ? launch_backward<float, __half>(p, s, v, n, precision, gates, slopes, upstream, gradients, gate_adjoint, stream)
        : launch_backward<float, float>(p, s, v, n, precision, gates, slopes, upstream, gradients, gate_adjoint, stream);
}

} // namespace cellerator::modeling::cuda
