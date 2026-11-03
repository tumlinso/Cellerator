#include "factorized_interaction_stable.cuh"
#include "factorized_interaction_numerics.cuh"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace cellerator::modeling::cuda::stable_detail {
namespace {

using View = layout::FactorizedInteractionView;
using Problem = FactorizedInteractionProblem;
using Precision = FactorizedInteractionPrecision;
using Values = FactorizedInteractionValues;
using Gradients = FactorizedInteractionGradients;
namespace mc = memory::cuda;

constexpr unsigned kThreads = 128;
constexpr unsigned kWarps = kThreads / 32;
constexpr std::uint64_t kMaxGrid = 65535;

// Each thread forms bounded compensated leaves, then merges them using an
// online binary tree. A single unbounded two-float sum can itself lose a small
// correction after sufficiently many additions, especially before sigmoid.
// Only occupied tree slots are read; this avoids a separate workspace and a
// global synchronization. Full-warp/CTA merges below complete the tree.
struct TreeSum {
    float high[64];
    float low[64];
    std::uint64_t occupied = 0;
    unsigned leaf_size = 0;
    Accurate leaf;

    __device__ __forceinline__ void flush() {
        Accurate carry = leaf;
        unsigned level = 0;
        while ((occupied & (std::uint64_t(1) << level)) != 0) {
            Accurate left;
            left.hi = high[level];
            left.lo = low[level];
            left.add(carry);
            carry = left;
            occupied &= ~(std::uint64_t(1) << level);
            ++level;
        }
        high[level] = carry.hi;
        low[level] = carry.lo;
        occupied |= std::uint64_t(1) << level;
        leaf = Accurate{};
        leaf_size = 0;
    }

    __device__ __forceinline__ void add(const Accurate& value) {
        leaf.add(value);
        if (++leaf_size == 32) flush();
    }
    __device__ __forceinline__ void add(float value) { add(Accurate(value)); }
    __device__ __forceinline__ void add_product(float a, float b) {
        Accurate value;
        value.add_product(a, b);
        add(value);
    }
    __device__ __forceinline__ Accurate finish() {
        if (leaf_size) flush();
        Accurate result;
        for (int level = 63; level >= 0; --level) {
            if ((occupied & (std::uint64_t(1) << level)) != 0) {
                Accurate item;
                item.hi = high[level];
                item.lo = low[level];
                result.add(item);
            }
        }
        return result;
    }
};

unsigned blocks_for_owners(std::uint64_t count) noexcept {
    return static_cast<unsigned>(std::min<std::uint64_t>(count, kMaxGrid));
}

unsigned blocks_for_warps(std::uint64_t count) noexcept {
    const auto blocks = count / kWarps + (count % kWarps != 0);
    return static_cast<unsigned>(std::min<std::uint64_t>(blocks, kMaxGrid));
}

__device__ __forceinline__ std::uint64_t warp_ordinal() {
    return std::uint64_t(blockIdx.x) * kWarps + threadIdx.x / 32;
}

__device__ __forceinline__ std::uint64_t warp_stride() {
    return std::uint64_t(gridDim.x) * kWarps;
}

// Returns the result to thread zero; every thread in the CTA must participate.
// Callers synchronize before reusing this shared storage in a grid-stride loop.
__device__ __forceinline__ Accurate block_sum(Accurate local) {
    __shared__ float high[kWarps];
    __shared__ float low[kWarps];
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = threadIdx.x / 32u;
    local = warp_sum(local);
    if (lane == 0) {
        high[warp] = local.hi;
        low[warp] = local.lo;
    }
    __syncthreads();
    Accurate result;
    if (warp == 0) {
        if (lane < kWarps) {
            result.hi = high[lane];
            result.lo = low[lane];
        }
        result = warp_sum(result);
    }
    return result;
}

template<class Activation>
__device__ __forceinline__ Accurate participant_product(
    const layout::FactorizedInteractionRecord& record, const Activation* x,
    Problem shape, std::uint64_t channel, std::uint64_t cell,
    unsigned excluded_role = 4) {
    Accurate result(1.0f);
    const auto tail = channel * shape.cells + cell;
    const auto stride = shape.channels * shape.cells;
    #pragma unroll
    for (unsigned role = 0; role < 4; ++role) {
        if (role < record.arity && role != excluded_role)
            result.multiply(load(x, record.inputs[role] * stride + tail));
    }
    return result;
}

template<class Activation, class Parameter>
__global__ void prepare_gates(View plan, Problem shape, const Activation* z,
                              const Parameter* bias, const Parameter* gamma,
                              float* gates, float* slopes,
                              std::uint64_t count) {
    const unsigned lane = threadIdx.x & 31u;
    for (auto owner = warp_ordinal(); owner < count; owner += warp_stride()) {
        const auto mechanism = owner / shape.cells;
        const auto cell = owner % shape.cells;
        const auto record = plan.mechanisms[mechanism];
        TreeSum dot;
        if (lane == 0) dot.add(load(bias, record.bias));
        for (std::uint64_t k = lane; k < record.term_count; k += 32) {
            const auto term = plan.terms[record.term_begin + k];
            dot.add_product(load(gamma, term.gamma),
                            load(z, term.context * shape.cells + cell));
        }
        const auto reduced = warp_sum(dot.finish());
        if (lane == 0) {
            const auto evaluated = sigmoid_and_slope(reduced.value());
            gates[owner] = evaluated.gate;
            slopes[owner] = evaluated.slope;
        }
    }
}

template<class Activation, class Parameter>
__global__ void prepare_gate_adjoint(View plan, Problem shape,
                                     const Activation* x, const Parameter* alpha,
                                     const float* dy, const float* slopes,
                                     float* adjoint, std::uint64_t count) {
    const unsigned lane = threadIdx.x & 31u;
    for (auto owner = warp_ordinal(); owner < count; owner += warp_stride()) {
        const auto mechanism = owner / shape.cells;
        const auto cell = owner % shape.cells;
        const auto record = plan.mechanisms[mechanism];
        TreeSum sum;
        for (std::uint64_t channel = lane; channel < shape.channels; channel += 32) {
            auto contribution = participant_product(record, x, shape, channel, cell);
            contribution.multiply(load(alpha, record.alpha * shape.channels + channel));
            contribution.multiply(dy[(record.target * shape.channels + channel) * shape.cells + cell]);
            sum.add(contribution);
        }
        auto reduced = warp_sum(sum.finish());
        if (lane == 0) {
            reduced.multiply(slopes[owner]);
            adjoint[owner] = reduced.value();
        }
    }
}

template<class Activation, class Parameter, class Output>
__global__ void reduce_output(View plan, Problem shape, const Activation* x,
                              const Parameter* alpha, const float* gates,
                              Output* output, std::uint64_t count) {
    for (std::uint64_t owner = blockIdx.x; owner < count; owner += gridDim.x) {
        const auto cell = owner % shape.cells;
        const auto channel = (owner / shape.cells) % shape.channels;
        const auto target = owner / shape.cells / shape.channels;
        const auto begin = plan.output_packets.offsets[target];
        const auto end = plan.output_packets.offsets[target + 1];
        TreeSum sum;
        for (auto k = begin + threadIdx.x; k < end; k += blockDim.x) {
            const auto local_output = plan.output_packets.entries[k];
            for (auto j = plan.local_output_mechanisms.offsets[local_output];
                 j < plan.local_output_mechanisms.offsets[local_output + 1]; ++j) {
                const auto mechanism = plan.packet_mechanisms[plan.local_output_mechanisms.entries[j]];
                const auto record = plan.mechanisms[mechanism];
                auto contribution = participant_product(record, x, shape, channel, cell);
                contribution.multiply(load(alpha, record.alpha * shape.channels + channel));
                contribution.multiply(gates[mechanism * shape.cells + cell]);
                sum.add(contribution);
            }
        }
        auto reduced = block_sum(sum.finish());
        if (threadIdx.x == 0) store(output, owner, reduced.value());
        __syncthreads();
    }
}

template<class Activation, class Parameter>
__global__ void reduce_input_gradient(View plan, Problem shape, const Activation* x,
                                      const Parameter* alpha, const float* gates,
                                      const float* dy, float* output, bool accumulate,
                                      std::uint64_t count) {
    for (std::uint64_t owner = blockIdx.x; owner < count; owner += gridDim.x) {
        const auto cell = owner % shape.cells;
        const auto channel = (owner / shape.cells) % shape.channels;
        const auto occurrence = owner / shape.cells / shape.channels;
        const auto begin = plan.input_packets.offsets[occurrence];
        const auto end = plan.input_packets.offsets[occurrence + 1];
        TreeSum sum;
        for (auto k = begin + threadIdx.x; k < end; k += blockDim.x) {
            const auto local_input = plan.input_packets.entries[k];
            for (auto j = plan.local_input_roles.offsets[local_input];
                 j < plan.local_input_roles.offsets[local_input + 1]; ++j) {
                const auto incidence = plan.local_input_roles.entries[j];
                const auto mechanism = plan.packet_mechanisms[incidence / 4];
                const auto role = static_cast<unsigned>(incidence % 4);
                const auto record = plan.mechanisms[mechanism];
                auto contribution = participant_product(record, x, shape, channel, cell, role);
                contribution.multiply(load(alpha, record.alpha * shape.channels + channel));
                contribution.multiply(gates[mechanism * shape.cells + cell]);
                contribution.multiply(dy[(record.target * shape.channels + channel) * shape.cells + cell]);
                sum.add(contribution);
            }
        }
        auto reduced = block_sum(sum.finish());
        if (threadIdx.x == 0) {
            if (accumulate) reduced.add(output[owner]);
            output[owner] = reduced.value();
        }
        __syncthreads();
    }
}

// Iterate the Cartesian product of a posting list and cells without allowing
// uint64 multiplication to wrap. The overflow branch remains correct for an
// enormous trusted layout, even though such a device allocation is unusual.
template<class Body>
__device__ __forceinline__ void posting_cells(
    std::uint64_t begin, std::uint64_t end, std::uint64_t cells, Body body) {
    const auto entries = end - begin;
    if (entries <= (~std::uint64_t(0)) / cells) {
        const auto total = entries * cells;
        for (std::uint64_t k = threadIdx.x; k < total; k += blockDim.x)
            body(begin + k / cells, k % cells);
    } else {
        for (auto k = begin + threadIdx.x; k < end; k += blockDim.x)
            for (std::uint64_t cell = 0; cell < cells; ++cell) body(k, cell);
    }
}

template<class Activation, class Parameter>
__global__ void reduce_alpha_gradient(View plan, Problem shape, const Activation* x,
                                      const float* gates, const float* dy,
                                      float* output, bool accumulate,
                                      std::uint64_t count) {
    for (std::uint64_t owner = blockIdx.x; owner < count; owner += gridDim.x) {
        const auto channel = owner % shape.channels;
        const auto parameter = owner / shape.channels;
        const auto begin = plan.alpha_mechanisms.offsets[parameter];
        const auto end = plan.alpha_mechanisms.offsets[parameter + 1];
        TreeSum sum;
        posting_cells(begin, end, shape.cells, [&](std::uint64_t k, std::uint64_t cell) {
            const auto mechanism = plan.alpha_mechanisms.entries[k];
            const auto record = plan.mechanisms[mechanism];
            auto contribution = participant_product(record, x, shape, channel, cell);
            contribution.multiply(gates[mechanism * shape.cells + cell]);
            contribution.multiply(dy[(record.target * shape.channels + channel) * shape.cells + cell]);
            sum.add(contribution);
        });
        auto reduced = block_sum(sum.finish());
        if (threadIdx.x == 0) {
            if (accumulate) reduced.add(output[owner]);
            output[owner] = reduced.value();
        }
        __syncthreads();
    }
}

template<class Parameter>
__global__ void reduce_context_gradient(View plan, Problem shape,
                                        const Parameter* gamma,
                                        const float* gate_adjoint, float* output,
                                        bool accumulate, std::uint64_t count) {
    for (std::uint64_t owner = blockIdx.x; owner < count; owner += gridDim.x) {
        const auto cell = owner % shape.cells;
        const auto context = owner / shape.cells;
        const auto begin = plan.context_terms.offsets[context];
        const auto end = plan.context_terms.offsets[context + 1];
        TreeSum sum;
        for (auto k = begin + threadIdx.x; k < end; k += blockDim.x) {
            const auto term = plan.terms[plan.context_terms.entries[k]];
            sum.add_product(gate_adjoint[term.mechanism * shape.cells + cell],
                            load(gamma, term.gamma));
        }
        auto reduced = block_sum(sum.finish());
        if (threadIdx.x == 0) {
            if (accumulate) reduced.add(output[owner]);
            output[owner] = reduced.value();
        }
        __syncthreads();
    }
}

__global__ void reduce_bias_gradient(View plan, Problem shape,
                                     const float* gate_adjoint, float* output,
                                     bool accumulate, std::uint64_t count) {
    for (std::uint64_t owner = blockIdx.x; owner < count; owner += gridDim.x) {
        const auto begin = plan.bias_mechanisms.offsets[owner];
        const auto end = plan.bias_mechanisms.offsets[owner + 1];
        TreeSum sum;
        posting_cells(begin, end, shape.cells, [&](std::uint64_t k, std::uint64_t cell) {
            const auto mechanism = plan.bias_mechanisms.entries[k];
            sum.add(gate_adjoint[mechanism * shape.cells + cell]);
        });
        auto reduced = block_sum(sum.finish());
        if (threadIdx.x == 0) {
            if (accumulate) reduced.add(output[owner]);
            output[owner] = reduced.value();
        }
        __syncthreads();
    }
}

template<class Activation>
__global__ void reduce_gamma_gradient(View plan, Problem shape, const Activation* z,
                                      const float* gate_adjoint, float* output,
                                      bool accumulate, std::uint64_t count) {
    for (std::uint64_t owner = blockIdx.x; owner < count; owner += gridDim.x) {
        const auto begin = plan.gamma_terms.offsets[owner];
        const auto end = plan.gamma_terms.offsets[owner + 1];
        TreeSum sum;
        posting_cells(begin, end, shape.cells, [&](std::uint64_t k, std::uint64_t cell) {
            const auto term = plan.terms[plan.gamma_terms.entries[k]];
            sum.add_product(gate_adjoint[term.mechanism * shape.cells + cell],
                            load(z, term.context * shape.cells + cell));
        });
        auto reduced = block_sum(sum.finish());
        if (threadIdx.x == 0) {
            if (accumulate) reduced.add(output[owner]);
            output[owner] = reduced.value();
        }
        __syncthreads();
    }
}

template<class Activation, class Parameter>
cudaError_t forward_typed(const View& plan, Problem shape, Precision precision,
                          Values values, mc::DeviceView output, float* gates,
                          float* slopes, cudaStream_t stream) noexcept {
    const auto gate_count = plan.mechanism_count * shape.cells;
    const auto output_count = plan.extents.outputs * shape.channels * shape.cells;
    prepare_gates<<<blocks_for_warps(gate_count), kThreads, 0, stream>>>(
        plan, shape, static_cast<const Activation*>(values.z.data),
        static_cast<const Parameter*>(values.bias.data),
        static_cast<const Parameter*>(values.gamma.data), gates, slopes, gate_count);
    auto status = cudaGetLastError();
    if (status != cudaSuccess || !output_count) return status;
    if (precision.output == mc::DType::float16)
        reduce_output<<<blocks_for_owners(output_count), kThreads, 0, stream>>>(
            plan, shape, static_cast<const Activation*>(values.x.data),
            static_cast<const Parameter*>(values.alpha.data), gates,
            static_cast<__half*>(output.data), output_count);
    else
        reduce_output<<<blocks_for_owners(output_count), kThreads, 0, stream>>>(
            plan, shape, static_cast<const Activation*>(values.x.data),
            static_cast<const Parameter*>(values.alpha.data), gates,
            static_cast<float*>(output.data), output_count);
    return cudaGetLastError();
}

template<class Activation, class Parameter>
cudaError_t gate_adjoint_typed(const View& plan, Problem shape, Values values,
                               const float* dy, const float* slopes,
                               float* adjoint, cudaStream_t stream) noexcept {
    const auto count = plan.mechanism_count * shape.cells;
    if (!count) return cudaSuccess;
    prepare_gate_adjoint<<<blocks_for_warps(count), kThreads, 0, stream>>>(
        plan, shape, static_cast<const Activation*>(values.x.data),
        static_cast<const Parameter*>(values.alpha.data), dy, slopes, adjoint, count);
    return cudaGetLastError();
}

template<class Activation, class Parameter>
cudaError_t backward_typed(const View& plan, Problem shape, Values values,
                           const float* dy, const float* gates, const float* slopes,
                           Gradients gradients, float* gate_adjoint,
                           FactorizedInteractionGradientMode mode,
                           cudaStream_t stream) noexcept {
    auto status = gate_adjoint_typed<Activation, Parameter>(
        plan, shape, values, dy, slopes, gate_adjoint, stream);
    if (status != cudaSuccess) return status;
    const bool accumulate = mode == FactorizedInteractionGradientMode::accumulate;
    const auto inputs = plan.extents.inputs * shape.channels * shape.cells;
    const auto contexts = plan.extents.contexts * shape.cells;
    const auto alphas = plan.extents.alpha_rows * shape.channels;
    if (inputs) {
        reduce_input_gradient<<<blocks_for_owners(inputs), kThreads, 0, stream>>>(
            plan, shape, static_cast<const Activation*>(values.x.data),
            static_cast<const Parameter*>(values.alpha.data), gates, dy,
            static_cast<float*>(gradients.x.data), accumulate, inputs);
        status = cudaGetLastError(); if (status != cudaSuccess) return status;
    }
    if (contexts) {
        reduce_context_gradient<<<blocks_for_owners(contexts), kThreads, 0, stream>>>(
            plan, shape, static_cast<const Parameter*>(values.gamma.data), gate_adjoint,
            static_cast<float*>(gradients.z.data), accumulate, contexts);
        status = cudaGetLastError(); if (status != cudaSuccess) return status;
    }
    if (alphas) {
        reduce_alpha_gradient<Activation, Parameter><<<blocks_for_owners(alphas), kThreads, 0, stream>>>(
            plan, shape, static_cast<const Activation*>(values.x.data), gates, dy,
            static_cast<float*>(gradients.alpha.data), accumulate, alphas);
        status = cudaGetLastError(); if (status != cudaSuccess) return status;
    }
    if (plan.extents.biases) {
        reduce_bias_gradient<<<blocks_for_owners(plan.extents.biases), kThreads, 0, stream>>>(
            plan, shape, gate_adjoint, static_cast<float*>(gradients.bias.data),
            accumulate, plan.extents.biases);
        status = cudaGetLastError(); if (status != cudaSuccess) return status;
    }
    if (plan.extents.gammas) {
        reduce_gamma_gradient<<<blocks_for_owners(plan.extents.gammas), kThreads, 0, stream>>>(
            plan, shape, static_cast<const Activation*>(values.z.data), gate_adjoint,
            static_cast<float*>(gradients.gamma.data), accumulate, plan.extents.gammas);
        status = cudaGetLastError(); if (status != cudaSuccess) return status;
    }
    return cudaSuccess;
}

} // namespace

cudaError_t launch_forward(const View& plan, Problem shape, Precision precision,
                           Values values, mc::DeviceView output,
                           float* gates, float* slopes, cudaStream_t stream) noexcept {
    if (!plan.mechanism_count || !shape.cells || !shape.channels) return cudaSuccess;
    if (precision.activations == mc::DType::float16) {
        if (precision.parameters == mc::DType::float16)
            return forward_typed<__half, __half>(plan, shape, precision, values, output, gates, slopes, stream);
        return forward_typed<__half, float>(plan, shape, precision, values, output, gates, slopes, stream);
    }
    if (precision.parameters == mc::DType::float16)
        return forward_typed<float, __half>(plan, shape, precision, values, output, gates, slopes, stream);
    return forward_typed<float, float>(plan, shape, precision, values, output, gates, slopes, stream);
}

cudaError_t launch_gate_adjoint(const View& plan, Problem shape, Precision precision,
                                Values values, const float* dy, const float* gates,
                                const float* slopes, float* adjoint,
                                cudaStream_t stream) noexcept {
    (void)gates;
    if (!plan.mechanism_count || !shape.cells || !shape.channels) return cudaSuccess;
    if (precision.activations == mc::DType::float16) {
        if (precision.parameters == mc::DType::float16)
            return gate_adjoint_typed<__half, __half>(plan, shape, values, dy, slopes, adjoint, stream);
        return gate_adjoint_typed<__half, float>(plan, shape, values, dy, slopes, adjoint, stream);
    }
    if (precision.parameters == mc::DType::float16)
        return gate_adjoint_typed<float, __half>(plan, shape, values, dy, slopes, adjoint, stream);
    return gate_adjoint_typed<float, float>(plan, shape, values, dy, slopes, adjoint, stream);
}

cudaError_t launch_backward(const View& plan, Problem shape, Precision precision,
                            Values values, const float* dy, const float* gates,
                            const float* slopes, Gradients gradients, float* gate_adjoint,
                            FactorizedInteractionGradientMode mode,
                            cudaStream_t stream) noexcept {
    if (!plan.mechanism_count || !shape.cells || !shape.channels) return cudaSuccess;
    if (precision.activations == mc::DType::float16) {
        if (precision.parameters == mc::DType::float16)
            return backward_typed<__half, __half>(plan, shape, values, dy, gates, slopes, gradients, gate_adjoint, mode, stream);
        return backward_typed<__half, float>(plan, shape, values, dy, gates, slopes, gradients, gate_adjoint, mode, stream);
    }
    if (precision.parameters == mc::DType::float16)
        return backward_typed<float, __half>(plan, shape, values, dy, gates, slopes, gradients, gate_adjoint, mode, stream);
    return backward_typed<float, float>(plan, shape, values, dy, gates, slopes, gradients, gate_adjoint, mode, stream);
}

} // namespace cellerator::modeling::cuda::stable_detail
