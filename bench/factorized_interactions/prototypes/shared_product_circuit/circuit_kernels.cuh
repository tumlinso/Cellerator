#pragma once

// Private fixed-topology experiment. The shared numerical helpers are included
// unchanged; neither arm changes the public native implementation.
#include "../../../../src/modeling/cuda/factorized_interaction_numerics.cuh"

namespace shared_product_circuit {

using Accurate = cellerator::modeling::cuda::stable_detail::Accurate;
namespace numerics = cellerator::modeling::cuda::stable_detail;
constexpr int mechanisms = 8;
constexpr int channels = 32;
constexpr int threads = 128;

struct Primals {
    const __half* x;
    const __half* z;
    const __half* alpha;
    const __half* bias;
    const __half* gamma;
    const float* dy;
};

struct Destinations {
    float* y;
    float* dx;
    float* dz;
    float* dalpha;
    float* dbias;
    float* dgamma;
};

struct Scratch {
    // Gates and slopes retain the existing two-plane FP32 tape convention.
    float* tape;
    Accurate* alpha_partials;  // [4,32,cells]
    Accurate* gate_partials;   // [8,32,cells]
    Accurate* gate_adjoint;    // [8,cells]
};

__device__ __forceinline__ float half_value(const __half* values, int index) {
    return __half2float(values[index]);
}

__device__ __forceinline__ void finish(float* destination, int index,
                                      Accurate value, bool accumulate) {
    if (accumulate) value.add(destination[index]);
    destination[index] = value.value();
}

__device__ __forceinline__ Accurate block_sum(Accurate value,
                                             Accurate* warp_partials) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x / 32;
    value = numerics::warp_sum(value);
    if (lane == 0) warp_partials[warp] = value;
    __syncthreads();
    Accurate total;
    if (warp == 0) {
        if (lane < threads / 32) total = warp_partials[lane];
        total = numerics::warp_sum(total);
    }
    return total;
}

__global__ void gates(Primals p, int cells, float* tape) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= mechanisms * cells) return;
    const int m = i / cells;
    const int cell = i % cells;
    Accurate logit(half_value(p.bias, m % 3));
    logit.add_product(half_value(p.gamma, m % 4), half_value(p.z, cell));
    logit.add_product(half_value(p.gamma, (m + 1) % 4),
                      half_value(p.z, cells + cell));
    const auto result = numerics::sigmoid_and_slope(logit.value());
    tape[i] = result.gate;
    tape[mechanisms * cells + i] = result.slope;
}

template<bool Circuit>
__global__ void forward(Primals p, int cells, bool repeated_tail,
                        const float* tape, float* y) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= channels * cells) return;
    const int d = i / cells;
    const int c = i % cells;
    const int plane = channels * cells;
    const float a = half_value(p.x, i);
    const float b = half_value(p.x, plane + i);
    const float common = a * b;
    Accurate sum;
    #pragma unroll
    for (int m = 0; m < mechanisms; ++m) {
        const int tail = repeated_tail && m == 0 ? 0 : 2 + m;
        Accurate contribution(tape[m * cells + c]);
        contribution.multiply(half_value(p.alpha, (m % 4) * channels + d));
        // The literal control is allowed the same common-subexpression
        // elimination: two FP16 operands have an exact FP32 product here.
        // It retains this multiply inside each full mechanism contribution.
        if constexpr (!Circuit) contribution.multiply(common);
        contribution.multiply(half_value(p.x, tail * plane + i));
        sum.add(contribution);
    }
    if constexpr (Circuit) sum.multiply(common);
    y[i] = sum.value();
}

template<bool Circuit>
__global__ void data_backward(Primals p, int cells, bool repeated_tail,
                              Scratch scratch, Destinations out,
                              bool accumulate) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= channels * cells) return;
    const int d = i / cells;
    const int c = i % cells;
    const int plane = channels * cells;
    const float a = half_value(p.x, i);
    const float b = half_value(p.x, plane + i);
    const float common = a * b;
    const float upstream = p.dy[i];
    Accurate sum, grad_a, grad_b, repeated_extra;
    Accurate alpha_partials[4];
    #pragma unroll
    for (int m = 0; m < mechanisms; ++m) {
        const int tail = repeated_tail && m == 0 ? 0 : 2 + m;
        const float value = half_value(p.x, tail * plane + i);
        const float coefficient = half_value(p.alpha, (m % 4) * channels + d);
        const float gate = scratch.tape[m * cells + c];
        Accurate weighted(gate);
        weighted.multiply(coefficient);

        if constexpr (Circuit) {
            Accurate s = weighted;
            s.multiply(value);
            sum.add(s);
        } else {
            Accurate left = weighted;
            left.multiply(upstream);
            left.multiply(b);
            left.multiply(value);
            grad_a.add(left);
            Accurate right = weighted;
            right.multiply(upstream);
            right.multiply(a);
            right.multiply(value);
            grad_b.add(right);
        }

        Accurate tail_gradient = weighted;
        tail_gradient.multiply(upstream);
        tail_gradient.multiply(common);
        if (repeated_tail && m == 0) repeated_extra.add(tail_gradient);
        else finish(out.dx, (2 + m) * plane + i, tail_gradient, accumulate);

        Accurate alpha_gradient(gate);
        alpha_gradient.multiply(upstream);
        alpha_gradient.multiply(common);
        alpha_gradient.multiply(value);
        alpha_partials[m % 4].add(alpha_gradient);

        Accurate gate_gradient(coefficient);
        gate_gradient.multiply(upstream);
        gate_gradient.multiply(common);
        gate_gradient.multiply(value);
        scratch.gate_partials[m * plane + i] = gate_gradient;
    }
    if constexpr (Circuit) {
        grad_a = sum;
        grad_a.multiply(upstream);
        grad_a.multiply(b);
        grad_b = sum;
        grad_b.multiply(upstream);
        grad_b.multiply(a);
    }
    grad_a.add(repeated_extra);
    finish(out.dx, i, grad_a, accumulate);
    finish(out.dx, plane + i, grad_b, accumulate);
    // The repeated-tail fixture leaves occurrence 2 unused. It is still a
    // declared input gradient and must be overwritten/accumulated correctly.
    if (repeated_tail) finish(out.dx, 2 * plane + i, Accurate{}, accumulate);
    #pragma unroll
    for (int bank = 0; bank < 4; ++bank)
        scratch.alpha_partials[bank * plane + i] = alpha_partials[bank];
}

__global__ void channel_adjoint(int cells, Scratch scratch) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= mechanisms * cells) return;
    const int m = i / cells;
    const int c = i % cells;
    const int plane = channels * cells;
    Accurate total;
    #pragma unroll
    for (int d = 0; d < channels; ++d)
        total.add(scratch.gate_partials[m * plane + d * cells + c]);
    total.multiply(scratch.tape[mechanisms * cells + i]);
    scratch.gate_adjoint[i] = total;
}

__global__ void context_backward(Primals p, int cells, Scratch scratch,
                                 float* dz, bool accumulate) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 2 * cells) return;
    const int q = i / cells;
    const int c = i % cells;
    Accurate total;
    #pragma unroll
    for (int m = 0; m < mechanisms; ++m) {
        Accurate term = scratch.gate_adjoint[m * cells + c];
        term.multiply(half_value(p.gamma, (m + q) % 4));
        total.add(term);
    }
    finish(dz, i, total, accumulate);
}

__global__ void alpha_backward(int cells, Scratch scratch, float* dalpha,
                               bool accumulate) {
    __shared__ __align__(8) unsigned char storage[sizeof(Accurate) * (threads / 32)];
    auto* partials = reinterpret_cast<Accurate*>(storage);
    const int bank_channel = blockIdx.x;
    Accurate total;
    for (int c = threadIdx.x; c < cells; c += blockDim.x)
        total.add(scratch.alpha_partials[bank_channel * cells + c]);
    total = block_sum(total, partials);
    if (threadIdx.x == 0) finish(dalpha, bank_channel, total, accumulate);
}

__global__ void gate_parameter_backward(Primals p, int cells, Scratch scratch,
                                        float* dbias, float* dgamma,
                                        bool accumulate) {
    __shared__ __align__(8) unsigned char storage[sizeof(Accurate) * (threads / 32)];
    auto* partials = reinterpret_cast<Accurate*>(storage);
    const int bank = blockIdx.x;
    Accurate total;
    for (int i = threadIdx.x; i < mechanisms * cells; i += blockDim.x) {
        const int m = i / cells;
        const int c = i % cells;
        const Accurate h = scratch.gate_adjoint[i];
        if (bank < 3) {
            if (m % 3 == bank) total.add(h);
        } else {
            #pragma unroll
            for (int q = 0; q < 2; ++q) if ((m + q) % 4 == bank - 3) {
                Accurate term = h;
                term.multiply(half_value(p.z, q * cells + c));
                total.add(term);
            }
        }
    }
    total = block_sum(total, partials);
    if (threadIdx.x == 0) {
        if (bank < 3) finish(dbias, bank, total, accumulate);
        else finish(dgamma, bank - 3, total, accumulate);
    }
}

template<bool Circuit>
cudaError_t launch_forward(Primals p, int cells, bool repeated_tail,
                           Scratch scratch, Destinations out,
                           cudaStream_t stream) {
    gates<<<(mechanisms * cells + threads - 1) / threads, threads, 0, stream>>>(
        p, cells, scratch.tape);
    forward<Circuit><<<(channels * cells + threads - 1) / threads,
                       threads, 0, stream>>>(p, cells, repeated_tail,
                                           scratch.tape, out.y);
    return cudaGetLastError();
}

template<bool Circuit>
cudaError_t launch_backward(Primals p, int cells, bool repeated_tail,
                            Scratch scratch, Destinations out,
                            bool accumulate, cudaStream_t stream) {
    data_backward<Circuit><<<(channels * cells + threads - 1) / threads,
                              threads, 0, stream>>>(p, cells, repeated_tail,
                                                  scratch, out, accumulate);
    channel_adjoint<<<(mechanisms * cells + threads - 1) / threads,
                      threads, 0, stream>>>(cells, scratch);
    context_backward<<<(2 * cells + threads - 1) / threads,
                       threads, 0, stream>>>(p, cells, scratch, out.dz, accumulate);
    alpha_backward<<<4 * channels, threads, 0, stream>>>(
        cells, scratch, out.dalpha, accumulate);
    gate_parameter_backward<<<7, threads, 0, stream>>>(
        p, cells, scratch, out.dbias, out.dgamma, accumulate);
    return cudaGetLastError();
}

} // namespace shared_product_circuit
