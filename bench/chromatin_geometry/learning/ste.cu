#include "learning/ste.hh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace cgs {
namespace {

constexpr unsigned threads = 128;
constexpr std::uint64_t max_blocks = 65535;

std::uint32_t bits(float x) noexcept {
    std::uint32_t out;
    std::memcpy(&out, &x, sizeof(out));
    return out;
}

bool product_ok(std::uint64_t a, std::uint64_t b) noexcept {
    return !b || a <= std::numeric_limits<std::uint64_t>::max() / b;
}

unsigned grid_for(std::uint64_t units, unsigned units_per_block) noexcept {
    const auto blocks = units / units_per_block + (units % units_per_block != 0);
    return static_cast<unsigned>(std::min(blocks, max_blocks));
}

__device__ float sigmoid(float x) {
    // This form avoids exponential overflow at either tail.
    if (x >= 0.0f) return 1.0f / (1.0f + expf(-x));
    const float e = expf(x);
    return e / (1.0f + e);
}

// Each warp owns a condition and 32 consecutive cells; its ballot directly
// writes the plane consumed by the numerical kernels. Inactive tail lanes
// still participate in the ballot, with false membership.
__global__ void forward_kernel(std::uint64_t cells, std::uint64_t conditions,
                               std::uint64_t words, float temperature,
                               float threshold, SteBuffers b) {
    const unsigned lane = threadIdx.x & 31u;
    const std::uint64_t first = (std::uint64_t(blockIdx.x) * blockDim.x +
                                 threadIdx.x) / 32u;
    const std::uint64_t stride = std::uint64_t(gridDim.x) * blockDim.x / 32u;
    const std::uint64_t total = conditions * words;
    for (std::uint64_t w = first; w < total; w += stride) {
        const std::uint64_t condition = w / words;
        const std::uint64_t cell = (w % words) * 32u + lane;
        const bool in_range = cell < cells;
        const std::uint32_t candidates = b.candidates ? b.candidates[w] : ~0u;
        const bool candidate = in_range && ((candidates >> lane) & 1u);
        bool active = false;
        if (in_range) {
            const std::uint64_t i = cell * conditions + condition;
            float p = 0.0f;
            if (candidate) {
                const float s = b.scores[i];
                if (!isfinite(s)) atomicOr(b.status, status_nonfinite);
                p = sigmoid(s / temperature);
                active = p >= threshold;
            }
            b.probabilities[i] = p;
            b.q[i] = active ? p : 0.0f;
            b.admitted[i] = static_cast<std::uint8_t>(active);
        }
        const auto pmask = __ballot_sync(0xffffffffu, active);
        const auto dmask = __ballot_sync(0xffffffffu, candidate);
        if (lane == 0) {
            b.primal_words[w] = pmask;
            b.derivative_words[w] = dmask;
        }
    }
}

__global__ void backward_kernel(std::uint64_t cells, std::uint64_t conditions,
                                std::uint64_t words, float temperature,
                                const float* p, const std::uint32_t* derivative,
                                const float* dq, float* ds, std::uint32_t* status) {
    __shared__ bool enabled;
    if (threadIdx.x == 0) enabled = atomicOr(status, 0u) == 0;
    __syncthreads();
    if (!enabled) return;
    const std::uint64_t count = cells * conditions;
    for (std::uint64_t i = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += std::uint64_t(blockDim.x) * gridDim.x) {
        const auto cell = i / conditions;
        const auto condition = i % conditions;
        const bool candidate = (derivative[condition * words + cell / 32u] >>
                                (cell % 32u)) & 1u;
        if (candidate) {
            const float probability = p[i];
            const float result = dq[i] * (probability * (1.0f - probability) / temperature);
            if (!isfinite(result)) atomicOr(status, unsigned(status_nonfinite));
            ds[i] = result;
        } else {
            ds[i] = 0.0f;
        }
    }
}

__global__ void reduce_kernel(std::uint64_t cells, std::uint64_t conditions,
                              const float* ds, float* reduced,
                              const std::uint32_t* status) {
    if (*status != 0) return;
    __shared__ float partial[threads];
    for (std::uint64_t c = blockIdx.x; c < conditions; c += gridDim.x) {
        float sum = 0;
        for (std::uint64_t cell = threadIdx.x; cell < cells; cell += blockDim.x)
            sum += ds[cell * conditions + c];
        partial[threadIdx.x] = sum;
        __syncthreads();
        for (unsigned offset = threads / 2; offset; offset /= 2) {
            if (threadIdx.x < offset) partial[threadIdx.x] += partial[threadIdx.x + offset];
            __syncthreads();
        }
        if (threadIdx.x == 0) reduced[c] = partial[0];
        __syncthreads();
    }
}

__global__ void validate_sgd_kernel(std::uint64_t count, const float* params,
                                    const float* grad, float rate,
                                    std::uint32_t* status) {
    __shared__ bool enabled;
    if (threadIdx.x == 0) enabled = atomicOr(status, 0u) == 0;
    __syncthreads();
    if (!enabled) return;
    for (std::uint64_t i = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += std::uint64_t(blockDim.x) * gridDim.x) {
        const float g = grad[i];
        if (g == 0.0f) continue;
        const float p = params[i];
        const float candidate = p - rate * g;
        if (!isfinite(p) || !isfinite(g) || !isfinite(candidate))
            atomicOr(status, unsigned(status_nonfinite));
    }
}

__global__ void commit_sgd_kernel(std::uint64_t count, float* params,
                                  const float* grad, float rate,
                                  const std::uint32_t* status) {
    // The previous kernel has finished across the device before this starts.
    // This kernel does not write status, so every block observes one decision.
    if (*status != 0) return;
    for (std::uint64_t i = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += std::uint64_t(blockDim.x) * gridDim.x) {
        const float g = grad[i];
        if (g != 0.0f) params[i] -= rate * g;
    }
}

} // namespace

cudaError_t ste_forward(std::uint64_t cells, std::uint64_t conditions,
                        float temperature, float threshold,
                        const GateRevisions& revisions, SteBuffers b,
                        SteSnapshot* snapshot, cudaStream_t stream) noexcept {
    if (!snapshot) return cudaErrorInvalidValue;
    snapshot->valid = false;
    const auto words = cells / 32u + (cells % 32u != 0);
    if (!std::isfinite(temperature) || temperature <= 0 ||
        !std::isfinite(threshold) || threshold < 0 || threshold > 1 ||
        revisions.temperature_bits != bits(temperature) ||
        revisions.threshold_bits != bits(threshold) ||
        !product_ok(cells, conditions) || !product_ok(words, conditions))
        return cudaErrorInvalidValue;
    const auto count = cells * conditions;
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
        words * conditions > std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t))
        return cudaErrorInvalidValue;
    if (count && (!b.scores || !b.probabilities || !b.q || !b.admitted ||
                  !b.primal_words || !b.derivative_words || !b.status))
        return cudaErrorInvalidValue;
    if (count) {
        forward_kernel<<<grid_for(words * conditions, 4), threads, 0, stream>>>(
            cells, conditions, words, temperature, threshold, b);
        const auto err = cudaPeekAtLastError();
        if (err != cudaSuccess) return err;
    }
    *snapshot = {revisions, cells, conditions, temperature, b.scores,
                 b.probabilities, b.derivative_words, b.status, true};
    return cudaSuccess;
}

cudaError_t ste_backward(const SteSnapshot& s, const GateRevisions& current,
                         const float* probabilities,
                         const std::uint32_t* derivative_words,
                         const float* dq, float* ds,
                         cudaStream_t stream) noexcept {
    if (!s.valid || !same_revisions(s.revisions, current) ||
        s.probabilities != probabilities || s.derivative_words != derivative_words)
        return cudaErrorInvalidValue;
    const auto count = s.cells * s.conditions;
    if (!count) return cudaSuccess;
    if (!probabilities || !derivative_words || !dq || !ds || !s.status) return cudaErrorInvalidValue;
    const auto words = s.cells / 32u + (s.cells % 32u != 0);
    backward_kernel<<<grid_for(count, threads), threads, 0, stream>>>(
        s.cells, s.conditions, words, s.temperature,
        probabilities, derivative_words, dq, ds, s.status);
    return cudaPeekAtLastError();
}

cudaError_t ste_reduce_per_condition(std::uint64_t cells,
                                     std::uint64_t conditions,
                                     const float* ds, float* reduced,
                                     std::uint32_t* status,
                                     cudaStream_t stream) noexcept {
    if (!product_ok(cells, conditions) ||
        cells * conditions > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
        conditions > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
        (conditions && (!reduced || !status)) ||
        (cells && conditions && !ds)) return cudaErrorInvalidValue;
    if (!conditions) return cudaSuccess;
    reduce_kernel<<<static_cast<unsigned>(std::min(conditions, max_blocks)),
                    threads, 0, stream>>>(cells, conditions, ds, reduced, status);
    return cudaPeekAtLastError();
}

cudaError_t ste_sgd_update(std::uint64_t count, float* parameters,
                           const float* gradients, float learning_rate,
                           SteSnapshot* snapshot, cudaStream_t stream) noexcept {
    if (!std::isfinite(learning_rate) || learning_rate < 0 ||
        count > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
        (count && (!parameters || !gradients || !snapshot || !snapshot->valid ||
                   !snapshot->status))) return cudaErrorInvalidValue;
    if (count) {
        validate_sgd_kernel<<<grid_for(count, threads), threads, 0, stream>>>(
            count, parameters, gradients, learning_rate, snapshot->status);
        auto err = cudaPeekAtLastError();
        if (err != cudaSuccess) return err;
        commit_sgd_kernel<<<grid_for(count, threads), threads, 0, stream>>>(
            count, parameters, gradients, learning_rate, snapshot->status);
        err = cudaPeekAtLastError();
        if (err != cudaSuccess) { snapshot->valid = false; return err; }
    }
    if (snapshot) snapshot->valid = false;
    return cudaSuccess;
}

} // namespace cgs
