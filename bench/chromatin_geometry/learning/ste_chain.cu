#include "learning/ste_chain.hh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace cgs {
namespace {

cudaError_t event(cudaEvent_t e, cudaStream_t stream) noexcept {
    return e ? cudaEventRecord(e, stream) : cudaSuccess;
}

std::uint32_t bits(float x) noexcept {
    std::uint32_t result;
    std::memcpy(&result, &x, sizeof(result));
    return result;
}

__global__ void expand_shared(std::uint64_t count, std::uint64_t conditions,
                              const float* parameters, float* scores) {
    for (std::uint64_t i = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += std::uint64_t(blockDim.x) * gridDim.x)
        scores[i] = parameters[i % conditions];
}

} // namespace

cudaError_t ste_chain_forward(std::uint64_t cells, std::uint64_t conditions,
                              float temperature, float threshold,
                              const GateRevisions& revisions, SteChainBuffers b,
                              SteChain* chain, StePhaseEvents events,
                              cudaStream_t stream) noexcept {
    if (!chain) return cudaErrorInvalidValue;
    chain->snapshot.valid = false;
    if (b.parameterization != SteParameters::per_cell_condition &&
        b.parameterization != SteParameters::shared_condition)
        return cudaErrorInvalidValue;
    if (!std::isfinite(temperature) || temperature <= 0 ||
        !std::isfinite(threshold) || threshold < 0 || threshold > 1 ||
        revisions.temperature_bits != bits(temperature) ||
        revisions.threshold_bits != bits(threshold) ||
        (conditions && cells > std::numeric_limits<std::uint64_t>::max() / conditions) ||
        cells * conditions > std::numeric_limits<std::size_t>::max() / sizeof(float))
        return cudaErrorInvalidValue;
    if (cells && conditions && (!b.dq || !b.ds || !b.parameters || !b.gates.status ||
        !b.gates.scores || !b.gates.probabilities || !b.gates.q || !b.gates.admitted ||
        !b.gates.primal_words || !b.gates.derivative_words ||
        (b.parameterization == SteParameters::shared_condition && !b.reduced)))
        return cudaErrorInvalidValue;
    if (cells && conditions &&
        ((b.parameterization == SteParameters::per_cell_condition && b.parameters != b.gates.scores) ||
         (b.parameterization == SteParameters::shared_condition &&
          (!b.shared_scores || b.shared_scores != b.gates.scores || b.shared_scores == b.parameters))))
        return cudaErrorInvalidValue;
    auto err = event(events.begin, stream);
    if (err != cudaSuccess) return err;
    if (cells && conditions && b.parameterization == SteParameters::shared_condition) {
        const auto count = cells * conditions;
        constexpr unsigned threads = 256;
        const auto nblocks = count / threads + (count % threads != 0);
        const auto blocks = static_cast<unsigned>(std::min<std::uint64_t>(nblocks, 65535));
        expand_shared<<<blocks, threads, 0, stream>>>(count, conditions, b.parameters, b.shared_scores);
        err = cudaPeekAtLastError();
        if (err != cudaSuccess) return err;
    }
    err = ste_forward(cells, conditions, temperature, threshold, revisions,
                      b.gates, &chain->snapshot, stream);
    if (err != cudaSuccess) return err;
    chain->buffers = b;
    chain->gates = {GateProfile::ste, b.gates.probabilities, b.gates.admitted,
                    b.gates.primal_words, b.gates.derivative_words,
                    cells / 32u + (cells % 32u != 0), revisions};
    err = event(events.end, stream);
    if (err != cudaSuccess) chain->snapshot.valid = false;
    return err;
}

cudaError_t ste_chain_backward_update(SteChain* chain, const GateRevisions& current,
                                      const ResultView& body_result,
                                      WorkspaceView body_workspace,
                                      float learning_rate, StePhaseEvents events,
                                      cudaStream_t stream) noexcept {
    if (!chain || !chain->snapshot.valid ||
        !same_revisions(chain->snapshot.revisions, current) ||
        body_result.dq != chain->buffers.dq ||
        body_workspace.status != chain->snapshot.status ||
        !std::isfinite(learning_rate) || learning_rate < 0)
        return cudaErrorInvalidValue;
    const auto cells = chain->snapshot.cells;
    const auto conditions = chain->snapshot.conditions;
    auto err = event(events.begin, stream);
    if (err != cudaSuccess) return err;
    err = ste_backward(chain->snapshot, current, chain->buffers.gates.probabilities,
                       chain->buffers.gates.derivative_words, chain->buffers.dq,
                       chain->buffers.ds, stream);
    if (err != cudaSuccess) return err;
    if (cells && conditions) {
        const bool shared = chain->buffers.parameterization == SteParameters::shared_condition;
        if (shared) {
            err = ste_reduce_per_condition(cells, conditions, chain->buffers.ds,
                                           chain->buffers.reduced,
                                           chain->snapshot.status, stream);
            if (err != cudaSuccess) return err;
        }
        err = ste_sgd_update(shared ? conditions : cells * conditions,
                             chain->buffers.parameters,
                             shared ? chain->buffers.reduced : chain->buffers.ds,
                             learning_rate, &chain->snapshot, stream);
        if (err != cudaSuccess) return err;
    } else {
        chain->snapshot.valid = false;
    }
    return event(events.end, stream);
}

} // namespace cgs
