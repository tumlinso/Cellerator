#pragma once

#include "learning/ste.hh"

namespace cgs {

enum class SteParameters : std::uint8_t {
    per_cell_condition,
    shared_condition
};

// All storage is caller-owned device memory. Gate scores, probabilities, q,
// admitted, dq and ds have shape [batch,conditions]; plane words have shape
// [conditions,ceil(batch/32)]. Shared parameters and reduced are [conditions].
// For shared_condition shared_scores must alias gates.scores and is populated
// on device from parameter[c] before the forward phase. For per-cell scores,
// parameters must alias gates.scores. Status is a single device
// word, cleared by the caller before this complete same-stream chain.
struct SteChainBuffers {
    SteBuffers gates{};
    float* dq = nullptr;
    float* ds = nullptr;
    float* reduced = nullptr;
    float* parameters = nullptr;
    SteParameters parameterization = SteParameters::per_cell_condition;
    float* shared_scores = nullptr; // writable [batch,conditions] for shared mode
};

struct StePhaseEvents {
    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
};

struct SteChain {
    SteChainBuffers buffers{};
    SteSnapshot snapshot{};
    GateView gates{};
};

// Enqueue the gate phase, exporting a GateView suitable for either ordered
// product or grouped-linear consumers. Follow with consumer prepare/forward/
// backward on the same stream, using the SAME status word, with ResultView.dq
// initially zeroed by the caller. Events are optional and never synchronized.
cudaError_t ste_chain_forward(std::uint64_t cells, std::uint64_t conditions,
                              float temperature, float threshold,
                              const GateRevisions&, SteChainBuffers,
                              SteChain*, StePhaseEvents, cudaStream_t) noexcept;

// Call after the body's native backward has accumulated its dq. Enqueues the
// declared analytic chain, optional shared-condition sum, and guarded SGD.
// The numerical workspace and gate construction must share a status word;
// any upstream or nonfinite failure prevents all parameter writes on device.
// A successful enqueue invalidates the snapshot, even when status later
// prevents the update. The caller must inspect status before publishing and
// increment revisions before reusing scores/descriptors.
cudaError_t ste_chain_backward_update(SteChain*, const GateRevisions& current,
                                      const ResultView& body_result,
                                      WorkspaceView body_workspace,
                                      float learning_rate, StePhaseEvents,
                                      cudaStream_t) noexcept;

} // namespace cgs
