#pragma once

#include "contract.hh"

#include <cstdint>
#include <string>

namespace cgs {

// Private study interface. All device buffers are caller-owned, nonoverlapping
// except that candidates and derivative_words may be the same allocation.
// Arrays are row-major [cells, conditions]; masks are condition-major words.
// A null candidates plane declares every in-range pair as a candidate.
struct SteBuffers {
    const float* scores = nullptr;
    const std::uint32_t* candidates = nullptr;
    float* probabilities = nullptr;
    float* q = nullptr;
    std::uint8_t* admitted = nullptr;
    std::uint32_t* primal_words = nullptr;
    std::uint32_t* derivative_words = nullptr;
    std::uint32_t* status = nullptr; // device word, cleared by caller
};

// Host-side receipt of an enqueued forward. It binds shape, exact scalar bits,
// revisions and saved device buffers; it is not a synchronization primitive.
// Callers must advance the appropriate revision on any mutation, and retain
// score/probability/mask storage until all enqueued users complete.
struct SteSnapshot {
    GateRevisions revisions{};
    std::uint64_t cells = 0;
    std::uint64_t conditions = 0;
    float temperature = 0;
    const float* scores = nullptr;
    const float* probabilities = nullptr;
    const std::uint32_t* derivative_words = nullptr;
    std::uint32_t* status = nullptr;
    bool valid = false;
};

// Threshold equality admits. p and q are computed only for declared candidates;
// other entries are zero. Sets status_nonfinite for nonfinite candidate scores.
cudaError_t ste_forward(std::uint64_t cells, std::uint64_t conditions,
                        float temperature, float threshold,
                        const GateRevisions&, SteBuffers, SteSnapshot*,
                        cudaStream_t) noexcept;

// Uses saved p and the complete derivative plane, not the primal plane.
// For each candidate ds=dq*p*(1-p)/T, including an excluded forward gate.
// Rejects changed revisions or buffer identities before enqueueing work.
cudaError_t ste_backward(const SteSnapshot&, const GateRevisions& current,
                         const float* probabilities,
                         const std::uint32_t* derivative_words,
                         const float* dq, float* ds,
                         cudaStream_t) noexcept;

// Shared score parameters may sum the per-cell gradients per condition. The
// independent-cell case uses ds directly. Output is overwritten, not added.
cudaError_t ste_reduce_per_condition(std::uint64_t cells,
                                     std::uint64_t conditions,
                                     const float* ds, float* reduced,
                                     std::uint32_t* status,
                                     cudaStream_t) noexcept;

// In-place SGD with a separate device validation/commit. If any preceding
// phase or any proposed update sets snapshot.status, no parameter is written.
// Snapshot and its status word are required for nonempty updates, and the
// status must not be cleared until this entire chain completes. Invalidates
// snapshot on enqueue; caller increments the relevant revision before reuse.
cudaError_t ste_sgd_update(std::uint64_t count, float* parameters,
                           const float* gradients, float learning_rate,
                           SteSnapshot*, cudaStream_t) noexcept;

// Allocates its own temporary device buffers; call only in an admitted GPU job.
bool run_ste_tests(cudaStream_t stream, std::string* error);

} // namespace cgs
