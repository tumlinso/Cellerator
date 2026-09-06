#pragma once
#include <Cellerator/execution/identity.hh>
#include <cuda_runtime_api.h>
#include <cstdint>

namespace cellerator::runtime {
enum class relation_readiness_status : std::uint8_t {
    success, invalid_argument, invalid_state, stale_generation, identity_mismatch,
    device_mismatch, wrong_stream, busy, invalid_ticket, capture_unsupported,
    producer_enqueue_failed, cuda_failure, poisoned
};
// Cold injection seam for deterministic runtime-failure testing. Defaults call
// CUDA directly; callbacks and their state must outlive this component.
struct relation_event_api {
    cudaError_t (*record)(cudaEvent_t, cudaStream_t) = cudaEventRecord;
    cudaError_t (*wait)(cudaStream_t, cudaEvent_t, unsigned) = cudaStreamWaitEvent;
};
struct relation_read_ticket {
    execution::structure_id structure{};
    execution::structure_epoch epoch{};
    execution::value_generation generation{};
    std::uint64_t incarnation = 0, nonce = 0;
    int device = -1;
};
// Runtime-only component owned by the existing prepared pair. Host calls are
// externally serialized. No stream ownership and no historical value snapshots.
class relation_value_readiness {
public:
    relation_value_readiness() noexcept = default;
    ~relation_value_readiness() noexcept;
    relation_value_readiness(const relation_value_readiness&) = delete;
    relation_value_readiness& operator=(const relation_value_readiness&) = delete;
    relation_readiness_status initialize(execution::structure_id,
        execution::structure_epoch, int device, cudaStream_t owner,
        relation_event_api = {}) noexcept;
    // Validate BEFORE enqueueing any producer work. Initial publication uses
    // expected=0; subsequent mutations require the exact published generation.
    relation_readiness_status validate_write(execution::value_generation expected,
        execution::value_generation next, cudaStream_t owner) const noexcept;
    // Called after all producer submissions; failures poison because in-place
    // writes cannot roll back. The old numeric generation remains diagnostic.
    relation_readiness_status publish(execution::value_generation next,
        cudaStream_t owner, cudaError_t producer_enqueue_status) noexcept;
    // Producer-ready edge only. External borrowing must use begin/end_read;
    // a bare wait does not protect storage from a later writer.
    relation_readiness_status wait_current(execution::structure_id,
        execution::structure_epoch, execution::value_generation, int device,
        cudaStream_t consumer) noexcept;
    // Cold teardown fences owner work only. Never frees an unreturned borrow.
    relation_readiness_status close() noexcept;
    bool initialized() const noexcept { return ready_ != nullptr; }
    bool poisoned() const noexcept { return poisoned_; }
    bool active_reader() const noexcept { return active_nonce_ != 0; }
    execution::value_generation generation() const noexcept { return generation_; }
    std::uint64_t incarnation() const noexcept { return incarnation_; }
    std::uint64_t ready_records() const noexcept { return ready_records_; }
    std::uint64_t reader_returns() const noexcept { return reader_returns_; }
private:
    relation_readiness_status check_stream(cudaStream_t) const noexcept;
    cudaEvent_t ready_ = nullptr, done_ = nullptr;
    cudaStream_t owner_ = nullptr, consumer_ = nullptr;
    execution::structure_id structure_{};
    execution::structure_epoch epoch_{};
    execution::value_generation generation_{};
    std::uint64_t incarnation_ = 0, nonce_ = 0, active_nonce_ = 0;
    std::uint64_t ready_records_ = 0, reader_returns_ = 0;
    int device_ = -1;
    bool poisoned_ = false;
    relation_event_api api_{};
};
} // namespace cellerator::runtime
