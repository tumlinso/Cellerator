#include <Cellerator/runtime/relation_value_readiness.hh>
#include <atomic>
#include <limits>

namespace cellerator::runtime {
namespace {
using result = relation_readiness_status;
std::atomic<std::uint64_t> next_incarnation{1};
bool equal(execution::structure_id a, execution::structure_id b) noexcept {
    return a.low == b.low && a.high == b.high;
}
}
relation_value_readiness::~relation_value_readiness() noexcept {
    // Checked close is mandatory for borrowed storage; refusing cleanup is safer
    // than freeing events while an unreturned reader may still be submitted.
    (void)close();
}
result relation_value_readiness::initialize(execution::structure_id structure,
    execution::structure_epoch epoch, int device, cudaStream_t owner,
    relation_event_api api) noexcept {
    if (initialized()) return result::invalid_state;
    if ((!structure.low && !structure.high) || !epoch.value || device < 0 ||
        !api.record || !api.wait) return result::invalid_argument;
    cudaStreamCaptureStatus capture{};
    if (cudaStreamIsCapturing(owner, &capture) != cudaSuccess) return result::cuda_failure;
    if (capture != cudaStreamCaptureStatusNone) return result::capture_unsupported;
    int current = -1;
    if (cudaGetDevice(&current) != cudaSuccess) return result::cuda_failure;
    if (current != device) return result::device_mismatch;
    int stream_device = -1;
    if (cudaStreamGetDevice(owner, &stream_device) != cudaSuccess) return result::cuda_failure;
    if (stream_device != device) return result::device_mismatch;
    // Saturating allocation never wraps and never reuses an earlier lifetime.
    auto candidate = next_incarnation.load(std::memory_order_relaxed);
    do {
        if (candidate == std::numeric_limits<std::uint64_t>::max())
            return result::invalid_state;
    } while (!next_incarnation.compare_exchange_weak(candidate, candidate + 1,
        std::memory_order_relaxed));
    if (cudaEventCreateWithFlags(&ready_, cudaEventDisableTiming) != cudaSuccess)
        return result::cuda_failure;
    if (cudaEventCreateWithFlags(&done_, cudaEventDisableTiming) != cudaSuccess) {
        cudaEventDestroy(ready_); ready_ = nullptr;
        return result::cuda_failure;
    }
    structure_ = structure; epoch_ = epoch; device_ = device; owner_ = owner;
    api_ = api; incarnation_ = candidate;
    return result::success;
}
result relation_value_readiness::check_stream(cudaStream_t stream) const noexcept {
    if (!initialized()) return result::invalid_state;
    if (poisoned_) return result::poisoned;
    // Query capture first: other runtime queries may reject an active capture.
    cudaStreamCaptureStatus capture{};
    if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess) return result::cuda_failure;
    if (capture != cudaStreamCaptureStatusNone) return result::capture_unsupported;
    int current = -1;
    if (cudaGetDevice(&current) != cudaSuccess) return result::cuda_failure;
    if (current != device_) return result::device_mismatch;
    int stream_device = -1;
    if (cudaStreamGetDevice(stream, &stream_device) != cudaSuccess) return result::cuda_failure;
    if (stream_device != device_) return result::device_mismatch;
    return result::success;
}
result relation_value_readiness::validate_write(execution::value_generation expected,
    execution::value_generation next, cudaStream_t owner) const noexcept {
    if (owner != owner_) return result::wrong_stream;
    const auto checked = check_stream(owner);
    if (checked != result::success) return checked;
    if (active_reader()) return result::busy;
    if (expected.value != generation_.value || next.value == 0 ||
        next.value <= generation_.value) return result::stale_generation;
    return result::success;
}
result relation_value_readiness::publish(execution::value_generation next,
    cudaStream_t owner, cudaError_t producer_status) noexcept {
    const auto checked = validate_write(generation_, next, owner);
    if (checked != result::success) return checked;
    if (producer_status != cudaSuccess) {
        poisoned_ = true; return result::producer_enqueue_failed;
    }
    if (api_.record(ready_, owner) != cudaSuccess) {
        poisoned_ = true; return result::cuda_failure;
    }
    generation_ = next; ++ready_records_;
    return result::success;
}
result relation_value_readiness::wait_current(execution::structure_id structure,
    execution::structure_epoch epoch, execution::value_generation generation,
    int device, cudaStream_t consumer) noexcept {
    if (!equal(structure, structure_) || epoch.value != epoch_.value)
        return result::identity_mismatch;
    if (device != device_) return result::device_mismatch;
    const auto checked = check_stream(consumer);
    if (checked != result::success) return checked;
    if (!generation_.value) return result::invalid_state;
    if (!generation.value || generation.value != generation_.value)
        return result::stale_generation;
    if (consumer != owner_ && api_.wait(consumer, ready_, 0) != cudaSuccess) {
        poisoned_ = true; return result::cuda_failure;
    }
    return result::success;
}
result relation_value_readiness::close() noexcept {
    if (active_reader()) return result::busy;
    if (!initialized()) return result::success;
    int current = -1;
    if (cudaGetDevice(&current) != cudaSuccess) return result::cuda_failure;
    if (current != device_) return result::device_mismatch;
    cudaStreamCaptureStatus capture{};
    if (cudaStreamIsCapturing(owner_, &capture) != cudaSuccess) return result::cuda_failure;
    if (capture != cudaStreamCaptureStatusNone) return result::capture_unsupported;
    if (cudaStreamSynchronize(owner_) != cudaSuccess) {
        poisoned_ = true; return result::cuda_failure;
    }
    // Keep failed destruction handles available for a checked retry.
    if (done_ && cudaEventDestroy(done_) != cudaSuccess) return result::cuda_failure;
    done_ = nullptr;
    if (cudaEventDestroy(ready_) != cudaSuccess) return result::cuda_failure;
    ready_ = nullptr; owner_ = nullptr; consumer_ = nullptr;
    structure_ = {}; epoch_ = {}; generation_ = {};
    incarnation_ = 0; nonce_ = 0; active_nonce_ = 0;
    ready_records_ = 0; reader_returns_ = 0; device_ = -1; poisoned_ = false;
    return result::success;
}
} // namespace cellerator::runtime
