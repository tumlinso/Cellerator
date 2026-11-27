#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace cellerator::runtime {

enum class value_readiness_status : std::uint8_t {
    success = 0,
    invalid_argument,
    invalid_state,
    stale_generation,
    device_mismatch,
    producer_enqueue_failed,
    cuda_failure
};

// Runtime-only readiness for one mutable value plane. This record owns its
// event, but never owns either stream. It is deliberately absent from the
// persistent biological ABI and from CPE2 serialization.
class value_readiness_record {
public:
    value_readiness_record() noexcept = default;
    ~value_readiness_record() noexcept;

    value_readiness_record(const value_readiness_record &) = delete;
    value_readiness_record &operator=(const value_readiness_record &) = delete;
    value_readiness_record(value_readiness_record &&) = delete;
    value_readiness_record &operator=(value_readiness_record &&) = delete;

    bool initialized() const noexcept { return event_ != nullptr; }
    bool published() const noexcept { return published_; }
    int device() const noexcept { return device_; }
    std::uint64_t structure_epoch() const noexcept { return structure_epoch_; }
    std::uint64_t generation() const noexcept { return generation_; }

private:
    friend value_readiness_status initialize_value_readiness(
        value_readiness_record *, int) noexcept;
    friend value_readiness_status clear_value_readiness(
        value_readiness_record *) noexcept;
    friend value_readiness_status publish_value_generation(
        value_readiness_record *, std::uint64_t, std::uint64_t,
        cudaStream_t, cudaError_t) noexcept;
    friend value_readiness_status wait_for_value_generation(
        const value_readiness_record &, std::uint64_t, std::uint64_t,
        cudaStream_t, int) noexcept;

    void reset() noexcept {
        event_ = nullptr;
        producer_stream_ = nullptr;
        structure_epoch_ = 0;
        generation_ = 0;
        device_ = -1;
        published_ = false;
    }

    cudaEvent_t event_ = nullptr;
    cudaStream_t producer_stream_ = nullptr;
    std::uint64_t structure_epoch_ = 0;
    std::uint64_t generation_ = 0;
    int device_ = -1;
    bool published_ = false;
};

// Initialization is preparation-time work and requires device to be current.
value_readiness_status initialize_value_readiness(
    value_readiness_record *record,
    int device) noexcept;

// Explicit cleanup reports CUDA errors. The destructor is a final idempotent
// fallback, but callers should clear while the owning execution session lives.
value_readiness_status clear_value_readiness(
    value_readiness_record *record) noexcept;

// Call only after attempting to enqueue all producer work. A failed producer
// status or failed event record leaves the previously published generation
// unchanged, so consumers can never observe a generation that was not queued.
value_readiness_status publish_value_generation(
    value_readiness_record *record,
    std::uint64_t structure_epoch,
    std::uint64_t generation,
    cudaStream_t producer_stream,
    cudaError_t producer_enqueue_status) noexcept;

// The same producer/consumer stream needs no CUDA call. A different stream gets
// an explicit wait on the published event. This function never synchronizes
// the host or the device.
value_readiness_status wait_for_value_generation(
    const value_readiness_record &record,
    std::uint64_t expected_structure_epoch,
    std::uint64_t expected_generation,
    cudaStream_t consumer_stream,
    int consumer_device) noexcept;

} // namespace cellerator::runtime
