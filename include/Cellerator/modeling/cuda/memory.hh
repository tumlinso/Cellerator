#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace cellerator::modeling::cuda {

// Borrowed device storage for lazy per-entity/per-channel event traces.
// values are anchored at anchor_time; tau is an explicit positive decay
// constant per channel. Entity indices are physical slots in the host layout.
template<class Storage>
struct EventMemoryView {
    Storage* values = nullptr;             // [entity_count, channel_count]
    std::size_t value_capacity = 0;
    float* anchor_time = nullptr;          // [entity_count]
    std::size_t anchor_capacity = 0;
    const float* time_constant = nullptr;  // [channel_count]
    std::size_t time_constant_count = 0;
    std::int64_t entity_count = 0;
    std::int64_t channel_count = 0;
};

// Apply sparse impulses in update-row order. update_entities contains unique
// physical entity slots; update_offsets delimits each entity's events. Events
// within a row must be sorted by nondecreasing absolute time, at/after that
// entity's anchor and no later than current_time. One CTA owns each updated
// entity, avoiding write races. State and event arrays are borrowed until the
// stream completes. Caller validates device-resident index/time contents.
cudaError_t apply_events_fp16(
    EventMemoryView<__half> memory, std::int64_t update_count,
    const std::int64_t* update_entities, const std::int64_t* update_offsets,
    const std::int64_t* event_channels, const float* event_amplitudes,
    const float* event_times, std::int64_t event_count, float current_time,
    cudaStream_t stream) noexcept;
cudaError_t apply_events_fp32(
    EventMemoryView<float> memory, std::int64_t update_count,
    const std::int64_t* update_entities, const std::int64_t* update_offsets,
    const std::int64_t* event_channels, const float* event_amplitudes,
    const float* event_times, std::int64_t event_count, float current_time,
    cudaStream_t stream) noexcept;

// Read selected rows at a later time without changing stored anchors. Output
// is row-major [read_count, channel_count]. Duplicate read slots are allowed.
cudaError_t read_traces_fp16(
    EventMemoryView<__half> memory, const std::int64_t* entity_slots,
    std::int64_t read_count, float read_time, __half* output,
    std::size_t output_capacity, cudaStream_t stream) noexcept;
cudaError_t read_traces_fp32(
    EventMemoryView<float> memory, const std::int64_t* entity_slots,
    std::int64_t read_count, float read_time, float* output,
    std::size_t output_capacity, cudaStream_t stream) noexcept;

}  // namespace cellerator::modeling::cuda
