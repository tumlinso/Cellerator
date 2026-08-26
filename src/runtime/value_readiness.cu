#include <Cellerator/runtime/value_readiness.cuh>

namespace cellerator::runtime {
namespace {

value_readiness_status current_device(int *device) noexcept {
    if (device == nullptr)
        return value_readiness_status::invalid_argument;
    return cudaGetDevice(device) == cudaSuccess
        ? value_readiness_status::success
        : value_readiness_status::cuda_failure;
}

} // namespace

value_readiness_record::~value_readiness_record() noexcept {
    if (event_ != nullptr)
        cudaEventDestroy(event_);
}

value_readiness_status initialize_value_readiness(
    value_readiness_record *record,
    int device) noexcept {
    if (record == nullptr || device < 0)
        return value_readiness_status::invalid_argument;
    if (record->event_ != nullptr)
        return value_readiness_status::invalid_state;
    int active_device = -1;
    const value_readiness_status device_status = current_device(&active_device);
    if (device_status != value_readiness_status::success)
        return device_status;
    if (active_device != device)
        return value_readiness_status::device_mismatch;

    cudaEvent_t event = nullptr;
    if (cudaEventCreateWithFlags(&event, cudaEventDisableTiming) != cudaSuccess)
        return value_readiness_status::cuda_failure;
    record->reset();
    record->event_ = event;
    record->device_ = device;
    return value_readiness_status::success;
}

value_readiness_status clear_value_readiness(
    value_readiness_record *record) noexcept {
    if (record == nullptr)
        return value_readiness_status::invalid_argument;
    if (record->event_ == nullptr) {
        record->reset();
        return value_readiness_status::success;
    }
    int active_device = -1;
    const value_readiness_status device_status = current_device(&active_device);
    if (device_status != value_readiness_status::success)
        return device_status;
    if (active_device != record->device_)
        return value_readiness_status::device_mismatch;
    if (cudaEventDestroy(record->event_) != cudaSuccess)
        return value_readiness_status::cuda_failure;
    record->reset();
    return value_readiness_status::success;
}

value_readiness_status publish_value_generation(
    value_readiness_record *record,
    std::uint64_t structure_epoch,
    std::uint64_t generation,
    cudaStream_t producer_stream,
    cudaError_t producer_enqueue_status) noexcept {
    if (record == nullptr || structure_epoch == 0 || generation == 0)
        return value_readiness_status::invalid_argument;
    if (record->event_ == nullptr)
        return value_readiness_status::invalid_state;
    if (producer_enqueue_status != cudaSuccess)
        return value_readiness_status::producer_enqueue_failed;
    if (record->published_
        && (structure_epoch < record->structure_epoch_
            || (structure_epoch == record->structure_epoch_
                && generation <= record->generation_)))
        return value_readiness_status::stale_generation;

    int active_device = -1;
    const value_readiness_status device_status = current_device(&active_device);
    if (device_status != value_readiness_status::success)
        return device_status;
    if (active_device != record->device_)
        return value_readiness_status::device_mismatch;
    if (cudaEventRecord(record->event_, producer_stream) != cudaSuccess)
        return value_readiness_status::cuda_failure;

    record->producer_stream_ = producer_stream;
    record->structure_epoch_ = structure_epoch;
    record->generation_ = generation;
    record->published_ = true;
    return value_readiness_status::success;
}

value_readiness_status wait_for_value_generation(
    const value_readiness_record &record,
    std::uint64_t expected_structure_epoch,
    std::uint64_t expected_generation,
    cudaStream_t consumer_stream,
    int consumer_device) noexcept {
    if (expected_structure_epoch == 0 || expected_generation == 0
        || consumer_device < 0)
        return value_readiness_status::invalid_argument;
    if (record.event_ == nullptr || !record.published_)
        return value_readiness_status::invalid_state;
    if (consumer_device != record.device_)
        return value_readiness_status::device_mismatch;
    if (expected_structure_epoch != record.structure_epoch_
        || expected_generation != record.generation_)
        return value_readiness_status::stale_generation;
    if (consumer_stream == record.producer_stream_)
        return value_readiness_status::success;
    return cudaStreamWaitEvent(consumer_stream, record.event_, 0) == cudaSuccess
        ? value_readiness_status::success
        : value_readiness_status::cuda_failure;
}

} // namespace cellerator::runtime
