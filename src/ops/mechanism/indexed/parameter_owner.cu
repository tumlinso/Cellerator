#include <Cellerator/ops/mechanism/indexed/training.hh>
#include "parameter_identity_validation.hh"
#include "indexed_cuda_support.cuh"

#include <cuda_fp16.h>

#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace cellerator::ops::mechanism::indexed {
using namespace detail;
namespace {
__global__ void refresh_half(const float* input, __half* output, std::uint64_t n) {
    for (auto i = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += std::uint64_t(blockDim.x) * gridDim.x)
        output[i] = __float2half_rn(input[i]);
}
} // namespace

struct mechanism_parameter_owner::implementation {
    indexed_axis axis;
    std::vector<identity> ids;
    session_storage storage;
    float* values;
    std::uint16_t* half;
    struct reader { std::unique_ptr<event> done; bool active = false, recorded = false; };
    std::vector<reader> readers;
    event ready;
    event failed_writer_done;
    std::uint64_t generation = 1;
    bool writing = false, poisoned = false;
    cudaStream_t writer_stream = nullptr;
    bool failed_writer_pending = false, failed_writer_recorded = false;
    mutable std::mutex mutex;
    implementation(indexed_axis a, std::vector<identity> i, int device) : axis(a), ids(std::move(i)), storage(device) {}
};
void mechanism_parameter_owner::poison_locked() noexcept {
    if (impl_->writing) {
        int previous = -1;
        const bool have_previous = cudaGetDevice(&previous) == cudaSuccess;
        bool device_ready = have_previous && previous == device();
        if (have_previous && previous != device())
            device_ready = cudaSetDevice(device()) == cudaSuccess;
        impl_->failed_writer_pending = true;
        impl_->failed_writer_recorded = device_ready &&
            cudaEventRecord(impl_->failed_writer_done.value, impl_->writer_stream) == cudaSuccess;
        if (have_previous && previous != device()) (void)cudaSetDevice(previous);
    }
    impl_->poisoned = true;
    impl_->writing = false;
    impl_->writer_stream = nullptr;
}
mechanism_parameter_owner::mechanism_parameter_owner(indexed_axis axis, std::vector<identity> ids,
        std::span<const float> initial, int device, std::uint32_t capacity) {
    device_scope scope(device);
    require(execution::validate_persistent_axis_identity(axis.identity) == execution::biological_validation_code::ok,
            "invalid coefficient axis");
    require(axis.extent && ids.size() == axis.extent && initial.size() == axis.extent && capacity, "invalid coefficient capacity");
    const auto input_validation = detail::validate_parameter_inputs(ids, initial);
    require(input_validation != detail::parameter_input_validation::invalid_identity_or_value,
            "invalid coefficient identity/value");
    require(input_validation != detail::parameter_input_validation::duplicate_identity,
            "duplicate coefficient identity");
    impl_ = std::make_unique<implementation>(axis,std::move(ids),device);
    auto bytes = checked_product(initial.size(),sizeof(float));
    impl_->values = static_cast<float*>(impl_->storage.reserve(bytes + checked_product(initial.size(),sizeof(std::uint16_t))));
    impl_->half = reinterpret_cast<std::uint16_t*>(impl_->values + initial.size());
    check(cudaMemcpy(impl_->values,initial.data(),bytes,cudaMemcpyHostToDevice));
    refresh_half<<<blocks(initial.size()),256>>>(impl_->values,reinterpret_cast<__half*>(impl_->half),initial.size());
    check(cudaGetLastError()); check(cudaEventRecord(impl_->ready.value)); check(cudaEventSynchronize(impl_->ready.value));
    impl_->readers.resize(capacity);
    for (auto& r : impl_->readers) r.done = std::make_unique<event>();
}
mechanism_parameter_owner::~mechanism_parameter_owner() = default;
float* mechanism_parameter_owner::data() const { return impl_->values; }
const std::uint16_t* mechanism_parameter_owner::half_data() const { return impl_->half; }
std::uint64_t mechanism_parameter_owner::generation() const {
    std::lock_guard lock(impl_->mutex);
    return impl_->generation;
}
std::uint64_t mechanism_parameter_owner::size() const { return impl_->axis.extent; }
int mechanism_parameter_owner::device() const { return impl_->storage.session.device; }
const indexed_axis& mechanism_parameter_owner::axis() const { return impl_->axis; }
const std::vector<identity>& mechanism_parameter_owner::logical_ids() const { return impl_->ids; }
bool mechanism_parameter_owner::poisoned() const {
    std::lock_guard lock(impl_->mutex);
    return impl_->poisoned;
}
void mechanism_parameter_owner::preflight_write() const {
    std::lock_guard lock(impl_->mutex);
    require(!impl_->poisoned && !impl_->writing, "coefficient owner poisoned or writer active");
    require(impl_->generation != std::numeric_limits<std::uint64_t>::max(), "coefficient generation exhausted");
    for (const auto& r : impl_->readers) require(!r.active, "unresolved mechanism forward prevents update");
}
void mechanism_parameter_owner::begin_write(void* stream_ptr) {
    std::lock_guard lock(impl_->mutex); device_scope scope(device());
    validate_stream_device(stream_ptr, device());
    require(!impl_->poisoned && !impl_->writing, "coefficient owner poisoned or writer active");
    require(impl_->generation != std::numeric_limits<std::uint64_t>::max(), "coefficient generation exhausted");
    for (const auto& r : impl_->readers) require(!r.active, "unresolved mechanism forward prevents update");
    auto stream = static_cast<cudaStream_t>(stream_ptr);
    check(cudaStreamWaitEvent(stream,impl_->ready.value));
    for (const auto& r : impl_->readers) if (r.recorded) check(cudaStreamWaitEvent(stream,r.done->value));
    impl_->writer_stream = stream;
    impl_->writing = true;
}
void mechanism_parameter_owner::publish_write(void* stream_ptr) {
    std::lock_guard lock(impl_->mutex); device_scope scope(device());
    require(impl_->writing && !impl_->poisoned, "no sanctioned coefficient writer");
    try {
        require(impl_->generation != std::numeric_limits<std::uint64_t>::max(), "coefficient generation exhausted");
        validate_stream_device(stream_ptr, device());
        auto stream = static_cast<cudaStream_t>(stream_ptr);
        require(stream == impl_->writer_stream, "parameter update must publish on its admission stream");
        refresh_half<<<blocks(size()),256,0,stream>>>(data(),reinterpret_cast<__half*>(impl_->half),size());
        check(cudaGetLastError()); check(cudaEventRecord(impl_->ready.value,stream));
        ++impl_->generation; impl_->writing = false; impl_->writer_stream = nullptr;
        impl_->failed_writer_pending = false; impl_->failed_writer_recorded = false;
    } catch (...) {
        const bool stream_mismatch = static_cast<cudaStream_t>(stream_ptr) != impl_->writer_stream;
        poison_locked();
        // The caller may have issued an unsupported update on the wrong
        // stream before publish detected it. Recovery then synchronizes the
        // whole device because that stream cannot be inferred safely.
        if (stream_mismatch) impl_->failed_writer_recorded = false;
        throw;
    }
}
void mechanism_parameter_owner::poison() noexcept {
    std::lock_guard lock(impl_->mutex);
    poison_locked();
}
void mechanism_parameter_owner::restore(std::span<const float> values, void* stream_ptr) {
    require(values.size() == size(), "checkpoint coefficient count mismatch");
    device_scope scope(device());
    validate_stream_device(stream_ptr, device());
    for (float value : values) require(std::isfinite(value), "nonfinite checkpoint coefficient");
    {
        std::lock_guard lock(impl_->mutex);
        // An admitted writer may still be modifying the master plane on its
        // stream. Restore cannot supersede it: only a poisoned writer has a
        // recorded completion event (or requires the cold device sync below).
        require(!impl_->writing, "checkpoint restoration cannot supersede an active writer");
        for (const auto& r : impl_->readers)
            require(!r.active, "checkpoint restoration has live tapes");
        require(impl_->generation != std::numeric_limits<std::uint64_t>::max(),
                "coefficient generation exhausted");
        auto stream = static_cast<cudaStream_t>(stream_ptr);
        if (impl_->failed_writer_pending) {
            if (impl_->failed_writer_recorded)
                check(cudaStreamWaitEvent(stream, impl_->failed_writer_done.value));
            else
                check(cudaDeviceSynchronize()); // Cold recovery when event recording itself failed.
        }
        check(cudaStreamWaitEvent(stream, impl_->ready.value));
        for (const auto& r : impl_->readers)
            if (r.recorded) check(cudaStreamWaitEvent(stream, r.done->value));
        impl_->writing = true;
        impl_->writer_stream = stream;
        impl_->poisoned = false;
        impl_->failed_writer_pending = false;
        impl_->failed_writer_recorded = false;
    }
    try {
        check(cudaMemcpyAsync(data(),values.data(),values.size_bytes(),cudaMemcpyHostToDevice,static_cast<cudaStream_t>(stream_ptr)));
        publish_write(stream_ptr);
        check(cudaEventSynchronize(impl_->ready.value)); // cold host checkpoint lifetime
    } catch (...) { poison(); throw; }
}
std::vector<float> mechanism_parameter_owner::snapshot(void* stream_ptr) const {
    std::lock_guard lock(impl_->mutex); device_scope scope(device());
    validate_stream_device(stream_ptr, device());
    require(!impl_->poisoned && !impl_->writing, "cannot checkpoint poisoned or updating owner");
    auto stream = static_cast<cudaStream_t>(stream_ptr);
    check(cudaStreamWaitEvent(stream,impl_->ready.value));
    std::vector<float> values(size());
    check(cudaMemcpyAsync(values.data(),data(),values.size()*sizeof(float),cudaMemcpyDeviceToHost,stream));
    check(cudaStreamSynchronize(stream)); return values;
}
std::uint32_t mechanism_parameter_owner::acquire_reader(void* stream_ptr) {
    std::lock_guard lock(impl_->mutex); device_scope scope(device());
    validate_stream_device(stream_ptr, device());
    require(!impl_->poisoned && !impl_->writing, "coefficient owner unavailable");
    auto stream = static_cast<cudaStream_t>(stream_ptr);
    for (std::uint32_t i = 0; i < impl_->readers.size(); ++i) {
        auto& r = impl_->readers[i]; if (r.active) continue;
        check(cudaStreamWaitEvent(stream,impl_->ready.value));
        if (r.recorded) check(cudaStreamWaitEvent(stream,r.done->value));
        r.active = true; return i;
    }
    throw std::runtime_error("coefficient reader capacity exceeded");
}
void mechanism_parameter_owner::complete_reader(std::uint32_t ticket, void* stream_ptr) noexcept {
    std::lock_guard lock(impl_->mutex);
    if (ticket >= impl_->readers.size() || !impl_->readers[ticket].active) {
        impl_->poisoned = true;
        return;
    }
    auto& r = impl_->readers[ticket];
    if (cudaEventRecord(r.done->value,static_cast<cudaStream_t>(stream_ptr)) != cudaSuccess) impl_->poisoned = true;
    r.recorded = true; r.active = false;
}


} // namespace cellerator::ops::mechanism::indexed
