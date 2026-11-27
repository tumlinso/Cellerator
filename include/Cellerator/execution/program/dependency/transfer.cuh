#pragma once

#include <Cellerator/memory/cuda/resources.hh>

#include <cstddef>

namespace cellerator::execution::dependency {

enum class TransferMode {
    // Always copy through a prepared pinned host buffer.
    staged,
    // Use peer access when the destination can directly access the source;
    // otherwise prepare a pinned host path.
    peer_if_supported,
    // Require direct peer access from the destination to the source.
    peer_required
};

// A reusable cross-device copy path with fixed endpoints and byte capacity.
// Preparation allocates any pinned staging storage and selects/enables the
// transfer path. enqueue() performs no allocation or peer-capability query.
//
// The source and destination views are borrowed. Their allocations and
// contents, and both supplied streams, must remain valid until completion()
// is ready. completion() marks copy completion only; destination consumers
// queued after that event need their own completion/lifetime tracking.
//
// A transfer object supports one in-flight copy at a time and is not
// thread-safe. Reuse is allowed after the prior copy completion event is
// ready. CUDA peer access enabled during preparation is process/device state
// and is intentionally not disabled when this object is destroyed.
class PreparedTransfer final {
public:
    PreparedTransfer(int source_device, int destination_device,
                     std::size_t capacity, TransferMode mode);
    ~PreparedTransfer() noexcept;

    PreparedTransfer(const PreparedTransfer&) = delete;
    PreparedTransfer& operator=(const PreparedTransfer&) = delete;
    PreparedTransfer(PreparedTransfer&&) = delete;
    PreparedTransfer& operator=(PreparedTransfer&&) = delete;

    void enqueue(memory::cuda::ConstDeviceView source,
                 memory::cuda::DeviceView destination,
                 std::size_t bytes,
                 const memory::cuda::Stream& source_stream,
                 const memory::cuda::Stream& destination_stream);

    // A zero-byte enqueue only records the completion event and permits null
    // data pointers. The returned event is valid after a successful enqueue;
    // a later enqueue re-records it. It marks transfer completion, not later
    // work on the destination stream.
    const memory::cuda::Event& completion() const;
    bool ready() const;
    void wait();

    bool uses_peer() const noexcept { return uses_peer_; }
    std::size_t capacity() const noexcept { return capacity_; }
    int source_device() const noexcept { return source_device_; }
    int destination_device() const noexcept { return destination_device_; }

private:
    void drain_last_streams() const noexcept;

    int source_device_;
    int destination_device_;
    std::size_t capacity_;
    bool uses_peer_;
    memory::cuda::PinnedBuffer staging_;
    memory::cuda::Event source_complete_;
    memory::cuda::Event completion_;

    bool completion_valid_ = false;
    bool last_streams_valid_ = false;
    cudaStream_t last_source_stream_ = nullptr;
    cudaStream_t last_destination_stream_ = nullptr;
};

} // namespace cellerator::execution::dependency
