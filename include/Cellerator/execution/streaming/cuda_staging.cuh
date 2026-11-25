#pragma once

#include <Cellerator/execution/streaming/fetch.hh>
#include <Cellerator/execution/streaming/placement.hh>
#include <Cellerator/memory/striped/StripedBuffer.hh>

namespace cellerator::execution::streaming {

struct CudaStaging;
struct DeviceTicket {
    const CudaStaging *owner = nullptr;
    uint64_t generation = 0;
    unsigned slot = 0;
};

struct StageEvents {
    memory::cuda::Event *ready = nullptr, *last_reader = nullptr;
    // Optional caller-provisioned timing-enabled pairs. Each brackets only
    // its copy. Read their timings once ready and before reusing the slot.
    memory::cuda::Event *early_begin = nullptr, *early_end = nullptr;
    memory::cuda::Event *deferred_begin = nullptr, *deferred_end = nullptr;
};

struct DeviceBatch {
    tensor::base::device::SlicedEllView view{};
    SourceInfo source{};
    uint64_t batch = 0;
    const layout::SlicedEllRow *row_map = nullptr;
    const memory::cuda::Event *ready = nullptr;
    PacketPlacement placement{};
};

struct StageReport {
    FetchState state = FetchState::stale;
    memory::status status = memory::status::invalid_argument;
    DeviceBatch batch{};
};

// The coordinator retains queue, pinned packet storage, arena, stream, and
// distinct events through drain. Source slots handed to stage must no
// longer be released by the caller. The caller still owns the row map.
struct CudaStaging {
    CudaStaging() noexcept = default;
    CudaStaging(const CudaStaging &) = delete;
    CudaStaging &operator=(const CudaStaging &) = delete;

    memory::status prepare(FetchQueue &queue, memory::StripedBuffer &arena,
                           const memory::cuda::Stream &copy, const StageEvents (&events)[2],
                           size_t minimum_early_bytes) noexcept;
    void set_minimum_early_bytes(size_t n) noexcept { minimum_early_ = n; }
    memory::status stage(FetchTicket packet, DeviceTicket *ticket) noexcept;
    // A ready view has a recorded copy event. Order all readers after it.
    StageReport poll(DeviceTicket ticket) noexcept;
    // Supply the stream ordered after the final GPU reader. A null stream
    // abandons a packet which has no GPU readers.
    memory::status release(DeviceTicket ticket, const memory::cuda::Stream *last_reader) noexcept;
    // Explicit cleanup waits for submitted copies and recorded uses. Release
    // any published view before draining. Packet queue drain remains separate.
    memory::status drain() noexcept;

private:
    struct Slot {
        bool live = false, high = false, ready_recorded = false, delivered = false;
        bool released = false, last_reader_recorded = false;
        uint64_t generation = 0, previous_generation = 0;
        FetchTicket host{};
        StageEvents events{};
        DeviceBatch batch{};
        const void *packet_data = nullptr;
        size_t packet_bytes = 0;
    };

    Slot *lookup(DeviceTicket ticket) noexcept;
    void upload(Slot &slot, size_t offset, size_t bytes, bool deferred);
    memory::status advance() noexcept;

    FetchQueue *queue_ = nullptr;
    memory::StripedBuffer *arena_ = nullptr;
    const memory::cuda::Stream *copy_ = nullptr;
    Slot slots_[2]{};
    uint64_t generation_ = 0;
    size_t minimum_early_ = 0, capacity_ = 0;
    bool high_next_ = false, failed_ = false;
};

} // namespace cellerator::execution::streaming
