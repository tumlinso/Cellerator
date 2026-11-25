#pragma once

#include <Cellerator/memory/tiled/Panel.hh>
#include <Cellerator/memory/tiled/TiledBuffer.hh>

namespace cellerator::memory {

// Bind each in-flight batch to disjoint pinned and device spans. Keep their
// owners bound to the original allocations until the last asynchronous use.
template<class T>
class PanelStagingSpan final {
public:
    PanelStagingSpan(TiledBuffer<T> &device, size_t first, size_t panels,
                     cuda::PinnedBuffer &host, size_t host_offset = 0)
        : device_(device), first_(first),
          device_range_(device.slot_bytes(first, cuda::checked_bytes(panels, panel_tiles))),
          host_(host_range(host, host_offset, device_range_.bytes)) {
        static_assert(sizeof(T) == 2);
    }

    // The caller waits for the previous copy before writing the pinned span.
    void *host_data() const noexcept { return host_; }
    size_t capacity() const noexcept { return device_range_.bytes / (panel_elements*sizeof(T)); }

    // Reuse requires a last-reader event (if any) already recorded. The compute
    // stream waits on ready before reading a view of this span. On a CUDA error
    // after a copy submission, drain the copy stream before reusing the spans.
    void upload_async(size_t panels, const cuda::Stream &copy, cuda::Event &ready,
                      const cuda::Event *last_reader = nullptr) const {
        check_count(panels);
        if (copy.device() != device_range_.device || ready.device() != copy.device())
            cuda::check(cudaErrorInvalidDevice, "panel staging stream/event device");
        if (last_reader) copy.wait(*last_reader);
        device_.upload_slots_async(host_, panels*panel_tiles, copy, first_);
        ready.record(copy);
    }

    const_array_view<T> view(size_t panels) const {
        check_count(panels);
        return {static_cast<const T *>(device_range_.data), panels*panel_elements,
                {domain::device, int16_t(device_range_.device)}};
    }

private:
    static void *host_range(cuda::PinnedBuffer &host, size_t offset, size_t bytes) {
        if (offset > host.bytes() || bytes > host.bytes() - offset || offset % alignof(T))
            cuda::check(cudaErrorInvalidValue, "panel staging host range");
        return bytes ? static_cast<unsigned char *>(host.data()) + offset : nullptr;
    }
    void check_count(size_t panels) const {
        if (panels > capacity()) cuda::check(cudaErrorInvalidValue, "panel staging extent");
    }

    TiledBuffer<T> &device_;
    size_t first_;
    cuda::DeviceView device_range_;
    void *host_;
};

} // namespace cellerator::memory
