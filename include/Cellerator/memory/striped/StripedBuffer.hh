#pragma once

#include <Cellerator/memory/cuda/resources.hh>
#include <Cellerator/memory/status.hh>

#include <stddef.h>

namespace cellerator::memory {

// One fixed-capacity CUDA allocation. The caller places views at aligned
// byte offsets, orders transfers and computation, and reuses a range after
// its last asynchronous consumer. Binding a strip does not access its payload.
class StripedBuffer final {
public:
    explicit StripedBuffer(size_t bytes, int device = 0) : storage_(bytes, device) {}

    StripedBuffer(StripedBuffer &&) noexcept = default;
    StripedBuffer &operator=(StripedBuffer &&) noexcept = default;
    StripedBuffer(const StripedBuffer &) = delete;
    StripedBuffer &operator=(const StripedBuffer &) = delete;

    size_t bytes() const noexcept { return storage_.bytes(); }
    int device() const noexcept { return storage_.device(); }

    cuda::DeviceView view() noexcept { return storage_.view(); }
    cuda::ConstDeviceView view() const noexcept { return storage_.view(); }

    template<class Layout, class View>
        requires requires(const Layout &layout, void *base, View *out) {
            { layout.bytes() } noexcept;
            { layout.bind(base, size_t(0), out) } noexcept;
        }
    status view(size_t offset, const Layout &layout, View *out) noexcept {
        return bind(storage_.view(), offset, layout, out);
    }

    template<class Layout, class View>
        requires requires(const Layout &layout, const void *base, View *out) {
            { layout.bytes() } noexcept;
            { layout.bind(base, size_t(0), out) } noexcept;
        }
    status view(size_t offset, const Layout &layout, View *out) const noexcept {
        return bind(storage_.view(), offset, layout, out);
    }

    // Use pinned host storage for overlap and retain it through copy completion.
    void upload_async(const void *source, size_t bytes, const cuda::Stream &stream, size_t offset = 0) {
        storage_.upload_async(source, bytes, stream, offset);
    }

    void download_async(void *destination, size_t bytes, const cuda::Stream &stream, size_t offset = 0) const {
        storage_.download_async(destination, bytes, stream, offset);
    }

private:
    template<class Resource, class Layout, class View>
    static status bind(Resource storage, size_t offset, const Layout &layout, View *out) noexcept {
        if (offset > storage.bytes || layout.bytes() > storage.bytes - offset)
            return status::capacity_exceeded;
        return layout.bind(storage.subview(offset, layout.bytes()).data, layout.bytes(), out);
    }

    cuda::DeviceBuffer storage_;
};

} // namespace cellerator::memory
