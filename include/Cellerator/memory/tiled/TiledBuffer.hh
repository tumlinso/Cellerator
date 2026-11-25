#pragma once

#include <Cellerator/memory/cuda/resources.hh>
#include <Cellerator/memory/tiled/Tile.hh>
#include <Cellerator/memory/view.hh>

#include <stddef.h>
#include <stdint.h>

namespace cellerator::memory {

// Owns one fixed-capacity CUDA allocation of uninitialized tile slots.
// Views and transfers borrow storage. The caller keeps each range and its
// contents live through its last asynchronous use and orders writes and reads.
template<class T>
class TiledBuffer final {
    using TileType = Tile<T>;
    static constexpr size_t tile_bytes = sizeof(TileType);
    static_assert(tile_bytes == TileType::bytes);
    static_assert(__is_trivially_assignable(TileType &, const TileType &),
        "TiledBuffer requires mutable tile values with trivial assignment");

public:
    explicit TiledBuffer(size_t capacity, int device = 0)
        : storage_(cuda::checked_bytes(capacity, tile_bytes), checked_device(device)) {}

    TiledBuffer(TiledBuffer &&) noexcept = default;
    TiledBuffer &operator=(TiledBuffer &&) noexcept = default;
    TiledBuffer(const TiledBuffer &) = delete;
    TiledBuffer &operator=(const TiledBuffer &) = delete;

    size_t capacity() const noexcept { return storage_.bytes() / tile_bytes; }
    int device() const noexcept { return storage_.device(); }

    TileType *data() noexcept { return static_cast<TileType *>(storage_.data()); }
    const TileType *data() const noexcept { return static_cast<const TileType *>(storage_.data()); }

    array_view<TileType> view() noexcept { return {data(), capacity(), where()}; }
    const_array_view<TileType> view() const noexcept { return {data(), capacity(), where()}; }

    // A raw range can hold an interleaved panel that spans several slots.
    cuda::DeviceView slot_bytes(size_t first, size_t count) {
        return storage_.view().subview(cuda::checked_bytes(first, tile_bytes),
                                       cuda::checked_bytes(count, tile_bytes));
    }
    cuda::ConstDeviceView slot_bytes(size_t first, size_t count) const {
        return storage_.view().subview(cuda::checked_bytes(first, tile_bytes),
                                       cuda::checked_bytes(count, tile_bytes));
    }

    // Offsets and counts are measured in physical tile slots.
    array_view<TileType> view(size_t first, size_t count) {
        const auto range = slot_bytes(first, count);
        return {static_cast<TileType *>(range.data), count, where()};
    }
    const_array_view<TileType> view(size_t first, size_t count) const {
        const auto range = slot_bytes(first, count);
        return {static_cast<const TileType *>(range.data), count, where()};
    }

    // Host arrays remain live through transfer completion. Use pinned host
    // storage when the transfer must overlap other work.
    void upload_async(const TileType *source, size_t count, const cuda::Stream &stream,
                      size_t first = 0) {
        upload_slots_async(source, count, stream, first);
    }
    void upload_slots_async(const void *source, size_t count, const cuda::Stream &stream,
                            size_t first = 0) {
        storage_.upload_async(source, cuda::checked_bytes(count, tile_bytes), stream,
                              cuda::checked_bytes(first, tile_bytes));
    }
    void download_async(TileType *destination, size_t count, const cuda::Stream &stream,
                        size_t first = 0) const {
        storage_.download_async(destination, cuda::checked_bytes(count, tile_bytes), stream,
                                cuda::checked_bytes(first, tile_bytes));
    }

private:
    static int checked_device(int device) {
        // Borrowed views store the device ordinal in a signed 16-bit field.
        if (device < 0 || device > INT16_MAX)
            cuda::check(cudaErrorInvalidDevice, "TiledBuffer device ordinal");
        return device;
    }
    placement where() const noexcept { return {domain::device, int16_t(device()), -1, 0}; }

    cuda::DeviceBuffer storage_;
};

} // namespace cellerator::memory
