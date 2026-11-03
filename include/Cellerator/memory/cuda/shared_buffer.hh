#pragma once

#include "resources.hh"
#include <Cellerator/memory/view.hh>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace cellerator::memory::cuda {

// Copyable ownership handle for device arrays that must be retained by more
// than one view or execution tape. The allocation itself is always a single
// canonical DeviceBuffer; this type only shares its lifetime.
template<class T>
class SharedDeviceBuffer final {
    static_assert(!std::is_const<T>::value && !std::is_volatile<T>::value,
                  "SharedDeviceBuffer owns mutable element storage");

public:
    using value_type = T;

    SharedDeviceBuffer() noexcept = default;

    SharedDeviceBuffer(const SharedDeviceBuffer&) = default;
    SharedDeviceBuffer& operator=(const SharedDeviceBuffer&) = default;

    SharedDeviceBuffer(SharedDeviceBuffer&& other) noexcept
        : owner_(std::move(other.owner_)), count_(std::exchange(other.count_, 0)) {}

    SharedDeviceBuffer& operator=(SharedDeviceBuffer&& other) noexcept {
        if (this != &other) {
            owner_ = std::move(other.owner_);
            count_ = std::exchange(other.count_, 0);
        }
        return *this;
    }

    SharedDeviceBuffer(std::size_t count, int device)
        : owner_(std::make_shared<DeviceBuffer>(checked_bytes(count, sizeof(T)), device)),
          count_(count) {}

    static SharedDeviceBuffer allocate(std::size_t count, int device) {
        return SharedDeviceBuffer(count, device);
    }

    T* data() noexcept {
        return owner_ ? static_cast<T*>(owner_->data()) : nullptr;
    }
    const T* data() const noexcept {
        return owner_ ? static_cast<const T*>(owner_->data()) : nullptr;
    }
    std::size_t size() const noexcept { return count_; }
    std::size_t count() const noexcept { return count_; }
    std::size_t bytes() const noexcept { return count_ * sizeof(T); }
    int device() const noexcept { return owner_ ? owner_->device() : -1; }
    bool empty() const noexcept { return count_ == 0; }

    DeviceView resource_view() noexcept {
        return owner_ ? owner_->view() : DeviceView{nullptr, 0, -1};
    }
    ConstDeviceView resource_view() const noexcept {
        return owner_ ? static_cast<const DeviceBuffer&>(*owner_).view()
                      : ConstDeviceView{nullptr, 0, -1};
    }
    ::cellerator::memory::array_view<T> view() noexcept {
        return {data(), count_, placement_for_device()};
    }
    ::cellerator::memory::const_array_view<T> view() const noexcept {
        return {data(), count_, placement_for_device()};
    }

    // Retaining this handle keeps the underlying allocation alive. Content
    // immutability and synchronization remain the responsibility of the
    // execution owner retaining it.
    std::shared_ptr<const DeviceBuffer> retained_owner() const noexcept {
        return owner_;
    }

    void zero_async(const Stream& stream) {
        require_owner_for_nonempty();
        if (owner_) owner_->zero_async(stream);
    }

    void upload_async(const T* source, std::size_t count, const Stream& stream,
                      std::size_t offset = 0) {
        require_range(count, offset);
        if (!count) return;
        require_owner_for_nonempty();
        owner_->upload_async(source, checked_bytes(count, sizeof(T)), stream,
                             checked_bytes(offset, sizeof(T)));
    }

    void download_async(T* destination, std::size_t count, const Stream& stream,
                        std::size_t offset = 0) const {
        require_range(count, offset);
        if (!count) return;
        require_owner_for_nonempty();
        owner_->download_async(destination, checked_bytes(count, sizeof(T)), stream,
                               checked_bytes(offset, sizeof(T)));
    }

private:
    ::cellerator::memory::placement placement_for_device() const noexcept {
        if (!owner_) return {};
        const auto ordinal = owner_->device();
        return {::cellerator::memory::domain::device,
                static_cast<std::int16_t>(ordinal), -1, 0};
    }

    void require_owner_for_nonempty() const {
        if (count_ && !owner_) throw std::logic_error("nonempty shared device buffer has no owner");
    }

    void require_range(std::size_t count, std::size_t offset) const {
        if (offset > count_ || count > count_ - offset)
            throw std::out_of_range("shared device buffer range exceeds allocation");
    }

    std::shared_ptr<DeviceBuffer> owner_;
    std::size_t count_ = 0;
};

} // namespace cellerator::memory::cuda
