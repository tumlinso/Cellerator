#pragma once

#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace cellerator::modeling::cuda {

class Error : public std::runtime_error {
public:
    Error(cudaError_t status, const char* operation);
    cudaError_t status() const noexcept { return status_; }
private:
    cudaError_t status_;
};
void check(cudaError_t status, const char* operation);
std::size_t checked_bytes(std::size_t count, std::size_t item_size);

enum class DType { float16, float32 };
std::size_t item_size(DType dtype);

// Restores the thread's previous CUDA device. Construction may initialize CUDA.
class DeviceGuard final {
public:
    explicit DeviceGuard(int device);
    ~DeviceGuard() noexcept;
    DeviceGuard(const DeviceGuard&) = delete;
    DeviceGuard& operator=(const DeviceGuard&) = delete;
private:
    int previous_ = -1;
    bool changed_ = false;
};

// Borrowed views never free memory. Callers retain the owner until all uses
// complete. bytes describes the accessible extent, not a logical shape.
struct ConstDeviceView {
    const void* data = nullptr;
    std::size_t bytes = 0;
    int device = 0;
    ConstDeviceView subview(std::size_t offset, std::size_t count) const;
};
struct DeviceView {
    void* data = nullptr;
    std::size_t bytes = 0;
    int device = 0;
    DeviceView subview(std::size_t offset, std::size_t count) const;
    operator ConstDeviceView() const noexcept { return {data, bytes, device}; }
};
void require_view(ConstDeviceView view, std::size_t bytes, int device,
                  const char* description);

class Event;
class Stream final {
public:
    explicit Stream(int device = 0, bool nonblocking = true);
    // Borrowed streams are never destroyed. Zero denotes the default stream.
    static Stream borrow(cudaStream_t stream, int device);
    ~Stream() noexcept;
    Stream(Stream&& other) noexcept;
    Stream& operator=(Stream&& other) noexcept;
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    cudaStream_t get() const noexcept { return stream_; }
    int device() const noexcept { return device_; }
    bool owns_handle() const noexcept { return owned_; }
    void synchronize() const;
    bool ready() const;
    void wait(const Event& event) const;
private:
    Stream(cudaStream_t stream, int device, bool owned);
    void release() noexcept;
    cudaStream_t stream_ = nullptr;
    int device_ = 0;
    bool owned_ = false;
};

class Event final {
public:
    explicit Event(int device = 0, bool timing = false);
    ~Event() noexcept;
    Event(Event&& other) noexcept;
    Event& operator=(Event&& other) noexcept;
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    void record(const Stream& stream);
    void synchronize() const;
    bool ready() const;
    float elapsed_ms(const Event& end) const;
    cudaEvent_t get() const noexcept { return event_; }
    int device() const noexcept { return device_; }
    bool recorded() const noexcept { return recorded_; }
private:
    void release() noexcept;
    cudaEvent_t event_ = nullptr;
    int device_ = 0;
    bool timing_ = false;
    bool recorded_ = false;
};

class DeviceBuffer final {
public:
    explicit DeviceBuffer(std::size_t bytes = 0, int device = 0);
    ~DeviceBuffer() noexcept;
    DeviceBuffer(DeviceBuffer&& other) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceView view() noexcept { return {data_, bytes_, device_}; }
    ConstDeviceView view() const noexcept { return {data_, bytes_, device_}; }
    void* data() noexcept { return data_; }
    const void* data() const noexcept { return data_; }
    std::size_t bytes() const noexcept { return bytes_; }
    int device() const noexcept { return device_; }
    void zero_async(const Stream& stream);
    void copy_from_async(ConstDeviceView source, const Stream& stream);
    // Host memory must remain live until stream completion. Pinned memory is
    // required for dependable asynchronous host transfers.
    void upload_async(const void* source, std::size_t bytes, const Stream& stream,
                      std::size_t offset = 0);
    void download_async(void* destination, std::size_t bytes, const Stream& stream,
                        std::size_t offset = 0) const;
private:
    void release() noexcept;
    void* data_ = nullptr;
    std::size_t bytes_ = 0;
    int device_ = 0;
};

class PinnedBuffer final {
public:
    explicit PinnedBuffer(std::size_t bytes = 0);
    ~PinnedBuffer() noexcept;
    PinnedBuffer(PinnedBuffer&& other) noexcept;
    PinnedBuffer& operator=(PinnedBuffer&& other) noexcept;
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    void* data() noexcept { return data_; }
    const void* data() const noexcept { return data_; }
    std::size_t bytes() const noexcept { return bytes_; }
private:
    void* data_ = nullptr;
    std::size_t bytes_ = 0;
};

} // namespace cellerator::modeling::cuda
