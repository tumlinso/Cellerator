#include <Cellerator/modeling/cuda/runtime.hh>
#include <limits>

namespace cellerator::modeling::cuda {
namespace {
void same_device(int expected, const Stream& stream) {
    if (expected != stream.device()) throw std::invalid_argument("stream/device mismatch");
}
void bounds(std::size_t size, std::size_t offset, std::size_t count) {
    if (offset > size || count > size - offset)
        throw std::out_of_range("buffer range exceeds allocation");
}
}
Error::Error(cudaError_t status, const char* operation)
    : std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status)), status_(status) {}
void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) throw Error(status, operation);
}
std::size_t checked_bytes(std::size_t count, std::size_t size) {
    if (size && count > std::numeric_limits<std::size_t>::max() / size)
        throw std::overflow_error("buffer size overflow");
    return count * size;
}
std::size_t item_size(DType dtype) {
    switch (dtype) { case DType::float16: return 2; case DType::float32: return 4; }
    throw std::invalid_argument("unsupported CUDA storage type");
}
DeviceGuard::DeviceGuard(int device) {
    if (device < 0) throw std::invalid_argument("device must be nonnegative");
    check(cudaGetDevice(&previous_), "cudaGetDevice");
    changed_ = previous_ != device;
    if (changed_) check(cudaSetDevice(device), "cudaSetDevice");
}
DeviceGuard::~DeviceGuard() noexcept { if (changed_) (void)cudaSetDevice(previous_); }
ConstDeviceView ConstDeviceView::subview(std::size_t offset, std::size_t count) const {
    bounds(bytes, offset, count);
    if (!data && (offset || count)) throw std::invalid_argument("null view");
    return {data ? static_cast<const char*>(data) + offset : nullptr, count, device};
}
DeviceView DeviceView::subview(std::size_t offset, std::size_t count) const {
    bounds(bytes, offset, count);
    if (!data && (offset || count)) throw std::invalid_argument("null view");
    return {data ? static_cast<char*>(data) + offset : nullptr, count, device};
}
void require_view(ConstDeviceView view, std::size_t bytes, int device, const char* description) {
    if (view.device != device || view.bytes < bytes || (bytes && !view.data))
        throw std::invalid_argument(std::string("invalid device view: ") + description);
}
Stream::Stream(int device, bool nonblocking) : device_(device), owned_(true) {
    DeviceGuard guard(device_);
    check(cudaStreamCreateWithFlags(&stream_, nonblocking ? cudaStreamNonBlocking : cudaStreamDefault), "cudaStreamCreateWithFlags");
}
Stream::Stream(cudaStream_t stream, int device, bool owned) : stream_(stream), device_(device), owned_(owned) {
    if (device < 0) throw std::invalid_argument("device must be nonnegative");
}
Stream Stream::borrow(cudaStream_t stream, int device) { return Stream(stream, device, false); }
Stream::~Stream() noexcept { release(); }
void Stream::release() noexcept {
    if (owned_) {
        int previous = 0;
        if (cudaGetDevice(&previous) == cudaSuccess) {
            if (cudaSetDevice(device_) == cudaSuccess) {
                (void)cudaStreamSynchronize(stream_);
                (void)cudaStreamDestroy(stream_);
            }
            (void)cudaSetDevice(previous);
        }
    }
    stream_ = nullptr; owned_ = false;
}
Stream::Stream(Stream&& other) noexcept : stream_(other.stream_), device_(other.device_), owned_(other.owned_) {
    other.stream_ = nullptr; other.owned_ = false;
}
Stream& Stream::operator=(Stream&& other) noexcept {
    if (this != &other) { release(); stream_ = other.stream_; device_ = other.device_; owned_ = other.owned_; other.stream_ = nullptr; other.owned_ = false; }
    return *this;
}
void Stream::synchronize() const { DeviceGuard guard(device_); check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize"); }
bool Stream::ready() const {
    DeviceGuard guard(device_); auto status = cudaStreamQuery(stream_);
    if (status == cudaErrorNotReady) return false;
    check(status, "cudaStreamQuery"); return true;
}
void Stream::wait(const Event& event) const {
    if (!event.recorded()) throw std::invalid_argument("cannot wait for an unrecorded event");
    DeviceGuard guard(device_); check(cudaStreamWaitEvent(stream_, event.get(), 0), "cudaStreamWaitEvent");
}
Event::Event(int device, bool timing) : device_(device), timing_(timing) {
    DeviceGuard guard(device_); check(cudaEventCreateWithFlags(&event_, timing ? cudaEventDefault : cudaEventDisableTiming), "cudaEventCreateWithFlags");
}
Event::~Event() noexcept { release(); }
void Event::release() noexcept {
    if (event_) { int previous = 0; if (cudaGetDevice(&previous) == cudaSuccess) { if (cudaSetDevice(device_) == cudaSuccess) (void)cudaEventDestroy(event_); (void)cudaSetDevice(previous); } }
    event_ = nullptr;
}
Event::Event(Event&& other) noexcept : event_(other.event_), device_(other.device_), timing_(other.timing_), recorded_(other.recorded_) { other.event_ = nullptr; other.recorded_ = false; }
Event& Event::operator=(Event&& other) noexcept {
    if (this != &other) { release(); event_ = other.event_; device_ = other.device_; timing_ = other.timing_; recorded_ = other.recorded_; other.event_ = nullptr; other.recorded_ = false; } return *this;
}
void Event::record(const Stream& stream) { same_device(device_, stream); DeviceGuard guard(device_); check(cudaEventRecord(event_, stream.get()), "cudaEventRecord"); recorded_ = true; }
void Event::synchronize() const { if (!recorded_) throw std::logic_error("event has not been recorded"); DeviceGuard guard(device_); check(cudaEventSynchronize(event_), "cudaEventSynchronize"); }
bool Event::ready() const { if (!recorded_) return false; DeviceGuard guard(device_); auto status = cudaEventQuery(event_); if (status == cudaErrorNotReady) return false; check(status, "cudaEventQuery"); return true; }
float Event::elapsed_ms(const Event& end) const {
    if (device_ != end.device_ || !timing_ || !end.timing_ || !recorded_ || !end.recorded_) throw std::invalid_argument("elapsed time requires recorded timing events on one device");
    DeviceGuard guard(device_); float value = 0; check(cudaEventElapsedTime(&value, event_, end.event_), "cudaEventElapsedTime"); return value;
}
DeviceBuffer::DeviceBuffer(std::size_t bytes, int device) : bytes_(bytes), device_(device) {
    if (device < 0) throw std::invalid_argument("device must be nonnegative");
    if (bytes) { DeviceGuard guard(device_); check(cudaMalloc(&data_, bytes), "cudaMalloc"); }
}
DeviceBuffer::~DeviceBuffer() noexcept { release(); }
void DeviceBuffer::release() noexcept {
    if (data_) { int previous = 0; if (cudaGetDevice(&previous) == cudaSuccess) { if (cudaSetDevice(device_) == cudaSuccess) (void)cudaFree(data_); (void)cudaSetDevice(previous); } }
    data_ = nullptr; bytes_ = 0;
}
DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept : data_(other.data_), bytes_(other.bytes_), device_(other.device_) { other.data_ = nullptr; other.bytes_ = 0; }
DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept { if (this != &other) { release(); data_ = other.data_; bytes_ = other.bytes_; device_ = other.device_; other.data_ = nullptr; other.bytes_ = 0; } return *this; }
void DeviceBuffer::zero_async(const Stream& stream) { same_device(device_, stream); if (!bytes_) return; DeviceGuard guard(device_); check(cudaMemsetAsync(data_, 0, bytes_, stream.get()), "cudaMemsetAsync"); }
void DeviceBuffer::copy_from_async(ConstDeviceView source, const Stream& stream) {
    same_device(device_, stream); require_view(source, bytes_, device_, "copy source"); if (!bytes_) return;
    DeviceGuard guard(device_); check(cudaMemcpyAsync(data_, source.data, bytes_, cudaMemcpyDeviceToDevice, stream.get()), "cudaMemcpyAsync D2D");
}
void DeviceBuffer::upload_async(const void* source, std::size_t bytes, const Stream& stream, std::size_t offset) {
    same_device(device_, stream); bounds(bytes_, offset, bytes); if (!bytes) return;
    if (!source) throw std::invalid_argument("null host upload source");
    DeviceGuard guard(device_); check(cudaMemcpyAsync(static_cast<char*>(data_) + offset, source, bytes, cudaMemcpyHostToDevice, stream.get()), "cudaMemcpyAsync H2D");
}
void DeviceBuffer::download_async(void* destination, std::size_t bytes, const Stream& stream, std::size_t offset) const {
    same_device(device_, stream); bounds(bytes_, offset, bytes); if (!bytes) return;
    if (!destination) throw std::invalid_argument("null host download destination");
    DeviceGuard guard(device_); check(cudaMemcpyAsync(destination, static_cast<const char*>(data_) + offset, bytes, cudaMemcpyDeviceToHost, stream.get()), "cudaMemcpyAsync D2H");
}
PinnedBuffer::PinnedBuffer(std::size_t bytes) : bytes_(bytes) { if (bytes) check(cudaHostAlloc(&data_, bytes, cudaHostAllocPortable), "cudaHostAlloc"); }
PinnedBuffer::~PinnedBuffer() noexcept { if (data_) (void)cudaFreeHost(data_); }
PinnedBuffer::PinnedBuffer(PinnedBuffer&& other) noexcept : data_(other.data_), bytes_(other.bytes_) { other.data_ = nullptr; other.bytes_ = 0; }
PinnedBuffer& PinnedBuffer::operator=(PinnedBuffer&& other) noexcept { if (this != &other) { if (data_) (void)cudaFreeHost(data_); data_ = other.data_; bytes_ = other.bytes_; other.data_ = nullptr; other.bytes_ = 0; } return *this; }
} // namespace cellerator::modeling::cuda
