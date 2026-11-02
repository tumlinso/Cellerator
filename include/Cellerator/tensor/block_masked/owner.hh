#pragma once

#include <Cellerator/memory/cuda/resources.hh>
#include <Cellerator/tensor/block_masked/structure.hh>
#include <Cellerator/tensor/block_masked/view.hh>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>

namespace cellerator::tensor::block_masked {

// One immutable metadata replica. Upload is preparation-time work and completes
// on the supplied stream before construction returns. Values have another owner.
class DeviceStructure final {
public:
    DeviceStructure(std::shared_ptr<const Structure> structure,
                    const memory::cuda::Stream& stream);
    DeviceStructure(const DeviceStructure&) = delete;
    DeviceStructure& operator=(const DeviceStructure&) = delete;
    DeviceStructure(DeviceStructure&&) = delete;
    DeviceStructure& operator=(DeviceStructure&&) = delete;
    const Structure& host() const & noexcept { return *host_; }
    const Structure& host() const && = delete;
    LayoutView<> view() const & noexcept { return view_; }
    LayoutView<> view() const && = delete;
    int device() const noexcept { return metadata_.device(); }
    std::size_t metadata_bytes() const noexcept { return metadata_.bytes(); }
private:
    std::shared_ptr<const Structure> host_;
    memory::cuda::DeviceBuffer metadata_;
    LayoutView<> view_{};
};

namespace detail {
struct ValueStorage;
std::shared_ptr<ValueStorage> make_value_storage(memory::cuda::DeviceBuffer);
void* value_data(const std::shared_ptr<ValueStorage>&);
memory::cuda::DeviceView value_buffer(const std::shared_ptr<ValueStorage>&);
std::uint64_t value_generation(const std::shared_ptr<ValueStorage>&);
void check_readable(const std::shared_ptr<ValueStorage>&);
std::uint64_t acquire_read(const std::shared_ptr<ValueStorage>&, std::uint64_t expected);
void release_read(const std::shared_ptr<ValueStorage>&) noexcept;
void begin_write(const std::shared_ptr<ValueStorage>&);
void publish_write(const std::shared_ptr<ValueStorage>&, std::uint64_t generation);
void abandon_write(const std::shared_ptr<ValueStorage>&) noexcept;
}

template<class T> class NativeTensor;

// Pins actual contents as well as allocation lifetime. The execution/tape owner
// must retain this handle through the final asynchronous consumer completion.
// Destruction does not infer CUDA completion or synchronize a device.
template<class T>
class RetainedRead final {
public:
    RetainedRead(RetainedRead&& other) noexcept
        : structure_(std::move(other.structure_)), values_(std::move(other.values_)),
          generation_(other.generation_) {}
    RetainedRead& operator=(RetainedRead&& other) noexcept {
        if (this != &other) {
            reset(); structure_ = std::move(other.structure_);
            values_ = std::move(other.values_); generation_ = other.generation_;
        }
        return *this;
    }
    RetainedRead(const RetainedRead&) = delete;
    RetainedRead& operator=(const RetainedRead&) = delete;
    ~RetainedRead() { reset(); }
    View<const T> view() const & {
        if (!values_) throw std::logic_error("empty block-masked read lease");
        return {structure_->view(), static_cast<const T*>(detail::value_data(values_))};
    }
    View<const T> view() const && = delete;
    std::uint64_t generation() const noexcept { return generation_; }
    int device() const { return structure_->device(); }
    const DeviceStructure& structure() const & { return *structure_; }
    const DeviceStructure& structure() const && = delete;
private:
    friend class NativeTensor<T>;
    RetainedRead(std::shared_ptr<const DeviceStructure> structure,
                 std::shared_ptr<detail::ValueStorage> values, std::uint64_t expected)
        : structure_(std::move(structure)), values_(std::move(values)),
          generation_(detail::acquire_read(values_, expected)) {}
    void reset() noexcept {
        if (values_) detail::release_read(values_);
        values_.reset(); structure_.reset();
    }
    std::shared_ptr<const DeviceStructure> structure_;
    std::shared_ptr<detail::ValueStorage> values_;
    std::uint64_t generation_ = 0;
};

// Exclusive numeric edit. Publish only after the producer work has been
// enqueued; the caller separately publishes its CUDA readiness association.
// An abandoned edit poisons the owner because its contents may have changed.
template<class T>
class WriteLease final {
public:
    WriteLease(WriteLease&& other) noexcept
        : structure_(std::move(other.structure_)), values_(std::move(other.values_)),
          active_(other.active_) { other.active_ = false; }
    WriteLease& operator=(WriteLease&& other) noexcept {
        if (this != &other) {
            abandon(); structure_ = std::move(other.structure_);
            values_ = std::move(other.values_); active_ = other.active_;
            other.active_ = false;
        }
        return *this;
    }
    WriteLease(const WriteLease&) = delete;
    WriteLease& operator=(const WriteLease&) = delete;
    ~WriteLease() { abandon(); }
    View<T> view() & {
        require_active();
        return {structure_->view(), static_cast<T*>(detail::value_data(values_))};
    }
    View<T> view() && = delete;
    memory::cuda::DeviceView buffer() & { require_active(); return detail::value_buffer(values_); }
    memory::cuda::DeviceView buffer() && = delete;
    void publish(std::uint64_t generation) {
        require_active(); detail::publish_write(values_, generation); active_ = false;
    }
private:
    friend class NativeTensor<T>;
    WriteLease(std::shared_ptr<const DeviceStructure> structure,
               std::shared_ptr<detail::ValueStorage> values)
        : structure_(std::move(structure)), values_(std::move(values)) {
        detail::begin_write(values_); active_ = true;
    }
    void require_active() const {
        if (!active_) throw std::logic_error("inactive block-masked write lease");
    }
    void abandon() noexcept {
        if (active_) detail::abandon_write(values_);
        active_ = false;
    }
    std::shared_ptr<const DeviceStructure> structure_;
    std::shared_ptr<detail::ValueStorage> values_;
    bool active_ = false;
};

template<class T>
class NativeTensor final {
public:
    explicit NativeTensor(std::shared_ptr<const DeviceStructure> structure)
        : NativeTensor(structure, memory::cuda::DeviceBuffer(required_bytes(structure),
                                                            required_device(structure))) {}
    NativeTensor(std::shared_ptr<const DeviceStructure> structure,
                 memory::cuda::DeviceBuffer values)
        : structure_(std::move(structure)) {
        const auto bytes = required_bytes(structure_);
        if (values.device() != structure_->device() || values.bytes() < bytes)
            throw std::invalid_argument("block-masked values placement/capacity mismatch");
        if (values.data() && reinterpret_cast<std::uintptr_t>(values.data()) % alignof(T))
            throw std::invalid_argument("misaligned block-masked values");
        values_ = detail::make_value_storage(std::move(values));
    }
    NativeTensor(NativeTensor&&) noexcept = default;
    NativeTensor& operator=(NativeTensor&&) noexcept = default;
    NativeTensor(const NativeTensor&) = delete;
    NativeTensor& operator=(const NativeTensor&) = delete;
    View<const T> view() const & {
        detail::check_readable(values_);
        return {structure_->view(), static_cast<const T*>(detail::value_data(values_))};
    }
    View<const T> view() const && = delete;
    RetainedRead<T> retain(std::uint64_t expected_generation = 0) const & {
        return RetainedRead<T>(structure_, values_, expected_generation);
    }
    RetainedRead<T> retain(std::uint64_t = 0) const && = delete;
    WriteLease<T> edit() & { return WriteLease<T>(structure_, values_); }
    WriteLease<T> edit() && = delete;
    std::uint64_t generation() const { return detail::value_generation(values_); }
    int device() const { return required_device(structure_); }
    const DeviceStructure& structure() const & { return *structure_; }
    const DeviceStructure& structure() const && = delete;
private:
    static int required_device(const std::shared_ptr<const DeviceStructure>& structure) {
        if (!structure) throw std::invalid_argument("null block-masked structure");
        return structure->device();
    }
    static std::size_t required_bytes(const std::shared_ptr<const DeviceStructure>& structure) {
        required_device(structure);
        return memory::cuda::checked_bytes(structure->view().stored_values, sizeof(T));
    }
    std::shared_ptr<const DeviceStructure> structure_;
    std::shared_ptr<detail::ValueStorage> values_;
};

} // namespace cellerator::tensor::block_masked
