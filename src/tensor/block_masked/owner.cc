#include <Cellerator/tensor/block_masked/owner.hh>

#include <limits>
#include <mutex>

namespace cellerator::tensor::block_masked {

DeviceStructure::DeviceStructure(std::shared_ptr<const Structure> structure,
                                 const memory::cuda::Stream& stream)
    : host_(static_cast<decltype(structure)&&>(structure)),
      metadata_(0, stream.device()) {
    if (!host_) throw std::invalid_argument("null block-masked host structure");
    view_ = host_->view();
    const auto bytes = host_->metadata_bytes();
    const auto* source = static_cast<const unsigned char*>(host_->metadata_data());
    metadata_ = memory::cuda::DeviceBuffer(bytes, stream.device());
    metadata_.upload_async(source, bytes, stream);
    stream.synchronize();
    const auto* destination = static_cast<const unsigned char*>(metadata_.data());
    const auto rebase = [&](const uint32_t* host_pointer) {
        const auto offset = reinterpret_cast<const unsigned char*>(host_pointer) - source;
        return reinterpret_cast<const uint32_t*>(destination + offset);
    };
    view_.tile_block_offsets = rebase(view_.tile_block_offsets);
    view_.block_ids = rebase(view_.block_ids);
    view_.participating_row_masks = rebase(view_.participating_row_masks);
    view_.block_row_entry_offsets = rebase(view_.block_row_entry_offsets);
    view_.feature_masks = rebase(view_.feature_masks);
    view_.row_block_value_offsets = rebase(view_.row_block_value_offsets);
    view_.feature_block_offsets = rebase(view_.feature_block_offsets);
}

namespace detail {
struct ValueStorage {
    explicit ValueStorage(memory::cuda::DeviceBuffer buffer) : buffer(std::move(buffer)) {}
    memory::cuda::DeviceBuffer buffer;
    std::mutex mutex;
    std::uint64_t generation = 0, readers = 0;
    bool writing = false, poisoned = false;
};
namespace {
ValueStorage& require(const std::shared_ptr<ValueStorage>& values) {
    if (!values) throw std::logic_error("empty block-masked values owner");
    return *values;
}
void readable(const ValueStorage& values) {
    if (values.poisoned) throw std::logic_error("poisoned block-masked values owner");
    if (values.writing || !values.generation)
        throw std::logic_error("unpublished block-masked values");
}
}
std::shared_ptr<ValueStorage> make_value_storage(memory::cuda::DeviceBuffer buffer) {
    return std::make_shared<ValueStorage>(std::move(buffer));
}
void* value_data(const std::shared_ptr<ValueStorage>& values) { return require(values).buffer.data(); }
memory::cuda::DeviceView value_buffer(const std::shared_ptr<ValueStorage>& values) {
    return require(values).buffer.view();
}
std::uint64_t value_generation(const std::shared_ptr<ValueStorage>& values) {
    auto& v = require(values); std::lock_guard lock(v.mutex); return v.generation;
}
void check_readable(const std::shared_ptr<ValueStorage>& values) {
    auto& v = require(values); std::lock_guard lock(v.mutex); readable(v);
}
std::uint64_t acquire_read(const std::shared_ptr<ValueStorage>& values, std::uint64_t expected) {
    auto& v = require(values); std::lock_guard lock(v.mutex); readable(v);
    if (expected && v.generation != expected)
        throw std::invalid_argument("stale block-masked value generation");
    if (v.readers == std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error("block-masked reader count overflow");
    ++v.readers; return v.generation;
}
void release_read(const std::shared_ptr<ValueStorage>& values) noexcept {
    if (!values) return;
    std::lock_guard lock(values->mutex);
    if (!values->readers) values->poisoned = true;
    else --values->readers;
}
void begin_write(const std::shared_ptr<ValueStorage>& values) {
    auto& v = require(values); std::lock_guard lock(v.mutex);
    if (v.poisoned || v.writing || v.readers)
        throw std::logic_error("block-masked write conflicts with live readers/writer");
    v.writing = true;
}
void publish_write(const std::shared_ptr<ValueStorage>& values, std::uint64_t generation) {
    auto& v = require(values); std::lock_guard lock(v.mutex);
    if (!v.writing || v.poisoned || !generation || generation <= v.generation)
        throw std::invalid_argument("invalid block-masked value publication");
    v.generation = generation; v.writing = false;
}
void abandon_write(const std::shared_ptr<ValueStorage>& values) noexcept {
    if (!values) return;
    std::lock_guard lock(values->mutex); values->poisoned = true; values->writing = false;
}
} // namespace detail
} // namespace cellerator::tensor::block_masked
