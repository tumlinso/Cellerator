#include <Cellerator/tensor/block_masked/owner.hh>

#include <cstring>
#include <limits>
#include <mutex>
#include <vector>

namespace cellerator::tensor::block_masked {
namespace {
std::size_t append_bytes(std::size_t& bytes, std::uint64_t count) {
    const auto size = memory::cuda::checked_bytes(count, sizeof(std::uint32_t));
    if (size > std::numeric_limits<std::size_t>::max() - bytes)
        throw std::overflow_error("block-masked metadata size overflow");
    const auto offset = bytes; bytes += size; return offset;
}
}

DeviceStructure::DeviceStructure(std::shared_ptr<const Structure> structure,
                                 const memory::cuda::Stream& stream)
    : host_(structure ? std::make_shared<const Structure>(*structure) : nullptr),
      metadata_(0, stream.device()) {
    if (!host_) throw std::invalid_argument("null block-masked host structure");
    view_ = host_->view();
    const auto n = view_;
    std::size_t bytes = 0;
    const auto tiles = append_bytes(bytes, n.tiles + 1);
    const auto blocks = append_bytes(bytes, n.tile_blocks);
    const auto rows = append_bytes(bytes, n.tile_blocks);
    const auto entries = append_bytes(bytes, n.tile_blocks + 1);
    const auto features = append_bytes(bytes, n.row_block_entries);
    const auto values = append_bytes(bytes, n.row_block_entries + 1);
    const auto feature_blocks = append_bytes(bytes, n.feature_blocks + 1);
    std::vector<unsigned char> staging(bytes);
    auto copy = [&](std::size_t offset, const std::uint32_t* source, std::uint64_t count) {
        if (count && !source) throw std::invalid_argument("missing block-masked metadata");
        if (count) std::memcpy(staging.data() + offset, source, count * sizeof(*source));
    };
    copy(tiles, n.tile_block_offsets, n.tiles + 1);
    copy(blocks, n.block_ids, n.tile_blocks);
    copy(rows, n.participating_row_masks, n.tile_blocks);
    copy(entries, n.block_row_entry_offsets, n.tile_blocks + 1);
    copy(features, n.feature_masks, n.row_block_entries);
    copy(values, n.row_block_value_offsets, n.row_block_entries + 1);
    copy(feature_blocks, n.feature_block_offsets, n.feature_blocks + 1);
    metadata_ = memory::cuda::DeviceBuffer(bytes, stream.device());
    metadata_.upload_async(staging.data(), bytes, stream);
    stream.synchronize(); // Explicit preparation boundary; staging can now retire.
    auto pointer = [&](std::size_t offset) {
        return reinterpret_cast<const std::uint32_t*>(
            static_cast<const unsigned char*>(metadata_.data()) + offset);
    };
    view_.tile_block_offsets = pointer(tiles);
    view_.block_ids = pointer(blocks);
    view_.participating_row_masks = pointer(rows);
    view_.block_row_entry_offsets = pointer(entries);
    view_.feature_masks = pointer(features);
    view_.row_block_value_offsets = pointer(values);
    view_.feature_block_offsets = pointer(feature_blocks);
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
