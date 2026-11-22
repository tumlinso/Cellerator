#include "Cellerator/tensor/block_masked/structure.hh"

#include "Cellerator/tensor/block_masked/view.hh"

namespace cellerator::tensor::block_masked {
namespace {

// The existing throwing host API reports checked preparation and lookup errors.
void require_extent(bool condition) {
    if (!condition) {
        throw std::overflow_error("block-masked offset exceeds the selected offset type");
    }
}

void require_support(bool condition, const char* description) {
    if (!condition) {
        throw std::invalid_argument(description);
    }
}

uint64_t group_count(uint64_t items, uint32_t width) noexcept {
    return items / width + (items % width != 0);
}

template<class T>
size_t append(size_t& bytes, uint64_t count) {
    const auto mask = alignof(T) - 1;
    require_extent(bytes <= SIZE_MAX - mask);
    const auto start = (bytes + mask) & ~mask;
    require_extent(count <= (SIZE_MAX - start) / sizeof(T));
    bytes = start + static_cast<size_t>(count) * sizeof(T);
    return start;
}

void check_ids(memory::const_array_view<uint64_t> ids, uint64_t required) {
    require_support(!ids.count || (ids.count == required && ids.data),
        "canonical coordinate map length must match its physical extent");
    require_support(!ids.count || ids.where.kind != memory::domain::device,
        "canonical coordinate maps must be host-readable");
}

struct ScratchSizes {
    size_t keys;
    size_t temporary;
    size_t sorted_rows;
};

ScratchSizes scratch_sizes(const StructureInput& input) {
    require_support(input.block_width != 0 && input.block_width <= 32,
        "block_width must be in [1, 32]");
    require_support(input.tile_row_width != 0 && input.tile_row_width <= 32,
        "tile_row_width must be in [1, 32]");

    const auto support = input.support;
    require_extent(input.features <= UINT32_MAX && support.entries <= UINT32_MAX &&
        support.rows < SIZE_MAX / sizeof(uint64_t));
    check_ids(input.canonical_features, input.features);
    check_ids(input.canonical_rows, support.rows);
    require_support((!support.rows || support.row_offsets) &&
        (!support.entries || (support.physical_features && support.rows)),
        "missing block-masked support arrays");

    uint64_t largest_tile = 0;
    if (support.row_offsets) {
        require_support(support.row_offsets[0] == 0 && support.row_offsets[support.rows] == support.entries,
            "support offsets must begin at zero and terminate at the entry count");
        uint64_t tile_begin = 0, previous = 0;
        for (uint64_t row = 1; row <= support.rows; ++row) {
            const auto end = support.row_offsets[row];
            require_support(end >= previous && end <= support.entries,
                "support offsets must be monotonic and within the entry count");
            if (row % input.tile_row_width == 0 || row == support.rows) {
                const auto tile_entries = end - tile_begin;
                if (tile_entries > largest_tile) largest_tile = tile_entries;
                tile_begin = end;
            }
            previous = end;
        }
    }

    // Canonical ID validation and tile sorting reuse the same buffers.
    auto keys = support.entries;
    auto temporary = largest_tile;
    const auto features = input.canonical_features.count;
    const auto rows = input.canonical_rows.count;
    const auto ids = features > rows ? features : rows;
    if (ids > keys) keys = ids;
    if (ids > temporary) temporary = ids;
    constexpr auto capacity = SIZE_MAX / sizeof(uint64_t);
    require_extent(keys <= capacity && temporary <= capacity - keys &&
        rows <= capacity - keys - temporary);
    return {static_cast<size_t>(keys), static_cast<size_t>(temporary), rows};
}

void sort_keys(uint64_t* keys, uint64_t* temporary, size_t count) noexcept {
    if (count < 2) return;
    if (count < 32) {
        for (size_t i = 1; i < count; ++i) {
            const auto value = keys[i];
            auto j = i;
            while (j && keys[j - 1] > value) {
                keys[j] = keys[j - 1];
                --j;
            }
            keys[j] = value;
        }
        return;
    }

    bool sorted = true;
    uint64_t differing_bits = 0;
    for (size_t i = 1; i < count; ++i) {
        sorted &= keys[i - 1] <= keys[i];
        differing_bits |= keys[i] ^ keys[0];
    }
    if (sorted) return;

    auto* from = keys;
    auto* to = temporary;
    for (unsigned shift = 0; shift < 64; shift += 8) {
        if (((differing_bits >> shift) & 255u) == 0) continue;
        size_t histogram[256]{};
        for (size_t i = 0; i < count; ++i) ++histogram[(from[i] >> shift) & 255u];
        size_t prefix = 0;
        for (auto& slot : histogram) {
            const auto next = prefix + slot;
            slot = prefix;
            prefix = next;
        }
        for (size_t i = 0; i < count; ++i) {
            const auto key = from[i];
            to[histogram[(key >> shift) & 255u]++] = key;
        }
        auto* swap = from;
        from = to;
        to = swap;
    }
    if (from != keys) __builtin_memcpy(keys, from, count * sizeof(uint64_t));
}

bool validate_ids(memory::const_array_view<uint64_t> ids, uint64_t* keys, uint64_t* temporary) {
    bool ordered = true, identity = true;
    for (size_t i = 0; i < ids.count; ++i) {
        identity &= ids.data[i] == i;
        if (i) {
            require_support(ids.data[i] != ids.data[i - 1], "canonical coordinates must be unique");
            ordered &= ids.data[i] > ids.data[i - 1];
        }
    }
    if (ordered) return identity;
    __builtin_memcpy(keys, ids.data, ids.count * sizeof(uint64_t));
    sort_keys(keys, temporary, ids.count);
    for (size_t i = 1; i < ids.count; ++i)
        require_support(keys[i] != keys[i - 1], "canonical coordinates must be unique");
    return identity;
}

void fill_ids(uint64_t* out, uint64_t count, memory::const_array_view<uint64_t> ids) noexcept {
    if (!count) return;
    if (ids.count) __builtin_memcpy(out, ids.data, count * sizeof(uint64_t));
    else for (uint64_t i = 0; i < count; ++i) out[i] = i;
}

uint64_t feature_hash(uint64_t id) noexcept {
    id ^= id >> 30;
    id *= UINT64_C(0xbf58476d1ce4e5b9);
    id ^= id >> 27;
    id *= UINT64_C(0x94d049bb133111eb);
    return id ^ (id >> 31);
}

size_t slots_for_features(uint64_t features) {
    size_t slots = 1;
    while (slots / 2 < features) {
        require_extent(slots <= SIZE_MAX / 2);
        slots *= 2;
    }
    return slots;
}

// Two five-bit lanes encode a tile-local row and a block-local feature.
// Tile ranges are already delimited by the input's row offsets.
void sort_support(const StructureInput& input, uint64_t* keys, uint64_t* temporary,
                  LayoutView<>& layout) {
    const auto support = input.support;
    for (uint64_t tile = 0; tile < layout.tiles; ++tile) {
        const auto first_row = tile * input.tile_row_width;
        const auto remaining_rows = support.rows - first_row;
        const auto active_rows = remaining_rows < input.tile_row_width ? remaining_rows : input.tile_row_width;
        const auto first = support.row_offsets[first_row];
        const auto last = support.row_offsets[first_row + active_rows];
        if (first == last) continue;
        for (uint64_t row = 0; row < active_rows; ++row) {
            const auto begin = support.row_offsets[first_row + row];
            const auto end = support.row_offsets[first_row + row + 1];
            for (auto entry = begin; entry < end; ++entry) {
                const auto feature = support.physical_features[entry];
                require_support(feature < input.features, "physical feature is outside the feature extent");
                keys[entry] = (static_cast<uint64_t>(feature / input.block_width) << 10)
                    | (row << 5) | (feature % input.block_width);
            }
        }

        sort_keys(keys + first, temporary, static_cast<size_t>(last - first));
        ++layout.tile_blocks;
        ++layout.row_block_entries;
        for (auto entry = first + 1; entry < last; ++entry) {
            const auto previous = keys[entry - 1];
            const auto current = keys[entry];
            require_support(previous != current, "a row cannot allocate the same feature more than once");
            layout.tile_blocks += (previous >> 10) != (current >> 10);
            layout.row_block_entries += (previous >> 5) != (current >> 5);
        }
    }
}

struct MetadataBuffers {
    uint32_t* tile_offsets;
    uint32_t* block_ids;
    uint32_t* row_masks;
    uint32_t* entry_offsets;
    uint32_t* feature_masks;
    uint32_t* value_offsets;
    uint32_t* feature_offsets;
};

void write_metadata(MetadataBuffers out, const StructureInput& input,
                    LayoutView<> layout, const uint64_t* keys) noexcept {
    size_t blocks = 0, entries = 0;
    out.tile_offsets[0] = 0;
    out.entry_offsets[0] = 0;
    out.value_offsets[0] = 0;
    const auto support = input.support;
    for (uint64_t tile = 0; tile < layout.tiles; ++tile) {
        const auto row = tile * layout.tile_row_width;
        const auto remaining = layout.rows - row;
        const auto active = remaining < layout.tile_row_width ? remaining : layout.tile_row_width;
        auto cursor = support.row_offsets[row];
        const auto end = support.row_offsets[row + active];
        while (cursor < end) {
            const auto block = keys[cursor] >> 10;
            uint32_t rows = 0;
            do {
                const auto entry = keys[cursor] >> 5;
                rows |= uint32_t{1} << (entry & 31);
                uint32_t features = 0;
                do {
                    features |= uint32_t{1} << (keys[cursor] & 31);
                    ++cursor;
                } while (cursor < end && (keys[cursor] >> 5) == entry);
                out.feature_masks[entries] = features;
                out.value_offsets[++entries] = static_cast<uint32_t>(cursor);
            } while (cursor < end && (keys[cursor] >> 10) == block);
            out.block_ids[blocks] = static_cast<uint32_t>(block);
            out.row_masks[blocks] = rows;
            out.entry_offsets[++blocks] = static_cast<uint32_t>(entries);
        }
        out.tile_offsets[tile + 1] = static_cast<uint32_t>(blocks);
    }
    for (uint64_t block = 0; block < layout.feature_blocks; ++block)
        out.feature_offsets[block] = static_cast<uint32_t>(block * input.block_width);
    out.feature_offsets[layout.feature_blocks] = static_cast<uint32_t>(layout.features);
}

} // namespace

size_t structure_scratch_bytes(const StructureInput& input) {
    const auto sizes = scratch_sizes(input);
    return (sizes.keys + sizes.temporary + sizes.sorted_rows) * sizeof(uint64_t);
}

Structure::Structure(const StructureInput& input, void* scratch, size_t scratch_bytes) {
    const auto sizes = scratch_sizes(input);
    const auto required = (sizes.keys + sizes.temporary + sizes.sorted_rows) * sizeof(uint64_t);
    require_support(!required || (scratch && scratch_bytes >= required &&
        reinterpret_cast<uintptr_t>(scratch) % alignof(uint64_t) == 0),
        "block-masked build scratch is missing, too small, or misaligned");

    auto* keys = static_cast<uint64_t*>(scratch);
    auto* temporary = sizes.temporary ? keys + sizes.keys : nullptr;
    auto* sorted_rows = sizes.sorted_rows ? keys + sizes.keys + sizes.temporary : nullptr;
    const bool identity_features = validate_ids(input.canonical_features, keys, temporary);
    // Sorted row IDs are shared by forward argument/baseline maps and by the
    // distributed path. Use the original row array directly if already ordered.
    bool rows_ordered = true;
    for (size_t i = 1; i < input.canonical_rows.count; ++i) {
        rows_ordered &= input.canonical_rows.data[i - 1] < input.canonical_rows.data[i];
    }
    if (!rows_ordered) {
        __builtin_memcpy(sorted_rows, input.canonical_rows.data, sizes.sorted_rows * sizeof(uint64_t));
        sort_keys(sorted_rows, temporary, sizes.sorted_rows);
        for (size_t i = 1; i < sizes.sorted_rows; ++i)
            require_support(sorted_rows[i] != sorted_rows[i - 1], "canonical coordinates must be unique");
    }

    auto& layout = layout_;
    layout.tile_row_width = input.tile_row_width;
    layout.rows = input.support.rows;
    layout.features = input.features;
    layout.tiles = group_count(layout.rows, layout.tile_row_width);
    layout.feature_blocks = group_count(layout.features, input.block_width);
    layout.stored_values = input.support.entries;
    sort_support(input, keys, temporary, layout);

    // The first seven arrays are in the existing device upload order. The
    // coordinate maps and reverse lookup table stay at the host.
    size_t bytes = 0;
    const auto tile_offsets = append<uint32_t>(bytes, layout.tiles + 1);
    const auto block_ids = append<uint32_t>(bytes, layout.tile_blocks);
    const auto row_masks = append<uint32_t>(bytes, layout.tile_blocks);
    const auto entry_offsets = append<uint32_t>(bytes, layout.tile_blocks + 1);
    const auto feature_masks = append<uint32_t>(bytes, layout.row_block_entries);
    const auto value_offsets = append<uint32_t>(bytes, layout.row_block_entries + 1);
    const auto feature_offsets = append<uint32_t>(bytes, layout.feature_blocks + 1);
    metadata_bytes_ = bytes;
    const auto feature_ids = append<uint64_t>(bytes, layout.features);
    const auto row_ids = append<uint64_t>(bytes, layout.rows);
    const auto ordered_row_ids = rows_ordered ? row_ids : append<uint64_t>(bytes, layout.rows);
    feature_slot_count_ = identity_features ? 0 : slots_for_features(layout.features);
    const auto slots_offset = append<uint32_t>(bytes, feature_slot_count_);
    image_ = ::operator new(bytes);
    image_bytes_ = bytes;
    auto* image = static_cast<unsigned char*>(image_);
    auto u32_at = [image](size_t offset) { return reinterpret_cast<uint32_t*>(image + offset); };
    MetadataBuffers metadata{
        u32_at(tile_offsets), u32_at(block_ids), u32_at(row_masks), u32_at(entry_offsets),
        u32_at(feature_masks), u32_at(value_offsets), u32_at(feature_offsets)};
    write_metadata(metadata, input, layout, keys);
    layout.tile_block_offsets = metadata.tile_offsets;
    layout.block_ids = metadata.block_ids;
    layout.participating_row_masks = metadata.row_masks;
    layout.block_row_entry_offsets = metadata.entry_offsets;
    layout.feature_masks = metadata.feature_masks;
    layout.row_block_value_offsets = metadata.value_offsets;
    layout.feature_block_offsets = metadata.feature_offsets;

    auto* features = reinterpret_cast<uint64_t*>(image + feature_ids);
    auto* rows = reinterpret_cast<uint64_t*>(image + row_ids);
    fill_ids(features, layout.features, input.canonical_features);
    fill_ids(rows, layout.rows, input.canonical_rows);
    canonical_features_ = {features, static_cast<size_t>(layout.features), {memory::domain::host}};
    canonical_rows_ = {rows, static_cast<size_t>(layout.rows), {memory::domain::host}};
    auto* row_order = reinterpret_cast<uint64_t*>(image + ordered_row_ids);
    if (!rows_ordered) __builtin_memcpy(row_order, sorted_rows, sizes.sorted_rows * sizeof(uint64_t));
    canonical_rows_in_order_ = {row_order, static_cast<size_t>(layout.rows), {memory::domain::host}};

    if (feature_slot_count_) {
        auto* slots = u32_at(slots_offset);
        __builtin_memset(slots, 0, feature_slot_count_ * sizeof(*slots));
        const auto mask = feature_slot_count_ - 1;
        for (uint64_t physical = 0; physical < layout.features; ++physical) {
            auto slot = static_cast<size_t>(feature_hash(features[physical])) & mask;
            while (slots[slot]) slot = (slot + 1) & mask;
            slots[slot] = static_cast<uint32_t>(physical + 1); // zero is empty
        }
        feature_slots_ = slots;
    }
}

Structure::~Structure() {
    ::operator delete(image_);
}

uint32_t Structure::physical_feature(uint64_t canonical_coordinate) const {
    if (feature_slot_count_) {
        const auto mask = feature_slot_count_ - 1;
        auto slot = static_cast<size_t>(feature_hash(canonical_coordinate)) & mask;
        while (const auto stored = feature_slots_[slot]) {
            const auto physical = stored - 1;
            if (canonical_features_.data[physical] == canonical_coordinate) return physical;
            slot = (slot + 1) & mask;
        }
    } else if (canonical_coordinate < layout_.features) {
        return static_cast<uint32_t>(canonical_coordinate);
    }
    throw std::out_of_range("canonical feature coordinate is not bound by this structure");
}

std::uint64_t Structure::value_offset(
    std::uint64_t row,
    std::uint64_t physical_feature) const {
    std::uint64_t offset = 0;
    if (!block_masked::value_offset(view(), row, physical_feature, &offset)) {
        throw std::out_of_range("coordinate is not in the allocated block-masked support");
    }
    return offset;
}

} // namespace cellerator::tensor::block_masked
