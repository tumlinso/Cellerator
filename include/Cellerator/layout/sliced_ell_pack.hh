#pragma once

#include <Cellerator/common/real/fp16.hh>
#include <Cellerator/tensor/base/compressed.hh>
#include <Cellerator/tensor/base/sliced_ell_binding.cuh>

namespace cellerator::layout {

struct SlicedEllPackOptions {
    unsigned slice_rows = 16, sort_window = 256;
    bool reorder_rows = false, allow_fp16_conversion = false;
};

struct SlicedEllRow {
    uint64_t logical_position, source_cell;
};

struct SlicedEllSlice {
    unsigned row_offset, slot_offset, width;
};

// The plan borrows order and slices. Keep them and the descriptor unchanged
// through packing. order maps each physical row to its input occurrence.
struct SlicedEllPackPlan {
    tensor::base::device::SlicedEllLayout layout{};
    tensor::base::device::SlicedEllShape shape{};
    const unsigned *order = nullptr;
    const SlicedEllSlice *slices = nullptr;
    bool allow_fp16_conversion = false;
};

template<class O>
memory::status sliced_ell_row_lengths(const O *offsets, size_t rows, size_t nnz,
                                      unsigned *lengths, size_t capacity) noexcept {
    static_assert(sizeof(O) <= sizeof(uint64_t) && requires(O x) { x % 2; });
    using memory::status;
    if (rows > capacity) return status::capacity_exceeded;
    if ((!rows && nnz) || (rows && (!offsets || !lengths))) return status::invalid_argument;
    if (!rows) return status::success;
    if (offsets[0] != 0 || uint64_t(offsets[rows]) != nnz) return status::invalid_argument;
    uint64_t begin = 0;
    for (size_t row = 0; row < rows; ++row) {
        const uint64_t end = uint64_t(offsets[row+1]);
        if (end < begin || end > nnz) return status::invalid_argument;
        if (end-begin > UINT32_MAX) return status::arithmetic_overflow;
        lengths[row] = unsigned(end-begin);
        begin = end;
    }
    return status::success;
}

namespace detail {

inline bool sell_row_before(const unsigned *lengths, unsigned a, unsigned b) noexcept {
    return lengths[a] > lengths[b] || (lengths[a] == lengths[b] && a < b);
}

inline void sift_sell_rows(const unsigned *lengths, unsigned *order, size_t root, size_t n) noexcept {
    const unsigned row = order[root];
    while (root < n/2) {
        size_t child = 2*root + 1;
        if (child+1 < n && sell_row_before(lengths, order[child], order[child+1])) ++child;
        if (!sell_row_before(lengths, row, order[child])) break;
        order[root] = order[child];
        root = child;
    }
    order[root] = row;
}

// Compare by input ordinal at equal lengths, so the bounded in-place heap sort
// preserves their order without another row-sized workspace.
inline void sort_sell_rows(const unsigned *lengths, unsigned *order, size_t n) noexcept {
    for (size_t root = n/2; root != 0; --root) sift_sell_rows(lengths, order, root-1, n);
    for (size_t end = n; end > 1;) {
        --end;
        const unsigned row = order[end];
        order[end] = order[0]; order[0] = row;
        sift_sell_rows(lengths, order, 0, end);
    }
}

} // namespace detail

// Supply rows order entries and ceil(rows / slice_rows) slice entries in
// disjoint storage. On failure out is unchanged and workspace may be written.
inline memory::status plan_sliced_ell(const unsigned *row_lengths, size_t rows, size_t cols,
                                      SlicedEllPackOptions options, unsigned *order, size_t order_capacity,
                                      SlicedEllSlice *slices, size_t slice_capacity, SlicedEllPackPlan *out) noexcept {
    using memory::status;
    if (!out || !options.slice_rows || (options.reorder_rows && !options.sort_window)) return status::invalid_argument;
    if (rows > UINT32_MAX || cols > UINT32_MAX) return status::arithmetic_overflow;
    const size_t slice_count = rows/options.slice_rows + (rows%options.slice_rows != 0);
    if (rows > order_capacity || slice_count > slice_capacity) return status::capacity_exceeded;
    if (rows && (!row_lengths || !order || !slices)) return status::invalid_argument;

    uint64_t nnz = 0;
    for (size_t row = 0; row < rows; ++row) {
        nnz += row_lengths[row];
        if (nnz > UINT32_MAX) return status::arithmetic_overflow;
        order[row] = unsigned(row);
    }
    if (nnz && !cols) return status::invalid_argument;
    if (options.reorder_rows) {
        for (size_t row = 0; row < rows;) {
            const size_t n = rows-row < options.sort_window ? rows-row : options.sort_window;
            detail::sort_sell_rows(row_lengths, order+row, n);
            row += n;
        }
    }

    size_t slots = 0, row = 0;
    for (size_t slice = 0; slice < slice_count; ++slice) {
        const size_t n = rows-row < options.slice_rows ? rows-row : options.slice_rows;
        unsigned width = 0;
        for (size_t r = row; r < row+n; ++r) {
            const unsigned length = row_lengths[order[r]];
            if (length > width) width = length;
        }
        if (slots > UINT32_MAX) return status::arithmetic_overflow;
        slices[slice] = {unsigned(row), unsigned(slots), width};
        size_t count;
        if (__builtin_mul_overflow(n, size_t(width), &count) ||
            __builtin_add_overflow(slots, count, &slots)) return status::arithmetic_overflow;
        row += n;
    }
    SlicedEllPackPlan plan;
    plan.shape = {unsigned(rows), unsigned(cols), unsigned(nnz), unsigned(slice_count), options.slice_rows, slots};
    const auto result = tensor::base::device::SlicedEllLayout::prepare(plan.shape, &plan.layout);
    if (result != status::success) return result;
    plan.order = order; plan.slices = slices; plan.allow_fp16_conversion = options.allow_fp16_conversion;
    *out = plan;
    return status::success;
}

// The compressed major axis lists cell occurrences and its offsets are rebased
// to zero. logical_rows labels each occurrence in the compressed input order.
// Source, plan, packet, and row-map storage must be host-accessible and disjoint.
// Publish the packet and map only on success; a failed fill may leave partial bytes.
template<class I, class V, class O>
memory::status pack_sliced_ell(const SlicedEllPackPlan &plan, tensor::base::CompressedView<I, V, O> source,
                               const SlicedEllRow *logical_rows, void *packet, size_t packet_bytes,
                               SlicedEllRow *row_map, size_t row_map_capacity) noexcept {
    using memory::status;
    using tensor::base::CompressedAxis;
    using tensor::base::device::SlicedEllView;
    static_assert(sizeof(I) <= sizeof(uint64_t) && requires(I x) { x % 2; });
    static_assert(sizeof(O) <= sizeof(uint64_t) && requires(O x) { x % 2; });
    constexpr bool fp16 = __is_same(V, real::f16_t) || __is_same(V, const real::f16_t);
    static_assert(fp16 || requires(V x) { x % 2; } || __is_same(V, float) || __is_same(V, const float) ||
                  __is_same(V, double) || __is_same(V, const double));
    if (!fp16 && !plan.allow_fp16_conversion) return status::invalid_argument;
    const auto &s = source.structure;
    if (s.axis != CompressedAxis::row && s.axis != CompressedAxis::column) return status::invalid_argument;
    const uint64_t rows = s.axis == CompressedAxis::row ? s.rows : s.cols;
    const uint64_t cols = s.axis == CompressedAxis::row ? s.cols : s.rows;
    if (rows != plan.shape.rows || cols != plan.shape.cols || s.nnz != plan.shape.nnz) return status::invalid_argument;
    if (rows > row_map_capacity) return status::capacity_exceeded;
    if ((s.nnz && (!s.minor_idx || !source.values)) ||
        (rows && (!s.major_ptr || !logical_rows || !row_map || !plan.order || !plan.slices)))
        return status::invalid_argument;
    if (s.major_ptr && (s.major_ptr[0] != 0 || uint64_t(s.major_ptr[rows]) != s.nnz))
        return status::invalid_argument;

    SlicedEllView out{};
    const auto result = plan.layout.bind(packet, packet_bytes, &out);
    if (result != status::success) return result;
    for (unsigned slice = 0; slice < out.slice_count; ++slice) {
        const auto &part = plan.slices[slice];
        const unsigned n = out.rows-part.row_offset < out.slice_rows ? out.rows-part.row_offset : out.slice_rows;
        out.slice_row_offsets[slice] = part.row_offset;
        out.slice_slot_offsets[slice] = part.slot_offset;
        out.slice_widths[slice] = part.width;
        for (unsigned row = 0; row < n; ++row) {
            const unsigned physical = part.row_offset + row;
            const unsigned input = plan.order[physical];
            const uint64_t begin = uint64_t(s.major_ptr[input]);
            const uint64_t end = uint64_t(s.major_ptr[input+1]);
            if (end < begin || end > s.nnz || end-begin > part.width) return status::invalid_argument;
            const size_t slot = size_t(part.slot_offset) + size_t(row)*part.width;
            for (uint64_t j = begin; j < end; ++j) {
                const uint64_t col = uint64_t(s.minor_idx[j]);
                if (col >= cols) return status::invalid_argument;
                const size_t k = slot + size_t(j-begin);
                if constexpr (fp16) {
                    static_assert(sizeof(real::f16_t) == sizeof(real::f16_bits));
                    const auto *values = reinterpret_cast<const unsigned char *>(source.values);
                    real::f16_bits bits;
                    __builtin_memcpy(&bits, values+j*sizeof(bits), sizeof(bits));
                    out.val[k] = __builtin_bit_cast(real::f16_t, bits);
                } else {
                    const double value = double(source.values[j]);
                    const auto bits = real::round_f16(value);
                    if ((bits & 0x7fffu) == 0x7c00u &&
                        (__builtin_bit_cast(uint64_t, value) & 0x7fffffffffffffffull) < 0x7ff0000000000000ull)
                        return status::arithmetic_overflow;
                    out.val[k] = __builtin_bit_cast(real::f16_t, bits);
                }
                out.col_idx[k] = unsigned(col);
            }
            for (size_t j = size_t(end-begin); j < part.width; ++j) {
                out.col_idx[slot+j] = tensor::base::sliced_ell_invalid_col;
                out.val[slot+j] = __builtin_bit_cast(real::f16_t, uint16_t(0));
            }
            row_map[physical] = logical_rows[input];
        }
    }
    out.slice_row_offsets[out.slice_count] = out.rows;

    // Initialize just the alignment gaps in the byte span handed to the copy.
    auto *meta_end = reinterpret_cast<unsigned char *>(out.slice_count ? out.slice_slot_offsets + out.slice_count
                                                                       : out.slice_row_offsets + 1);
    auto *packet_end = static_cast<unsigned char *>(packet) + plan.layout.bytes();
    if (plan.shape.slots) {
        auto *columns = reinterpret_cast<unsigned char *>(out.col_idx);
        auto *columns_end = reinterpret_cast<unsigned char *>(out.col_idx + plan.shape.slots);
        auto *values = reinterpret_cast<unsigned char *>(out.val);
        auto *values_end = reinterpret_cast<unsigned char *>(out.val + plan.shape.slots);
        __builtin_memset(meta_end, 0, uintptr_t(columns)-uintptr_t(meta_end));
        __builtin_memset(columns_end, 0, uintptr_t(values)-uintptr_t(columns_end));
        meta_end = values_end;
    }
    __builtin_memset(meta_end, 0, uintptr_t(packet_end)-uintptr_t(meta_end));
    return status::success;
}

} // namespace cellerator::layout
