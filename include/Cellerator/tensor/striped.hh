#pragma once

#include <Cellerator/common/compiler_annotations.hh>
#include <Cellerator/memory/status.hh>

#include <stddef.h>
#include <stdint.h>

namespace cellerator::tensor::striped {

enum class StripOrder : uint8_t { row_major, slot_major };

// A strip serves a subset of 32 batch-local rows. Its physical rows follow the
// set bits of row_mask. first_entry is an ordinal in each source sparse row.
// width includes entry padding. pitch is the leading stride in element slots.
struct StripShape {
    uint32_t row_mask = 0, first_entry = 0, width = 0, pitch = 0;
};

namespace detail {

CELLERATOR_HOST_DEVICE CELLERATOR_CUDA_FORCEINLINE uint32_t count_rows(uint32_t mask) noexcept {
#if defined(__CUDA_ARCH__)
    return __popc(mask);
#else
    return __builtin_popcount(mask);
#endif
}

} // namespace detail

template<class Value, class Column = uint16_t, StripOrder order = StripOrder::row_major>
struct StripView {
    static_assert(__is_same(Column, uint16_t) || __is_same(Column, const uint16_t) ||
                  __is_same(Column, uint32_t) || __is_same(Column, const uint32_t));
    static constexpr uint32_t missing_row = 32;
    // The largest column code denotes padding. Skip it before a dense gather.
    static constexpr Column padding_column = Column(-1);

    Value *values = nullptr;
    Column *column_indices = nullptr;
    StripShape shape{};

    CELLERATOR_HOST_DEVICE uint32_t rows() const noexcept {
        return detail::count_rows(shape.row_mask);
    }

    // Resolve the batch row once before traversing its entries.
    CELLERATOR_HOST_DEVICE uint32_t packed_row(uint32_t batch_row) const noexcept {
        if (batch_row >= missing_row) return missing_row;
        const uint32_t bit = uint32_t(1) << batch_row;
        return shape.row_mask & bit ? detail::count_rows(shape.row_mask & (bit - 1u)) : missing_row;
    }

    // Caller supplies a present packed row and an entry below shape.width.
    CELLERATOR_HOST_DEVICE constexpr size_t slot(uint32_t packed_row, uint32_t entry) const noexcept {
        if constexpr (order == StripOrder::row_major) return size_t(packed_row)*shape.pitch + entry;
        else return size_t(entry)*shape.pitch + packed_row;
    }
};

// Checked byte layout for one independent, relocatable strip. Each plane and
// the complete span have 32-byte alignment. Pitch selection belongs to the
// caller so the compact layout and transaction-aligned pitches are both usable.
template<class Value, class Column = uint16_t, StripOrder order = StripOrder::row_major>
class StripLayout {
public:
    static constexpr size_t alignment = 32;
    static_assert(__is_trivially_copyable(Value) && alignof(Value) <= alignment);
    static_assert(__is_same(Column, uint16_t) || __is_same(Column, uint32_t));
    static_assert(order == StripOrder::row_major || order == StripOrder::slot_major);

    // A failed preparation or binding leaves the output unchanged.
    static memory::status prepare(StripShape shape, StripLayout *out) noexcept {
        using memory::status;
        if (!out) return status::invalid_argument;
        if (shape.width > UINT32_MAX - shape.first_entry) return status::arithmetic_overflow;

        StripLayout layout;
        layout.shape_ = shape;
        const uint32_t rows = detail::count_rows(shape.row_mask);
        if (rows) {
            const size_t outer = order == StripOrder::row_major ? rows : shape.width;
            const size_t inner = order == StripOrder::row_major ? shape.width : rows;
            if (!shape.width || shape.pitch < inner) return status::invalid_argument;

            size_t value_bytes, column_bytes, end;
            if (__builtin_mul_overflow(outer, size_t(shape.pitch), &layout.slots_) ||
                __builtin_mul_overflow(layout.slots_, sizeof(Value), &value_bytes) ||
                __builtin_mul_overflow(layout.slots_, sizeof(Column), &column_bytes) ||
                !align_up(value_bytes, &layout.columns_offset_) ||
                __builtin_add_overflow(layout.columns_offset_, column_bytes, &end) ||
                !align_up(end, &layout.bytes_)) return status::arithmetic_overflow;
        }
        *out = layout;
        return status::success;
    }

    constexpr StripShape shape() const noexcept { return shape_; }
    constexpr size_t slots() const noexcept { return slots_; }
    constexpr size_t columns_offset() const noexcept { return columns_offset_; }
    constexpr size_t bytes() const noexcept { return bytes_; }

    memory::status bind(void *base, size_t capacity, StripView<Value, Column, order> *out) const noexcept {
        return bind_planes(static_cast<unsigned char *>(base), capacity, out);
    }

    memory::status bind(const void *base, size_t capacity,
                        StripView<const Value, const Column, order> *out) const noexcept {
        return bind_planes(static_cast<const unsigned char *>(base), capacity, out);
    }

private:
    static bool align_up(size_t count, size_t *out) noexcept {
        size_t padded;
        if (__builtin_add_overflow(count, alignment - 1u, &padded)) return false;
        *out = padded & ~(alignment - 1u);
        return true;
    }

    template<class Byte, class View>
    memory::status bind_planes(Byte *base, size_t capacity, View *out) const noexcept {
        using memory::status;
        if (!out) return status::invalid_argument;
        if (!bytes_) {
            *out = {nullptr, nullptr, shape_};
            return status::success;
        }
        if (bytes_ > capacity) return status::capacity_exceeded;
        if (!base) return status::invalid_argument;
        if (reinterpret_cast<uintptr_t>(base) % alignment) return status::invalid_alignment;
        *out = {reinterpret_cast<decltype(out->values)>(base),
                reinterpret_cast<decltype(out->column_indices)>(base + columns_offset_), shape_};
        return status::success;
    }

    StripShape shape_{};
    size_t slots_ = 0, columns_offset_ = 0, bytes_ = 0;
};

} // namespace cellerator::tensor::striped
