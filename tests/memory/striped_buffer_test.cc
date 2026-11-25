#define CELLERATOR_ENABLE_CUDA 0
#include <Cellerator/tensor/base/sliced_ell_binding.cuh>
#include <Cellerator/tensor/striped.hh>
#include <cstdio>
#include <cstdlib>

using cellerator::memory::status;
using namespace cellerator::tensor::striped;

static void require(bool condition, const char *label) {
    if (!condition) {
        std::fprintf(stderr, "Strip layout: %s\n", label);
        std::exit(1);
    }
}

template<StripOrder order>
static void check_order(uint32_t pitch, size_t slots, size_t last) {
    using Layout = StripLayout<float, uint16_t, order>;
    using View = StripView<float, uint16_t, order>;
    constexpr uint32_t mask = (1u << 1) | (1u << 8) | (1u << 31);
    Layout layout;
    require(Layout::prepare({mask, 32, 5, pitch}, &layout) == status::success, "prepare");
    require(layout.slots() == slots, "physical slots");
    alignas(32) unsigned char bytes[512]{};
    View a, b;
    require(layout.bind(bytes, layout.bytes(), &a) == status::success, "first stripe");
    require(layout.bind(bytes + 256, layout.bytes(), &b) == status::success, "second stripe");
    require(a.rows() == 3 && a.packed_row(1) == 0 && a.packed_row(8) == 1 &&
        a.packed_row(31) == 2 && a.packed_row(32) == View::missing_row &&
        a.packed_row(3) == View::missing_row, "nonprefix mask rank");
    require(a.shape.first_entry == 32 && a.slot(2, 4) == last, "entry address");
    require(layout.bytes() % 32 == 0 && layout.columns_offset() % 32 == 0 &&
        reinterpret_cast<unsigned char *>(a.column_indices) == bytes + layout.columns_offset(),
        "plane alignment");
    a.column_indices[last] = View::padding_column;
    b.column_indices[last] = 65000;
    b.values[last] = 9;
    require(a.column_indices[last] == 65535 && b.column_indices[last] == 65000 &&
        b.values[last] == 9, "narrow columns and disjoint views");
    require(layout.bind(bytes + 1, layout.bytes(), &a) == status::invalid_alignment, "unaligned");
    require(layout.bind(bytes, layout.bytes() - 1, &a) == status::capacity_exceeded, "short range");
    StripView<const float, const uint16_t, order> read_only;
    require(layout.bind(static_cast<const void *>(bytes + 256), layout.bytes(), &read_only) ==
        status::success && read_only.column_indices[last] == 65000, "const view");
    require(layout.bind(bytes, layout.bytes(), &a) == status::success &&
        a.column_indices[last] == View::padding_column, "rebind");
}

static void check_sliced_ell() {
    using namespace cellerator::tensor::base::device;
    SlicedEllLayout layout;
    SlicedEllView view{};
    alignas(32) unsigned char storage[256]{};
    require(layout.bind(storage, sizeof(storage), &view) == status::invalid_state,
        "unprepared SELL");
    require(SlicedEllLayout::prepare({3, 5, 5, 2, 0, 6}, &layout) == status::success &&
        layout.bytes() <= sizeof(storage) && layout.bytes() % 32 == 0, "prepare SELL");
    require(layout.bind(storage, sizeof(storage), &view) == status::success &&
        view.rows == 3 && view.cols == 5 && view.nnz == 5 && view.slice_count == 2,
        "bind ordinary SELL shape");
    require(reinterpret_cast<uintptr_t>(view.col_idx) % 32 == 0 &&
        reinterpret_cast<uintptr_t>(view.val) % 32 == 0, "SELL plane alignment");
    view.slice_row_offsets[0] = 0;
    view.slice_row_offsets[1] = 1;
    view.slice_row_offsets[2] = 3;
    view.slice_widths[0] = view.slice_widths[1] = 2;
    view.slice_slot_offsets[0] = 0;
    view.slice_slot_offsets[1] = 2;
    view.col_idx[4] = 4;
    view.val[4] = cellerator::real::store_f16_storage(40);
    auto slot = view.slice_slot_offsets[1] +
        (2 - view.slice_row_offsets[1]) * view.slice_widths[1];
    require(view.col_idx[slot] == 4 && cellerator::real::load_f16_storage(view.val[slot]) == 40,
        "nonuniform SELL row lookup");
    auto *columns = view.col_idx;
    require(layout.bind(storage, layout.bytes() - 1, &view) == status::capacity_exceeded &&
        layout.bind(storage + 1, sizeof(storage) - 1, &view) == status::invalid_alignment &&
        view.col_idx == columns, "failed SELL bind preserves view");
    const size_t bytes = layout.bytes();
    require(SlicedEllLayout::prepare({3, 5, 5, 2, 0, SIZE_MAX}, &layout) ==
        status::arithmetic_overflow && layout.bytes() == bytes, "SELL size overflow preserves layout");
    require(SlicedEllLayout::prepare({}, &layout) == status::success &&
        layout.bind(storage, sizeof(storage), &view) == status::success &&
        view.slice_row_offsets && !view.col_idx && !view.val, "empty SELL span");
}

int main() {
    check_sliced_ell();
    check_order<StripOrder::row_major>(8, 24, 20);
    check_order<StripOrder::slot_major>(4, 20, 18);

    StripLayout<double, uint32_t, StripOrder::slot_major> large;
    require(decltype(large)::prepare({1, 0, UINT32_MAX, UINT32_MAX}, &large) ==
        status::arithmetic_overflow, "slot byte overflow");
    StripLayout<float, uint32_t> layout;
    StripView<float, uint32_t> view;
    require(decltype(view)::padding_column == UINT32_MAX, "wide column padding");
    require(decltype(layout)::prepare({1, 0, 8, 7}, &layout) == status::invalid_argument, "bad pitch");
    require(decltype(layout)::prepare({UINT32_MAX, 0, 0, 0}, &layout) == status::invalid_argument,
        "nonempty zero width");
    require(decltype(layout)::prepare({0, 32, 16, 16}, &layout) == status::success &&
        layout.bytes() == 0 && layout.slots() == 0 &&
        layout.bind(static_cast<void *>(nullptr), 0, &view) == status::success &&
        !view.values && !view.column_indices, "empty layout");
}
