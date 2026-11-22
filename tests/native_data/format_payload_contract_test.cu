#include <Cellerator/tensor/base/compressed.hh>
#include <Cellerator/tensor/base/coo.hh>
#include <Cellerator/memory/view.hh>
#include <Cellerator/tensor/base/device_views.cuh>

#include <cuda_fp16.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace formats = ::cellerator::tensor::base;

namespace {

void require(bool condition, const char *message) {
    if (condition) return;
    std::fprintf(stderr, "%s\n", message);
    std::exit(1);
}

bool close_half(__half value, float expected) {
    return std::fabs(__half2float(value) - expected) < 1.0e-3f;
}

void check_dense() {
    __half values[8]{};
    const cellerator::memory::matrix_view<__half> columns{values, 3, 2, 1, {}, 4};
    const auto read = cellerator::memory::as_const(columns);
    require(read.data == values && read.rows == 3 && read.columns == 2 &&
            read.row_stride == 1 && read.column_stride == 4,
            "const dense view should borrow the same layout and data");
    columns.data[2 * columns.row_stride + columns.column_stride] = __float2half(9.0f);
    require(close_half(values[6], 9.0f) && close_half(read.data[6], 9.0f),
            "borrowed dense writes should reach the original array");
}

void check_compressed() {
    const int32_t offsets[] = {0, 2, 2, 3};
    const int32_t columns[] = {0, 3, 2};
    __half values[] = {__float2half(1.0f), __float2half(3.0f), __float2half(6.0f)};
    const formats::CompressedView<int32_t, __half> csr{
        {3, 4, 3, formats::CompressedAxis::row, offsets, columns}, values};
    const formats::CompressedView<int32_t, const __half> csc{
        {4, 3, 3, formats::CompressedAxis::column, offsets, columns}, values};
    require(csr.structure.major_ptr == csc.structure.major_ptr &&
            csr.structure.minor_idx == csc.structure.minor_idx &&
            csr.values == csc.values, "compressed views should borrow their arrays");
    csr.values[1] = __float2half(9.0f);
    require(close_half(csc.values[1], 9.0f), "compressed views should see shared value updates");
}

void check_dia() {
    int offsets[] = {0, 1};
    __half values[] = {__float2half(2.0f), __float2half(3.0f),
        __float2half(4.0f), __float2half(0.0f)};
    const formats::device::DiaView view{2u, 2u, 4u, 2u, offsets, values};
    require(view.offsets == offsets && view.val == values,
            "DIA view should borrow its arrays");
    require(view.offsets[0] == 0 && close_half(view.val[1u], 3.0f),
            "DIA main diagonal mapping changed");
    require(view.offsets[1] == 1 && close_half(view.val[view.rows], 4.0f)
                && close_half(view.val[view.rows + 1u], 0.0f),
            "DIA upper diagonal stride or padding changed");
}

static_assert(alignof(formats::device::DiaView) == 16);

void check_coo() {
    const uint32_t rows[] = {1, 0, 1};
    const uint32_t cols[] = {2, 1, 2};
    __half values[] = {__float2half(5.0f), __float2half(4.0f), __float2half(7.0f)};
    const formats::CooView<uint32_t, const __half> view{2, 3, 3, rows, cols, values};
    require(view.row_idx == rows && view.col_idx == cols && view.values == values,
            "COO view should borrow its arrays");
    require(view.row_idx[0] == view.row_idx[2] && view.col_idx[0] == view.col_idx[2] &&
            close_half(view.values[0], 5.0f) && close_half(view.values[2], 7.0f),
            "COO view should retain repeated coordinates as separate entries");
}

void check_blocked_ell() {
    unsigned int columns[] = {0u, 1u, 1u, formats::blocked_ell_invalid_col};
    __half values[12];
    for (std::size_t i = 0u; i < 12u; ++i) values[i] = __float2half(static_cast<float>(i));
    const formats::device::BlockedEllView view{3u, 4u, 3u, 2u, 4u, columns, values};
    require(view.blockColIdx == columns && view.val == values,
            "Blocked-ELL view should borrow its arrays");
    const unsigned int row = 2u;
    const unsigned int column = 3u;
    const unsigned int block_row = row / view.block_size;
    const unsigned int blocks_per_row = view.ell_cols / view.block_size;
    const unsigned int block_row_base = block_row * blocks_per_row;
    const unsigned int slot = row * view.ell_cols + column % view.block_size;
    require(view.blockColIdx[block_row_base] == column / view.block_size
                && close_half(view.val[slot], 9.0f),
            "Blocked-ELL value mapping changed");
    require(view.blockColIdx[block_row_base + 1u] == formats::blocked_ell_invalid_col,
            "Blocked-ELL empty block slot changed");
}

void check_sliced_ell() {
    unsigned int row_offsets[] = {0u, 1u, 3u};
    unsigned int widths[] = {2u, 2u};
    unsigned int slot_offsets[] = {0u, 2u};
    unsigned int columns[] = {1u, 3u, 2u, formats::sliced_ell_invalid_col, 4u, 0u};
    __half values[] = {__float2half(10.0f), __float2half(20.0f), __float2half(30.0f),
        __float2half(0.0f), __float2half(40.0f), __float2half(50.0f)};
    const formats::device::SlicedEllView view{3u, 5u, 5u, 2u, 0u,
        row_offsets, widths, slot_offsets, columns, values};
    require(view.slice_row_offsets == row_offsets && view.slice_widths == widths
                && view.slice_slot_offsets == slot_offsets
                && view.col_idx == columns && view.val == values,
            "Sliced-ELL view should borrow its arrays");
    const unsigned int slice = 1u;
    const unsigned int row = 2u;
    const unsigned int slot = view.slice_slot_offsets[slice]
        + (row - view.slice_row_offsets[slice]) * view.slice_widths[slice];
    require(view.col_idx[slot] == 4u && close_half(view.val[slot], 40.0f)
                && view.col_idx[slot + 1u] == 0u && close_half(view.val[slot + 1u], 50.0f),
            "Sliced-ELL row packing changed");
    const unsigned int padding_slot = view.slice_slot_offsets[slice] + 1u;
    require(view.col_idx[padding_slot] == formats::sliced_ell_invalid_col
                && close_half(view.val[padding_slot], 0.0f),
            "Sliced-ELL padding slot changed");
}

} // namespace

int main() {
    check_dense();
    check_compressed();
    check_coo();
    check_dia();
    check_blocked_ell();
    check_sliced_ell();
    return 0;
}
