#include <Cellerator/tensor/base/blocked_ell.cuh>
#include <Cellerator/tensor/base/compressed.hh>
#include <Cellerator/tensor/base/coo.hh>
#include <Cellerator/tensor/base/diagonal.cuh>
#include <Cellerator/tensor/base/device_views.cuh>
#include <Cellerator/tensor/base/sliced_ell.cuh>

#include <cuda_fp16.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace formats = ::cellerator::tensor::base;
namespace types = ::cellerator::types;

namespace {

void require(bool condition, const char *message) {
    if (condition) return;
    std::fprintf(stderr, "%s\n", message);
    std::exit(1);
}

bool close_half(__half value, float expected) {
    return std::fabs(__half2float(value) - expected) < 1.0e-3f;
}

std::size_t align_up(std::size_t value, std::size_t alignment) {
    return ((value + alignment - 1u) / alignment) * alignment;
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
    formats::dia matrix{};
    formats::init(&matrix, 2u, 2u, 2u);
    matrix.num_diagonals = 1u;
    require(formats::allocate(&matrix) != 0, "DIA allocation failed");
    matrix.offsets[0] = 0; matrix.val[0] = __float2half(2.0f); matrix.val[1] = __float2half(3.0f);
    require(close_half(*formats::at(&matrix, 1u, 1u), 3.0f), "DIA lookup changed");
    require(formats::at(&matrix, 0u, 1u) == nullptr, "DIA absent lookup changed");
    formats::clear(&matrix);
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
    formats::blocked_ell matrix{};
    formats::init(&matrix, 3u, 4u, 3u, 2u, 4u);
    require(formats::allocate(&matrix) != 0, "Blocked-ELL allocation failed");
    require(matrix.storage == matrix.blockColIdx, "Blocked-ELL column-index base changed");
    const types::idx_t columns[] = {0u, 1u, 1u, formats::blocked_ell_invalid_col};
    std::memcpy(matrix.blockColIdx, columns, sizeof(columns));
    for (std::size_t i = 0u; i < 12u; ++i) matrix.val[i] = __float2half(static_cast<float>(i));
    require(formats::bytes(&matrix) == sizeof(matrix) + 4u * sizeof(types::idx_t)
                + 12u * sizeof(types::storage_value_t),
            "Blocked-ELL descriptor/payload byte count changed");
    require(formats::row_block_count(&matrix) == 2u, "Blocked-ELL row-block count changed");
    require(formats::ell_width_blocks(&matrix) == 2u, "Blocked-ELL width changed");
    require(formats::pack_blocked_ell_aux(matrix.block_size, formats::ell_width_blocks(&matrix)) == 0x00020002ul,
            "Blocked-ELL auxiliary encoding changed");
    require(close_half(*formats::at(&matrix, 2u, 3u), 9.0f), "Blocked-ELL value mapping changed");
    require(formats::at(&matrix, 2u, 0u) == nullptr, "Blocked-ELL missing block lookup changed");
    formats::clear(&matrix);
}

void check_sliced_ell() {
    formats::sliced_ell matrix{};
    formats::init(&matrix, 3u, 5u, 4u);
    const types::u32 row_offsets[] = {0u, 1u, 3u};
    const types::u32 widths[] = {2u, 1u};
    require(formats::allocate(&matrix, 2u, row_offsets, widths) != 0, "Sliced-ELL allocation failed");
    require(matrix.storage == matrix.slice_row_offsets, "Sliced-ELL row-offset base changed");
    require(formats::total_slots(&matrix) == 4u, "Sliced-ELL slot count changed");
    const std::size_t widths_offset = align_up(3u * sizeof(types::u32), alignof(types::u32));
    const std::size_t columns_offset = align_up(widths_offset + 2u * sizeof(types::u32), alignof(types::idx_t));
    const std::size_t values_offset = align_up(columns_offset + 4u * sizeof(types::idx_t), alignof(types::storage_value_t));
    require(reinterpret_cast<unsigned char *>(matrix.slice_widths) == static_cast<unsigned char *>(matrix.storage) + widths_offset,
            "Sliced-ELL slice-width offset changed");
    require(reinterpret_cast<unsigned char *>(matrix.col_idx) == static_cast<unsigned char *>(matrix.storage) + columns_offset,
            "Sliced-ELL column-index offset changed");
    require(reinterpret_cast<unsigned char *>(matrix.val) == static_cast<unsigned char *>(matrix.storage) + values_offset,
            "Sliced-ELL value offset changed");
    require(formats::bytes(&matrix) == sizeof(matrix) + values_offset + 4u * sizeof(types::storage_value_t),
            "Sliced-ELL descriptor/payload byte count changed");
    matrix.col_idx[0] = 1u;
    matrix.col_idx[1] = 3u;
    matrix.col_idx[2] = 2u;
    matrix.col_idx[3] = 4u;
    for (std::size_t i = 0u; i < 4u; ++i) matrix.val[i] = __float2half(static_cast<float>((i + 1u) * 10u));
    require(formats::find_slice(&matrix, 2u) == 1u, "Sliced-ELL row-to-slice lookup changed");
    require(formats::row_nnz(&matrix, 0u) == 2u, "Sliced-ELL row occupancy changed");
    require(close_half(*formats::at(&matrix, 2u, 4u), 40.0f), "Sliced-ELL value mapping changed");
    require(formats::pack_sliced_ell_aux(matrix.slice_count, formats::total_slots(&matrix)) == 0x0000000400000002ul,
            "Sliced-ELL auxiliary encoding changed");
    formats::clear(&matrix);
}

} // namespace

int main() {
    check_compressed();
    check_coo();
    check_dia();
    check_blocked_ell();
    check_sliced_ell();
    return 0;
}
