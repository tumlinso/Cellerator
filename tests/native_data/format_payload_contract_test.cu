#include <Cellerator/interop/formats/blocked_ell.cuh>
#include <Cellerator/interop/formats/compressed.cuh>
#include <Cellerator/interop/formats/coo.cuh>
#include <Cellerator/interop/formats/dense.cuh>
#include <Cellerator/interop/formats/sliced_ell.cuh>

#include <cuda_fp16.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace formats = ::cellerator::interop::formats;
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

void check_dense() {
    formats::DensePayload matrix{};
    formats::init(&matrix, 2u, 3u);
    require(formats::payload_elements(&matrix) == 6u, "dense element count changed");
    require(formats::payload_bytes(&matrix) == 6u * sizeof(types::storage_value_t), "dense payload byte count changed");
    require(formats::bytes(&matrix) == sizeof(matrix) + 6u * sizeof(types::storage_value_t), "dense descriptor byte count changed");
    require(formats::allocate(&matrix) != 0, "dense allocation failed");
    require(matrix.storage == matrix.val, "dense value pointer no longer begins at owned storage");
    *formats::at(&matrix, 1u, 2u) = __float2half(7.0f);
    require(close_half(*formats::at(&matrix, 1u, 2u), 7.0f), "dense row-major mapping changed");
    require(formats::at(&matrix, 2u, 0u) == nullptr, "dense bounds check changed");
    formats::clear(&matrix);

    __half columns[8]{};
    formats::attach(&matrix, 3u, 2u, columns, formats::dense_col_major, 4u);
    columns[1u * 4u + 2u] = __float2half(9.0f);
    require(formats::offset(&matrix, 2u, 1u) == 6u, "dense column-major offset changed");
    require(close_half(*formats::at(&matrix, 2u, 1u), 9.0f), "dense column-major mapping changed");
}

void check_compressed() {
    formats::CompressedPayload matrix{};
    formats::init(&matrix, 3u, 4u, 3u, formats::compressed_by_row);
    require(formats::allocate(&matrix) != 0, "compressed allocation failed");
    require(matrix.storage == matrix.majorPtr, "compressed major pointer base changed");
    const std::size_t ptr_bytes = 4u * sizeof(types::ptr_t);
    const std::size_t minor_offset = align_up(ptr_bytes, alignof(types::idx_t));
    const std::size_t value_offset = align_up(minor_offset + 3u * sizeof(types::idx_t), alignof(types::storage_value_t));
    require(reinterpret_cast<unsigned char *>(matrix.minorIdx) == static_cast<unsigned char *>(matrix.storage) + minor_offset,
            "compressed minor-index offset changed");
    require(reinterpret_cast<unsigned char *>(matrix.val) == static_cast<unsigned char *>(matrix.storage) + value_offset,
            "compressed value offset changed");
    const types::ptr_t row_ptr[] = {0u, 2u, 2u, 3u};
    const types::idx_t columns[] = {0u, 3u, 2u};
    std::memcpy(matrix.majorPtr, row_ptr, sizeof(row_ptr));
    std::memcpy(matrix.minorIdx, columns, sizeof(columns));
    matrix.val[0] = __float2half(1.0f);
    matrix.val[1] = __float2half(3.0f);
    matrix.val[2] = __float2half(6.0f);
    require(formats::bytes(&matrix) == sizeof(matrix) + 4u * sizeof(types::ptr_t)
                + 3u * sizeof(types::idx_t) + 3u * sizeof(types::storage_value_t),
            "compressed descriptor/payload byte count changed");
    require(formats::major_dim(&matrix) == 3u && formats::minor_dim(&matrix) == 4u, "compressed axes changed");
    require(close_half(*formats::at(&matrix, 0u, 3u), 3.0f), "compressed row lookup changed");
    require(formats::at(&matrix, 1u, 0u) == nullptr, "compressed absent lookup changed");
    formats::clear(&matrix);
}

void check_coo() {
    formats::CooPayload matrix{};
    formats::init(&matrix, 2u, 3u, 2u);
    require(formats::allocate(&matrix) != 0, "COO allocation failed");
    require(matrix.storage == matrix.rowIdx, "COO row-index base changed");
    require(formats::bytes(&matrix) == sizeof(matrix) + 4u * sizeof(types::idx_t)
                + 2u * sizeof(types::storage_value_t),
            "COO descriptor/payload byte count changed");
    matrix.rowIdx[0] = 1u;
    matrix.colIdx[0] = 2u;
    matrix.val[0] = __float2half(5.0f);
    matrix.rowIdx[1] = 0u;
    matrix.colIdx[1] = 1u;
    matrix.val[1] = __float2half(4.0f);
    require(close_half(*formats::at(&matrix, 1u, 2u), 5.0f), "COO coordinate lookup changed");
    require(formats::at(&matrix, 1u, 1u) == nullptr, "COO absent lookup changed");
    formats::clear(&matrix);
}

void check_blocked_ell() {
    formats::BlockedEllPayload matrix{};
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
    formats::SlicedEllPayload matrix{};
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
    check_dense();
    check_compressed();
    check_coo();
    check_blocked_ell();
    check_sliced_ell();
    return 0;
}
