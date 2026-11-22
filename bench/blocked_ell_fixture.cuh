#pragma once

#include <Cellerator/tensor/base/compressed.hh>
#include <Cellerator/tensor/base/device_views.cuh>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace cellerator::bench {

using BlockedEllColumns = std::vector<std::vector<std::uint32_t>>;

// Discover the sorted block columns and return the maximum number of blocks
// in a row block. The caller uses this extent to provision the packed storage.
inline std::uint32_t plan_blocked_ell_columns(
    const tensor::base::CompressedStructureView<std::uint32_t> &src,
    std::uint32_t block_size, BlockedEllColumns &block_cols) {
    if (block_size == 0u || src.rows == 0u || src.cols == 0u) return 0u;
    const std::uint32_t row_blocks = (src.rows + block_size - 1u) / block_size;
    const std::uint32_t col_blocks = (src.cols + block_size - 1u) / block_size;
    block_cols.resize(row_blocks);
    std::vector<unsigned char> seen(col_blocks, 0u);
    std::uint32_t width = 1u;

    for (std::uint32_t rb = 0u; rb < row_blocks; ++rb) {
        auto &columns = block_cols[rb];
        columns.clear();
        const std::uint32_t begin = rb * block_size;
        const auto end = begin + std::min<std::uint64_t>(src.rows - begin, block_size);
        for (std::uint32_t row = begin; row < end; ++row) {
            for (std::uint32_t off = src.major_ptr[row]; off < src.major_ptr[row + 1u]; ++off) {
                const std::uint32_t cb = src.minor_idx[off] / block_size;
                if (cb >= col_blocks || seen[cb]) continue;
                seen[cb] = 1u;
                columns.push_back(cb);
            }
        }
        std::sort(columns.begin(), columns.end());
        width = std::max(width, static_cast<std::uint32_t>(columns.size()));
        for (std::uint32_t cb : columns) seen[cb] = 0u;
    }
    return width;
}

inline std::size_t blocked_ell_index_count(const tensor::base::device::BlockedEllView &matrix) {
    const std::size_t row_blocks = (std::size_t(matrix.rows) + matrix.block_size - 1u) / matrix.block_size;
    return row_blocks * (matrix.ell_cols / matrix.block_size);
}

// dst borrows the caller's fully sized host arrays. Keep CSR visit order
// so duplicate scalar entries retain the legacy last-write behavior.
inline void fill_blocked_ell(
    const tensor::base::CompressedView<std::uint32_t, const __half> &src,
    const BlockedEllColumns &block_cols, tensor::base::device::BlockedEllView dst) {
    const std::uint32_t width = dst.ell_cols / dst.block_size;
    std::fill_n(dst.blockColIdx, blocked_ell_index_count(dst), tensor::base::blocked_ell_invalid_col);
    std::memset(dst.val, 0, std::size_t(dst.rows) * dst.ell_cols * sizeof(__half));
    for (std::size_t rb = 0; rb < block_cols.size(); ++rb) {
        const auto &columns = block_cols[rb];
        std::copy(columns.begin(), columns.end(), dst.blockColIdx + rb * width);
    }
    for (std::uint32_t row = 0u; row < dst.rows; ++row) {
        const auto &columns = block_cols[row / dst.block_size];
        for (std::uint32_t off = src.structure.major_ptr[row]; off < src.structure.major_ptr[row + 1u]; ++off) {
            const std::uint32_t col = src.structure.minor_idx[off];
            const std::uint32_t cb = col / dst.block_size;
            const auto slot = std::find(columns.begin(), columns.end(), cb);
            if (slot == columns.end()) continue;
            const std::uint32_t local_col = col - cb * dst.block_size;
            dst.val[std::size_t(row) * dst.ell_cols + (slot - columns.begin()) * dst.block_size + local_col] =
                src.values[off];
        }
    }
}

} // namespace cellerator::bench
