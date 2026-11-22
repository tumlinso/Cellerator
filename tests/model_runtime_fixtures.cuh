#pragma once

#include <cstddef>

#if defined(__CUDACC__)
#include <Cellerator/memory/cuda/resources.hh>
#include <Cellerator/tensor/base/device_views.cuh>

#include <cuda_fp16.h>

#include <stdexcept>

#endif

namespace model_test {

inline void derive_sliced_slot_offsets(
    unsigned int slice_count,
    const unsigned int *slice_row_offsets,
    const unsigned int *slice_widths,
    unsigned int *slot_offsets) {
    unsigned int slots = 0u;
    for (unsigned int slice = 0u; slice < slice_count; ++slice) {
        slot_offsets[slice] = slots;
        slots += (slice_row_offsets[slice + 1u] - slice_row_offsets[slice])
            * slice_widths[slice];
    }
}

#if defined(__CUDACC__)

namespace memory_cuda = ::cellerator::memory::cuda;
using BlockedEllView = ::cellerator::tensor::base::device::BlockedEllView;
using SlicedEllView = ::cellerator::tensor::base::device::SlicedEllView;

// The model fixtures use one-by-one blocks and one row per slice. Both
// layouts then borrow the same row-packed columns and values on the host.
template<unsigned int Rows, unsigned int Columns>
struct DenseEllFixture {
    explicit DenseEllFixture(const float *dense) {
        for (unsigned int row = 0u; row < Rows; ++row) {
            row_offsets[row] = row;
            widths[row] = Columns;
            for (unsigned int column = 0u; column < Columns; ++column) {
                const std::size_t slot = std::size_t(row) * Columns + column;
                indices[slot] = column;
                values[slot] = __float2half_rn(dense[slot]);
            }
        }
        row_offsets[Rows] = Rows;
        derive_sliced_slot_offsets(Rows, row_offsets, widths, slot_offsets);
    }

    BlockedEllView blocked() {
        return {Rows, Columns, Rows * Columns, 1u, Columns, indices, values};
    }

    SlicedEllView sliced() {
        return {Rows, Columns, Rows * Columns, Rows, 1u,
            row_offsets, widths, slot_offsets, indices, values};
    }

private:
    unsigned int indices[Rows * Columns];
    __half values[Rows * Columns];
    unsigned int row_offsets[Rows + 1u];
    unsigned int widths[Rows];
    unsigned int slot_offsets[Rows];
};

struct BlockedEllUpload {
    memory_cuda::DeviceBuffer indices;
    memory_cuda::DeviceBuffer values;
    BlockedEllView view{};

    BlockedEllUpload() = default;
    BlockedEllUpload(const BlockedEllUpload &) = delete;
    BlockedEllUpload &operator=(const BlockedEllUpload &) = delete;
    BlockedEllUpload(BlockedEllUpload &&) noexcept = default;
    BlockedEllUpload &operator=(BlockedEllUpload &&) noexcept = default;
};

struct SlicedEllUpload {
    memory_cuda::DeviceBuffer slice_row_offsets;
    memory_cuda::DeviceBuffer slice_widths;
    memory_cuda::DeviceBuffer slice_slot_offsets;
    memory_cuda::DeviceBuffer col_idx;
    memory_cuda::DeviceBuffer values;
    SlicedEllView view{};

    SlicedEllUpload() = default;
    SlicedEllUpload(const SlicedEllUpload &) = delete;
    SlicedEllUpload &operator=(const SlicedEllUpload &) = delete;
    SlicedEllUpload(SlicedEllUpload &&) noexcept = default;
    SlicedEllUpload &operator=(SlicedEllUpload &&) noexcept = default;
};

inline BlockedEllUpload upload_blocked_ell(
    const BlockedEllView &host,
    memory_cuda::Stream &stream) {
    if (host.block_size == 0u)
        throw std::invalid_argument("Blocked ELL block size is zero");
    const std::size_t block_rows =
        (std::size_t(host.rows) + host.block_size - 1u) / host.block_size;
    const std::size_t index_count = block_rows * (host.ell_cols / host.block_size);
    const std::size_t value_count = std::size_t(host.rows) * std::size_t(host.ell_cols);
    const std::size_t index_bytes = memory_cuda::checked_bytes(
        index_count, sizeof(*host.blockColIdx));
    const std::size_t value_bytes = memory_cuda::checked_bytes(value_count, sizeof(__half));

    BlockedEllUpload out;
    out.indices = memory_cuda::DeviceBuffer(index_bytes, stream.device());
    out.values = memory_cuda::DeviceBuffer(value_bytes, stream.device());
    out.indices.upload_async(host.blockColIdx, index_bytes, stream);
    out.values.upload_async(host.val, value_bytes, stream);
    out.view = {
        host.rows,
        host.cols,
        host.nnz,
        host.block_size,
        host.ell_cols,
        static_cast<unsigned int *>(out.indices.data()),
        static_cast<__half *>(out.values.data())
    };
    stream.synchronize();
    return out;
}

inline SlicedEllUpload upload_sliced_ell(
    const SlicedEllView &host,
    memory_cuda::Stream &stream) {
    if (host.slice_count != 0u && (host.slice_row_offsets == nullptr
        || host.slice_widths == nullptr || host.slice_slot_offsets == nullptr))
        throw std::invalid_argument("sliced ELL metadata is missing");

    const unsigned int empty_row_offset = 0u;
    const unsigned int *row_offsets = host.slice_row_offsets != nullptr
        ? host.slice_row_offsets
        : &empty_row_offset;
    const std::size_t offset_count = std::size_t(host.slice_count) + 1u;
    std::size_t slot_count = 0u;
    if (host.slice_count != 0u) {
        const auto last = host.slice_count - 1u;
        const auto last_rows = row_offsets[last + 1u] - row_offsets[last];
        slot_count = host.slice_slot_offsets[last] + std::size_t(last_rows) * host.slice_widths[last];
    }
    const std::size_t offset_bytes = memory_cuda::checked_bytes(
        offset_count, sizeof(*host.slice_row_offsets));
    const std::size_t width_bytes = memory_cuda::checked_bytes(
        host.slice_count, sizeof(*host.slice_widths));
    const std::size_t slot_offset_bytes = memory_cuda::checked_bytes(
        host.slice_count, sizeof(*host.slice_slot_offsets));
    const std::size_t index_bytes = memory_cuda::checked_bytes(
        slot_count, sizeof(*host.col_idx));
    const std::size_t value_bytes = memory_cuda::checked_bytes(slot_count, sizeof(__half));

    SlicedEllUpload out;
    out.slice_row_offsets = memory_cuda::DeviceBuffer(offset_bytes, stream.device());
    out.slice_widths = memory_cuda::DeviceBuffer(width_bytes, stream.device());
    out.slice_slot_offsets = memory_cuda::DeviceBuffer(slot_offset_bytes, stream.device());
    out.col_idx = memory_cuda::DeviceBuffer(index_bytes, stream.device());
    out.values = memory_cuda::DeviceBuffer(value_bytes, stream.device());
    out.slice_row_offsets.upload_async(row_offsets, offset_bytes, stream);
    out.slice_widths.upload_async(host.slice_widths, width_bytes, stream);
    out.slice_slot_offsets.upload_async(host.slice_slot_offsets, slot_offset_bytes, stream);
    out.col_idx.upload_async(host.col_idx, index_bytes, stream);
    out.values.upload_async(host.val, value_bytes, stream);
    out.view = {
        host.rows,
        host.cols,
        host.nnz,
        host.slice_count,
        host.slice_rows,
        static_cast<unsigned int *>(out.slice_row_offsets.data()),
        static_cast<unsigned int *>(out.slice_widths.data()),
        static_cast<unsigned int *>(out.slice_slot_offsets.data()),
        static_cast<unsigned int *>(out.col_idx.data()),
        static_cast<__half *>(out.values.data())
    };
    stream.synchronize();
    return out;
}

#endif

} // namespace model_test
