#pragma once

#include <CellShard/access/adapter.cuh>

#include <Cellerator/tensor/base/blocked_ell.cuh>
#include <Cellerator/tensor/base/compressed.cuh>
#include <Cellerator/tensor/base/dense.cuh>
#include <Cellerator/interop/formats/affine_blocked_ell.hh>
#include <Cellerator/tensor/base/sliced_ell.cuh>

#include <cstddef>
#include <cstdint>

namespace cellerator::interop {

template<class MatrixT>
struct cellshard_matrix_binding {
    const MatrixT *archive;
    MatrixT *pack;
    std::uint32_t assay_id;
};

template<class MatrixT>
struct cellshard_matrix_traits;

template<>
struct cellshard_matrix_traits<tensor::base::dense> {
    static constexpr cellshard::disk_format archive_format = cellshard::disk_format_dense;
    static constexpr std::uint32_t execution_format = cellshard::dataset_execution_format_dense;
    static constexpr std::uint64_t capabilities =
        cellshard::access::capability_contiguous_payload |
        cellshard::access::capability_cell_span_copy |
        cellshard::access::capability_feature_span_copy;

    static std::uint64_t rows(const tensor::base::dense *m) { return m != nullptr ? m->rows : 0u; }
    static std::uint64_t cols(const tensor::base::dense *m) { return m != nullptr ? m->cols : 0u; }
    static std::uint64_t nnz(const tensor::base::dense *m) { return m != nullptr ? (std::uint64_t) m->rows * (std::uint64_t) m->cols : 0u; }
    static std::size_t payload_bytes(const tensor::base::dense *m) { return m != nullptr ? tensor::base::payload_bytes(m) : 0u; }
    static std::uint32_t value_code(const tensor::base::dense *) { return cellerator::types::value_code<cellerator::types::storage_value_t>::code; }
};

template<>
struct cellshard_matrix_traits<tensor::base::compressed> {
    static constexpr cellshard::disk_format archive_format = cellshard::disk_format_compressed;
    static constexpr std::uint32_t execution_format = cellshard::dataset_execution_format_compressed;
    static constexpr std::uint64_t capabilities =
        cellshard::access::capability_contiguous_payload |
        cellshard::access::capability_cell_span_copy |
        cellshard::access::capability_feature_span_copy |
        cellshard::access::capability_pinned_host_staging;

    static std::uint64_t rows(const tensor::base::compressed *m) { return m != nullptr ? m->rows : 0u; }
    static std::uint64_t cols(const tensor::base::compressed *m) { return m != nullptr ? m->cols : 0u; }
    static std::uint64_t nnz(const tensor::base::compressed *m) { return m != nullptr ? m->nnz : 0u; }
    static std::size_t payload_bytes(const tensor::base::compressed *m) { return m != nullptr ? tensor::base::bytes(m) : 0u; }
    static std::uint32_t value_code(const tensor::base::compressed *) { return cellerator::types::value_code<cellerator::types::storage_value_t>::code; }
};

template<>
struct cellshard_matrix_traits<tensor::base::blocked_ell> {
    static constexpr cellshard::disk_format archive_format = cellshard::disk_format_blocked_ell;
    static constexpr std::uint32_t execution_format = cellshard::dataset_execution_format_blocked_ell;
    static constexpr std::uint64_t capabilities =
        cellshard::access::capability_contiguous_payload |
        cellshard::access::capability_pinned_host_staging;

    static std::uint64_t rows(const tensor::base::blocked_ell *m) { return m != nullptr ? m->rows : 0u; }
    static std::uint64_t cols(const tensor::base::blocked_ell *m) { return m != nullptr ? m->cols : 0u; }
    static std::uint64_t nnz(const tensor::base::blocked_ell *m) { return m != nullptr ? m->nnz : 0u; }
    static std::size_t payload_bytes(const tensor::base::blocked_ell *m) { return m != nullptr ? tensor::base::bytes(m) : 0u; }
    static std::uint32_t value_code(const tensor::base::blocked_ell *) { return cellerator::types::value_code<cellerator::types::storage_value_t>::code; }
};

template<>
struct cellshard_matrix_traits<tensor::base::sliced_ell> {
    static constexpr cellshard::disk_format archive_format = cellshard::disk_format_sliced_ell;
    static constexpr std::uint32_t execution_format = cellshard::dataset_execution_format_sliced_ell;
    static constexpr std::uint64_t capabilities =
        cellshard::access::capability_contiguous_payload |
        cellshard::access::capability_pinned_host_staging;

    static std::uint64_t rows(const tensor::base::sliced_ell *m) { return m != nullptr ? m->rows : 0u; }
    static std::uint64_t cols(const tensor::base::sliced_ell *m) { return m != nullptr ? m->cols : 0u; }
    static std::uint64_t nnz(const tensor::base::sliced_ell *m) { return m != nullptr ? m->nnz : 0u; }
    static std::size_t payload_bytes(const tensor::base::sliced_ell *m) { return m != nullptr ? tensor::base::bytes(m) : 0u; }
    static std::uint32_t value_code(const tensor::base::sliced_ell *) { return cellerator::types::value_code<cellerator::types::storage_value_t>::code; }
};

template<>
struct cellshard_matrix_traits<formats::AffineBlockedEllCodec> {
    static constexpr cellshard::disk_format archive_format = cellshard::disk_format_quantized_blocked_ell;
    static constexpr std::uint32_t execution_format = cellshard::dataset_execution_format_quantized_blocked_ell;
    static constexpr std::uint64_t capabilities =
        cellshard::access::capability_contiguous_payload |
        cellshard::access::capability_pinned_host_staging;

    static std::uint64_t rows(const formats::AffineBlockedEllCodec *m) { return m != nullptr ? m->rows : 0u; }
    static std::uint64_t cols(const formats::AffineBlockedEllCodec *m) { return m != nullptr ? m->cols : 0u; }
    static std::uint64_t nnz(const formats::AffineBlockedEllCodec *m) { return m != nullptr ? m->nnz : 0u; }
    static std::size_t payload_bytes(const formats::AffineBlockedEllCodec *m) { return m != nullptr ? formats::bytes(m) : 0u; }
    static std::uint32_t value_code(const formats::AffineBlockedEllCodec *) { return 0u; }
};

template<class MatrixT>
inline cellshard_matrix_binding<MatrixT> make_cellshard_matrix_binding(
    const MatrixT &archive,
    std::uint32_t assay_id = 0u) {
    return cellshard_matrix_binding<MatrixT>{&archive, nullptr, assay_id};
}

template<class MatrixT>
inline cellshard_matrix_binding<MatrixT> make_cellshard_matrix_binding(
    const MatrixT &archive,
    MatrixT &pack,
    std::uint32_t assay_id = 0u) {
    return cellshard_matrix_binding<MatrixT>{&archive, &pack, assay_id};
}

} // namespace cellerator::interop

namespace cellshard::access {

template<>
struct payload_traits<cellerator::tensor::base::blocked_ell> {
    using matrix_type = cellerator::tensor::base::blocked_ell;

    __host__ __device__ __forceinline__ static std::uint64_t rows(const matrix_type *matrix) { return matrix != nullptr ? matrix->rows : 0u; }
    __host__ __device__ __forceinline__ static std::uint64_t cols(const matrix_type *matrix) { return matrix != nullptr ? matrix->cols : 0u; }
    __host__ __device__ __forceinline__ static std::uint64_t nnz(const matrix_type *matrix) { return matrix != nullptr ? matrix->nnz : 0u; }

    __host__ __device__ __forceinline__ static std::uint64_t aux(const matrix_type *matrix) {
        return matrix != nullptr
            ? cellerator::tensor::base::pack_blocked_ell_aux(matrix->block_size, cellerator::tensor::base::ell_width_blocks(matrix))
            : 0u;
    }

    __host__ __device__ __forceinline__ static std::size_t host_bytes(
        const matrix_type *matrix,
        std::uint64_t rows,
        std::uint64_t,
        std::uint64_t,
        std::uint64_t aux_value) {
        const std::uint32_t block_size = cellerator::tensor::base::unpack_blocked_ell_block_size(aux_value);
        const std::uint64_t ell_width = cellerator::tensor::base::unpack_blocked_ell_ell_width(aux_value);
        return matrix != nullptr ? cellerator::tensor::base::bytes(matrix)
                                 : sizeof(matrix_type)
                                     + static_cast<std::size_t>((rows + block_size - 1u) / block_size) * static_cast<std::size_t>(ell_width) * sizeof(cellerator::types::idx_t)
                                     + static_cast<std::size_t>(rows) * static_cast<std::size_t>(ell_width * block_size) * sizeof(cellerator::real::storage_t);
    }

    __host__ __device__ __forceinline__ static const cellshard::types::storage_value_t *debug_at(
        const matrix_type *matrix,
        std::uint64_t row,
        cellshard::types::idx_t col) {
        return cellerator::tensor::base::at(matrix, static_cast<cellerator::types::dim_t>(row), col);
    }
};

template<>
struct payload_traits<cellerator::tensor::base::sliced_ell> {
    using matrix_type = cellerator::tensor::base::sliced_ell;

    __host__ __device__ __forceinline__ static std::uint64_t rows(const matrix_type *matrix) { return matrix != nullptr ? matrix->rows : 0u; }
    __host__ __device__ __forceinline__ static std::uint64_t cols(const matrix_type *matrix) { return matrix != nullptr ? matrix->cols : 0u; }
    __host__ __device__ __forceinline__ static std::uint64_t nnz(const matrix_type *matrix) { return matrix != nullptr ? matrix->nnz : 0u; }

    __host__ __device__ __forceinline__ static std::uint64_t aux(const matrix_type *matrix) {
        return matrix != nullptr
            ? cellerator::tensor::base::pack_sliced_ell_aux(matrix->slice_count, cellerator::tensor::base::total_slots(matrix))
            : 0u;
    }

    __host__ __device__ __forceinline__ static std::size_t host_bytes(
        const matrix_type *matrix,
        std::uint64_t,
        std::uint64_t,
        std::uint64_t,
        std::uint64_t aux_value) {
        const std::uint32_t slice_count = cellerator::tensor::base::unpack_sliced_ell_slice_count(aux_value);
        const std::uint32_t total_slot_count = cellerator::tensor::base::unpack_sliced_ell_total_slots(aux_value);
        return matrix != nullptr ? cellerator::tensor::base::bytes(matrix)
                                 : sizeof(matrix_type)
                                     + static_cast<std::size_t>(slice_count + 1u) * sizeof(cellerator::types::u32)
                                     + static_cast<std::size_t>(slice_count) * sizeof(cellerator::types::u32)
                                     + static_cast<std::size_t>(total_slot_count) * sizeof(cellerator::types::idx_t)
                                     + static_cast<std::size_t>(total_slot_count) * sizeof(cellerator::real::storage_t);
    }

    __host__ __device__ __forceinline__ static const cellshard::types::storage_value_t *debug_at(
        const matrix_type *matrix,
        std::uint64_t row,
        cellshard::types::idx_t col) {
        return cellerator::tensor::base::at(matrix, static_cast<cellerator::types::dim_t>(row), col);
    }
};

template<>
struct payload_traits<cellerator::interop::formats::AffineBlockedEllCodec> {
    using matrix_type = cellerator::interop::formats::AffineBlockedEllCodec;

    __host__ __device__ __forceinline__ static std::uint64_t rows(const matrix_type *matrix) { return matrix != nullptr ? matrix->rows : 0u; }
    __host__ __device__ __forceinline__ static std::uint64_t cols(const matrix_type *matrix) { return matrix != nullptr ? matrix->cols : 0u; }
    __host__ __device__ __forceinline__ static std::uint64_t nnz(const matrix_type *matrix) { return matrix != nullptr ? matrix->nnz : 0u; }

    __host__ __device__ __forceinline__ static std::uint64_t aux(const matrix_type *matrix) {
        return matrix != nullptr
            ? cellerator::interop::formats::pack_quantized_blocked_ell_aux(matrix->bits, matrix->block_size, cellerator::interop::formats::ell_width_blocks(matrix))
            : 0u;
    }

    __host__ __device__ __forceinline__ static std::size_t host_bytes(
        const matrix_type *matrix,
        std::uint64_t rows,
        std::uint64_t cols,
        std::uint64_t,
        std::uint64_t aux_value) {
        const std::uint64_t bits = cellerator::interop::formats::unpack_quantized_blocked_ell_bits(aux_value);
        const std::uint64_t block_size = cellerator::interop::formats::unpack_quantized_blocked_ell_block_size(aux_value);
        const std::uint64_t ell_cols = cellerator::interop::formats::unpack_quantized_blocked_ell_cols(aux_value);
        const std::uint64_t row_stride_bytes = cellerator::interop::formats::quantized_blocked_ell_aligned_row_bytes(
            static_cast<cellerator::types::u32>(bits),
            static_cast<cellerator::types::u32>(ell_cols));
        const std::uint64_t row_blocks = block_size == 0u ? 0u : (rows + block_size - 1u) / block_size;
        const std::uint64_t ell_width = block_size == 0u ? 0u : ell_cols / block_size;
        return matrix != nullptr ? cellerator::interop::formats::bytes(matrix)
                                 : sizeof(matrix_type)
                                     + static_cast<std::size_t>(row_blocks * ell_width) * sizeof(cellerator::types::idx_t)
                                     + static_cast<std::size_t>(rows * row_stride_bytes)
                                     + static_cast<std::size_t>(cols) * sizeof(float)
                                     + static_cast<std::size_t>(cols) * sizeof(float)
                                     + static_cast<std::size_t>(rows) * sizeof(float);
    }

    __host__ __device__ __forceinline__ static const cellshard::types::storage_value_t *debug_at(
        const matrix_type *,
        std::uint64_t,
        cellshard::types::idx_t) {
        return nullptr;
    }
};

template<class MatrixT>
struct archive_adapter<cellerator::interop::cellshard_matrix_binding<MatrixT>> {
    using binding_type = cellerator::interop::cellshard_matrix_binding<MatrixT>;
    using traits = cellerator::interop::cellshard_matrix_traits<MatrixT>;

    static archive_descriptor describe(const adapter_view<binding_type> &view) {
        const MatrixT *matrix = view.binding != nullptr ? view.binding->archive : nullptr;
        if (matrix == nullptr) return archive_descriptor{};
        return archive_descriptor{
            traits::rows(matrix),
            traits::cols(matrix),
            traits::nnz(matrix),
            view.binding->assay_id,
            traits::value_code(matrix),
            traits::archive_format,
            traits::execution_format,
            traits::capabilities | capability_archive_to_pack
        };
    }
};

template<class MatrixT>
struct pack_adapter<cellerator::interop::cellshard_matrix_binding<MatrixT>> {
    using binding_type = cellerator::interop::cellshard_matrix_binding<MatrixT>;
    using traits = cellerator::interop::cellshard_matrix_traits<MatrixT>;

    static pack_descriptor describe(const adapter_view<binding_type> &view) {
        const MatrixT *matrix = view.binding != nullptr
            ? (view.binding->pack != nullptr ? view.binding->pack : view.binding->archive)
            : nullptr;
        if (matrix == nullptr) return pack_descriptor{};
        return pack_descriptor{
            traits::rows(matrix),
            traits::cols(matrix),
            traits::nnz(matrix),
            traits::value_code(matrix),
            traits::execution_format,
            traits::payload_bytes(matrix),
            traits::capabilities | capability_pack_read | capability_pack_write
        };
    }
};

} // namespace cellshard::access
