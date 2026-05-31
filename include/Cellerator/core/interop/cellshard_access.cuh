#pragma once

#include <CellShard/access/adapter.cuh>

#include <Cellerator/core/matrix/blocked_ell.cuh>
#include <Cellerator/core/matrix/compressed.cuh>
#include <Cellerator/core/matrix/dense.cuh>
#include <Cellerator/core/matrix/quantized_blocked_ell.cuh>
#include <Cellerator/core/matrix/sliced_ell.cuh>

#include <cstddef>
#include <cstdint>

namespace cellerator::core::interop {

template<class MatrixT>
struct cellshard_matrix_binding {
    const MatrixT *archive;
    MatrixT *pack;
    std::uint32_t assay_id;
};

template<class MatrixT>
struct cellshard_matrix_traits;

template<>
struct cellshard_matrix_traits<matrix::dense> {
    static constexpr cellshard::disk_format archive_format = cellshard::disk_format_dense;
    static constexpr std::uint32_t execution_format = cellshard::dataset_execution_format_dense;
    static constexpr std::uint64_t capabilities =
        cellshard::access::capability_contiguous_payload |
        cellshard::access::capability_cell_span_copy |
        cellshard::access::capability_feature_span_copy;

    static std::uint64_t rows(const matrix::dense *m) { return m != nullptr ? m->rows : 0u; }
    static std::uint64_t cols(const matrix::dense *m) { return m != nullptr ? m->cols : 0u; }
    static std::uint64_t nnz(const matrix::dense *m) { return m != nullptr ? (std::uint64_t) m->rows * (std::uint64_t) m->cols : 0u; }
    static std::size_t payload_bytes(const matrix::dense *m) { return m != nullptr ? matrix::payload_bytes(m) : 0u; }
    static std::uint32_t value_code(const matrix::dense *) { return cellerator::core::types::value_code<cellerator::core::types::storage_value_t>::code; }
};

template<>
struct cellshard_matrix_traits<matrix::compressed> {
    static constexpr cellshard::disk_format archive_format = cellshard::disk_format_compressed;
    static constexpr std::uint32_t execution_format = cellshard::dataset_execution_format_compressed;
    static constexpr std::uint64_t capabilities =
        cellshard::access::capability_contiguous_payload |
        cellshard::access::capability_cell_span_copy |
        cellshard::access::capability_feature_span_copy |
        cellshard::access::capability_pinned_host_staging;

    static std::uint64_t rows(const matrix::compressed *m) { return m != nullptr ? m->rows : 0u; }
    static std::uint64_t cols(const matrix::compressed *m) { return m != nullptr ? m->cols : 0u; }
    static std::uint64_t nnz(const matrix::compressed *m) { return m != nullptr ? m->nnz : 0u; }
    static std::size_t payload_bytes(const matrix::compressed *m) { return m != nullptr ? matrix::bytes(m) : 0u; }
    static std::uint32_t value_code(const matrix::compressed *) { return cellerator::core::types::value_code<cellerator::core::types::storage_value_t>::code; }
};

template<>
struct cellshard_matrix_traits<matrix::blocked_ell> {
    static constexpr cellshard::disk_format archive_format = cellshard::disk_format_blocked_ell;
    static constexpr std::uint32_t execution_format = cellshard::dataset_execution_format_blocked_ell;
    static constexpr std::uint64_t capabilities =
        cellshard::access::capability_contiguous_payload |
        cellshard::access::capability_pinned_host_staging;

    static std::uint64_t rows(const matrix::blocked_ell *m) { return m != nullptr ? m->rows : 0u; }
    static std::uint64_t cols(const matrix::blocked_ell *m) { return m != nullptr ? m->cols : 0u; }
    static std::uint64_t nnz(const matrix::blocked_ell *m) { return m != nullptr ? m->nnz : 0u; }
    static std::size_t payload_bytes(const matrix::blocked_ell *m) { return m != nullptr ? matrix::bytes(m) : 0u; }
    static std::uint32_t value_code(const matrix::blocked_ell *) { return cellerator::core::types::value_code<cellerator::core::types::storage_value_t>::code; }
};

template<>
struct cellshard_matrix_traits<matrix::sliced_ell> {
    static constexpr cellshard::disk_format archive_format = cellshard::disk_format_sliced_ell;
    static constexpr std::uint32_t execution_format = cellshard::dataset_execution_format_sliced_ell;
    static constexpr std::uint64_t capabilities =
        cellshard::access::capability_contiguous_payload |
        cellshard::access::capability_pinned_host_staging;

    static std::uint64_t rows(const matrix::sliced_ell *m) { return m != nullptr ? m->rows : 0u; }
    static std::uint64_t cols(const matrix::sliced_ell *m) { return m != nullptr ? m->cols : 0u; }
    static std::uint64_t nnz(const matrix::sliced_ell *m) { return m != nullptr ? m->nnz : 0u; }
    static std::size_t payload_bytes(const matrix::sliced_ell *m) { return m != nullptr ? matrix::bytes(m) : 0u; }
    static std::uint32_t value_code(const matrix::sliced_ell *) { return cellerator::core::types::value_code<cellerator::core::types::storage_value_t>::code; }
};

template<>
struct cellshard_matrix_traits<matrix::quantized_blocked_ell> {
    static constexpr cellshard::disk_format archive_format = cellshard::disk_format_quantized_blocked_ell;
    static constexpr std::uint32_t execution_format = cellshard::dataset_execution_format_quantized_blocked_ell;
    static constexpr std::uint64_t capabilities =
        cellshard::access::capability_contiguous_payload |
        cellshard::access::capability_pinned_host_staging;

    static std::uint64_t rows(const matrix::quantized_blocked_ell *m) { return m != nullptr ? m->rows : 0u; }
    static std::uint64_t cols(const matrix::quantized_blocked_ell *m) { return m != nullptr ? m->cols : 0u; }
    static std::uint64_t nnz(const matrix::quantized_blocked_ell *m) { return m != nullptr ? m->nnz : 0u; }
    static std::size_t payload_bytes(const matrix::quantized_blocked_ell *m) { return m != nullptr ? matrix::bytes(m) : 0u; }
    static std::uint32_t value_code(const matrix::quantized_blocked_ell *) { return 0u; }
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

} // namespace cellerator::core::interop

namespace cellshard::access {

template<class MatrixT>
struct archive_adapter<cellerator::core::interop::cellshard_matrix_binding<MatrixT>> {
    using binding_type = cellerator::core::interop::cellshard_matrix_binding<MatrixT>;
    using traits = cellerator::core::interop::cellshard_matrix_traits<MatrixT>;

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
struct pack_adapter<cellerator::core::interop::cellshard_matrix_binding<MatrixT>> {
    using binding_type = cellerator::core::interop::cellshard_matrix_binding<MatrixT>;
    using traits = cellerator::core::interop::cellshard_matrix_traits<MatrixT>;

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
