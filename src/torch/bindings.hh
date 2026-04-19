#pragma once

#include "../../extern/CellShard/include/CellShard/formats/compressed.cuh"
#include "../../extern/CellShard/include/CellShard/runtime/layout/sharded.cuh"

#include <torch/torch.h>

#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace cellerator::torch_bindings {

// This bridge is intentionally narrow and intentionally explicit about cost.
//
// It exists to let downstream model code treat Cellerator / CellShard data as a
// libtorch-facing library boundary. It is not a replacement for the sparse
// runtime, staging layer, or preprocess kernels.
//
// Important behavioral contract:
// - export always materializes a brand-new CPU tensor
// - CSR / CellShard 32-bit metadata is widened to Torch 64-bit indices
// - values are copied out of CellShard-owned storage
// - no fetch, stage, upload, or device-side execution happens here
//
// In practice this means:
// - good: explicit interop at dataset or model boundaries
// - bad: calling this inside per-step training loops or other hot paths
struct ExportOptions {
    // Default to the current storage dtype so the bridge stays as faithful as
    // possible to the host payload. Callers can widen to float32 if they want
    // easier downstream math at the cost of more host traffic.
    torch::ScalarType value_dtype = torch::kFloat16;
};

namespace detail {

inline std::int64_t checked_i64_(unsigned long value, const char *label) {
    if (value > static_cast<unsigned long>(std::numeric_limits<std::int64_t>::max())) {
        throw std::overflow_error(std::string(label) + " does not fit into int64");
    }
    return static_cast<std::int64_t>(value);
}

inline std::int64_t checked_i64_(std::uint32_t value, const char *) {
    return static_cast<std::int64_t>(value);
}

inline void validate_export_dtype_(torch::ScalarType dtype) {
    if (dtype != torch::kFloat16 && dtype != torch::kFloat32) {
        throw std::invalid_argument("torch_bindings export only supports float16 or float32 values");
    }
}

inline void require_row_compressed_part_(const cellshard::sparse::compressed &part, const char *label) {
    if (part.axis != cellshard::sparse::compressed_by_row) {
        throw std::invalid_argument(std::string(label) + " requires row-compressed CSR input");
    }
    if ((part.rows != 0 && part.majorPtr == 0) || (part.nnz != 0 && (part.minorIdx == 0 || part.val == 0))) {
        throw std::invalid_argument(std::string(label) + " requires materialized host arrays");
    }
}

inline torch::Tensor allocate_value_tensor_(std::int64_t nnz, const ExportOptions &options) {
    validate_export_dtype_(options.value_dtype);
    return torch::empty(
        { nnz },
        torch::TensorOptions().dtype(options.value_dtype).device(torch::kCPU));
}

inline void copy_value_payload_(
    const ::real::storage_t *src,
    std::int64_t nnz,
    const ExportOptions &options,
    void *dst,
    std::int64_t dst_offset = 0) {
    if (dst == nullptr) throw std::invalid_argument("destination buffer pointer must be non-null");
    if (nnz == 0) return;

    static_assert(std::is_same<::real::storage_t, __half>::value,
                  "torch_bindings currently assumes CellShard host storage_t is __half");

    if (options.value_dtype == torch::kFloat16) {
        static_assert(sizeof(at::Half) == sizeof(::real::storage_t), "ATen half storage must match CellShard storage");
        auto *dst_ptr = static_cast<at::Half *>(dst) + dst_offset;
        std::memcpy(dst_ptr, src, static_cast<std::size_t>(nnz) * sizeof(::real::storage_t));
        return;
    }

    float *dst_ptr = static_cast<float *>(dst) + dst_offset;
    for (std::int64_t i = 0; i < nnz; ++i) {
        dst_ptr[i] = __half2float(src[i]);
    }
}

} // namespace detail

// Export one already-materialized CellShard CSR part into a new CPU
// torch::sparse_csr_tensor.
//
// This helper is deliberately copy-based. Using from_blob-style aliasing here
// would hide lifetime coupling to CellShard-owned buffers and make accidental
// hot-path usage easier. The explicit copy keeps ownership and performance
// semantics obvious.
inline torch::Tensor export_as_tensor(
    const cellshard::sparse::compressed &part,
    const ExportOptions &options = ExportOptions()) {
    detail::require_row_compressed_part_(part, "export_as_tensor(part)");

    const std::int64_t rows = detail::checked_i64_(part.rows, "rows");
    const std::int64_t cols = detail::checked_i64_(part.cols, "cols");
    const std::int64_t nnz = detail::checked_i64_(part.nnz, "nnz");

    torch::Tensor crow_tensor = torch::empty(
        { rows + 1 },
        torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    torch::Tensor col_tensor = torch::empty(
        { nnz },
        torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    torch::Tensor value_tensor = detail::allocate_value_tensor_(nnz, options);

    std::int64_t *crow_ptr = crow_tensor.data_ptr<std::int64_t>();
    std::int64_t *col_ptr = col_tensor.data_ptr<std::int64_t>();

    for (std::int64_t row = 0; row <= rows; ++row) {
        crow_ptr[row] = static_cast<std::int64_t>(part.majorPtr[row]);
    }
    for (std::int64_t i = 0; i < nnz; ++i) {
        col_ptr[i] = static_cast<std::int64_t>(part.minorIdx[i]);
    }
    detail::copy_value_payload_(part.val, nnz, options, value_tensor.data_ptr(), 0);

    return torch::sparse_csr_tensor(
        crow_tensor,
        col_tensor,
        value_tensor,
        { rows, cols },
        torch::TensorOptions().dtype(options.value_dtype).device(torch::kCPU));
}

// Export a loaded sharded CSR view into one stitched CPU
// torch::sparse_csr_tensor.
//
// This is even more expensive than the single-part export because it walks
// every loaded part, widens every index, and builds a fresh global CSR payload.
// The only sane place to use this is at an explicit interop boundary, for
// example "hand this matrix to a Torch-native prototype model".
inline torch::Tensor export_as_tensor(
    const cellshard::sharded<cellshard::sparse::compressed> &view,
    const ExportOptions &options = ExportOptions()) {
    const std::int64_t rows = detail::checked_i64_(view.rows, "rows");
    const std::int64_t cols = detail::checked_i64_(view.cols, "cols");
    const std::int64_t nnz = detail::checked_i64_(view.nnz, "nnz");

    torch::Tensor crow_tensor = torch::empty(
        { rows + 1 },
        torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    torch::Tensor col_tensor = torch::empty(
        { nnz },
        torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    torch::Tensor value_tensor = detail::allocate_value_tensor_(nnz, options);

    std::int64_t *crow_ptr = crow_tensor.data_ptr<std::int64_t>();
    std::int64_t *col_ptr = col_tensor.data_ptr<std::int64_t>();
    std::int64_t nnz_cursor = 0;
    std::int64_t row_cursor = 0;

    crow_ptr[0] = 0;
    for (unsigned long partition_id = 0; partition_id < view.num_partitions; ++partition_id) {
        const cellshard::sparse::compressed *part = view.parts != 0 ? view.parts[partition_id] : 0;
        if (part == 0) {
            throw std::invalid_argument("export_as_tensor(sharded) requires every part to already be loaded on host");
        }
        detail::require_row_compressed_part_(*part, "export_as_tensor(sharded)");

        for (std::uint32_t local_row = 0; local_row < part->rows; ++local_row) {
            crow_ptr[row_cursor + static_cast<std::int64_t>(local_row) + 1] =
                nnz_cursor + static_cast<std::int64_t>(part->majorPtr[local_row + 1]);
        }
        for (std::uint32_t i = 0; i < part->nnz; ++i) {
            col_ptr[nnz_cursor + static_cast<std::int64_t>(i)] = static_cast<std::int64_t>(part->minorIdx[i]);
        }
        detail::copy_value_payload_(
            part->val,
            static_cast<std::int64_t>(part->nnz),
            options,
            value_tensor.data_ptr(),
            nnz_cursor);
        nnz_cursor += static_cast<std::int64_t>(part->nnz);
        row_cursor += static_cast<std::int64_t>(part->rows);
    }

    if (row_cursor != rows) {
        throw std::runtime_error("export_as_tensor(sharded) observed a row count mismatch while stitching parts");
    }
    if (nnz_cursor != nnz) {
        throw std::runtime_error("export_as_tensor(sharded) observed an nnz mismatch while stitching parts");
    }

    return torch::sparse_csr_tensor(
        crow_tensor,
        col_tensor,
        value_tensor,
        { rows, cols },
        torch::TensorOptions().dtype(options.value_dtype).device(torch::kCPU));
}

} // namespace cellerator::torch_bindings
