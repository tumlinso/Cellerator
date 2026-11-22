#pragma once

#include <Cellerator/tensor/base/coo.hh>

#include <ATen/core/Tensor.h>

namespace cellerator::interop::torch {

enum class CooEntryOrder : uint8_t { preserve, assume_sorted_unique };

namespace detail {

tensor::base::CooView<int64_t, const void> borrow_coo_storage(
    const at::Tensor &matrix, at::ScalarType value_type);

} // namespace detail

// Read stored entries of rank-two scalar COO in their original order, including
// repeated coordinates. The caller keeps the source storage live and unchanged,
// and orders CUDA streams through completion. This raw view does not record
// autograd operations. It requires contiguous int64[2, nnz] and values[nnz].
template<class Value>
tensor::base::CooView<int64_t, const Value> borrow_coo_entries(const at::Tensor &matrix) {
    const auto view = detail::borrow_coo_storage(matrix, c10::CppTypeToScalarType<Value>::value);
    return {view.rows, view.cols, view.nnz, view.row_idx, view.col_idx,
            static_cast<const Value *>(view.values)};
}

template<class Value>
tensor::base::CooView<int64_t, const Value> borrow_coo_entries(const at::Tensor &&) = delete;

inline tensor::base::CooView<int64_t, const void> borrow_coo_entries_fp16(const at::Tensor &matrix) {
    static_assert(sizeof(at::Half) == sizeof(uint16_t), "native FP16 views require binary16 storage");
    return detail::borrow_coo_storage(matrix, at::kHalf);
}

inline tensor::base::CooView<int64_t, const void> borrow_coo_entries_fp16(const at::Tensor &&) = delete;

// Wrap contiguous int64[2, nnz] and values[nnz] on the same CPU or CUDA device.
// Coordinates must be in range. preserve marks the result uncoalesced and keeps
// all entries. assume_sorted_unique asserts lexicographically sorted, distinct pairs.
// Use ATen coalesce() separately to sum duplicates. The ATen result retains the
// supplied storages, while externally owned memory follows its owner's lifetime.
// ATen's factory determines autograd behavior and honors its invariant settings.
at::Tensor wrap_coo_entries(CooEntryOrder order, int64_t rows, int64_t cols,
                            const at::Tensor &coordinates, const at::Tensor &values);

// Explicitly pack separate contiguous int32, int64, or uint32 coordinate vectors
// into the supplied contiguous int64[2, nnz] tensor. Input widths may differ.
// All tensors must be on the same device, and the output must be disjoint from
// both inputs. This copy preserves entry order. For CUDA it runs on ATen's
// current stream for this device, and the caller keeps all storage live through
// completion and orders any access from other streams.
void pack_coo_indices(at::Tensor &packed, const at::Tensor &row_idx, const at::Tensor &col_idx);

} // namespace cellerator::interop::torch
