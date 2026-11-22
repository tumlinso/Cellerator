#pragma once

#include <Cellerator/memory/view.hh>

#include <ATen/core/Tensor.h>

#include <stddef.h>

namespace cellerator::interop::torch {

namespace detail {

memory::const_matrix_view<void> borrow_dense_storage(const at::Tensor &matrix,
    at::ScalarType value_type, size_t alignment, bool writable);

at::Tensor wrap_dense_storage(memory::matrix_view<void> view, at::ScalarType value_type,
                             size_t alignment);

} // namespace detail

// Borrow a rank-two strided CPU or CUDA tensor using its actual element strides
// and storage offset. Read-only views can have overlapping entries. The caller
// keeps the source storage live and orders CUDA streams through completion.
// CPU tensors are classified as host without inferring pinning or NUMA placement.
template<class Value>
memory::const_matrix_view<Value> borrow_dense(const at::Tensor &matrix) {
    const auto view = detail::borrow_dense_storage(
        matrix, c10::CppTypeToScalarType<Value>::value, alignof(Value), false);
    return {static_cast<const Value *>(view.data), view.rows, view.columns,
            view.row_stride, view.where, view.column_stride};
}

template<class Value>
memory::const_matrix_view<Value> borrow_dense(const at::Tensor &&) = delete;

// The caller must have exclusive write access and keep all aliases out of
// active or saved autograd graphs. Raw writes bypass ATen version tracking.
// The layout must have distinct addresses. Lifetime and stream rules apply.
template<class Value>
memory::matrix_view<Value> borrow_mutable_dense(at::Tensor &matrix) {
    const auto view = detail::borrow_dense_storage(
        matrix, c10::CppTypeToScalarType<Value>::value, alignof(Value), true);
    return {static_cast<Value *>(matrix.mutable_data_ptr()), view.rows, view.columns,
            view.row_stride, view.where, view.column_stride};
}

// Expose a writable native view as an ATen tensor. The view must specify host
// placement or a concrete CUDA device and have distinct element addresses.
// The caller owns the entire addressed buffer and retains it through the life
// of the tensor and its aliases and through completion on any CUDA stream.
// The wrapper creates separate ATen metadata and version tracking. The caller
// must coordinate mutations through all aliases and keep them out of active or
// saved autograd graphs when mutating through an independently tracked alias.
template<class Value>
at::Tensor wrap_dense(memory::matrix_view<Value> view) {
    return detail::wrap_dense_storage(
        {view.data, view.rows, view.columns, view.row_stride, view.where, view.column_stride},
        c10::CppTypeToScalarType<Value>::value, alignof(Value));
}

} // namespace cellerator::interop::torch
