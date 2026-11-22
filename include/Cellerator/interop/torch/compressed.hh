#pragma once

#include <Cellerator/tensor/base/compressed.hh>

#include <ATen/core/Tensor.h>

namespace cellerator::interop::torch {

namespace detail {

static_assert(sizeof(at::Half) == sizeof(uint16_t), "native FP16 views require binary16 storage");

struct CompressedComponents {
    tensor::base::CompressedAxis axis;
    at::Tensor major_ptr, minor_idx, values;
};

CompressedComponents compressed_components(const at::Tensor &matrix);

template<class Index>
tensor::base::CompressedStructureView<Index> compressed_structure(
    const at::Tensor &matrix, const CompressedComponents &components) {
    static_assert(__is_same(Index, int32_t) || __is_same(Index, int64_t),
                  "ATen compressed indices must be signed int32 or int64");
    TORCH_CHECK(components.major_ptr.scalar_type() == c10::CppTypeToScalarType<Index>::value,
                "compressed index dtype does not match the requested view");
    return {uint64_t(matrix.size(0)), uint64_t(matrix.size(1)),
            uint64_t(components.values.size(0)), components.axis,
            static_cast<const Index *>(components.major_ptr.const_data_ptr()),
            static_cast<const Index *>(components.minor_idx.const_data_ptr())};
}

} // namespace detail

// Borrow rank-two scalar CSR or CSC with contiguous CPU or CUDA components.
// The matrix must have valid compressed indices and the exact requested dtypes.
// Keep its storage alive and unchanged through the last use of the view. For
// CUDA, the caller orders producer and consumer streams and retains the storage
// through completion. The pointers remain in the matrix's original placement.
template<class Index, class Value>
tensor::base::CompressedView<Index, const Value> borrow_compressed(
    const at::Tensor &matrix) {
    const auto components = detail::compressed_components(matrix);
    TORCH_CHECK(components.values.scalar_type() == c10::CppTypeToScalarType<Value>::value,
                "compressed value dtype does not match the requested view");
    return {detail::compressed_structure<Index>(matrix, components),
            static_cast<const Value *>(components.values.const_data_ptr())};
}

template<class Index, class Value>
tensor::base::CompressedView<Index, const Value> borrow_compressed(
    const at::Tensor &&) = delete;

// FP16 storage for native kernels that load binary16 through the byte view.
template<class Index>
tensor::base::CompressedView<Index, const void> borrow_compressed_fp16(
    const at::Tensor &matrix) {
    const auto view = borrow_compressed<Index, at::Half>(matrix);
    return {view.structure, view.values};
}

template<class Index>
tensor::base::CompressedView<Index, const void> borrow_compressed_fp16(
    const at::Tensor &&) = delete;

// Only the values may be written. The caller must have exclusive write access
// and must keep all aliases out of active or saved autograd graphs. Raw writes
// bypass ATen version tracking. Caller lifetime and stream requirements apply.
template<class Index>
tensor::base::CompressedView<Index, void> borrow_mutable_compressed_fp16(
    at::Tensor &matrix) {
    auto components = detail::compressed_components(matrix);
    TORCH_CHECK(!matrix.requires_grad() && !components.values.requires_grad(),
                "mutable compressed borrowing requires values outside autograd");
    TORCH_CHECK(components.values.scalar_type() == at::kHalf,
                "mutable FP16 borrowing requires FP16 compressed values");
    return {detail::compressed_structure<Index>(matrix, components), components.values.mutable_data_ptr()};
}

// Construct ATen sparse tensor metadata over the three supplied ATen storages.
// All components must be contiguous rank-one tensors on the same CPU or CUDA
// device. Offsets and indices must have the same signed int32 or int64 dtype.
// Offsets start at zero, increase monotonically, and end at nnz. Each segment
// has sorted, distinct minor indices in range. These are caller preconditions.
// The result retains the component storage. Any memory borrowed by the three
// tensors still follows its external owner's lifetime. ATen's sparse factory
// determines autograd behavior and honors its sparse invariant-check setting.
at::Tensor wrap_compressed(tensor::base::CompressedAxis axis, int64_t rows,
                           int64_t cols, const at::Tensor &major_ptr,
                           const at::Tensor &minor_idx, const at::Tensor &values);

} // namespace cellerator::interop::torch
