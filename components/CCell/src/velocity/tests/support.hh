#pragma once

#include <array>
#include <Cellerator/memory/view.hh>
#include <Cellerator/tensor/base/compressed.hh>

using SparseAxis = cellerator::tensor::base::CompressedAxis;

template<class T>
velocitypp::AcquiredMatrix borrow_dense(const T* data, std::array<size_t, 2> shape,
                                       std::array<ptrdiff_t, 2> stride) {
    return velocitypp::borrow_matrix(cellerator::memory::const_matrix_view<T, ptrdiff_t>{
        data, shape[0], shape[1], stride[0], {}, stride[1]});
}

template<class T, class Index, class Offset>
velocitypp::AcquiredMatrix borrow_sparse(SparseAxis axis, size_t rows, size_t cols, size_t nnz,
    const T* values, const Index* indices, const Offset* offsets) {
    return velocitypp::borrow_matrix(
        cellerator::tensor::base::CompressedView<Index, const T, Offset>{
            {rows, cols, nnz, axis, offsets, indices}, values});
}
