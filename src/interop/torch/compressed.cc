#include <Cellerator/interop/torch/compressed.hh>

#include <ATen/ops/sparse_compressed_tensor.h>

namespace cellerator::interop::torch {
namespace {

using tensor::base::CompressedAxis;

bool storage_vector(const at::Tensor &array) {
    return array.defined() && array.layout() == at::kStrided &&
           array.dim() == 1 && array.is_contiguous() &&
           !array.is_conj() && !array.is_neg();
}

void validate_components(CompressedAxis axis, int64_t rows, int64_t cols,
                         const at::Tensor &major_ptr, const at::Tensor &minor_idx,
                         const at::Tensor &values) {
    TORCH_CHECK(axis == CompressedAxis::row || axis == CompressedAxis::column,
                "expected row or column compression");
    TORCH_CHECK(rows >= 0 && cols >= 0, "compressed matrix extents must be nonnegative");
    const int64_t major = axis == CompressedAxis::row ? rows : cols;
    TORCH_CHECK(major < INT64_MAX, "compressed major extent plus one must fit int64");
    TORCH_CHECK(storage_vector(major_ptr) && storage_vector(minor_idx) && storage_vector(values),
                "compressed components must be contiguous one-dimensional storage vectors");
    TORCH_CHECK(values.is_cpu() || values.is_cuda(), "compressed storage must be on CPU or CUDA");
    TORCH_CHECK(major_ptr.device() == values.device() && minor_idx.device() == values.device(),
                "compressed components must be on the same device");
    const at::ScalarType index_type = major_ptr.scalar_type();
    TORCH_CHECK((index_type == at::kInt || index_type == at::kLong) &&
                index_type == minor_idx.scalar_type(),
                "compressed offsets and minor indices must use matching int32 or int64 dtypes");
    const int64_t nnz = values.size(0);
    TORCH_CHECK(major_ptr.size(0) == major + 1 && minor_idx.size(0) == nnz,
                "compressed component lengths do not match the matrix shape and nnz");
    TORCH_CHECK(index_type == at::kLong ||
                (rows <= INT32_MAX && cols <= INT32_MAX && nnz <= INT32_MAX),
                "compressed matrix extents and nnz must fit its index dtype");
}

} // namespace

detail::CompressedComponents detail::compressed_components(const at::Tensor &matrix) {
    TORCH_CHECK(matrix.defined() && matrix.dim() == 2 &&
                (matrix.layout() == at::kSparseCsr || matrix.layout() == at::kSparseCsc),
                "expected a rank-two CSR or CSC tensor");
    const bool row = matrix.layout() == at::kSparseCsr;
    CompressedComponents components{
        row ? CompressedAxis::row : CompressedAxis::column,
        row ? matrix.crow_indices() : matrix.ccol_indices(),
        row ? matrix.col_indices() : matrix.row_indices(),
        matrix.values()};
    validate_components(components.axis, matrix.size(0), matrix.size(1),
                        components.major_ptr, components.minor_idx, components.values);
    return components;
}

at::Tensor wrap_compressed(CompressedAxis axis, int64_t rows, int64_t cols,
                           const at::Tensor &major_ptr, const at::Tensor &minor_idx,
                           const at::Tensor &values) {
    validate_components(axis, rows, cols, major_ptr, minor_idx, values);
    const auto layout = axis == CompressedAxis::row ? at::kSparseCsr : at::kSparseCsc;
    return at::sparse_compressed_tensor(major_ptr, minor_idx, values, {rows, cols},
                                        values.options().layout(layout));
}

} // namespace cellerator::interop::torch
