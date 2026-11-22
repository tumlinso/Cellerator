#include <Cellerator/interop/torch/coo.hh>

#include <ATen/ops/sparse_coo_tensor.h>

namespace cellerator::interop::torch {
namespace {

bool contiguous_storage(const at::Tensor &array, int64_t rank) {
    return array.defined() && array.layout() == at::kStrided &&
           array.dim() == rank && array.is_contiguous() &&
           !array.is_conj() && !array.is_neg();
}

bool coo_coordinate_storage(const at::Tensor &coordinates) {
    return contiguous_storage(coordinates, 2) && coordinates.scalar_type() == at::kLong &&
           coordinates.size(0) == 2;
}

int64_t validate_components(const at::Tensor &coordinates, const at::Tensor &values) {
    TORCH_CHECK(coo_coordinate_storage(coordinates) && contiguous_storage(values, 1),
                "COO requires contiguous int64[2, nnz] coordinates and rank-one values");
    TORCH_CHECK(values.is_cpu() || values.is_cuda(), "COO storage must be on CPU or CUDA");
    TORCH_CHECK(coordinates.device() == values.device(), "COO components must be on the same device");
    const int64_t nnz = coordinates.size(1);
    TORCH_CHECK(values.size(0) == nnz, "COO coordinate and value counts must match");
    return nnz;
}

bool split_coordinate_storage(const at::Tensor &indices) {
    if (!contiguous_storage(indices, 1)) return false;
    const auto type = indices.scalar_type();
    return type == at::kInt || type == at::kLong || type == at::kUInt32;
}

bool storage_overlaps(const at::Tensor &a, const at::Tensor &b) {
    if (!a.numel() || !b.numel()) return false;
    const auto p = reinterpret_cast<uintptr_t>(a.const_data_ptr());
    const auto q = reinterpret_cast<uintptr_t>(b.const_data_ptr());
    return p <= q ? q - p < a.nbytes() : p - q < b.nbytes();
}

} // namespace

tensor::base::CooView<int64_t, const void> detail::borrow_coo_storage(
    const at::Tensor &matrix, at::ScalarType value_type) {
    TORCH_CHECK(matrix.defined() && matrix.layout() == at::kSparse &&
                matrix.dim() == 2 && matrix.sparse_dim() == 2 && matrix.dense_dim() == 0,
                "expected a rank-two scalar COO tensor");
    const auto coordinates = matrix._indices(), values = matrix._values();
    const int64_t nnz = validate_components(coordinates, values);
    TORCH_CHECK(values.scalar_type() == value_type, "COO value dtype does not match the requested view");
    const auto *row = static_cast<const int64_t *>(coordinates.const_data_ptr());
    const auto *col = nnz ? row + nnz : row;
    return {uint64_t(matrix.size(0)), uint64_t(matrix.size(1)), uint64_t(nnz),
            row, col, values.const_data_ptr()};
}

at::Tensor wrap_coo_entries(CooEntryOrder order, int64_t rows, int64_t cols,
                            const at::Tensor &coordinates, const at::Tensor &values) {
    TORCH_CHECK(order == CooEntryOrder::preserve || order == CooEntryOrder::assume_sorted_unique,
                "expected preserve or sorted-unique COO entries");
    TORCH_CHECK(rows >= 0 && cols >= 0, "COO matrix extents must be nonnegative");
    validate_components(coordinates, values);
    return at::sparse_coo_tensor(coordinates, values, {rows, cols}, values.options().layout(at::kSparse),
                                 order == CooEntryOrder::assume_sorted_unique);
}

void pack_coo_indices(at::Tensor &packed, const at::Tensor &row_idx, const at::Tensor &col_idx) {
    TORCH_CHECK(coo_coordinate_storage(packed), "COO output must be contiguous int64[2, nnz]");
    TORCH_CHECK(split_coordinate_storage(row_idx) && split_coordinate_storage(col_idx),
                "COO input coordinates must be contiguous int32, int64, or uint32 vectors");
    TORCH_CHECK(packed.is_cpu() || packed.is_cuda(), "COO storage must be on CPU or CUDA");
    TORCH_CHECK(row_idx.device() == packed.device() && col_idx.device() == packed.device(),
                "COO pack operands must be on the same device");
    const int64_t nnz = packed.size(1);
    TORCH_CHECK(row_idx.size(0) == nnz && col_idx.size(0) == nnz,
                "COO coordinate input and output counts must match");
    TORCH_CHECK(!storage_overlaps(packed, row_idx) && !storage_overlaps(packed, col_idx),
                "COO packed coordinates must not overlap split input coordinates");
    if (!nnz) return;
    packed.select(0, 0).copy_(row_idx);
    packed.select(0, 1).copy_(col_idx);
}

} // namespace cellerator::interop::torch
