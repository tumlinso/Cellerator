#include <Cellerator/interop/torch/dense.hh>

#include <ATen/ops/from_blob.h>

#include <stdint.h>

namespace cellerator::interop::torch {
namespace {

size_t nonnegative_size(int64_t value) {
    TORCH_CHECK(value >= 0 && uint64_t(value) <= SIZE_MAX,
                "dense extents, strides and storage offsets must fit nonnegative size_t");
    return size_t(value);
}

bool distinct_addresses(size_t rows, size_t cols, size_t rs, size_t cs) {
    if (!rows || !cols) return true;
    if ((rows > 1 && !rs) || (cols > 1 && !cs)) return false;
    if (rows == 1 || cols == 1) return true;
    size_t a = rs, b = cs;
    while (b) {
        const size_t r = a%b;
        a = b; b = r;
    }
    // The first possible collision shifts row by cs/gcd and column by rs/gcd.
    return rows <= cs/a || cols <= rs/a;
}

size_t dense_span(size_t rows, size_t cols, size_t rs, size_t cs, size_t item_bytes) {
    TORCH_CHECK(rows <= uint64_t(INT64_MAX) && cols <= uint64_t(INT64_MAX) &&
                rs <= uint64_t(INT64_MAX) && cs <= uint64_t(INT64_MAX),
                "dense extents and strides must fit int64");
    if (!rows || !cols) return 0;
    TORCH_CHECK(rows <= uint64_t(INT64_MAX)/cols, "dense element count overflows int64");
    constexpr uint64_t limit = SIZE_MAX < uint64_t(INT64_MAX) ? SIZE_MAX : uint64_t(INT64_MAX);
    const uint64_t max_offset = limit/item_bytes - 1u;
    TORCH_CHECK(rows == 1 || rs <= max_offset/(rows - 1u), "dense row stride overflows the storage span");
    const uint64_t row_offset = uint64_t(rows - 1u)*rs;
    TORCH_CHECK(cols == 1 || cs <= (max_offset - row_offset)/(cols - 1u),
                "dense column stride overflows the storage span");
    return size_t((row_offset + uint64_t(cols - 1u)*cs + 1u)*item_bytes);
}

bool plain_dtype(at::ScalarType type) { return !c10::isQIntType(type) && !c10::isBitsType(type); }

void validate_address(const void *data, size_t bytes, size_t alignment) {
    if (!bytes) return;
    const auto address = reinterpret_cast<uintptr_t>(data);
    TORCH_CHECK(data && address%alignment == 0 && bytes - 1u <= UINTPTR_MAX - address,
                "dense matrix storage must be aligned, non-null, and addressable");
}

at::Device tensor_device(memory::placement where) {
    using memory::domain;
    switch (where.kind) {
    case domain::host:
    case domain::host_numa:
    case domain::host_pinned:
    case domain::host_pinned_write_combined:
        return {at::kCPU};
    case domain::device: {
        const auto idx = static_cast<c10::DeviceIndex>(where.device_ordinal);
        TORCH_CHECK(where.device_ordinal >= 0 && idx == where.device_ordinal,
                    "dense matrix placement requires a valid CUDA device ordinal");
        return {at::kCUDA, idx};
    }
    default:
        TORCH_CHECK(false, "dense matrix placement must specify host or CUDA device storage");
    }
}

} // namespace

memory::const_matrix_view<void> detail::borrow_dense_storage(const at::Tensor &matrix,
    at::ScalarType value_type, size_t alignment, bool writable) {
    TORCH_CHECK(matrix.defined() && matrix.layout() == at::kStrided && matrix.dim() == 2 &&
                (matrix.is_cpu() || matrix.is_cuda()), "expected a rank-two strided CPU or CUDA tensor");
    TORCH_CHECK(matrix.scalar_type() == value_type && plain_dtype(value_type),
                "dense value dtype does not match the requested scalar view");
    TORCH_CHECK(matrix.has_storage() && !matrix.is_conj() && !matrix.is_neg(),
                "dense tensor must expose its values directly through its storage");
    const size_t rows = nonnegative_size(matrix.size(0)), cols = nonnegative_size(matrix.size(1));
    const size_t rs = nonnegative_size(matrix.stride(0)), cs = nonnegative_size(matrix.stride(1));
    const size_t item_bytes = matrix.element_size();
    const size_t bytes = dense_span(rows, cols, rs, cs, item_bytes);
    if (writable) {
        TORCH_CHECK(!matrix.requires_grad(), "mutable dense borrowing requires values outside autograd");
        TORCH_CHECK(distinct_addresses(rows, cols, rs, cs), "mutable dense entries must have distinct addresses");
    }
    if (bytes) {
        const size_t offset = nonnegative_size(matrix.storage_offset());
        const size_t capacity = matrix.storage().nbytes();
        TORCH_CHECK(offset <= capacity/item_bytes && bytes <= capacity - offset*item_bytes,
                    "dense tensor view exceeds its backing storage");
    }
    const void *data = matrix.const_data_ptr();
    validate_address(data, bytes, alignment);
    const memory::placement where = matrix.is_cuda()
        ? memory::placement{memory::domain::device, int16_t(matrix.get_device())}
        : memory::placement{memory::domain::host};
    return {data, rows, cols, rs, where, cs};
}

at::Tensor detail::wrap_dense_storage(memory::matrix_view<void> view, at::ScalarType value_type,
                                     size_t alignment) {
    TORCH_CHECK(plain_dtype(value_type), "dense wrapping requires a plain scalar storage dtype");
    const size_t bytes = dense_span(view.rows, view.columns, view.row_stride, view.column_stride,
                                    c10::elementSize(value_type));
    validate_address(view.data, bytes, alignment);
    TORCH_CHECK(distinct_addresses(view.rows, view.columns, view.row_stride, view.column_stride),
                "wrapped dense entries must have distinct addresses");
    const auto device = tensor_device(view.where);
    const int64_t sizes[] = {int64_t(view.rows), int64_t(view.columns)};
    const int64_t strides[] = {int64_t(view.row_stride), int64_t(view.column_stride)};
    const auto options = at::TensorOptions().dtype(value_type).device(device);
    if (device.is_cuda() && !view.data)
        return at::for_blob(view.data, sizes).strides(strides).options(options).target_device(device).make_tensor();
    return at::from_blob(view.data, sizes, strides, options);
}

} // namespace cellerator::interop::torch
