module;

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <Cellerator/memory/allocation.hh>
#include <Cellerator/memory/view.hh>

module cellerator.anndata;

namespace cellerator::anndata {
namespace {

using ::anndatapp::BackedArray;
using ::anndatapp::DenseBufferView;
using ::anndatapp::DType;
using ::anndatapp::Error;
using ::anndatapp::ErrorCode;
using ::anndatapp::Layout;
using ::anndatapp::PreparedRead;
using ::anndatapp::ReadRequirements;
using ::anndatapp::SparseBuffer;
using ::anndatapp::dtype_size;

[[noreturn]] void fail(ErrorCode code, const char *message) {
    throw Error(code, "cellerator.anndata", message);
}

bool host(memory::placement where) noexcept {
    using memory::domain;
    return where.kind == domain::host || where.kind == domain::host_numa ||
           where.kind == domain::host_pinned || where.kind == domain::host_pinned_write_combined;
}

size_t bytes_for(size_t count, size_t width) {
    size_t bytes;
    if (!width || __builtin_mul_overflow(count, width, &bytes) || bytes > size_t(PTRDIFF_MAX))
        fail(ErrorCode::invalid_argument, "destination byte size exceeds ptrdiff_t");
    return bytes;
}

size_t add_bytes(size_t a, size_t b) {
    if (b > size_t(PTRDIFF_MAX) - a)
        fail(ErrorCode::invalid_argument, "destination byte size exceeds ptrdiff_t");
    return a + b;
}

size_t aligned_offset(size_t start, size_t width) {
    return add_bytes(start, (width - start % width) % width);
}

size_t magnitude(ptrdiff_t x) noexcept {
    return x < 0 ? size_t(-(x + 1)) + 1 : size_t(x);
}

// An overlap requires r rows per step of c columns, after reducing both strides.
bool dense_overlaps(size_t rows, size_t cols, ptrdiff_t rs, ptrdiff_t cs) noexcept {
    if ((rows > 1 && !rs) || (cols > 1 && !cs)) return true;
    if (rows < 2 || cols < 2) return false;
    size_t r = magnitude(rs), c = magnitude(cs), a = r, b = c;
    while (b) {
        size_t rem = a % b;
        a = b;
        b = rem;
    }
    return rows > c/a && cols > r/a;
}

void check_dense_span(void *p, size_t rows, size_t cols, ptrdiff_t rs, ptrdiff_t cs,
                      size_t width, const memory::allocation& storage) {
    if (!rows || !cols) return;
    uintptr_t begin = reinterpret_cast<uintptr_t>(storage.base), first = reinterpret_cast<uintptr_t>(p);
    ptrdiff_t r, c, lo, hi;
    if (!p || !storage.base || first % width || first < begin ||
        storage.bytes > UINTPTR_MAX - begin || rows-1 > size_t(PTRDIFF_MAX) || cols-1 > size_t(PTRDIFF_MAX) ||
        __builtin_mul_overflow(rs, ptrdiff_t(rows-1), &r) || __builtin_mul_overflow(cs, ptrdiff_t(cols-1), &c) ||
        __builtin_add_overflow(r < 0 ? r : 0, c < 0 ? c : 0, &lo) ||
        __builtin_add_overflow(r > 0 ? r : 0, c > 0 ? c : 0, &hi) ||
        first - begin < magnitude(lo) || first - begin > storage.bytes ||
        add_bytes(size_t(hi), width) > storage.bytes - (first - begin))
        fail(ErrorCode::invalid_argument, "dense host allocation does not cover the addressed span");
    if (dense_overlaps(rows, cols, rs, cs))
        fail(ErrorCode::invalid_argument, "dense destination elements overlap");
}

struct ByteRange {
    uintptr_t start, end;
};

ByteRange sparse_range(memory::array_view<void> storage, size_t count, size_t width) {
    uintptr_t start = reinterpret_cast<uintptr_t>(storage.data);
    size_t bytes = bytes_for(count, width);
    if (!host(storage.where) || storage.count < count ||
        (count && (!start || start % width || bytes > UINTPTR_MAX - start)))
        fail(ErrorCode::invalid_argument, "invalid or insufficient sparse host buffer");
    return {start, start + bytes};
}

bool overlap(ByteRange a, ByteRange b) noexcept {
    return a.start != a.end && b.start != b.end && a.start < b.end && b.start < a.end;
}

detail::MatrixShape matrix_shape(const ReadRequirements& req) {
    if (req.shape.size() != 2) fail(ErrorCode::shape_mismatch, "expected a rank-two selection");
    return {req.shape[0], req.shape[1], req.nnz, req.layout};
}

size_t offset_count(detail::MatrixShape shape) {
    size_t major = shape.layout == Layout::csr ? shape.rows : shape.cols;
    if (major == SIZE_MAX) fail(ErrorCode::invalid_argument, "sparse offset count overflows");
    return major + 1;
}

}

namespace detail {

void read_dense(const PreparedRead& source, void *p, size_t rows, size_t cols, ptrdiff_t rs, ptrdiff_t cs,
                DType dtype, size_t width, memory::placement where, const memory::allocation& backing) {
    const auto& req = source.requirements_ref();
    auto shape = matrix_shape(req);
    if (shape.layout != Layout::dense) fail(ErrorCode::unsupported_layout, "expected a dense array");
    if (dtype != req.dtype || width != dtype_size(dtype)) fail(ErrorCode::dtype_mismatch, "dense dtype differs");
    if (shape.rows != rows || shape.cols != cols) fail(ErrorCode::shape_mismatch, "dense shape differs");
    if (!host(where) || !host(backing.where)) fail(ErrorCode::invalid_argument, "expected host placement");

    ptrdiff_t strides[2];
    if (__builtin_mul_overflow(rs, ptrdiff_t(width), strides) ||
        __builtin_mul_overflow(cs, ptrdiff_t(width), strides+1))
        fail(ErrorCode::invalid_argument, "destination byte stride exceeds ptrdiff_t");
    check_dense_span(p, rows, cols, strides[0], strides[1], width, backing);
    const size_t dimensions[] = {rows, cols};
    source.read_into(DenseBufferView{p, dtype, 2, dimensions, strides});
}

MatrixShape read_sparse(const PreparedRead& source, memory::array_view<void> values, DType value_type,
                        memory::array_view<void> indices, DType index_type,
                        memory::array_view<void> offsets, DType offset_type) {
    const auto& req = source.requirements_ref();
    auto shape = matrix_shape(req);
    if (shape.layout == Layout::dense) fail(ErrorCode::unsupported_layout, "expected CSR or CSC");
    if (value_type != req.dtype || index_type != req.index_dtype || offset_type != req.offset_dtype)
        fail(ErrorCode::dtype_mismatch, "sparse value, index, or offset dtype differs");
    auto v = sparse_range(values, shape.nnz, dtype_size(value_type));
    auto i = sparse_range(indices, shape.nnz, dtype_size(index_type));
    auto o = sparse_range(offsets, offset_count(shape), dtype_size(offset_type));
    if (overlap(v, i) || overlap(v, o) || overlap(i, o))
        fail(ErrorCode::invalid_argument, "sparse destination arrays overlap");
    source.read_into(SparseBuffer{values.data, indices.data, offsets.data, value_type, index_type,
                                 offset_type, values.count, indices.count, offsets.count});
    return shape;
}

void bad_view() {
    fail(ErrorCode::dtype_mismatch, "requested native view does not match the buffered matrix");
}

void bad_stride() {
    fail(ErrorCode::invalid_argument, "destination stride exceeds ptrdiff_t");
}

}

BufferedMatrix::~BufferedMatrix() noexcept { free(storage_); }

BufferedMatrix::BufferedMatrix(BufferedMatrix&& other) noexcept { *this = static_cast<BufferedMatrix&&>(other); }

BufferedMatrix& BufferedMatrix::operator=(BufferedMatrix&& other) noexcept {
    if (this == &other) return *this;
    free(storage_);
    storage_ = other.storage_; indices_ = other.indices_; offsets_ = other.offsets_;
    shape_ = other.shape_;
    dtype_ = other.dtype_; index_dtype_ = other.index_dtype_; offset_dtype_ = other.offset_dtype_;
    other.storage_ = nullptr; other.indices_ = nullptr; other.offsets_ = nullptr; other.shape_ = {};
    return *this;
}

BufferedMatrix read_allocated(const PreparedRead& source) {
    const auto& req = source.requirements_ref();
    BufferedMatrix out;
    auto shape = matrix_shape(req);
    out.shape_ = shape;
    out.dtype_ = req.dtype; out.index_dtype_ = req.index_dtype; out.offset_dtype_ = req.offset_dtype;

    size_t width = dtype_size(req.dtype), index_start = 0, offset_start = 0, offsets = 0, bytes;
    if (shape.layout == Layout::dense) {
        size_t row_bytes = bytes_for(shape.cols, width);
        bytes = row_bytes ? bytes_for(shape.rows, row_bytes) : 0;
    } else {
        size_t iw = dtype_size(req.index_dtype), ow = dtype_size(req.offset_dtype);
        offsets = offset_count(shape);
        index_start = aligned_offset(bytes_for(shape.nnz, width), iw);
        offset_start = aligned_offset(add_bytes(index_start, bytes_for(shape.nnz, iw)), ow);
        bytes = add_bytes(offset_start, bytes_for(offsets, ow));
    }
    constexpr size_t alignment = memory::default_allocation_alignment;
    if (bytes) {
        out.storage_ = aligned_alloc(alignment, aligned_offset(bytes, alignment));
        if (!out.storage_) fail(ErrorCode::io_error, "could not allocate host storage for the selection");
    }

    constexpr memory::placement host_memory{memory::domain::host};
    if (shape.layout == Layout::dense) {
        memory::allocation backing{out.storage_, bytes, alignment, host_memory};
        detail::read_dense(source, out.storage_, shape.rows, shape.cols, ptrdiff_t(shape.cols), 1,
                           req.dtype, width, host_memory, backing);
    } else {
        auto *p = static_cast<unsigned char*>(out.storage_);
        out.indices_ = p + index_start; out.offsets_ = p + offset_start;
        detail::read_sparse(source, {p, shape.nnz, host_memory}, req.dtype,
                            {out.indices_, shape.nnz, host_memory}, req.index_dtype,
                            {out.offsets_, offsets, host_memory}, req.offset_dtype);
    }
    return out;
}

BufferedMatrix read_allocated(const BackedArray& source) { return read_allocated(source.prepare_read()); }

}
