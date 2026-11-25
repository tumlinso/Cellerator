module;

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <Cellerator/execution/streaming/source.hh>
#include <Cellerator/memory/allocation.hh>
#include <Cellerator/memory/view.hh>
#include <Cellerator/tensor/base/compressed.hh>

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

namespace {

using namespace execution::streaming;

ScalarType fetch_type(DType type) noexcept {
    switch (type) {
    case DType::boolean: return ScalarType::boolean;
    case DType::int8: return ScalarType::i8;
    case DType::uint8: return ScalarType::u8;
    case DType::int16: return ScalarType::i16;
    case DType::uint16: return ScalarType::u16;
    case DType::int32: return ScalarType::i32;
    case DType::uint32: return ScalarType::u32;
    case DType::int64: return ScalarType::i64;
    case DType::uint64: return ScalarType::u64;
    case DType::float16: return ScalarType::f16;
    case DType::float32: return ScalarType::f32;
    case DType::float64: return ScalarType::f64;
    }
    return ScalarType::invalid;
}

memory::status fetch_status(ErrorCode code) noexcept {
    switch (code) {
    case ErrorCode::out_of_bounds:
    case ErrorCode::shape_mismatch:
    case ErrorCode::dtype_mismatch:
    case ErrorCode::invalid_argument: return memory::status::invalid_argument;
    case ErrorCode::unsupported_layout:
    case ErrorCode::unsupported_encoding:
    case ErrorCode::unsupported_dtype: return memory::status::unsupported_domain;
    default: return memory::status::invalid_state;
    }
}

} // namespace

FetchReader::FetchReader(::anndatapp::TransferContext& context, tensor::base::CompressedAxis cells,
                         uint64_t feature_axis)
    : context_(context), start_(2), count_(2), major_(cells == tensor::base::CompressedAxis::row ? 0 : 1) {
    const auto g = context_.geometry();
    const auto minor = 1 - major_;
    if ((cells != tensor::base::CompressedAxis::row && cells != tensor::base::CompressedAxis::column) ||
        g.layout != (major_ == 0 ? Layout::csr : Layout::csc) ||
        g.shape.size() != 2 || g.source_shape.size() != 2 || g.axes.size() != 2 ||
        g.shape[major_] != g.source_shape[major_] || !g.axes[major_].affine ||
        g.axes[major_].start != 0 || g.axes[major_].step != 1)
        fail(ErrorCode::unsupported_layout, "fetch requires the complete native cell axis");
    count_[minor] = g.shape[minor];

    // An empty block obtains the stored widths without reading values.
    const auto req = context_.requirements(start_, count_);
    value_ = req.dtype; index_ = req.index_dtype; offset_ = req.offset_dtype;
    info_ = {g.shape[major_], g.shape[minor], feature_axis,
             fetch_type(value_), fetch_type(index_), fetch_type(offset_)};
    if (!scalar_bytes(info_.value) || !scalar_bytes(info_.index) || !scalar_bytes(info_.offset))
        fail(ErrorCode::unsupported_dtype, "sparse fetch type has no native representation");
}

execution::streaming::FetchSource FetchReader::source() noexcept {
    return {this, info_, requirements, read_into, nullptr};
}

bool FetchReader::select(CellRange cells) noexcept {
    if (cells.first > info_.cells || cells.count > info_.cells - cells.first || cells.count == SIZE_MAX)
        return false;
    if (start_[major_] != cells.first || count_[major_] != cells.count) {
        start_[major_] = size_t(cells.first); count_[major_] = size_t(cells.count);
        entries_ = SIZE_MAX;
    }
    return true;
}

size_t FetchReader::prepare_block() {
    entries_ = SIZE_MAX;
    const auto req = context_.requirements(start_, count_);
    if (req.layout != (major_ == 0 ? Layout::csr : Layout::csc) ||
        req.dtype != value_ || req.index_dtype != index_ || req.offset_dtype != offset_)
        fail(ErrorCode::dtype_mismatch, "fetch source structure differs from its prepared widths");
    return entries_ = req.nnz;
}

memory::status FetchReader::requirements(void *context, CellRange cells, uint64_t *entries) noexcept {
    auto& reader = *static_cast<FetchReader*>(context);
    if (!entries || !reader.select(cells)) return memory::status::invalid_argument;
    try {
        *entries = reader.prepare_block();
        return memory::status::success;
    } catch (const Error& error) {
        return fetch_status(error.code());
    } catch (...) {
        return memory::status::invalid_state;
    }
}

memory::status FetchReader::read_into(void *context, CellRange cells, SparseBuffers out,
                                     ReadStats *stats) noexcept {
    auto& reader = *static_cast<FetchReader*>(context);
    if (!reader.select(cells)) return memory::status::invalid_argument;
    const size_t vw = scalar_bytes(reader.info_.value), iw = scalar_bytes(reader.info_.index),
                 ow = scalar_bytes(reader.info_.offset), offsets = size_t(cells.count) + 1;
    try {
        const auto before = stats ? reader.context_.stats().read_bytes : 0;
        const auto nnz = reader.entries_ == SIZE_MAX ? reader.prepare_block() : reader.entries_;
        const size_t vcap = out.value_bytes/vw, icap = out.index_bytes/iw, ocap = out.offset_bytes/ow;
        if (vcap < nnz || icap < nnz || ocap < offsets) return memory::status::capacity_exceeded;
        constexpr memory::placement host_memory{memory::domain::host};
        const auto v = sparse_range({out.values, vcap, host_memory}, nnz, vw);
        const auto i = sparse_range({out.indices, icap, host_memory}, nnz, iw);
        const auto o = sparse_range({out.offsets, ocap, host_memory}, offsets, ow);
        if (overlap(v, i) || overlap(v, o) || overlap(i, o)) return memory::status::invalid_argument;
        reader.context_.read_into(reader.start_, reader.count_,
            {out.values, out.indices, out.offsets, reader.value_, reader.index_, reader.offset_, vcap, icap, ocap});
        // The backend exposes decoded logical bytes and no compressed count.
        if (stats) *stats = {0, reader.context_.stats().read_bytes - before};
        return memory::status::success;
    } catch (const Error& error) {
        reader.entries_ = SIZE_MAX;
        return fetch_status(error.code());
    } catch (...) {
        reader.entries_ = SIZE_MAX;
        return memory::status::invalid_state;
    }
}

}
