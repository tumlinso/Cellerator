module;
#include <anndatapp/device_abi.h>
#include "cuda/bridge.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

module anndatapp.cuda;
import anndatapp.core;

namespace anndatapp::cuda {
namespace {

void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(error));
}
void bad(const char* message) { throw std::invalid_argument(message); }

struct DeviceGuard {
    int previous = -1;
    bool changed = false;
    explicit DeviceGuard(int device) {
        check(cudaGetDevice(&previous));
        changed = previous != device;
        if (changed) check(cudaSetDevice(device));
    }
    ~DeviceGuard() { if (changed) cudaSetDevice(previous); }
};

std::uint64_t multiply(std::uint64_t a, std::uint64_t b) {
    if (a && b > std::numeric_limits<std::uint64_t>::max()/a) bad("device extent overflows");
    return a*b;
}
std::uint64_t add(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max()-a) bad("device extent overflows");
    return a+b;
}
std::uint64_t magnitude(std::int64_t x) {
    return x < 0 ? std::uint64_t(-(x+1))+1 : std::uint64_t(x);
}
struct Range { std::uint64_t first, last; };
Range geometry(std::uint64_t origin, std::uint64_t rows, std::uint64_t columns,
               std::int64_t row_stride, std::int64_t column_stride,
               std::uint64_t capacity, std::size_t width) {
    if (origin > capacity) bad("logical origin is outside its allocation span");
    if (!rows || !columns) return {origin, origin};
    auto r = multiply(multiply(rows-1,magnitude(row_stride)),width);
    auto c = multiply(multiply(columns-1,magnitude(column_stride)),width);
    auto before = add(row_stride < 0 ? r : 0, column_stride < 0 ? c : 0);
    auto after = add(row_stride > 0 ? r : 0, column_stride > 0 ? c : 0);
    if (before > std::uint64_t(std::numeric_limits<std::int64_t>::max()) ||
        after > std::uint64_t(std::numeric_limits<std::int64_t>::max()))
        bad("dense byte offset exceeds signed index range");
    if (before > origin || after > capacity-origin || width > capacity-origin-after)
        bad("dense view exceeds its allocation span");
    return {origin-before, origin+after+width};
}
void check_address(void* p, std::uint64_t n) {
    if (n && !p) bad("nonempty span has a null pointer");
    if (n > std::numeric_limits<std::uintptr_t>::max()-reinterpret_cast<std::uintptr_t>(p))
        bad("allocation span address overflows");
}
void check_offset(Span v, std::size_t offset, std::size_t n) {
    if (offset > v.bytes || n > v.bytes-offset) bad("transfer exceeds its allocation span");
}
void check_stream(Stream stream, int device) {
    if (stream.device < 0 || stream.device != device) bad("stream and view devices differ");
}
void check_aligned(void* p, std::uint64_t offset, std::size_t width) {
    if ((reinterpret_cast<std::uintptr_t>(p)+offset)%width) bad("device address is not dtype-aligned");
}
bool index_type(ScalarType type) {
    switch (type) {
    case ScalarType::int8: case ScalarType::uint8: case ScalarType::int16: case ScalarType::uint16:
    case ScalarType::int32: case ScalarType::uint32: case ScalarType::int64: case ScalarType::uint64: return true;
    default: return false;
    }
}
bool integral_type(ScalarType type) { return type == ScalarType::boolean || index_type(type); }
bool signed_type(ScalarType type) {
    return type == ScalarType::int8 || type == ScalarType::int16 || type == ScalarType::int32 || type == ScalarType::int64;
}
unsigned integer_bits(ScalarType type) { return type == ScalarType::boolean ? 1u : unsigned(8*scalar_size(type)); }
bool conversion_can_fail(ScalarType target, ScalarType source) {
    if (!integral_type(target) || target == source) return false;
    if (!integral_type(source)) return true;
    if (source == ScalarType::boolean) return false;
    if (target == ScalarType::boolean) return true;
    auto a = integer_bits(source), b = integer_bits(target);
    auto s = signed_type(source), t = signed_type(target);
    return s == t ? b < a : s ? true : b <= a;
}
bool overlaps(void* a, std::uint64_t a0, std::uint64_t a1,
              void* b, std::uint64_t b0, std::uint64_t b1) {
    if (a0 == a1 || b0 == b1) return false;
    auto ap = reinterpret_cast<std::uintptr_t>(a), bp = reinterpret_cast<std::uintptr_t>(b);
    return ap+a0 < bp+b1 && bp+b0 < ap+a1;
}
Range dense_range(DenseView v) {
    return geometry(v.origin,v.rows,v.columns,v.row_stride,v.column_stride,v.values.bytes,scalar_size(v.dtype));
}
void check_status(Span status, int device) {
    validate(status,true);
    if (status.device != device || status.bytes < sizeof(std::uint32_t)) bad("invalid device status span");
    check_aligned(status.data,0,alignof(std::uint32_t));
}
void reject_status_overlap(Span status, Span input, Range r) {
    if (overlaps(status.data,0,sizeof(std::uint32_t),input.data,r.first,r.last))
        bad("status storage overlaps numerical data");
}
std::uint64_t offset_at(std::uint64_t origin, std::uint64_t i, std::uint64_t j,
                        std::int64_t row_stride, std::int64_t column_stride, std::size_t width) {
    auto r = i*magnitude(row_stride)*width, c = j*magnitude(column_stride)*width;
    if (row_stride < 0) origin -= r;
    if (column_stride < 0) origin -= c;
    if (row_stride > 0) origin += r;
    if (column_stride > 0) origin += c;
    return origin;
}

template<class T> std::uint64_t integer_pattern(double value) {
    auto x = std::trunc(static_cast<long double>(value));
    auto upper = std::ldexp(1.0L,std::numeric_limits<T>::digits);
    auto lower = std::numeric_limits<T>::is_signed ? -upper : 0.0L;
    if (!std::isfinite(value) || x < lower || x >= upper) bad("fill value is out of integer range");
    T t = static_cast<T>(x);
    std::uint64_t bits = 0;
    std::memcpy(&bits,&t,sizeof(T));
    return bits;
}

} // namespace

namespace detail {
struct BufferState {
    Span span{};
    Allocator allocator{};
    void* stream = nullptr;
    bool builtin = false;
    ~BufferState() {
        if (!span.data) return;
        if (!builtin) {
            if (allocator.release) allocator.release(allocator.context,span.data,span.bytes,span.device,stream);
            return;
        }
        int previous = -1;
        if (cudaGetDevice(&previous) != cudaSuccess) return;
        if (previous != span.device && cudaSetDevice(span.device) != cudaSuccess) return;
        cudaFree(span.data);
        if (previous != span.device) cudaSetDevice(previous);
    }
};
}

DeviceBuffer::DeviceBuffer() noexcept = default;
DeviceBuffer::DeviceBuffer(std::size_t bytes, int device) : DeviceBuffer(bytes,device,{},nullptr) {}
DeviceBuffer::DeviceBuffer(std::size_t bytes, int device, Allocator allocator, void* stream) {
    if (device < 0) bad("device must be nonnegative");
    if ((allocator.allocate == nullptr) != (allocator.release == nullptr)) bad("allocator callbacks must be paired");
    auto state = std::make_unique<detail::BufferState>();
    state->span = {nullptr,bytes,device,Access::read_write};
    state->stream = stream;
    state->builtin = !allocator.allocate;
    state->allocator = std::move(allocator);
    if (bytes) {
        if (state->builtin) {
            DeviceGuard guard(device);
            check(cudaMalloc(&state->span.data,bytes));
        } else {
            state->span.data = state->allocator.allocate(state->allocator.context,bytes,device,stream);
            if (!state->span.data) throw std::bad_alloc();
        }
    }
    state_ = std::move(state);
}
DeviceBuffer DeviceBuffer::adopt(Span span, Allocator allocator, void* stream) {
    validate(span);
    if (span.data && !allocator.release) bad("adopt requires a release callback");
    DeviceBuffer out;
    out.state_ = std::make_unique<detail::BufferState>();
    out.state_->span = span; out.state_->allocator = std::move(allocator); out.state_->stream = stream;
    return out;
}
DeviceBuffer DeviceBuffer::retain(Span span, std::shared_ptr<void> keepalive) {
    validate(span);
    if (span.data && !keepalive) bad("retain requires an owner for a nonempty span");
    DeviceBuffer out;
    out.state_ = std::make_unique<detail::BufferState>();
    out.state_->span = span; out.state_->allocator.keepalive = std::move(keepalive);
    return out;
}
DeviceBuffer::~DeviceBuffer() = default;
DeviceBuffer::DeviceBuffer(DeviceBuffer&&) noexcept = default;
DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&&) noexcept = default;
void* DeviceBuffer::data() const noexcept { return state_ ? state_->span.data : nullptr; }
std::size_t DeviceBuffer::size_bytes() const noexcept { return state_ ? std::size_t(state_->span.bytes) : 0; }
int DeviceBuffer::device() const noexcept { return state_ ? state_->span.device : -1; }
Span DeviceBuffer::span(Access access) const {
    if (access != Access::read_only && access != Access::read_write) bad("invalid access");
    if (!state_) return {nullptr,0,-1,access};
    if (access == Access::read_write && state_->span.access != Access::read_write) bad("cannot upgrade read-only access");
    auto out = state_->span; out.access = access; return out;
}

ScalarType scalar_type(DType type) {
    switch (type) {
    case DType::boolean: return ScalarType::boolean;
    case DType::int8: return ScalarType::int8;
    case DType::uint8: return ScalarType::uint8;
    case DType::int16: return ScalarType::int16;
    case DType::uint16: return ScalarType::uint16;
    case DType::int32: return ScalarType::int32;
    case DType::uint32: return ScalarType::uint32;
    case DType::int64: return ScalarType::int64;
    case DType::uint64: return ScalarType::uint64;
    case DType::float16: return ScalarType::float16;
    case DType::float32: return ScalarType::float32;
    case DType::float64: return ScalarType::float64;
    }
    bad("invalid anndatapp dtype"); return ScalarType::float32;
}
DType dtype(ScalarType type) {
    switch (type) {
    case ScalarType::boolean: return DType::boolean;
    case ScalarType::int8: return DType::int8;
    case ScalarType::uint8: return DType::uint8;
    case ScalarType::int16: return DType::int16;
    case ScalarType::uint16: return DType::uint16;
    case ScalarType::int32: return DType::int32;
    case ScalarType::uint32: return DType::uint32;
    case ScalarType::int64: return DType::int64;
    case ScalarType::uint64: return DType::uint64;
    case ScalarType::float16: return DType::float16;
    case ScalarType::float32: return DType::float32;
    case ScalarType::float64: return DType::float64;
    }
    bad("invalid device dtype"); return DType::float32;
}
std::size_t scalar_size(ScalarType type) {
    switch (type) {
    case ScalarType::boolean: case ScalarType::int8: case ScalarType::uint8: return 1;
    case ScalarType::int16: case ScalarType::uint16: case ScalarType::float16: return 2;
    case ScalarType::int32: case ScalarType::uint32: case ScalarType::float32: return 4;
    case ScalarType::int64: case ScalarType::uint64: case ScalarType::float64: return 8;
    }
    bad("invalid device dtype"); return 0;
}
void validate(Span v, bool write) {
    if (v.device < 0) bad("device must be nonnegative");
    if (v.access != Access::read_only && v.access != Access::read_write) bad("invalid span access");
    if (write && v.access != Access::read_write) bad("span is read-only");
    check_address(v.data,v.bytes);
}
void validate(DenseView v, bool write) {
    validate(v.values,write);
    auto width = scalar_size(v.dtype);
    if (v.rows && v.columns && (v.rows > std::uint64_t(std::numeric_limits<std::int64_t>::max()) ||
        v.columns > std::uint64_t(std::numeric_limits<std::int64_t>::max()) ||
        v.rows > std::numeric_limits<std::uint64_t>::max()/v.columns))
        bad("dense shape exceeds device index range");
    if (v.rows && v.columns && !v.values.data) bad("nonempty dense view has a null pointer");
    if (v.rows && v.columns) check_aligned(v.values.data,v.origin,width);
    geometry(v.origin,v.rows,v.columns,v.row_stride,v.column_stride,v.values.bytes,width);
    if (write && v.rows && v.columns) {
        auto r = magnitude(v.row_stride), c = magnitude(v.column_stride);
        if ((v.rows > 1 && !r) || (v.columns > 1 && !c)) bad("writable dense view aliases itself");
        if (v.rows > 1 && v.columns > 1) {
            auto g = std::gcd(r,c);
            if (c/g < v.rows && r/g < v.columns) bad("writable dense view aliases itself");
        }
    }
}
void validate(SparseView v, bool write) {
    validate(v.values,write); validate(v.indices,write); validate(v.offsets,write);
    if (v.values.device != v.indices.device || v.values.device != v.offsets.device) bad("sparse span devices differ");
    if (!index_type(v.index_type) || !index_type(v.offset_type)) bad("sparse index and offset types must be integers");
    if (v.layout != SparseLayout::csr && v.layout != SparseLayout::csc) bad("invalid sparse layout");
    auto value_width = scalar_size(v.dtype), index_width = scalar_size(v.index_type), offset_width = scalar_size(v.offset_type);
    auto major = v.layout == SparseLayout::csr ? v.rows : v.columns;
    if (multiply(v.nnz,value_width) > v.values.bytes || multiply(v.nnz,index_width) > v.indices.bytes ||
        multiply(add(major,1),offset_width) > v.offsets.bytes) bad("sparse component capacity is too small");
    if (v.nnz && (!v.values.data || !v.indices.data)) bad("nonempty sparse data has a null pointer");
    if (!v.offsets.data) bad("sparse offsets have a null pointer");
    if (v.nnz) { check_aligned(v.values.data,0,value_width); check_aligned(v.indices.data,0,index_width); }
    check_aligned(v.offsets.data,0,offset_width);
}
void validate_sparse_contents(SparseView v, Span status, Stream stream) {
    validate(v); check_status(status,v.values.device); check_stream(stream,v.values.device);
    reject_status_overlap(status,v.values,{0,v.values.bytes});
    reject_status_overlap(status,v.indices,{0,v.indices.bytes});
    reject_status_overlap(status,v.offsets,{0,v.offsets.bytes});
    DeviceGuard guard(stream.device);
    check(cudaMemsetAsync(status.data,0,sizeof(std::uint32_t),reinterpret_cast<cudaStream_t>(stream.native)));
    check(static_cast<cudaError_t>(anndatapp_cuda_validate_sparse(v,status.data,stream.native)));
}

void copy_to_device(Span dst, std::size_t offset, const void* src, std::size_t n, Stream stream) {
    validate(dst,true); check_offset(dst,offset,n); check_stream(stream,dst.device);
    if (n && !src) bad("host source is null");
    if (!n) return;
    DeviceGuard guard(stream.device);
    auto* p = static_cast<unsigned char*>(dst.data)+offset;
    check(cudaMemcpyAsync(p,src,n,cudaMemcpyHostToDevice,reinterpret_cast<cudaStream_t>(stream.native)));
}
void copy_to_host(void* dst, Span src, std::size_t offset, std::size_t n, Stream stream) {
    validate(src); check_offset(src,offset,n); check_stream(stream,src.device);
    if (n && !dst) bad("host destination is null");
    if (!n) return;
    DeviceGuard guard(stream.device);
    auto* p = static_cast<const unsigned char*>(src.data)+offset;
    check(cudaMemcpyAsync(dst,p,n,cudaMemcpyDeviceToHost,reinterpret_cast<cudaStream_t>(stream.native)));
}
void copy_device(Span dst, std::size_t doff, Span src, std::size_t soff, std::size_t n, Stream stream) {
    validate(dst,true); validate(src); check_offset(dst,doff,n); check_offset(src,soff,n);
    if (dst.device != src.device) bad("device copy requires the same device");
    check_stream(stream,dst.device);
    if (!n) return;
    auto* d = static_cast<unsigned char*>(dst.data)+doff;
    auto* s = static_cast<const unsigned char*>(src.data)+soff;
    if (d == s) return;
    if (overlaps(dst.data,doff,doff+n,src.data,soff,soff+n)) bad("device copy regions overlap");
    DeviceGuard guard(stream.device);
    check(cudaMemcpyAsync(d,s,n,cudaMemcpyDeviceToDevice,reinterpret_cast<cudaStream_t>(stream.native)));
}
void zero(Span dst, std::size_t offset, std::size_t n, Stream stream) {
    validate(dst,true); check_offset(dst,offset,n); check_stream(stream,dst.device);
    if (!n) return;
    DeviceGuard guard(stream.device);
    check(cudaMemsetAsync(static_cast<unsigned char*>(dst.data)+offset,0,n,reinterpret_cast<cudaStream_t>(stream.native)));
}
void synchronize(Stream stream) {
    if (stream.device < 0) bad("stream device must be nonnegative");
    DeviceGuard guard(stream.device);
    check(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream.native)));
}
void fill_bits(DenseView dst, std::uint64_t bits, Stream stream) {
    validate(dst,true); check_stream(stream,dst.values.device);
    if (!dst.rows || !dst.columns) return;
    if (dst.rows > std::numeric_limits<std::uint64_t>::max()/dst.columns) bad("dense element count overflows");
    DeviceGuard guard(stream.device);
    check(static_cast<cudaError_t>(anndatapp_cuda_fill_bits(dst,bits,stream.native)));
}
void fill(DenseView dst, double value, Stream stream) {
    validate(dst,true); check_stream(stream,dst.values.device);
    switch (dst.dtype) {
    case ScalarType::boolean:
        if (value != 0 && value != 1) bad("boolean fill must be zero or one");
        return fill_bits(dst,std::uint64_t(value),stream);
    case ScalarType::int8: return fill_bits(dst,integer_pattern<std::int8_t>(value),stream);
    case ScalarType::uint8: return fill_bits(dst,integer_pattern<std::uint8_t>(value),stream);
    case ScalarType::int16: return fill_bits(dst,integer_pattern<std::int16_t>(value),stream);
    case ScalarType::uint16: return fill_bits(dst,integer_pattern<std::uint16_t>(value),stream);
    case ScalarType::int32: return fill_bits(dst,integer_pattern<std::int32_t>(value),stream);
    case ScalarType::uint32: return fill_bits(dst,integer_pattern<std::uint32_t>(value),stream);
    case ScalarType::int64: return fill_bits(dst,integer_pattern<std::int64_t>(value),stream);
    case ScalarType::uint64: return fill_bits(dst,integer_pattern<std::uint64_t>(value),stream);
    case ScalarType::float16: case ScalarType::float32: case ScalarType::float64: break;
    }
    if (!dst.rows || !dst.columns) return;
    if (dst.rows > std::numeric_limits<std::uint64_t>::max()/dst.columns) bad("dense element count overflows");
    DeviceGuard guard(stream.device);
    check(static_cast<cudaError_t>(anndatapp_cuda_fill_float(dst,value,stream.native)));
}
void cast(DenseView dst, DenseView src, Stream stream, Span status) {
    validate(dst,true); validate(src); check_stream(stream,dst.values.device);
    if (src.values.device != dst.values.device || dst.rows != src.rows || dst.columns != src.columns)
        bad("cast devices or shapes differ");
    if (dst.rows && dst.columns && dst.rows > std::numeric_limits<std::uint64_t>::max()/dst.columns)
        bad("dense element count overflows");
    bool has_status = status.data || status.bytes;
    if (!has_status && dst.rows && dst.columns && conversion_can_fail(dst.dtype,src.dtype))
        bad("cast requires a device status span");
    if (has_status) {
        check_status(status,dst.values.device);
        reject_status_overlap(status,dst.values,dense_range(dst));
        reject_status_overlap(status,src.values,dense_range(src));
    }
    bool same = dst.values.data == src.values.data && dst.origin == src.origin && dst.dtype == src.dtype &&
                dst.row_stride == src.row_stride && dst.column_stride == src.column_stride;
    auto d = dense_range(dst), s = dense_range(src);
    if (!same && overlaps(dst.values.data,d.first,d.last,src.values.data,s.first,s.last)) bad("cast views overlap");
    if ((!dst.rows || !dst.columns || same) && !has_status) return;
    DeviceGuard guard(stream.device);
    if (has_status) check(cudaMemsetAsync(status.data,0,sizeof(std::uint32_t),reinterpret_cast<cudaStream_t>(stream.native)));
    if (!dst.rows || !dst.columns || same) return;
    check(static_cast<cudaError_t>(anndatapp_cuda_cast(dst,src,has_status ? status.data : nullptr,stream.native)));
}

namespace {
void gather(DenseView dst, DenseView src, Span columns, ScalarType index, Stream stream,
            Span status, bool checked) {
    validate(dst,true); validate(src); validate(columns);
    check_stream(stream,dst.values.device);
    if (src.values.device != dst.values.device || columns.device != dst.values.device ||
        dst.rows != src.rows || dst.dtype != src.dtype) bad("gather devices, shapes, or dtypes differ");
    if (!index_type(index)) bad("gather indices must be signed or unsigned integers");
    auto bytes = multiply(dst.columns,scalar_size(index));
    if (bytes > columns.bytes || (bytes && !columns.data)) bad("gather index capacity is too small");
    if (bytes) check_aligned(columns.data,0,scalar_size(index));
    if (dst.rows && dst.columns && dst.rows > std::numeric_limits<std::uint64_t>::max()/dst.columns)
        bad("dense element count overflows");
    auto d = dense_range(dst), s = dense_range(src);
    if (overlaps(dst.values.data,d.first,d.last,src.values.data,s.first,s.last) ||
        overlaps(dst.values.data,d.first,d.last,columns.data,0,bytes)) bad("gather output overlaps an input");
    if (checked) {
        check_status(status,dst.values.device);
        reject_status_overlap(status,dst.values,d);
        reject_status_overlap(status,src.values,s);
        reject_status_overlap(status,columns,{0,bytes});
    } else if (dst.columns && !src.columns) bad("trusted gather has no valid source columns");
    if (!checked && (!dst.rows || !dst.columns)) return;
    DeviceGuard guard(stream.device);
    if (checked) check(cudaMemsetAsync(status.data,0,sizeof(std::uint32_t),reinterpret_cast<cudaStream_t>(stream.native)));
    if (!dst.columns) return;
    check(static_cast<cudaError_t>(anndatapp_cuda_gather_columns(dst,src,columns,index,
        checked ? status.data : nullptr,stream.native)));
}
}
void gather_columns(DenseView dst, DenseView src, Span columns, ScalarType index, Stream stream, Span status) {
    gather(dst,src,columns,index,stream,status,true);
}
void gather_columns_trusted(DenseView dst, DenseView src, Span columns, ScalarType index, Stream stream) {
    gather(dst,src,columns,index,stream,{},false);
}

void pack_host_selected(HostDenseView src, const std::uint64_t* columns, std::size_t count,
                        void* destination, std::size_t capacity) {
    auto width = scalar_size(src.dtype);
    auto range = geometry(src.origin,src.rows,src.columns,src.row_stride,src.column_stride,src.bytes,width);
    check_address(const_cast<void*>(src.data),src.bytes);
    if (src.rows && src.columns && !src.data) bad("nonempty host view has a null pointer");
    if (!columns && count != src.columns) bad("null selection requires all source columns");
    auto n = multiply(multiply(src.rows,count),width);
    if (n > capacity) bad("host packing destination capacity is too small");
    check_address(destination,n);
    if (columns) {
        check_address(const_cast<std::uint64_t*>(columns),multiply(count,sizeof(std::uint64_t)));
        for (std::size_t j=0; j<count; ++j) if (columns[j] >= src.columns) bad("selected column is outside source");
    }
    if (overlaps(const_cast<void*>(src.data),range.first,range.last,destination,0,n) ||
        (columns && overlaps(const_cast<std::uint64_t*>(columns),0,count*sizeof(std::uint64_t),destination,0,n)))
        bad("host packing regions overlap");
    if (!n) return;
    auto* input = static_cast<const unsigned char*>(src.data);
    auto* output = static_cast<unsigned char*>(destination);
    if (!columns && src.column_stride == 1) {
        auto row_bytes = count*width;
        for (std::uint64_t i=0; i<src.rows; ++i) {
            auto pos = offset_at(src.origin,i,0,src.row_stride,src.column_stride,width);
            std::memcpy(output+i*row_bytes,input+pos,row_bytes);
        }
        return;
    }
    auto copy = [&]<std::size_t W>() {
        for (std::uint64_t i=0; i<src.rows; ++i) for (std::size_t j=0; j<count; ++j) {
            auto column = columns ? columns[j] : j;
            auto pos = offset_at(src.origin,i,column,src.row_stride,src.column_stride,W);
            std::memcpy(output+(i*count+j)*W,input+pos,W);
        }
    };
    switch (width) {
    case 1: return copy.template operator()<1>();
    case 2: return copy.template operator()<2>();
    case 4: return copy.template operator()<4>();
    case 8: return copy.template operator()<8>();
    default: bad("invalid packing dtype width");
    }
}

} // namespace anndatapp::cuda
