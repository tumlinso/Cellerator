module;
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

module velocity;
import :numeric;

namespace velocity {
namespace {
using anndatapp::DType;
using anndatapp::Layout;

std::ptrdiff_t mul(std::ptrdiff_t a, std::ptrdiff_t b) {
    constexpr auto min = std::numeric_limits<std::ptrdiff_t>::min(), max = std::numeric_limits<std::ptrdiff_t>::max();
    if ((a > 0 && (b > 0 ? a > max/b : b < min/a)) ||
        (a < 0 && (b > 0 ? a < min/b : b < 0 && a < max/b)))
        throw std::length_error("numeric stride overflow");
    return a*b;
}
std::ptrdiff_t add(std::ptrdiff_t a, std::ptrdiff_t b) {
    constexpr auto min = std::numeric_limits<std::ptrdiff_t>::min(), max = std::numeric_limits<std::ptrdiff_t>::max();
    if ((b > 0 && a > max - b) || (b < 0 && a < min - b)) throw std::length_error("numeric offset overflow");
    return a + b;
}
std::ptrdiff_t as_signed(std::size_t n) {
    if (n > std::size_t(std::numeric_limits<std::ptrdiff_t>::max())) throw std::length_error("numeric dimension overflow");
    return std::ptrdiff_t(n);
}
std::size_t product(std::size_t n, std::size_t width) {
    if (width && n > std::numeric_limits<std::size_t>::max()/width) throw std::length_error("numeric buffer size overflow");
    return n*width;
}
void address(const void* p, std::size_t n) {
    if (!n) return;
    if (!p) throw std::invalid_argument("null numeric buffer");
    auto base = reinterpret_cast<std::uintptr_t>(p);
    if (n > std::numeric_limits<std::uintptr_t>::max() - base) throw std::length_error("numeric address overflow");
}
std::size_t width(StorageType type) {
    return type == StorageType::float16 ? sizeof(std::uint16_t) : type == StorageType::float32 ? sizeof(float) : sizeof(double);
}
std::size_t alignment(StorageType type) {
    return type == StorageType::float16 ? alignof(std::uint16_t) : type == StorageType::float32 ? alignof(float) : alignof(double);
}
void dense_validate(const detail::NumericStorage& h) {
    if (h.rank == 0 || h.rank > 2)
        throw std::invalid_argument("numeric dense storage requires rank one or two and matching strides");
    bool empty = false;
    for (std::size_t a = 0; a < h.rank; ++a) { (void)as_signed(h.shape[a]); empty |= h.shape[a] == 0; }
    std::ptrdiff_t lo = 0, hi = 0;
    for (std::size_t a = 0; a < h.rank; ++a) {
        auto stride = mul(h.strides[a], std::ptrdiff_t(width(h.type)));
        if (empty || !stride || h.shape[a] <= 1) continue;
        auto d = mul(as_signed(h.shape[a] - 1), stride);
        if (d < 0) lo = add(lo, d); else hi = add(hi, d);
    }
    if (empty) return;
    if (!h.data) throw std::invalid_argument("null numeric dense buffer");
    if (reinterpret_cast<std::uintptr_t>(h.data) % alignment(h.type)) throw std::invalid_argument("unaligned numeric dense buffer");
    if (lo == std::numeric_limits<std::ptrdiff_t>::min()) throw std::length_error("numeric span overflow");
    (void)add(add(hi, -lo), std::ptrdiff_t(width(h.type)));
    auto base = reinterpret_cast<std::uintptr_t>(h.data);
    auto before = lo < 0 ? std::uintptr_t(-(lo + 1)) + 1 : 0;
    if (base < before || std::uintptr_t(hi) > std::numeric_limits<std::uintptr_t>::max() - base ||
        width(h.type) > std::numeric_limits<std::uintptr_t>::max() - (base + std::uintptr_t(hi)))
        throw std::length_error("numeric dense address overflow");
}
std::size_t integer_width(DType t) {
    switch (t) {
    case DType::int8: case DType::uint8: return 1;
    case DType::int16: case DType::uint16: return 2;
    case DType::int32: case DType::uint32: return 4;
    case DType::int64: case DType::uint64: return 8;
    default: throw std::invalid_argument("sparse index and offset types must be integers");
    }
}
template<class T> std::size_t integer_value(const std::byte* p) {
    T x;
    std::memcpy(&x, p, sizeof x);
    if constexpr (std::is_signed_v<T>) if (x < 0) throw std::invalid_argument("negative sparse index or offset");
    auto n = std::uint64_t(x);
    if (n > std::numeric_limits<std::size_t>::max()) throw std::length_error("sparse index exceeds size_t");
    return std::size_t(n);
}
std::size_t integer(const void* data, DType t, std::size_t i) {
    auto p = static_cast<const std::byte*>(data) + product(i, integer_width(t));
    switch (t) {
    case DType::int8: return integer_value<std::int8_t>(p);
    case DType::uint8: return integer_value<std::uint8_t>(p);
    case DType::int16: return integer_value<std::int16_t>(p);
    case DType::uint16: return integer_value<std::uint16_t>(p);
    case DType::int32: return integer_value<std::int32_t>(p);
    case DType::uint32: return integer_value<std::uint32_t>(p);
    case DType::int64: return integer_value<std::int64_t>(p);
    case DType::uint64: return integer_value<std::uint64_t>(p);
    default: throw std::invalid_argument("invalid sparse integer type");
    }
}
void sparse_validate(const detail::NumericStorage& h) {
    if (h.layout != Layout::csr && h.layout != Layout::csc) throw std::invalid_argument("sparse numeric layout must be CSR or CSC");
    auto major = h.layout == Layout::csr ? h.shape[0] : h.shape[1];
    auto minor = h.layout == Layout::csr ? h.shape[1] : h.shape[0];
    if (major == std::numeric_limits<std::size_t>::max()) throw std::length_error("sparse dimension overflow");
    auto iw = integer_width(h.sparse.index_type), ow = integer_width(h.sparse.offset_type);
    if (h.sparse.values_count < h.nnz || h.sparse.indices_count < h.nnz || h.sparse.offsets_count < major + 1)
        throw std::invalid_argument("sparse buffer capacity is too small");
    address(h.sparse.values, product(h.nnz, width(h.type)));
    if (h.nnz && reinterpret_cast<std::uintptr_t>(h.sparse.values) % alignment(h.type))
        throw std::invalid_argument("unaligned numeric sparse values");
    address(h.sparse.indices, product(h.nnz, iw));
    address(h.sparse.offsets, product(major + 1, ow));
    std::size_t previous = integer(h.sparse.offsets, h.sparse.offset_type, 0);
    if (previous != 0) throw std::invalid_argument("sparse offsets must start at zero");
    for (std::size_t i = 1; i <= major; ++i) {
        auto next = integer(h.sparse.offsets, h.sparse.offset_type, i);
        if (next < previous || next > h.nnz) throw std::invalid_argument("invalid sparse offsets");
        previous = next;
    }
    if (previous != h.nnz) throw std::invalid_argument("sparse offsets must end at nnz");
    for (std::size_t i = 0; i < h.nnz; ++i)
        if (integer(h.sparse.indices, h.sparse.index_type, i) >= minor) throw std::invalid_argument("sparse index out of bounds");
}
void require_owner(const std::shared_ptr<const void>& owner) {
    if (!owner.use_count()) throw std::invalid_argument("retained numeric buffer requires an owner");
}
}

NumericView::NumericView(anndatapp::ArrayView array) : array_(std::move(array)) {}
NumericView NumericView::dense_raw(const void* p, StorageType type, std::initializer_list<std::size_t> shape,
    std::initializer_list<std::ptrdiff_t> strides, std::shared_ptr<const void> owner) {
    if (!shape.size() || shape.size() > 2 || shape.size() != strides.size())
        throw std::invalid_argument("numeric dense storage requires rank one or two and matching strides");
    NumericView v; v.raw_ = true; v.owner_ = std::move(owner);
    auto& h = v.storage_; h.data = p; h.type = type; h.rank = shape.size();
    std::copy(shape.begin(), shape.end(), h.shape); std::copy(strides.begin(), strides.end(), h.strides);
    dense_validate(h); return v;
}
#define DENSE_VIEW(T, SUFFIX, TYPE) \
NumericView NumericView::borrow_dense##SUFFIX(const T* p, std::initializer_list<std::size_t> s, std::initializer_list<std::ptrdiff_t> d) { \
    return dense_raw(p, StorageType::TYPE, s, d, {}); \
} \
NumericView NumericView::retain_dense##SUFFIX(const T* p, std::initializer_list<std::size_t> s, std::initializer_list<std::ptrdiff_t> d, std::shared_ptr<const void> o) { \
    require_owner(o); return dense_raw(p, StorageType::TYPE, s, d, std::move(o)); \
}
DENSE_VIEW(std::uint16_t, _f16, float16)
DENSE_VIEW(float, , float32)
DENSE_VIEW(double, , float64)
#undef DENSE_VIEW
NumericView NumericView::sparse_raw(Layout layout, std::size_t rows, std::size_t cols, std::size_t nnz,
    SparseBuffers<void> b, StorageType type, std::shared_ptr<const void> owner) {
    NumericView v; v.raw_ = true; v.owner_ = std::move(owner); auto& h = v.storage_; h.layout = layout; h.rank = 2;
    h.shape[0] = rows; h.shape[1] = cols; h.nnz = nnz; h.sparse = b; h.type = type;
    sparse_validate(h); return v;
}
#define SPARSE_VIEW(T, SUFFIX, TYPE) \
NumericView NumericView::borrow_sparse##SUFFIX(Layout l, std::size_t r, std::size_t c, std::size_t n, SparseBuffers<T> b) { \
    return sparse_raw(l, r, c, n, {b.values,b.values_count,b.indices,b.indices_count,b.index_type,b.offsets,b.offsets_count,b.offset_type}, StorageType::TYPE, {}); \
} \
NumericView NumericView::retain_sparse##SUFFIX(Layout l, std::size_t r, std::size_t c, std::size_t n, SparseBuffers<T> b, std::shared_ptr<const void> o) { \
    require_owner(o); return sparse_raw(l, r, c, n, {b.values,b.values_count,b.indices,b.indices_count,b.index_type,b.offsets,b.offsets_count,b.offset_type}, StorageType::TYPE, std::move(o)); \
}
SPARSE_VIEW(std::uint16_t, _f16, float16)
SPARSE_VIEW(float, , float32)
SPARSE_VIEW(double, , float64)
#undef SPARSE_VIEW
std::size_t NumericView::rank() const noexcept { return raw_ ? storage_.rank : array_.rank(); }
std::size_t NumericView::extent(std::size_t axis) const {
    if (!raw_) return array_.extent(axis);
    if (axis >= storage_.rank) throw std::out_of_range("numeric axis");
    return storage_.shape[axis];
}
StorageType NumericView::dtype() const {
    if (raw_) return storage_.type;
    auto t = array_.dtype();
    if (t == DType::float32) return StorageType::float32;
    if (t == DType::float64) return StorageType::float64;
    throw std::invalid_argument("numeric storage requires float32 or float64 AnnData array");
}
Layout NumericView::layout() const { return raw_ ? storage_.layout : array_.layout(); }
DType NumericView::index_dtype() const {
    if (!raw_) return array_.index_dtype();
    if (layout() == Layout::dense) throw std::invalid_argument("dense storage has no sparse indices");
    return storage_.sparse.index_type;
}
DType NumericView::offset_dtype() const {
    if (!raw_) return array_.offset_dtype();
    if (layout() == Layout::dense) throw std::invalid_argument("dense storage has no sparse offsets");
    return storage_.sparse.offset_type;
}
void NumericView::validate() const {
    if (!raw_) { (void)dtype(); array_.validate(); return; }
    if (layout() == Layout::dense) dense_validate(storage_); else sparse_validate(storage_);
}
NumericView NumericView::subview(std::size_t axis, std::size_t start, std::size_t count, std::ptrdiff_t step) const {
    if (layout() != Layout::dense) throw std::invalid_argument("sparse numeric storage has no affine subview");
    if (axis >= rank()) throw std::out_of_range("numeric axis");
    auto n = extent(axis);
    if (!step || start > n || (count && start == n)) throw std::out_of_range("numeric subview bounds");
    if (count) {
        auto last = add(as_signed(start), mul(as_signed(count - 1), step));
        if (last < 0 || std::size_t(last) >= n) throw std::out_of_range("numeric subview bounds");
    }
    NumericView out = *this;
    if (!raw_) {
        auto b = bytes(); out.raw_ = true; out.storage_.data = b.data(); out.storage_.type = dtype(); out.storage_.rank = b.rank();
        for (std::size_t a = 0; a < b.rank(); ++a) {
            out.storage_.shape[a] = b.extent(a);
            out.storage_.strides[a] = b.stride_bytes(a)/std::ptrdiff_t(b.element_size());
        }
    }
    auto& h = out.storage_;
    bool empty = !count;
    for (std::size_t a = 0; a < h.rank; ++a) if (a != axis) empty |= h.shape[a] == 0;
    auto offset = empty ? 0 : mul(mul(as_signed(start), h.strides[axis]), std::ptrdiff_t(width(h.type)));
    auto p = reinterpret_cast<std::uintptr_t>(h.data);
    if (offset < 0) { auto d = std::uintptr_t(-(offset + 1)) + 1; if (p < d) throw std::length_error("numeric address overflow"); p -= d; }
    else { if (std::uintptr_t(offset) > std::numeric_limits<std::uintptr_t>::max() - p) throw std::length_error("numeric address overflow"); p += std::uintptr_t(offset); }
    h.data = reinterpret_cast<const void*>(p); h.shape[axis] = count; h.strides[axis] = mul(h.strides[axis], step);
    dense_validate(h); return out;
}
NumericBytes<const std::byte> NumericView::bytes() const {
    NumericBytes<const std::byte> out;
    if (!raw_) {
        (void)dtype(); auto b = array_.bytes();
        if (b.rank() > 2) throw std::invalid_argument("numeric dense storage requires rank one or two");
        out.data_ = b.data(); out.width_ = b.element_size(); out.contiguous_ = b.contiguous(); out.rank_ = b.rank();
        for (std::size_t a = 0; a < b.rank(); ++a) { out.shape_[a] = b.extent(a); out.strides_[a] = b.stride_bytes(a); }
        return out;
    }
    if (layout() != Layout::dense) throw std::invalid_argument("sparse numeric storage has no dense bytes");
    auto& h = storage_; out.data_ = static_cast<const std::byte*>(h.data); out.width_ = width(h.type); out.rank_ = h.rank;
    out.contiguous_ = true; std::ptrdiff_t expected = std::ptrdiff_t(out.width_);
    for (std::size_t a = h.rank; a-- > 0;) {
        out.shape_[a] = h.shape[a]; out.strides_[a] = mul(h.strides[a], std::ptrdiff_t(out.width_));
        if (h.shape[a] > 1 && out.strides_[a] != expected) out.contiguous_ = false;
        if (h.shape[a] && expected > std::numeric_limits<std::ptrdiff_t>::max()/as_signed(h.shape[a])) out.contiguous_ = false;
        else expected *= as_signed(h.shape[a]);
    }
    return out;
}
NumericSparseBytes NumericView::sparse_bytes() const {
    NumericSparseBytes out;
    if (!raw_) {
        (void)dtype(); auto b = array_.sparse_bytes();
        out.values_ = b.values(); out.indices_ = b.indices(); out.offsets_ = b.offsets();
        out.rows_ = b.rows(); out.columns_ = b.columns(); out.nnz_ = b.nnz();
        out.value_size_ = b.value_size(); out.index_size_ = b.index_size(); out.offset_size_ = b.offset_size();
        out.values_count_ = out.indices_count_ = b.nnz();
        out.offsets_count_ = (layout() == Layout::csr ? b.rows() : b.columns()) + 1;
        out.index_type_ = index_dtype(); out.offset_type_ = offset_dtype(); return out;
    }
    auto& h = storage_;
    if (h.layout == Layout::dense) throw std::invalid_argument("dense numeric storage has no sparse bytes");
    out.values_ = reinterpret_cast<const std::byte*>(h.sparse.values);
    out.indices_ = static_cast<const std::byte*>(h.sparse.indices); out.offsets_ = static_cast<const std::byte*>(h.sparse.offsets);
    out.rows_ = h.shape[0]; out.columns_ = h.shape[1]; out.nnz_ = h.nnz; out.value_size_ = width(h.type);
    out.index_size_ = integer_width(h.sparse.index_type); out.offset_size_ = integer_width(h.sparse.offset_type);
    out.values_count_ = h.sparse.values_count; out.indices_count_ = h.sparse.indices_count; out.offsets_count_ = h.sparse.offsets_count;
    out.index_type_ = h.sparse.index_type; out.offset_type_ = h.sparse.offset_type; return out;
}
NumericEdit::NumericEdit(anndatapp::ArrayEdit array) : array_(std::move(array)) {}
#define DENSE_EDIT(T, SUFFIX) \
NumericEdit NumericEdit::borrow_dense##SUFFIX(T* p, std::initializer_list<std::size_t> s, std::initializer_list<std::ptrdiff_t> d) { \
    NumericEdit e; e.raw_ = NumericView::borrow_dense##SUFFIX(p, s, d); return e; \
} \
NumericEdit NumericEdit::retain_dense##SUFFIX(T* p, std::initializer_list<std::size_t> s, std::initializer_list<std::ptrdiff_t> d, std::shared_ptr<const void> o) { \
    NumericEdit e; e.raw_ = NumericView::retain_dense##SUFFIX(p, s, d, std::move(o)); return e; \
}
DENSE_EDIT(std::uint16_t, _f16)
DENSE_EDIT(float, )
DENSE_EDIT(double, )
#undef DENSE_EDIT
NumericView NumericEdit::view() const { return raw_.raw_ ? raw_ : NumericView(array_.view()); }
NumericBytes<std::byte> NumericEdit::bytes() const {
    NumericBytes<std::byte> out;
    if (!raw_.raw_) {
        (void)view().dtype(); auto b = array_.bytes();
        if (b.rank() > 2) throw std::invalid_argument("numeric dense storage requires rank one or two");
        out.data_ = b.data(); out.width_ = b.element_size(); out.contiguous_ = b.contiguous(); out.rank_ = b.rank();
        for (std::size_t a = 0; a < b.rank(); ++a) { out.shape_[a] = b.extent(a); out.strides_[a] = b.stride_bytes(a); }
    } else {
        auto b = raw_.bytes(); out.data_ = const_cast<std::byte*>(b.data()); out.rank_ = b.rank(); out.width_ = b.element_size(); out.contiguous_ = b.contiguous();
        for (std::size_t a = 0; a < b.rank(); ++a) { out.shape_[a] = b.extent(a); out.strides_[a] = b.stride_bytes(a); }
    }
    return out;
}
}
