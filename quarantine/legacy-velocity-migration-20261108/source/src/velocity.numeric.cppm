module;
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <type_traits>

export module velocity:numeric;
export import anndatapp.core;

export namespace velocity {

enum class StorageType { float16, float32, float64 };

// Capacities are element counts. All buffers must remain stable while used.
template<class T> struct SparseBuffers {
    const T* values = nullptr;
    std::size_t values_count = 0;
    const void* indices = nullptr;
    std::size_t indices_count = 0;
    anndatapp::DType index_type = anndatapp::DType::int32;
    const void* offsets = nullptr;
    std::size_t offsets_count = 0;
    anndatapp::DType offset_type = anndatapp::DType::int32;
};
using SparseF16 = SparseBuffers<std::uint16_t>;

namespace detail {
struct NumericStorage {
    const void* data = nullptr;
    std::size_t shape[2] = {}, rank = 0, nnz = 0;
    std::ptrdiff_t strides[2] = {}; // elements
    StorageType type = StorageType::float16;
    anndatapp::Layout layout = anndatapp::Layout::dense;
    SparseBuffers<void> sparse;
};
}

// Acquired descriptors borrow their data. Keep the source view/edit alive.
template<class Byte> class NumericBytes {
public:
    NumericBytes() = default;
    // Trusted metadata: caller supplies valid, aligned storage and byte strides.
    NumericBytes(Byte* data, std::size_t rows, std::size_t columns, std::ptrdiff_t row_stride,
                 std::ptrdiff_t column_stride, std::size_t width, bool contiguous) noexcept
        : data_(data), shape_{rows, columns}, rank_(2), width_(width),
          strides_{row_stride, column_stride}, contiguous_(contiguous) {}
    Byte* data() const noexcept { return data_; }
    std::size_t rank() const noexcept { return rank_; }
    std::size_t extent(std::size_t axis) const { if (axis >= rank_) throw std::out_of_range("numeric axis"); return shape_[axis]; }
    std::ptrdiff_t stride_bytes(std::size_t axis) const { if (axis >= rank_) throw std::out_of_range("numeric axis"); return strides_[axis]; }
    std::size_t element_size() const noexcept { return width_; }
    bool contiguous() const noexcept { return contiguous_; }
private:
    template<class> friend class NumericDense;
    friend class NumericView;
    friend class NumericEdit;
    Byte* data_ = nullptr;
    std::size_t shape_[2] = {}, rank_ = 0, width_ = 0;
    std::ptrdiff_t strides_[2] = {};
    bool contiguous_ = false;
};

template<class T> class NumericDense {
public:
    explicit NumericDense(NumericBytes<std::conditional_t<std::is_const_v<T>, const std::byte, std::byte>> bytes) : bytes_(bytes) {}
    T* data() const noexcept { return reinterpret_cast<T*>(bytes_.data()); }
    std::size_t rank() const noexcept { return bytes_.rank(); }
    std::size_t extent(std::size_t axis) const { return bytes_.extent(axis); }
    std::ptrdiff_t stride(std::size_t axis) const { return bytes_.stride_bytes(axis)/std::ptrdiff_t(sizeof(T)); }
    bool contiguous() const noexcept { return bytes_.contiguous(); }
    T* contiguous_data() const {
        if (!contiguous()) throw std::invalid_argument("numeric view is not contiguous");
        return data();
    }
private:
    friend class NumericView;
    friend class NumericEdit;
    NumericBytes<std::conditional_t<std::is_const_v<T>, const std::byte, std::byte>> bytes_;
};

class NumericSparseBytes {
public:
    const std::byte* values() const noexcept { return values_; }
    const std::byte* indices() const noexcept { return indices_; }
    const std::byte* offsets() const noexcept { return offsets_; }
    std::size_t rows() const noexcept { return rows_; }
    std::size_t columns() const noexcept { return columns_; }
    std::size_t nnz() const noexcept { return nnz_; }
    std::size_t value_size() const noexcept { return value_size_; }
    std::size_t index_size() const noexcept { return index_size_; }
    std::size_t offset_size() const noexcept { return offset_size_; }
    std::size_t values_count() const noexcept { return values_count_; }
    std::size_t indices_count() const noexcept { return indices_count_; }
    std::size_t offsets_count() const noexcept { return offsets_count_; }
    anndatapp::DType index_type() const noexcept { return index_type_; }
    anndatapp::DType offset_type() const noexcept { return offset_type_; }
private:
    friend class NumericView;
    const std::byte *values_ = nullptr, *indices_ = nullptr, *offsets_ = nullptr;
    std::size_t rows_ = 0, columns_ = 0, nnz_ = 0, value_size_ = 0, index_size_ = 0, offset_size_ = 0;
    std::size_t values_count_ = 0, indices_count_ = 0, offsets_count_ = 0;
    anndatapp::DType index_type_ = anndatapp::DType::int32, offset_type_ = anndatapp::DType::int32;
};

class NumericView {
public:
    NumericView() = default;
    NumericView(anndatapp::ArrayView array);
    static NumericView borrow_dense_f16(const std::uint16_t*, std::initializer_list<std::size_t>, std::initializer_list<std::ptrdiff_t>);
    static NumericView retain_dense_f16(const std::uint16_t*, std::initializer_list<std::size_t>, std::initializer_list<std::ptrdiff_t>, std::shared_ptr<const void> owner);
    static NumericView borrow_dense(const float*, std::initializer_list<std::size_t>, std::initializer_list<std::ptrdiff_t>);
    static NumericView borrow_dense(const double*, std::initializer_list<std::size_t>, std::initializer_list<std::ptrdiff_t>);
    static NumericView retain_dense(const float*, std::initializer_list<std::size_t>, std::initializer_list<std::ptrdiff_t>, std::shared_ptr<const void> owner);
    static NumericView retain_dense(const double*, std::initializer_list<std::size_t>, std::initializer_list<std::ptrdiff_t>, std::shared_ptr<const void> owner);
    static NumericView borrow_sparse_f16(anndatapp::Layout, std::size_t rows, std::size_t columns, std::size_t nnz, SparseF16);
    static NumericView retain_sparse_f16(anndatapp::Layout, std::size_t rows, std::size_t columns, std::size_t nnz, SparseF16, std::shared_ptr<const void> owner);
    static NumericView borrow_sparse(anndatapp::Layout, std::size_t rows, std::size_t columns, std::size_t nnz, SparseBuffers<float>);
    static NumericView borrow_sparse(anndatapp::Layout, std::size_t rows, std::size_t columns, std::size_t nnz, SparseBuffers<double>);
    static NumericView retain_sparse(anndatapp::Layout, std::size_t rows, std::size_t columns, std::size_t nnz, SparseBuffers<float>, std::shared_ptr<const void> owner);
    static NumericView retain_sparse(anndatapp::Layout, std::size_t rows, std::size_t columns, std::size_t nnz, SparseBuffers<double>, std::shared_ptr<const void> owner);
    std::size_t rank() const noexcept;
    std::size_t extent(std::size_t axis) const;
    StorageType dtype() const;
    anndatapp::Layout layout() const;
    anndatapp::DType index_dtype() const;
    anndatapp::DType offset_dtype() const;
    void validate() const;
    // Dense affine selection. step may be negative; count zero permits start == extent.
    NumericView subview(std::size_t axis, std::size_t start, std::size_t count, std::ptrdiff_t step = 1) const;
    NumericBytes<const std::byte> bytes() const;
    NumericSparseBytes sparse_bytes() const;
    template<class T> NumericDense<const T> dense() const {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, std::uint16_t>);
        auto expected = std::is_same_v<T, float> ? StorageType::float32 : std::is_same_v<T, double> ? StorageType::float64 : StorageType::float16;
        if (dtype() != expected) throw std::invalid_argument("numeric view dtype mismatch");
        if constexpr (!std::is_same_v<T, std::uint16_t>) if (!raw_) (void)array_.dense<T>();
        return NumericDense<const T>(bytes());
    }
private:
    friend class NumericEdit;
    static NumericView dense_raw(const void*, StorageType, std::initializer_list<std::size_t>, std::initializer_list<std::ptrdiff_t>, std::shared_ptr<const void>);
    static NumericView sparse_raw(anndatapp::Layout, std::size_t, std::size_t, std::size_t, SparseBuffers<void>, StorageType, std::shared_ptr<const void>);
    detail::NumericStorage storage_;
    bool raw_ = false;
    std::shared_ptr<const void> owner_;
    anndatapp::ArrayView array_;
};

class NumericEdit {
public:
    NumericEdit() = default;
    NumericEdit(anndatapp::ArrayEdit array);
    static NumericEdit borrow_dense_f16(std::uint16_t*, std::initializer_list<std::size_t>, std::initializer_list<std::ptrdiff_t>);
    static NumericEdit retain_dense_f16(std::uint16_t*, std::initializer_list<std::size_t>, std::initializer_list<std::ptrdiff_t>, std::shared_ptr<const void> owner);
    static NumericEdit borrow_dense(float*, std::initializer_list<std::size_t>, std::initializer_list<std::ptrdiff_t>);
    static NumericEdit borrow_dense(double*, std::initializer_list<std::size_t>, std::initializer_list<std::ptrdiff_t>);
    static NumericEdit retain_dense(float*, std::initializer_list<std::size_t>, std::initializer_list<std::ptrdiff_t>, std::shared_ptr<const void> owner);
    static NumericEdit retain_dense(double*, std::initializer_list<std::size_t>, std::initializer_list<std::ptrdiff_t>, std::shared_ptr<const void> owner);
    NumericView view() const;
    NumericBytes<std::byte> bytes() const;
    template<class T> NumericDense<T> dense() const {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, std::uint16_t>);
        auto expected = std::is_same_v<T, float> ? StorageType::float32 : std::is_same_v<T, double> ? StorageType::float64 : StorageType::float16;
        if (view().dtype() != expected) throw std::invalid_argument("numeric edit dtype mismatch");
        if constexpr (!std::is_same_v<T, std::uint16_t>) if (!raw_.raw_) (void)array_.dense<T>();
        return NumericDense<T>(bytes());
    }
private:
    NumericView raw_;
    anndatapp::ArrayEdit array_;
};

}
