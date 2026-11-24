module;

#include <stddef.h>
#include <Cellerator/memory/allocation.hh>
#include <Cellerator/memory/view.hh>
#include <Cellerator/tensor/base/compressed.hh>

export module cellerator.anndata;
export import anndatapp;

namespace cellerator::anndata::detail {
using ::anndatapp::DType;
using ::anndatapp::Layout;
using ::anndatapp::PreparedRead;
using tensor::base::CompressedAxis;

struct MatrixShape {
    size_t rows = 0, cols = 0, nnz = 0;
    Layout layout = Layout::dense;
};

void read_dense(const PreparedRead&, void*, size_t, size_t, ptrdiff_t, ptrdiff_t,
                DType, size_t, memory::placement, const memory::allocation&);
MatrixShape read_sparse(const PreparedRead&, memory::array_view<void>, DType,
                        memory::array_view<void>, DType, memory::array_view<void>, DType);
[[noreturn]] void bad_view();
[[noreturn]] void bad_stride();
}

export namespace cellerator::anndata {

// The allocation covers the entire addressed byte span, including reversed strides.
template<class T, class S>
void read_into(const ::anndatapp::PreparedRead& source,
               memory::matrix_view<T, S> out, const memory::allocation& backing) {
    ptrdiff_t rs, cs;
    if (__builtin_add_overflow(out.row_stride, 0, &rs) ||
        __builtin_add_overflow(out.column_stride, 0, &cs)) detail::bad_stride();
    detail::read_dense(source, out.data, out.rows, out.columns, rs, cs,
                       ::anndatapp::dtype_of<T>(), sizeof(T), out.where, backing);
}

template<class V, class I, class O>
[[nodiscard]] tensor::base::CompressedView<I, V, O> read_into(
    const ::anndatapp::PreparedRead& source, memory::array_view<V> values,
    memory::array_view<I> indices, memory::array_view<O> offsets) {
    auto shape = detail::read_sparse(source, {values.data, values.count, values.where}, ::anndatapp::dtype_of<V>(),
        {indices.data, indices.count, indices.where}, ::anndatapp::dtype_of<I>(),
        {offsets.data, offsets.count, offsets.where}, ::anndatapp::dtype_of<O>());
    using tensor::base::CompressedAxis;
    auto axis = shape.layout == ::anndatapp::Layout::csr ? CompressedAxis::row : CompressedAxis::column;
    return {{shape.rows, shape.cols, shape.nnz, axis, offsets.data, indices.data}, values.data};
}

template<class T, class S>
void read_into(const ::anndatapp::BackedArray& source,
               memory::matrix_view<T, S> out, const memory::allocation& backing) {
    read_into(source.prepare_read(), out, backing);
}

template<class V, class I, class O>
[[nodiscard]] tensor::base::CompressedView<I, V, O> read_into(
    const ::anndatapp::BackedArray& source, memory::array_view<V> values,
    memory::array_view<I> indices, memory::array_view<O> offsets) {
    return read_into(source.prepare_read(), values, indices, offsets);
}

class BufferedMatrix {
public:
    ~BufferedMatrix() noexcept;
    BufferedMatrix(BufferedMatrix&&) noexcept;
    BufferedMatrix& operator=(BufferedMatrix&&) noexcept;
    BufferedMatrix(const BufferedMatrix&) = delete;
    BufferedMatrix& operator=(const BufferedMatrix&) = delete;

    size_t rows() const noexcept { return shape_.rows; }
    size_t columns() const noexcept { return shape_.cols; }
    size_t nnz() const noexcept { return shape_.nnz; }
    ::anndatapp::Layout layout() const noexcept { return shape_.layout; }
    ::anndatapp::DType dtype() const noexcept { return dtype_; }
    ::anndatapp::DType index_dtype() const noexcept { return index_dtype_; }
    ::anndatapp::DType offset_dtype() const noexcept { return offset_dtype_; }

    template<class T>
    [[nodiscard]] memory::const_matrix_view<T, ptrdiff_t> dense() const & {
        if (layout() != ::anndatapp::Layout::dense || dtype() != ::anndatapp::dtype_of<T>())
            detail::bad_view();
        return {static_cast<const T*>(storage_), rows(), columns(), ptrdiff_t(columns()), {memory::domain::host}};
    }

    template<class I, class V, class O = I>
    [[nodiscard]] tensor::base::CompressedView<I, const V, O> compressed() const & {
        if (layout() == ::anndatapp::Layout::dense || dtype() != ::anndatapp::dtype_of<V>() ||
            index_dtype() != ::anndatapp::dtype_of<I>() || offset_dtype() != ::anndatapp::dtype_of<O>())
            detail::bad_view();
        using tensor::base::CompressedAxis;
        auto axis = layout() == ::anndatapp::Layout::csr ? CompressedAxis::row : CompressedAxis::column;
        return {{rows(), columns(), nnz(), axis, static_cast<const O*>(offsets_),
                 static_cast<const I*>(indices_)}, static_cast<const V*>(storage_)};
    }

    template<class T> void dense() const && = delete;
    template<class I, class V, class O = I> void compressed() const && = delete;

private:
    BufferedMatrix() noexcept = default;
    friend BufferedMatrix read_allocated(const ::anndatapp::PreparedRead&);

    void *storage_ = nullptr, *indices_ = nullptr, *offsets_ = nullptr;
    detail::MatrixShape shape_{};
    ::anndatapp::DType dtype_ = ::anndatapp::DType::float64;
    ::anndatapp::DType index_dtype_ = ::anndatapp::DType::int32;
    ::anndatapp::DType offset_dtype_ = ::anndatapp::DType::int32;
};

[[nodiscard]] BufferedMatrix read_allocated(const ::anndatapp::PreparedRead&);
[[nodiscard]] BufferedMatrix read_allocated(const ::anndatapp::BackedArray&);

}
