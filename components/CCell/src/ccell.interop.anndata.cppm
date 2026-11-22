module;
#include "CCell/detail/native_views.hh"

export module ccell.interop.anndata;
import anndata;

namespace anndata = ::anndatapp;

static_assert(__is_standard_layout(anndata::Float16) &&
    sizeof(anndata::Float16) == sizeof(uint16_t) &&
    __builtin_offsetof(anndata::Float16, bits) == 0);

export namespace ccell::io {

enum class BorrowCode : uint8_t { ok, rank, layout, dtype, non_contiguous, extent, invalid };

}

namespace ccell::io::detail {

using Axis = cellerator::tensor::base::CompressedAxis;

inline BorrowCode borrow_error(const anndata::Error& error) noexcept {
    using Code = anndata::ErrorCode;
    Code code = error.code();
    return code == Code::unsupported_layout ? BorrowCode::layout
         : code == Code::dtype_mismatch ? BorrowCode::dtype
         : code == Code::non_contiguous ? BorrowCode::non_contiguous
         : code == Code::out_of_bounds ? BorrowCode::extent : BorrowCode::invalid;
}

inline size_t rank(const anndata::ArrayView& source) noexcept { return source.rank(); }
inline size_t rank(const anndata::ArrayEdit& source) { return source.view().rank(); }

template<class T, class Source, class View>
BorrowCode dense(const Source& source, View& out) {
    out = {};
    try {
        if (rank(source) != 2) return BorrowCode::rank;
        const auto data = source.template dense<T>();
        const View view{data.data(), data.extent(0), data.extent(1), data.stride(0),
            {cellerator::memory::domain::host}, data.stride(1)};
        if (!::ccell::detail::erase(view).rank()) return BorrowCode::extent;
        out = view;
        return BorrowCode::ok;
    } catch (const anndata::Error& error) {
        return borrow_error(error);
    }
}

template<class T, class I, class O, class Source, class View>
BorrowCode compressed(const Source& source, Axis axis, View& out) {
    out = {};
    try {
        if (rank(source) != 2) return BorrowCode::rank;
        if (axis != Axis::row && axis != Axis::column) return BorrowCode::layout;
        const auto data = axis == Axis::row ? source.template csr<T, I, O>()
                                           : source.template csc<T, I, O>();
        const View view{{data.rows(), data.columns(), data.nnz(), axis, data.offsets(), data.indices()},
            data.values()};
        if (!::ccell::detail::erase(view).value_size()) return BorrowCode::extent;
        out = view;
        return BorrowCode::ok;
    } catch (const anndata::Error& error) {
        return borrow_error(error);
    }
}

}

export namespace ccell::io {

// Borrow existing storage without materializing selection or changing order.
// The caller keeps the source owner live and its structure stable during use.
template<class T>
BorrowCode borrow_dense(const anndata::ArrayView& source,
        cellerator::memory::const_matrix_view<T, ptrdiff_t>& out) {
    return detail::dense<T>(source, out);
}

template<class T>
BorrowCode borrow_dense(const anndata::ArrayEdit& source,
        cellerator::memory::matrix_view<T, ptrdiff_t>& out) {
    return detail::dense<T>(source, out);
}

template<class I, class T, class O>
BorrowCode borrow_compressed(const anndata::ArrayView& source,
        cellerator::tensor::base::CompressedAxis axis,
        cellerator::tensor::base::CompressedView<I, const T, O>& out) {
    return detail::compressed<T, I, O>(source, axis, out);
}

template<class I, class T, class O>
BorrowCode borrow_compressed(const anndata::ArrayEdit& source,
        cellerator::tensor::base::CompressedAxis axis,
        cellerator::tensor::base::CompressedView<I, T, O>& out) {
    return detail::compressed<T, I, O>(source, axis, out);
}

}
