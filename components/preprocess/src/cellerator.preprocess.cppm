module;
#include <stddef.h>
#include <stdint.h>
#include <Cellerator/tensor/detail/native_views.hh>
export module cellerator.preprocess;
export import :qc;
export import :hvg;

namespace cellerator::preprocess::detail {
using ::cellerator::tensor::detail::DType;
using ::cellerator::tensor::detail::Layout;
using ::cellerator::tensor::detail::dtype;
using ::cellerator::tensor::detail::erase;
using ::cellerator::tensor::detail::layout;
using DenseInput = ::cellerator::tensor::detail::DenseBytes<const unsigned char>;
using SparseInput = ::cellerator::tensor::detail::SparseBytes<const unsigned char>;
using DenseOutput = ::cellerator::tensor::detail::DenseBytes<unsigned char>;
using SparseOutput = ::cellerator::tensor::detail::SparseBytes<unsigned char>;
template<class I, class T, class O>
SparseInput transform_source(cellerator::tensor::base::CompressedView<I,T,O> x) {
    return erase(cellerator::tensor::base::CompressedView<I,const T,O>{x.structure,x.values});
}
}

export namespace cellerator::preprocess {

// Compose arbitrary caller state without imposing reference layouts. A zero
// block size reports invalid_argument; bindings and synchronization are caller-owned.
template<class Operation> QcStatus for_blocks(size_t total, size_t block, Operation& operation) {
    if (!block) return {QcCode::invalid_argument};
    for (size_t first = 0; first < total;) {
        size_t count = total-first < block ? total-first : block;
        operation(first,count); first += count;
    }
    return {};
}
QcStatus for_blocks(size_t total, size_t block, void (*operation)(void*, size_t, size_t), void* context);

// Callers supply borrowed host matrix views and keep the storage live.
// Sparse inputs must have canonical sorted-unique indices and valid offsets.
// Buffers must not overlap except for identical input and output storage with
// matching shape and strides.
struct NormalizationOptions {
    bool target_is_set = false;
    double target_sum = 1;
    bool exclude_highly_expressed = false;
    double max_fraction = .05;
    bool trusted = false;
};
// Required capacities are rows factors, columns included flags, rows scratch.
// Inputs and buffers are borrowed; factor and included contents are overwritten.
// trusted=true omits per-element finite/nonnegative checks. Caller guarantees
// finite nonnegative values, representable finite row sums and valid structure.
struct NormalizationBuffers {
    double* factors = nullptr; size_t factors_capacity = 0;
    uint8_t* included = nullptr; size_t included_capacity = 0;
    double* scratch = nullptr; size_t scratch_capacity = 0;
    double target_sum = 1;
};
struct ScaleOptions {
    bool zero_center = true, clip = false;
    double max_value = 0;
    const uint8_t* observations = nullptr;
    bool trusted = false;
};

}

namespace cellerator::preprocess::detail {
QcStatus normalization_factors(const DenseInput&, DType, const NormalizationOptions&, NormalizationBuffers&);
QcStatus normalization_factors(const SparseInput&, DType, Layout, const NormalizationOptions&, NormalizationBuffers&);
QcStatus normalize_into(const DenseInput&, DType, const DenseOutput&, DType, const NormalizationOptions&, NormalizationBuffers&);
QcStatus normalize_into(const SparseInput&, DType, Layout, const DenseOutput&, DType, const NormalizationOptions&, NormalizationBuffers&);
QcStatus normalize_into(const SparseInput&, DType, Layout, const SparseOutput&, DType, Layout destination_layout, const NormalizationOptions&, NormalizationBuffers&);
QcStatus apply_normalization(const DenseInput&, DType, const DenseOutput&, DType, const double* factors, size_t capacity, bool trusted);
QcStatus apply_normalization(const SparseInput&, DType, Layout, const DenseOutput&, DType, const double* factors, size_t capacity, bool trusted);
QcStatus apply_normalization(const SparseInput&, DType, Layout, const SparseOutput&, DType, Layout destination_layout, const double* factors, size_t capacity, bool trusted);
QcStatus log1p_into(const DenseInput&, DType, const DenseOutput&, DType, double base, bool trusted);
QcStatus log1p_into(const SparseInput&, DType, Layout, const DenseOutput&, DType, double base, bool trusted);
QcStatus log1p_into(const SparseInput&, DType, Layout, const SparseOutput&, DType, Layout destination_layout, double base, bool trusted);
QcStatus scale_into(const DenseInput&, DType, const DenseOutput&, DType, const ScaleOptions&, double* mean, double* stddev, size_t capacity, uint64_t* scratch, size_t scratch_capacity);
QcStatus scale_into(const SparseInput&, DType, Layout, const DenseOutput&, DType, const ScaleOptions&, double* mean, double* stddev, size_t capacity, uint64_t* scratch, size_t scratch_capacity);
QcStatus scale_into(const SparseInput&, DType, Layout, const SparseOutput&, DType, Layout destination_layout, const ScaleOptions&, double* mean, double* stddev, size_t capacity, uint64_t* scratch, size_t scratch_capacity);
QcStatus apply_scale(const DenseInput&, DType, const DenseOutput&, DType, const ScaleOptions&, const double* mean, const double* stddev, size_t capacity);
QcStatus apply_scale(const SparseInput&, DType, Layout, const DenseOutput&, DType, const ScaleOptions&, const double* mean, const double* stddev, size_t capacity);
QcStatus apply_scale(const SparseInput&, DType, Layout, const SparseOutput&, DType, Layout destination_layout, const ScaleOptions&, const double* mean, const double* stddev, size_t capacity);
}

export namespace cellerator::preprocess {

// Sparse output must declare the same orientation and contain exactly the input
// index/offset structure. Checked calls compare structures; trusted calls assume it.

template<class T, class S>
QcStatus normalization_factors(cellerator::memory::const_matrix_view<T,S> x, const NormalizationOptions& options, NormalizationBuffers& buffers) {
    return detail::normalization_factors(detail::erase(x),detail::dtype<T>(),options,buffers);
}
template<class I, class T, class O>
QcStatus normalization_factors(cellerator::tensor::base::CompressedView<I,T,O> x, const NormalizationOptions& options, NormalizationBuffers& buffers) {
    return detail::normalization_factors(detail::transform_source(x),detail::dtype<T>(),detail::layout(x),options,buffers);
}

template<class T, class S, class U, class R>
QcStatus normalize_into(cellerator::memory::const_matrix_view<T,S> x, cellerator::memory::matrix_view<U,R> y, const NormalizationOptions& options, NormalizationBuffers& buffers) {
    return detail::normalize_into(detail::erase(x),detail::dtype<T>(),detail::erase(y),detail::dtype<U>(),options,buffers);
}
template<class I, class T, class O, class U, class R>
QcStatus normalize_into(cellerator::tensor::base::CompressedView<I,T,O> x, cellerator::memory::matrix_view<U,R> y, const NormalizationOptions& options, NormalizationBuffers& buffers) {
    return detail::normalize_into(detail::transform_source(x),detail::dtype<T>(),detail::layout(x),detail::erase(y),detail::dtype<U>(),options,buffers);
}
template<class I, class T, class O, class J, class U, class P>
QcStatus normalize_into(cellerator::tensor::base::CompressedView<I,T,O> x, cellerator::tensor::base::CompressedView<J,U,P> y, const NormalizationOptions& options, NormalizationBuffers& buffers) {
    return detail::normalize_into(detail::transform_source(x),detail::dtype<T>(),detail::layout(x),detail::erase(y),detail::dtype<U>(),detail::layout(y),options,buffers);
}

// Apply caller-supplied per-row factors; a zero factor leaves that row unchanged.
// trusted=true skips the finite/nonnegative factor scan. Input and output contracts
// are the same as normalize_into. The factors array has at least rows elements.

template<class T, class S, class U, class R>
QcStatus apply_normalization(cellerator::memory::const_matrix_view<T,S> x, cellerator::memory::matrix_view<U,R> y, const double* factors, size_t capacity, bool trusted = false) {
    return detail::apply_normalization(detail::erase(x),detail::dtype<T>(),detail::erase(y),detail::dtype<U>(),factors,capacity,trusted);
}
template<class I, class T, class O, class U, class R>
QcStatus apply_normalization(cellerator::tensor::base::CompressedView<I,T,O> x, cellerator::memory::matrix_view<U,R> y, const double* factors, size_t capacity, bool trusted = false) {
    return detail::apply_normalization(detail::transform_source(x),detail::dtype<T>(),detail::layout(x),detail::erase(y),detail::dtype<U>(),factors,capacity,trusted);
}
template<class I, class T, class O, class J, class U, class P>
QcStatus apply_normalization(cellerator::tensor::base::CompressedView<I,T,O> x, cellerator::tensor::base::CompressedView<J,U,P> y, const double* factors, size_t capacity, bool trusted = false) {
    return detail::apply_normalization(detail::transform_source(x),detail::dtype<T>(),detail::layout(x),detail::erase(y),detail::dtype<U>(),detail::layout(y),factors,capacity,trusted);
}

// base=0 specifies the natural logarithm; other bases must be positive and != 1.
// trusted=true assumes the destination has valid nonoverlapping elements.

template<class T, class S, class U, class R>
QcStatus log1p_into(cellerator::memory::const_matrix_view<T,S> x, cellerator::memory::matrix_view<U,R> y, double base = 0, bool trusted = false) {
    return detail::log1p_into(detail::erase(x),detail::dtype<T>(),detail::erase(y),detail::dtype<U>(),base,trusted);
}
template<class I, class T, class O, class U, class R>
QcStatus log1p_into(cellerator::tensor::base::CompressedView<I,T,O> x, cellerator::memory::matrix_view<U,R> y, double base = 0, bool trusted = false) {
    return detail::log1p_into(detail::transform_source(x),detail::dtype<T>(),detail::layout(x),detail::erase(y),detail::dtype<U>(),base,trusted);
}
template<class I, class T, class O, class J, class U, class P>
QcStatus log1p_into(cellerator::tensor::base::CompressedView<I,T,O> x, cellerator::tensor::base::CompressedView<J,U,P> y, double base = 0, bool trusted = false) {
    return detail::log1p_into(detail::transform_source(x),detail::dtype<T>(),detail::layout(x),detail::erase(y),detail::dtype<U>(),detail::layout(y),base,trusted);
}

// Fit requires columns elements in mean/stddev and uint64 scratch. Input mask
// selects rows for both estimation and application. Sparse centered output is dense.
// trusted=true omits per-element finite/structure validation in statistics.
// Caller guarantees finite input and canonical structure, valid mask and parameters.

template<class T, class S, class U, class R>
QcStatus scale_into(cellerator::memory::const_matrix_view<T,S> x, cellerator::memory::matrix_view<U,R> y, const ScaleOptions& options, double* mean, double* stddev,
    size_t capacity, uint64_t* scratch, size_t scratch_capacity) {
    return detail::scale_into(detail::erase(x),detail::dtype<T>(),detail::erase(y),detail::dtype<U>(),options,mean,stddev,capacity,scratch,scratch_capacity);
}
template<class I, class T, class O, class U, class R>
QcStatus scale_into(cellerator::tensor::base::CompressedView<I,T,O> x, cellerator::memory::matrix_view<U,R> y, const ScaleOptions& options, double* mean, double* stddev,
    size_t capacity, uint64_t* scratch, size_t scratch_capacity) {
    return detail::scale_into(detail::transform_source(x),detail::dtype<T>(),detail::layout(x),detail::erase(y),detail::dtype<U>(),options,mean,stddev,capacity,scratch,scratch_capacity);
}
template<class I, class T, class O, class J, class U, class P>
QcStatus scale_into(cellerator::tensor::base::CompressedView<I,T,O> x, cellerator::tensor::base::CompressedView<J,U,P> y, const ScaleOptions& options, double* mean, double* stddev,
    size_t capacity, uint64_t* scratch, size_t scratch_capacity) {
    return detail::scale_into(detail::transform_source(x),detail::dtype<T>(),detail::layout(x),detail::erase(y),detail::dtype<U>(),detail::layout(y),options,mean,stddev,capacity,scratch,scratch_capacity);
}

template<class T, class S, class U, class R>
QcStatus apply_scale(cellerator::memory::const_matrix_view<T,S> x, cellerator::memory::matrix_view<U,R> y, const ScaleOptions& options, const double* mean,
    const double* stddev, size_t capacity) {
    return detail::apply_scale(detail::erase(x),detail::dtype<T>(),detail::erase(y),detail::dtype<U>(),options,mean,stddev,capacity);
}
template<class I, class T, class O, class U, class R>
QcStatus apply_scale(cellerator::tensor::base::CompressedView<I,T,O> x, cellerator::memory::matrix_view<U,R> y, const ScaleOptions& options, const double* mean,
    const double* stddev, size_t capacity) {
    return detail::apply_scale(detail::transform_source(x),detail::dtype<T>(),detail::layout(x),detail::erase(y),detail::dtype<U>(),options,mean,stddev,capacity);
}
template<class I, class T, class O, class J, class U, class P>
QcStatus apply_scale(cellerator::tensor::base::CompressedView<I,T,O> x, cellerator::tensor::base::CompressedView<J,U,P> y, const ScaleOptions& options, const double* mean,
    const double* stddev, size_t capacity) {
    return detail::apply_scale(detail::transform_source(x),detail::dtype<T>(),detail::layout(x),detail::erase(y),detail::dtype<U>(),detail::layout(y),options,mean,stddev,capacity);
}
}
