module;
#include <stddef.h>
#include <stdint.h>
#include <Cellerator/tensor/detail/native_views.hh>
export module cellerator.preprocess:qc;
namespace cellerator::preprocess::detail {
using ::cellerator::tensor::detail::DType;
using ::cellerator::tensor::detail::Layout;
using ::cellerator::tensor::detail::dtype;
using ::cellerator::tensor::detail::erase;
using ::cellerator::tensor::detail::layout;
using DenseInput = ::cellerator::tensor::detail::DenseBytes<const unsigned char>;
using SparseInput = ::cellerator::tensor::detail::SparseBytes<const unsigned char>;
template<class I, class T, class O>
SparseInput source(cellerator::tensor::base::CompressedView<I,T,O> x) {
    return erase(cellerator::tensor::base::CompressedView<I,const T,O>{x.structure,x.values});
}

}

export namespace cellerator::preprocess {

// Host views and all result/scratch storage are borrowed until the call returns.
// Dense strides address elements, including signed strides for reversed views.
// Compressed indices must be sorted and unique within each major vector.

enum class QcCode : uint8_t { ok, invalid_argument, insufficient_capacity, invalid_value, invalid_structure, unsupported_type };
struct QcStatus { QcCode code = QcCode::ok; size_t index = 0, required = 0; explicit operator bool() const { return code == QcCode::ok; } };

// Group masks are contiguous group-major boolean bytes with `columns` bytes per group.
struct QcOptions { const uint8_t* masks = nullptr; size_t groups = 0; const size_t* percent_top = nullptr; size_t tops = 0; };
// All arrays are caller-owned; optional arrays may be null. Cell arrays have rows
// entries, gene arrays columns entries; group arrays use group-major rows;
// top_percent uses row-major tops. Capacities are element counts.
struct QcBuffers {
    double* cell_total = nullptr; uint64_t* cell_detected = nullptr; size_t cells_capacity = 0;
    double* gene_total = nullptr; uint64_t* gene_detected = nullptr; size_t genes_capacity = 0;
    double* mean = nullptr; double* dropout = nullptr;
    double* log_cell_total = nullptr; double* log_cell_detected = nullptr;
    double* log_gene_total = nullptr; double* log_gene_mean = nullptr;
    double* group_total = nullptr; double* group_percent = nullptr; double* group_log_total = nullptr; size_t group_capacity = 0;
    double* top_percent = nullptr; size_t top_capacity = 0;
};
// Welford partials represent selected rows in [first,last). Each partial owns
// no memory: caller supplies per-gene mean/M2 arrays and count scratch. Separate
// workers use disjoint arrays; merge in the caller's chosen order.
struct StatsPartial { size_t observations = 0, columns = 0; double* mean = nullptr; double* m2 = nullptr; size_t capacity = 0; };

}

namespace cellerator::preprocess::detail {
QcStatus qc_validate(const DenseInput&, DType, bool nonnegative);
QcStatus qc_validate(const SparseInput&, DType, Layout, bool nonnegative);
QcStatus qc_accumulate(const DenseInput&, DType, size_t first, size_t last, const QcOptions&, const QcBuffers&, double* scratch, size_t scratch_elements);
QcStatus qc_accumulate(const SparseInput&, DType, Layout, size_t first, size_t last, const QcOptions&, const QcBuffers&, double* scratch, size_t scratch_elements);
QcStatus qc_accumulate_unchecked(const DenseInput&, DType, size_t first, size_t last, const QcOptions&, const QcBuffers&, double* scratch);
QcStatus qc_accumulate_unchecked(const SparseInput&, DType, Layout, size_t first, size_t last, const QcOptions&, const QcBuffers&, double* scratch);
QcStatus qc(const DenseInput&, DType, const QcOptions&, const QcBuffers&, double* scratch, size_t scratch_elements);
QcStatus qc(const SparseInput&, DType, Layout, const QcOptions&, const QcBuffers&, double* scratch, size_t scratch_elements);
QcStatus stats_partial(const DenseInput&, DType, size_t first, size_t last, const uint8_t* mask, StatsPartial&, uint64_t* scratch, size_t scratch_elements);
QcStatus stats_partial(const SparseInput&, DType, Layout, size_t first, size_t last, const uint8_t* mask, StatsPartial&, uint64_t* scratch, size_t scratch_elements);
}

export namespace cellerator::preprocess::detail {
// Compiled preprocessing also uses these two statistics paths and the structure check.
QcStatus check_sparse_structure(const SparseInput&, Layout);
QcStatus feature_stats(const DenseInput&, DType, const uint8_t* mask, double* mean, double* variance, size_t capacity, uint64_t* scratch, size_t scratch_elements, size_t* observations);
QcStatus feature_stats(const SparseInput&, DType, Layout, const uint8_t* mask, double* mean, double* variance, size_t capacity, uint64_t* scratch, size_t scratch_elements, size_t* observations);
QcStatus feature_stats_unchecked(const DenseInput&, DType, const uint8_t* mask, double* mean, double* variance, size_t capacity, uint64_t* scratch, size_t scratch_elements, size_t* observations);
QcStatus feature_stats_unchecked(const SparseInput&, DType, Layout, const uint8_t* mask, double* mean, double* variance, size_t capacity, uint64_t* scratch, size_t scratch_elements, size_t* observations);
}

export namespace cellerator::preprocess {

QcStatus qc_check(size_t rows, size_t columns, const QcOptions&, const QcBuffers&, size_t scratch_elements);

// Validate finite values and canonical sparse structure without allocation.
template<class T, class S>
QcStatus qc_validate(cellerator::memory::const_matrix_view<T,S> x, bool nonnegative) {
    return detail::qc_validate(detail::erase(x),detail::dtype<T>(),nonnegative);
}
template<class I, class T, class O>
QcStatus qc_validate(cellerator::tensor::base::CompressedView<I,T,O> x, bool nonnegative) {
    return detail::qc_validate(detail::source(x),detail::dtype<T>(),detail::layout(x),nonnegative);
}
void qc_zero(size_t rows, size_t columns, size_t groups, const QcBuffers&);

// Accumulation writes the cell rows in [first,last) and updates gene totals.
// The caller zeros the buffers first. Invalid input may leave partial values.
template<class T, class S>
QcStatus qc_accumulate(cellerator::memory::const_matrix_view<T,S> x, size_t first, size_t last, const QcOptions& options, const QcBuffers& buffers, double* scratch, size_t capacity) {
    return detail::qc_accumulate(detail::erase(x),detail::dtype<T>(),first,last,options,buffers,scratch,capacity);
}
template<class I, class T, class O>
QcStatus qc_accumulate(cellerator::tensor::base::CompressedView<I,T,O> x, size_t first, size_t last, const QcOptions& options, const QcBuffers& buffers, double* scratch, size_t capacity) {
    return detail::qc_accumulate(detail::source(x),detail::dtype<T>(),detail::layout(x),first,last,options,buffers,scratch,capacity);
}
// Caller guarantees valid finite nonnegative data, canonical sparse structure,
// valid outputs and ranges; no per-entry validation occurs.
template<class T, class S>
QcStatus qc_accumulate_unchecked(cellerator::memory::const_matrix_view<T,S> x, size_t first, size_t last, const QcOptions& options, const QcBuffers& buffers, double* scratch) {
    return detail::qc_accumulate_unchecked(detail::erase(x),detail::dtype<T>(),first,last,options,buffers,scratch);
}
template<class I, class T, class O>
QcStatus qc_accumulate_unchecked(cellerator::tensor::base::CompressedView<I,T,O> x, size_t first, size_t last, const QcOptions& options, const QcBuffers& buffers, double* scratch) {
    return detail::qc_accumulate_unchecked(detail::source(x),detail::dtype<T>(),detail::layout(x),first,last,options,buffers,scratch);
}
void qc_merge(size_t columns, double* total, uint64_t* detected, const double* part_total, const uint64_t* part_detected);
void qc_finish(size_t rows, size_t columns, const QcOptions&, const QcBuffers&);
template<class T, class S>
QcStatus qc(cellerator::memory::const_matrix_view<T,S> x, const QcOptions& options, const QcBuffers& buffers, double* scratch, size_t capacity) {
    return detail::qc(detail::erase(x),detail::dtype<T>(),options,buffers,scratch,capacity);
}
template<class I, class T, class O>
QcStatus qc(cellerator::tensor::base::CompressedView<I,T,O> x, const QcOptions& options, const QcBuffers& buffers, double* scratch, size_t capacity) {
    return detail::qc(detail::source(x),detail::dtype<T>(),detail::layout(x),options,buffers,scratch,capacity);
}

struct FilterBounds { bool has_min_counts = false, has_max_counts = false, has_min_detected = false, has_max_detected = false; double min_counts = 0, max_counts = 0; uint64_t min_detected = 0, max_detected = 0; };
QcStatus filter(const double* totals, const uint64_t* detected, size_t count, const FilterBounds&, uint8_t* keep, size_t capacity);
template<class T, class S>
QcStatus stats_partial(cellerator::memory::const_matrix_view<T,S> x, size_t first, size_t last, const uint8_t* mask, StatsPartial& result, uint64_t* scratch, size_t capacity) {
    return detail::stats_partial(detail::erase(x),detail::dtype<T>(),first,last,mask,result,scratch,capacity);
}
template<class I, class T, class O>
QcStatus stats_partial(cellerator::tensor::base::CompressedView<I,T,O> x, size_t first, size_t last, const uint8_t* mask, StatsPartial& result, uint64_t* scratch, size_t capacity) {
    return detail::stats_partial(detail::source(x),detail::dtype<T>(),detail::layout(x),first,last,mask,result,scratch,capacity);
}
QcStatus stats_merge(StatsPartial&, const StatsPartial&);
// Finalizing an empty partial sets its mean and the output variance to NaN.
QcStatus stats_finish(StatsPartial&, double* variance, size_t capacity);
// Scratch holds `columns` uint64_t values; mask is optional contiguous rows bytes.
// Inputs must be canonical sparse with sorted unique indices. All values must be finite.
template<class T, class S>
QcStatus feature_stats(cellerator::memory::const_matrix_view<T,S> x, const uint8_t* mask, double* mean, double* variance, size_t capacity,
    uint64_t* scratch, size_t scratch_capacity, size_t* observations) {
    return detail::feature_stats(detail::erase(x),detail::dtype<T>(),mask,mean,variance,capacity,scratch,scratch_capacity,observations);
}
template<class I, class T, class O>
QcStatus feature_stats(cellerator::tensor::base::CompressedView<I,T,O> x, const uint8_t* mask, double* mean, double* variance, size_t capacity,
    uint64_t* scratch, size_t scratch_capacity, size_t* observations) {
    return detail::feature_stats(detail::source(x),detail::dtype<T>(),detail::layout(x),mask,mean,variance,capacity,scratch,scratch_capacity,observations);
}
// Assumes finite values and canonical sparse structure; capacity checks remain.
template<class T, class S>
QcStatus feature_stats_unchecked(cellerator::memory::const_matrix_view<T,S> x, const uint8_t* mask, double* mean, double* variance, size_t capacity,
    uint64_t* scratch, size_t scratch_capacity, size_t* observations) {
    return detail::feature_stats_unchecked(detail::erase(x),detail::dtype<T>(),mask,mean,variance,capacity,scratch,scratch_capacity,observations);
}
template<class I, class T, class O>
QcStatus feature_stats_unchecked(cellerator::tensor::base::CompressedView<I,T,O> x, const uint8_t* mask, double* mean, double* variance, size_t capacity,
    uint64_t* scratch, size_t scratch_capacity, size_t* observations) {
    return detail::feature_stats_unchecked(detail::source(x),detail::dtype<T>(),detail::layout(x),mask,mean,variance,capacity,scratch,scratch_capacity,observations);
}

}
