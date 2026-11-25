module;
#include <stddef.h>
#include <stdint.h>
#include <Cellerator/tensor/detail/native_views.hh>

export module cellerator.preprocess:hvg;

namespace cellerator::preprocess::detail {
using ::cellerator::tensor::detail::DType;
using ::cellerator::tensor::detail::Layout;
using ::cellerator::tensor::detail::dtype;
using ::cellerator::tensor::detail::erase;
using ::cellerator::tensor::detail::layout;
using DenseInput = ::cellerator::tensor::detail::DenseBytes<const unsigned char>;
using SparseInput = ::cellerator::tensor::detail::SparseBytes<const unsigned char>;
template<class I, class T, class O>
SparseInput hvg_source(cellerator::tensor::base::CompressedView<I,T,O> x) {
    return erase(cellerator::tensor::base::CompressedView<I,const T,O>{x.structure,x.values});
}
}

export namespace cellerator::preprocess {

enum class HvgMethod { seurat, cell_ranger };
enum class HvgCode { ok, invalid_argument, insufficient_capacity, overflow, invalid_input, empty_batch, no_scores, duplicate_bin_edges };
struct HvgStatus { HvgCode code = HvgCode::ok; size_t index = 0, required_bytes = 0; };
struct HvgOptions {
    HvgMethod method = HvgMethod::seurat;
    size_t n_top_genes = 0, n_bins = 20;
    double min_mean = .0125, max_mean = 3, min_disp = .5, max_disp = __builtin_inf();
    double log_base = 0;
    bool has_top = false, has_log_base = false, batched = false;
};
struct HvgOutput {
    double *mean = nullptr, *dispersion = nullptr, *normalized_dispersion = nullptr;
    uint8_t *selected = nullptr, *intersection = nullptr;
    size_t* nbatches = nullptr;
    size_t capacity = 0;
};
// Batch-major moment arrays have capacity >= genes*batches; counts has
// capacity >= batches. Arrays are owned by the caller and must not overlap.
struct HvgMoments {
    double *sum = nullptr, *square = nullptr;
    uint8_t* expressed = nullptr;
    size_t* counts = nullptr;
    size_t capacity = 0, batch_capacity = 0;
};

}

namespace cellerator::preprocess::detail {
HvgStatus hvg_accumulate_dense(const DenseInput&, DType, const size_t* batch, size_t batches,
    const HvgOptions&, HvgMoments);
HvgStatus hvg_accumulate_dense_unchecked(const DenseInput&, DType, const size_t* batch, size_t batches,
    const HvgOptions&, HvgMoments);
HvgStatus hvg_accumulate_sparse(const SparseInput&, DType, Layout, const size_t* batch, size_t batches,
    const HvgOptions&, HvgMoments);
HvgStatus hvg_accumulate_sparse_unchecked(const SparseInput&, DType, Layout, const size_t* batch, size_t batches,
    const HvgOptions&, HvgMoments);
HvgStatus hvg_dense(const DenseInput&, DType, const size_t* batch, size_t batches,
    const HvgOptions&, HvgOutput, void* scratch, size_t scratch_bytes);
HvgStatus hvg_sparse(const SparseInput&, DType, Layout, const size_t* batch, size_t batches,
    const HvgOptions&, HvgOutput, void* scratch, size_t scratch_bytes);
HvgStatus hvg_dense_unchecked(const DenseInput&, DType, const size_t* batch, size_t batches,
    const HvgOptions&, HvgOutput, void* scratch, size_t scratch_bytes);
HvgStatus hvg_sparse_unchecked(const SparseInput&, DType, Layout, const size_t* batch, size_t batches,
    const HvgOptions&, HvgOutput, void* scratch, size_t scratch_bytes);
}

export namespace cellerator::preprocess {

// Scratch is suitably aligned raw or byte storage permitting implicit-lifetime
// double and size_t objects, aligned to alignof(size_t) and alignof(double).
// Batch codes are contiguous [0,batches), caller-chosen; every batch is used.
// Input, output, and scratch storage must not overlap; all pointers remain valid
// until return. Capacities are in elements for output, bytes for scratch.
// On insufficient capacity no output is written; other failures may leave
// partial output and scratch contents. Batched calls also fill nbatches and
// intersection; unbatched calls leave those two arrays untouched.
HvgStatus hvg_scratch_bytes(size_t genes, size_t batches, const HvgOptions&, size_t& bytes);
// Rank batched genes by descending selected-batch count, then descending
// normalized dispersion, then original gene position. order has capacity genes.
HvgStatus hvg_rank_batched(const double* normalized_dispersion, const size_t* nbatches,
    size_t genes, size_t n_top_genes, uint8_t* selected, size_t* order, size_t capacity);
// Clear before the first accumulation; successive calls add rows. Input row
// batch codes address this same batch space. Any failed accumulation invalidates
// all moments: clear and replay before further accumulation, merge, or score.
HvgStatus hvg_moments_clear(size_t genes, size_t batches, HvgMoments);
// Add source into destination. Both refer to the same gene/batch order; arrays
// may be identical pairwise or disjoint, and counts must not overflow.
HvgStatus hvg_moments_merge(size_t genes, size_t batches, HvgMoments destination, const HvgMoments& source);
template<class T, class S>
HvgStatus hvg_accumulate_dense(cellerator::memory::const_matrix_view<T,S> view, const size_t* batch, size_t batches,
    const HvgOptions& options, HvgMoments moments) {
    return detail::hvg_accumulate_dense(detail::erase(view),detail::dtype<T>(),batch,batches,options,moments);
}
template<class T, class S>
HvgStatus hvg_accumulate_dense_unchecked(cellerator::memory::const_matrix_view<T,S> view, const size_t* batch, size_t batches,
    const HvgOptions& options, HvgMoments moments) {
    return detail::hvg_accumulate_dense_unchecked(detail::erase(view),detail::dtype<T>(),batch,batches,options,moments);
}
template<class I, class T, class O>
HvgStatus hvg_accumulate_sparse(cellerator::tensor::base::CompressedView<I,T,O> view, const size_t* batch, size_t batches,
    const HvgOptions& options, HvgMoments moments) {
    return detail::hvg_accumulate_sparse(detail::hvg_source(view),detail::dtype<T>(),detail::layout(view),batch,batches,options,moments);
}
template<class I, class T, class O>
HvgStatus hvg_accumulate_sparse_unchecked(cellerator::tensor::base::CompressedView<I,T,O> view, const size_t* batch, size_t batches,
    const HvgOptions& options, HvgMoments moments) {
    return detail::hvg_accumulate_sparse_unchecked(detail::hvg_source(view),detail::dtype<T>(),detail::layout(view),batch,batches,options,moments);
}
// Score caller-constructed moments without copying them. Input/scratch/output
// must be disjoint. Each batch must have nonzero count. Use the scoring scratch
// query; it excludes input moments and requires the alignment documented above.
HvgStatus hvg_score_scratch_bytes(size_t genes, const HvgOptions&, size_t& bytes);
HvgStatus hvg_score(const HvgMoments&, size_t genes, size_t batches, const HvgOptions&,
    HvgOutput, void* scratch, size_t scratch_bytes);
HvgStatus hvg_score_unchecked(const HvgMoments&, size_t genes, size_t batches, const HvgOptions&,
    HvgOutput, void* scratch, size_t scratch_bytes);
template<class T, class S>
HvgStatus hvg_dense(cellerator::memory::const_matrix_view<T,S> view, const size_t* batch, size_t batches,
    const HvgOptions& options, HvgOutput output, void* scratch, size_t scratch_bytes) {
    return detail::hvg_dense(detail::erase(view),detail::dtype<T>(),batch,batches,options,output,scratch,scratch_bytes);
}
template<class I, class T, class O>
HvgStatus hvg_sparse(cellerator::tensor::base::CompressedView<I,T,O> view, const size_t* batch, size_t batches,
    const HvgOptions& options, HvgOutput output, void* scratch, size_t scratch_bytes) {
    return detail::hvg_sparse(detail::hvg_source(view),detail::dtype<T>(),detail::layout(view),batch,batches,options,output,scratch,scratch_bytes);
}

// Trusted variants omit value and sparse-structure checks. The caller supplies
// finite, nonnegative transformed values and canonical compressed structures.
template<class T, class S>
HvgStatus hvg_dense_unchecked(cellerator::memory::const_matrix_view<T,S> view, const size_t* batch, size_t batches,
    const HvgOptions& options, HvgOutput output, void* scratch, size_t scratch_bytes) {
    return detail::hvg_dense_unchecked(detail::erase(view),detail::dtype<T>(),batch,batches,options,output,scratch,scratch_bytes);
}
template<class I, class T, class O>
HvgStatus hvg_sparse_unchecked(cellerator::tensor::base::CompressedView<I,T,O> view, const size_t* batch, size_t batches,
    const HvgOptions& options, HvgOutput output, void* scratch, size_t scratch_bytes) {
    return detail::hvg_sparse_unchecked(detail::hvg_source(view),detail::dtype<T>(),detail::layout(view),batch,batches,options,output,scratch,scratch_bytes);
}

} // namespace preprocess
