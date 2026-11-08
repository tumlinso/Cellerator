module;
#include <stddef.h>
#include <stdint.h>
export module cellerator.velocity;
export import velocitypp;
export import velocitypp.precision;

export namespace cellerator::velocity {
// Binary16 values are raw IEEE encodings, never AnnData++ integer storage.
// Dense strides are signed bytes; sparse indices and offsets are typed separately.
struct HalfMatrix {
    anndatapp::Layout layout = anndatapp::Layout::dense;
    const uint16_t* values = nullptr;
    const void* indices = nullptr;
    const void* offsets = nullptr;
    anndatapp::DType index_type = anndatapp::DType::int32, offset_type = anndatapp::DType::int64;
    size_t rows = 0, columns = 0, nnz = 0;
    ptrdiff_t row_stride_bytes = 0, column_stride_bytes = 0;
};
// Exactly one source is nonnull. All descriptors and storage are borrowed.
struct Matrix {
    const HalfMatrix* half = nullptr;
    const velocitypp::AcquiredMatrix* acquired = nullptr;
};
struct Input { Matrix spliced, unspliced, weights; velocitypp::FitDomain domain; };
velocitypp::Status validate(const Input&);
// Trusted after validate: output is row-major rows x count with stride count.
// Caller supplies rows*count elements in each requested field, and two rows
// of source scratch. All buffers must be disjoint from each other and inputs.
template<class T> void moments(const Input&, size_t first, size_t count,
    T* spliced, T* unspliced, T* source_s, T* source_u);
template<class T> void second_moments(const Input&, size_t first, size_t count,
    T* spliced, T* unspliced, T* spliced_second, T* cross, T* unspliced_second,
    T* source_s, T* source_u);
// Residual scratch is rows*count T elements; output is row/column strided in
// uint16_t elements. Check capacities, strides, and non-overlap before calling.
template<class T> void residuals_f16(const velocitypp::BasicMomentBlock<T>&,
    const velocitypp::BasicFitBlock<T>&, T* scratch, uint16_t* out, ptrdiff_t row_stride, ptrdiff_t column_stride = 1);
template<class T> void stochastic_residuals_f16(const velocitypp::BasicStochasticMomentBlock<T>&,
    const velocitypp::BasicStochasticFitBlock<T>&, T* scratch, uint16_t* out, uint16_t* second,
    ptrdiff_t row_stride, ptrdiff_t column_stride = 1);
// Use as precision::MomentReplacement; upstream original-column promotion owns
// selection and status. Set one active block before each corresponding pass.
template<class T> struct PromotionMoments {
    const Input* input = nullptr;
    velocitypp::BasicMomentBlock<T>* initial = nullptr;
    velocitypp::BasicStochasticMomentBlock<T>* second = nullptr;
    T* source_s = nullptr; T* source_u = nullptr; // rows elements each
};
velocitypp::precision::PrecisionCode replace32(void*, size_t, size_t, const velocitypp::precision::OriginalColumn&, bool);
velocitypp::precision::PrecisionCode replace64(void*, size_t, size_t, const velocitypp::precision::OriginalColumn&, bool);
}
