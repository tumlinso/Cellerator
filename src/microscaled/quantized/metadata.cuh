#pragma once

#include "numeric.cuh"

namespace cellerator::microscaled::quantized {

template<typename Real>
struct alignas(16) per_gene_affine {
    using real_type = Real;
    const Real* scales;
    const Real* offsets;

    struct row_cache {
    };

    __host__ __device__ __forceinline__ row_cache prepare_row(int) const {
        return {};
    }

    __host__ __device__ __forceinline__ Real scale_for(row_cache, int column) const {
        return scales != nullptr ? load_scalar(scales + column) : from_float<Real>(1.0f);
    }

    __host__ __device__ __forceinline__ Real offset_for(row_cache, int column) const {
        return offsets != nullptr ? load_scalar(offsets + column) : from_float<Real>(0.0f);
    }
};

template<typename Real>
struct alignas(16) column_scale_row_offset {
    using real_type = Real;
    const Real* column_scales;
    const Real* row_offsets;

    struct row_cache {
        Real offset;
    };

    __host__ __device__ __forceinline__ row_cache prepare_row(int row) const {
        row_cache cache;

        cache.offset = row_offsets != nullptr ? load_scalar(row_offsets + row) : from_float<Real>(0.0f);
        return cache;
    }

    __host__ __device__ __forceinline__ Real scale_for(row_cache, int column) const {
        return column_scales != nullptr ? load_scalar(column_scales + column) : from_float<Real>(1.0f);
    }

    __host__ __device__ __forceinline__ Real offset_for(row_cache cache, int) const {
        return cache.offset;
    }
};

template<typename Real>
__host__ __device__ __forceinline__ per_gene_affine<Real> make_per_gene_affine(
    const Real* scales,
    const Real* offsets = nullptr) {
    per_gene_affine<Real> metadata;

    metadata.scales = scales;
    metadata.offsets = offsets;
    return metadata;
}

template<typename Real>
__host__ __device__ __forceinline__ column_scale_row_offset<Real> make_column_scale_row_offset(
    const Real* column_scales,
    const Real* row_offsets = nullptr) {
    column_scale_row_offset<Real> metadata;

    metadata.column_scales = column_scales;
    metadata.row_offsets = row_offsets;
    return metadata;
}

} // namespace cellerator::microscaled::quantized
