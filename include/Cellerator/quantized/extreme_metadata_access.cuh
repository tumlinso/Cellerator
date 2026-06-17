#pragma once

#include "metadata.cuh"
#include "extreme_ptx_primitives.cuh"

namespace cellerator::quantized::extreme_backend {

template<typename Metadata>
struct prepared_metadata_sm70;

template<typename Real>
struct prepared_metadata_sm70<per_gene_affine<Real>> {
    const Real *scales = nullptr;
    const Real *offsets = nullptr;

    __device__ __forceinline__ float scale_for(int column) const {
        return scales != nullptr ? to_float(load_scalar(scales + column)) : 1.0f;
    }

    __device__ __forceinline__ float offset_for(int column) const {
        return offsets != nullptr ? to_float(load_scalar(offsets + column)) : 0.0f;
    }
};

template<typename Real>
struct prepared_metadata_sm70<column_scale_row_offset<Real>> {
    const Real *column_scales = nullptr;
    Real row_offset = from_float<Real>(0.0f);

    __device__ __forceinline__ float scale_for(int column) const {
        return column_scales != nullptr ? to_float(load_scalar(column_scales + column)) : 1.0f;
    }

    __device__ __forceinline__ float offset_for(int) const {
        return to_float(row_offset);
    }
};

template<typename Real>
__device__ __forceinline__ prepared_metadata_sm70<per_gene_affine<Real>> prepare_metadata_sm70(
    const per_gene_affine<Real> &metadata,
    typename per_gene_affine<Real>::row_cache) {
    prepared_metadata_sm70<per_gene_affine<Real>> prepared;
    prepared.scales = metadata.scales;
    prepared.offsets = metadata.offsets;
    return prepared;
}

template<typename Real>
__device__ __forceinline__ prepared_metadata_sm70<column_scale_row_offset<Real>> prepare_metadata_sm70(
    const column_scale_row_offset<Real> &metadata,
    typename column_scale_row_offset<Real>::row_cache row_cache) {
    prepared_metadata_sm70<column_scale_row_offset<Real>> prepared;
    prepared.column_scales = metadata.column_scales;
    prepared.row_offset = row_cache.offset;
    return prepared;
}

template<int Bits, typename Prepared, typename Real>
__device__ __forceinline__ unsigned int quantize_code_sm70_extreme_prepared(
    Real value,
    const Prepared &prepared,
    int column) {
    const float scale = prepared.scale_for(column);

    if (scale == 0.0f) {
        return 0u;
    }

    const float offset = prepared.offset_for(column);
    const float centered = ptx_sub_f32(to_float(value), offset);
    const float q = ptx_mul_f32(centered, ptx_rcp_nr1_f32(scale));
    return ptx_clamp_code(static_cast<unsigned int>(format_traits<Bits>::code_mask), ptx_round_rni_s32(q));
}

template<int Bits, typename Prepared, typename Real>
__device__ __forceinline__ Real dequantize_code_sm70_extreme_prepared(
    unsigned int code,
    const Prepared &prepared,
    int column) {
    const float scale = prepared.scale_for(column);
    const float offset = prepared.offset_for(column);
    const float reconstructed = ptx_fma_f32(ptx_cvt_f32_u32(code), scale, offset);
    return from_float<Real>(reconstructed);
}

} // namespace cellerator::quantized::extreme_backend
