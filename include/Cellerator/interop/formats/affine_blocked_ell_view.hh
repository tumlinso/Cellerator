#pragma once

#include "affine_blocked_ell.hh"
#include <Cellerator/tensor/quantized/blocked_ell.hh>

namespace cellerator::interop::formats {

// Borrow the legacy archive's packed rows as a runtime-dispatched quantized
// BELL view. The archive object and every pointed-to buffer must outlive the
// returned view. No bytes or affine metadata are copied or reinterpreted.
inline tensor::quantized::RuntimeBlockedEllView make_runtime_view(
    const AffineBlockedEllCodec& archive) noexcept {
    return tensor::quantized::RuntimeBlockedEllView{
        archive.rows,
        archive.cols,
        archive.nnz,
        archive.block_size,
        archive.ell_cols,
        archive.row_stride_bytes,
        archive.bits,
        archive.decode_policy,
        archive.blockColIdx,
        archive.packed_values,
        archive.column_scales,
        archive.column_offsets,
        archive.row_offsets};
}

} // namespace cellerator::interop::formats
