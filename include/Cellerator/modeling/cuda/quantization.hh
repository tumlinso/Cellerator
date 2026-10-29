#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace cellerator::modeling::cuda {

// A read-only view of a reusable, identity-indexed profile. Offsets delimit
// one sorted codebook per profile feature; levels is a concatenated array.
// feature_map maps each physical state column to a profile feature ordinal.
struct QuantizationProfileView {
  const std::uint32_t* codebook_offsets{};  // profile_feature_count + 1
  const float* levels{};                    // codebook_offsets.back()
  const std::uint32_t* feature_map{};       // state_feature_count
  std::uint32_t profile_feature_count{};
  std::uint32_t state_feature_count{};
  std::uint32_t level_count{};              // bounds device-side offset checks
  std::uint32_t code_bits{};                // one of 1, 2, 4, 8
};

struct PackedStateView {
  std::uint32_t* words{};
  std::size_t word_capacity{};
};
struct ConstPackedStateView {
  const std::uint32_t* words{};
  std::size_t word_capacity{};
};

// Exact/context-sensitive values are a sparse sidecar. Allocate capacity for
// the largest selected mask (n_rows * n_state_features is always sufficient).
// The kernel appends index/value pairs using override_count. The pair order is
// unspecified; consumers must use each stored logical index.
struct OverrideView {
  std::uint32_t* logical_indices{};
  float* values{};
  std::uint32_t* count{};  // device scalar, cleared by reset_override_count
  std::uint32_t capacity{};
};
struct ConstOverrideView {
  const std::uint32_t* logical_indices{};
  const float* values{};
  const std::uint32_t* count{};
  std::uint32_t capacity{};
};

// Caller-owned asynchronous validation bits. Bit 0: malformed profile
// (bad offsets, unsorted/non-finite levels, or invalid feature mapping).
// Bit 1: an exact override exceeded caller-provided capacity.
// Bit 2: a non-finite source/update value was encountered.
struct DeviceStatus {
  std::uint32_t* bits{};
};

// These launches do not allocate, synchronize, or own their input/output.
// Their pointers, profile arrays, status, and stream must remain valid until
// the stream reaches the returned work. State is row-major [row, feature]; a
// caller supplies feature_map when physical columns use another ordering.
// Before data operations, enqueue reset_status and validate_profile on the
// same stream. When encoding with exact_mask, reset_override_count first;
// ensure the sidecar capacity is sufficient and inspect status after execution.
cudaError_t reset_status(DeviceStatus status, cudaStream_t stream) noexcept;
cudaError_t reset_override_count(OverrideView overrides,
                                 cudaStream_t stream) noexcept;
cudaError_t validate_profile(QuantizationProfileView profile,
                             DeviceStatus status,
                             cudaStream_t stream) noexcept;

cudaError_t encode_state_fp32(const float* state,
                              std::uint32_t row_count,
                              QuantizationProfileView profile,
                              PackedStateView packed,
                              const std::uint8_t* exact_mask,
                              OverrideView overrides,
                              DeviceStatus status,
                              cudaStream_t stream) noexcept;
cudaError_t encode_state_fp16(const __half* state,
                              std::uint32_t row_count,
                              QuantizationProfileView profile,
                              PackedStateView packed,
                              const std::uint8_t* exact_mask,
                              OverrideView overrides,
                              DeviceStatus status,
                              cudaStream_t stream) noexcept;

cudaError_t decode_state_fp32(ConstPackedStateView packed,
                              std::uint32_t row_count,
                              QuantizationProfileView profile,
                              ConstOverrideView overrides,
                              float* output,
                              DeviceStatus status,
                              cudaStream_t stream) noexcept;
cudaError_t decode_state_fp16(ConstPackedStateView packed,
                              std::uint32_t row_count,
                              QuantizationProfileView profile,
                              ConstOverrideView overrides,
                              __half* output,
                              DeviceStatus status,
                              cudaStream_t stream) noexcept;

// Sparse value edits use the same logical row/physical feature axis as encode.
// Distinct indices may share a packed word: compare/exchange preserves every
// neighboring sub-word code. Duplicate logical_indices in one launch are
// unsupported; validate and coalesce them before calling. The exact-value
// sidecar is intentionally not modified by this operation.
cudaError_t update_codes_fp32(const std::uint32_t* logical_indices,
                              const float* values,
                              std::uint32_t update_count,
                              std::uint32_t row_count,
                              QuantizationProfileView profile,
                              PackedStateView packed,
                              DeviceStatus status,
                              cudaStream_t stream) noexcept;
cudaError_t update_codes_fp16(const std::uint32_t* logical_indices,
                              const __half* values,
                              std::uint32_t update_count,
                              std::uint32_t row_count,
                              QuantizationProfileView profile,
                              PackedStateView packed,
                              DeviceStatus status,
                              cudaStream_t stream) noexcept;

}  // namespace cellerator::modeling::cuda
