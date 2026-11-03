#pragma once

#include <Cellerator/tensor/state_codes/view.hh>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstdint>

namespace cellerator::tensor::state_codes {

// These launches do not allocate, synchronize, or own their input/output.
// Their pointers and stream must remain valid until the work completes.
// For raw views, enqueue reset_status and validate_profile before the data
// operation on the same stream. Profile validates its host codebook before
// upload. When encoding with exact_mask, reset the override count first,
// provide sufficient sidecar capacity, and inspect status after execution.
cudaError_t reset_status(DeviceStatus status, cudaStream_t stream) noexcept;
cudaError_t reset_exact_override_count(ExactOverrideView overrides,
                                       cudaStream_t stream) noexcept;
cudaError_t validate_profile(ProfileView profile, DeviceStatus status,
                             cudaStream_t stream) noexcept;

cudaError_t encode_fp32(const float* state, std::uint32_t row_count,
                        ProfileView profile, PackedView packed,
                        const std::uint8_t* exact_mask,
                        ExactOverrideView exact_overrides,
                        DeviceStatus status, cudaStream_t stream) noexcept;
cudaError_t encode_fp16(const __half* state, std::uint32_t row_count,
                        ProfileView profile, PackedView packed,
                        const std::uint8_t* exact_mask,
                        ExactOverrideView exact_overrides,
                        DeviceStatus status, cudaStream_t stream) noexcept;

cudaError_t decode_fp32(ConstPackedView packed, std::uint32_t row_count,
                        ProfileView profile,
                        ConstExactOverrideView exact_overrides, float* output,
                        DeviceStatus status, cudaStream_t stream) noexcept;
cudaError_t decode_fp16(ConstPackedView packed, std::uint32_t row_count,
                        ProfileView profile,
                        ConstExactOverrideView exact_overrides, __half* output,
                        DeviceStatus status, cudaStream_t stream) noexcept;

// Distinct logical indices may share a packed word; compare/exchange retains
// neighboring sub-word codes. Duplicate indices in one launch are unsupported
// and must be coalesced by the caller. These updates do not touch exact values.
cudaError_t update_codes_fp32(const std::uint32_t* logical_indices,
                              const float* values,
                              std::uint32_t update_count,
                              std::uint32_t row_count,
                              ProfileView profile, PackedView packed,
                              DeviceStatus status,
                              cudaStream_t stream) noexcept;
cudaError_t update_codes_fp16(const std::uint32_t* logical_indices,
                              const __half* values,
                              std::uint32_t update_count,
                              std::uint32_t row_count,
                              ProfileView profile, PackedView packed,
                              DeviceStatus status,
                              cudaStream_t stream) noexcept;

}  // namespace cellerator::tensor::state_codes
