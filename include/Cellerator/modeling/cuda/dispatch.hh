#pragma once

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>

namespace cellerator::modeling::cuda {

// Borrowed arrays describing one immutable host dispatch-plan projection.
// Arrays reside on the same CUDA device as the payload. Offsets have one
// terminal element; origin_dispatch stores dispatch ordinals, ordered exactly
// as the logical selections were declared for each origin.
//
// The host plan is authoritative. Convert the native plan with:
//   selection_origin[d] = selection_to_origin[dispatch_to_selection[d]]
//   dispatch_weight[d] = selection_weights[dispatch_to_selection[d]]
// and copy origin_offsets/origin_dispatch verbatim. Selection routes are
// discrete model decisions: these kernels don't infer, normalize, or change
// them. Pointer contents must remain alive and unmodified until stream work
// completes. Indices must be valid for the declared row counts; validating
// device-resident plan contents would require synchronization and is the
// caller/runtime's responsibility.
struct DispatchPlanView {
    const std::int64_t* selection_origin = nullptr;  // [selection_count]
    const std::int64_t* origin_offsets = nullptr;    // [origin_count + 1]
    const std::int64_t* origin_dispatch = nullptr;   // [selection_count]
    const float* dispatch_weight = nullptr;          // [selection_count], optional (ones)
    std::int64_t selection_count = 0;
    std::int64_t origin_count = 0;
};

// Gather each origin's feature row into dispatch order. Storage is row-major:
// input [origin, feature], output [dispatch selection, feature]. Repeated
// selections copy the row repeatedly. No per-cell state changes owner.
cudaError_t gather_fp16(const __half* input, std::size_t input_elements,
                        __half* gathered, std::size_t gathered_elements,
                        DispatchPlanView plan, std::int64_t feature_count,
                        cudaStream_t stream) noexcept;
cudaError_t gather_fp32(const float* input, std::size_t input_elements,
                        float* gathered, std::size_t gathered_elements,
                        DispatchPlanView plan, std::int64_t feature_count,
                        cudaStream_t stream) noexcept;

// Weighted expert results are returned to origin-major state order. Each
// origin-feature output is owned by exactly one thread and its selections are
// accumulated in stable logical-selection order, so duplicate routing never
// needs floating-point atomics. Weights are used verbatim (caller may provide
// normalized weights or additive contributions). An origin with no selections
// receives zeros.
cudaError_t combine_fp16(const __half* expert_output, std::size_t expert_elements,
                         __half* origin_output, std::size_t origin_elements,
                         DispatchPlanView plan, std::int64_t feature_count,
                         cudaStream_t stream) noexcept;
cudaError_t combine_fp32(const float* expert_output, std::size_t expert_elements,
                         float* origin_output, std::size_t origin_elements,
                         DispatchPlanView plan, std::int64_t feature_count,
                         cudaStream_t stream) noexcept;

// Reverse-mode for `origin_output = sum_d weight[d] * expert_output[d]`.
// Writes one expert gradient per dispatch occurrence and one gate gradient per
// occurrence (a feature dot product). Gradient of the discrete routing indices
// is intentionally undefined. The reduced feature dot uses FP32 for both
// storage types. Set grad_weight to null to omit that derivative; in this case
// expert_output may also be null. Outputs are fully assigned, including empty
// dimensions.
cudaError_t combine_backward_fp16(
    const __half* grad_origin, std::size_t grad_origin_elements,
    const __half* expert_output, std::size_t expert_elements,
    __half* grad_expert, std::size_t grad_expert_elements,
    float* grad_weight, std::size_t grad_weight_elements,
    DispatchPlanView plan, std::int64_t feature_count,
    cudaStream_t stream) noexcept;
cudaError_t combine_backward_fp32(
    const float* grad_origin, std::size_t grad_origin_elements,
    const float* expert_output, std::size_t expert_elements,
    float* grad_expert, std::size_t grad_expert_elements,
    float* grad_weight, std::size_t grad_weight_elements,
    DispatchPlanView plan, std::int64_t feature_count,
    cudaStream_t stream) noexcept;

// Reverse-mode for row gather. Repeated selections from one origin accumulate
// into its input gradient in stable logical-selection order. No atomics.
cudaError_t gather_backward_fp16(const __half* grad_gathered,
                                 std::size_t gathered_elements,
                                 __half* grad_input,
                                 std::size_t input_elements,
                                 DispatchPlanView plan,
                                 std::int64_t feature_count,
                                 cudaStream_t stream) noexcept;
cudaError_t gather_backward_fp32(const float* grad_gathered,
                                 std::size_t gathered_elements,
                                 float* grad_input,
                                 std::size_t input_elements,
                                 DispatchPlanView plan,
                                 std::int64_t feature_count,
                                 cudaStream_t stream) noexcept;

}  // namespace cellerator::modeling::cuda
