#include "Cellerator/geometry/optimizer/device/proposal_scoring.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <limits>

namespace cellerator::geometry::optimizer::device {
namespace {

__device__ bool checked_add(std::int64_t left, std::int64_t right,
                            std::int64_t* result) {
    if ((right > 0 && left > INT64_MAX - right) ||
        (right < 0 && left < INT64_MIN - right)) return false;
    *result = left + right;
    return true;
}

__device__ bool checked_multiply(std::int64_t left, std::int64_t right,
                                 std::int64_t* result) {
    if (left == 0 || right == 0) {
        *result = 0;
        return true;
    }
    if ((left > 0 && ((right > 0 && left > INT64_MAX / right) ||
                      (right < 0 && right < INT64_MIN / left))) ||
        (left < 0 && ((right > 0 && left < INT64_MIN / right) ||
                      (right < 0 && right < INT64_MAX / left)))) return false;
    *result = left * right;
    return true;
}

__global__ void score_proposals_kernel(
        proposal_scoring_problem_v1 problem,
        proposal_score_result_v1* results) {
    const std::uint64_t stride =
            static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
    for (std::uint64_t proposal_index =
                 static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         proposal_index < problem.proposal_count; proposal_index += stride) {
        const auto span = problem.proposals[proposal_index];
        proposal_score_result_v1 output{};
        output.stable_proposal_id = span.stable_proposal_id;
        if (span.first_term > problem.term_count ||
            span.term_count > problem.term_count - span.first_term) {
            output.flags = proposal_score_invalid_span;
            results[proposal_index] = output;
            continue;
        }
        for (std::uint64_t local = 0; local < span.term_count; ++local) {
            const auto term = problem.terms[span.first_term + local];
            for (std::uint32_t component = 0;
                 component < score_component_count; ++component) {
                std::int64_t weighted = 0;
                std::int64_t next = 0;
                if (!checked_multiply(term.component_delta[component],
                                      problem.weights.component[component],
                                      &weighted) ||
                    !checked_add(output.weighted_objective_delta, weighted,
                                 &next)) {
                    output.flags |= proposal_score_arithmetic_overflow;
                    break;
                }
                output.weighted_objective_delta = next;
            }
            if (output.flags != proposal_score_valid ||
                !checked_add(output.mma_interaction_delta,
                             term.mma_interaction_delta,
                             &output.mma_interaction_delta) ||
                !checked_add(output.residual_interaction_delta,
                             term.residual_interaction_delta,
                             &output.residual_interaction_delta)) {
                output.flags |= proposal_score_arithmetic_overflow;
                break;
            }
        }
        results[proposal_index] = output;
    }
}

}  // namespace

proposal_scoring_status launch_proposal_scoring_v1(
        const proposal_scoring_problem_v1& device_problem,
        proposal_score_result_v1* device_results,
        std::uint64_t result_capacity,
        void* caller_stream) noexcept {
    if ((device_problem.proposal_count != 0 &&
         (device_problem.proposals == nullptr || device_results == nullptr)) ||
        (device_problem.term_count != 0 && device_problem.terms == nullptr)) {
        return proposal_scoring_status::invalid_argument;
    }
    if (result_capacity < device_problem.proposal_count) {
        return proposal_scoring_status::insufficient_capacity;
    }
    if (device_problem.proposal_count == 0) {
        return proposal_scoring_status::success;
    }
    constexpr std::uint32_t threads = 128;
    const auto requested_blocks = device_problem.proposal_count / threads +
            (device_problem.proposal_count % threads == 0 ? 0U : 1U);
    const auto blocks = static_cast<std::uint32_t>(
            requested_blocks > 65535U ? 65535U : requested_blocks);
    score_proposals_kernel<<<blocks, threads, 0,
            static_cast<cudaStream_t>(caller_stream)>>>(
                    device_problem, device_results);
    return cudaPeekAtLastError() == cudaSuccess
            ? proposal_scoring_status::success
            : proposal_scoring_status::launch_failed;
}

}  // namespace cellerator::geometry::optimizer::device
