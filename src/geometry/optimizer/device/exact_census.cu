#include "Cellerator/geometry/optimizer/device/exact_census.h"

#include <cuda_runtime_api.h>

#include <cstdint>

namespace cellerator::geometry::optimizer::device {
namespace {

__device__ bool add_u64(std::uint64_t value, std::uint64_t* total) {
    if (value > UINT64_MAX - *total) return false;
    *total += value;
    return true;
}

__device__ bool signed_delta(std::uint64_t after, std::uint64_t before,
                             std::int64_t* result) {
    const auto magnitude = after >= before ? after - before : before - after;
    if (magnitude > static_cast<std::uint64_t>(INT64_MAX)) return false;
    const auto value = static_cast<std::int64_t>(magnitude);
    *result = after >= before ? value : -value;
    return true;
}

__global__ void exact_census_kernel(exact_census_problem_v1 problem,
                                    exact_census_result_v1* results) {
    const std::uint64_t stride =
            static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
    for (std::uint64_t proposal =
                 static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         proposal < problem.proposal_count; proposal += stride) {
        const auto span = problem.proposals[proposal];
        exact_census_result_v1 output{};
        output.stable_proposal_id = span.stable_proposal_id;
        if (span.first_change > problem.change_count ||
            span.change_count > problem.change_count - span.first_change) {
            output.flags = exact_census_invalid_span;
            results[proposal] = output;
            continue;
        }
        std::uint64_t before_mma = 0;
        std::uint64_t before_residual = 0;
        std::uint64_t after_mma = 0;
        std::uint64_t after_residual = 0;
        for (std::uint64_t local = 0; local < span.change_count; ++local) {
            const auto change = problem.changes[span.first_change + local];
            if (change.destination_count != 0 &&
                change.source_count > UINT64_MAX / change.destination_count) {
                output.flags |= exact_census_arithmetic_overflow;
                break;
            }
            const auto slots = change.source_count * change.destination_count;
            if (!add_u64(change.before_mma, &before_mma) ||
                !add_u64(change.before_residual, &before_residual) ||
                !add_u64(change.after_mma, &after_mma) ||
                !add_u64(change.after_residual, &after_residual) ||
                !add_u64(slots, &output.after_physical_slots)) {
                output.flags |= exact_census_arithmetic_overflow;
                break;
            }
            if (change.before_residual > UINT64_MAX - change.before_mma ||
                change.after_residual > UINT64_MAX - change.after_mma) {
                output.flags |= exact_census_arithmetic_overflow;
            } else if (change.before_mma + change.before_residual !=
                       change.after_mma + change.after_residual) {
                output.flags |= exact_census_nonunique_contribution;
            }
            if (change.after_mma > slots) {
                output.flags |= exact_census_rectangle_overfull;
            }
        }
        if (output.flags == exact_census_valid) {
            if (!add_u64(before_mma, &output.before_interactions) ||
                !add_u64(before_residual, &output.before_interactions) ||
                !add_u64(after_mma, &output.after_interactions) ||
                !add_u64(after_residual, &output.after_interactions) ||
                after_mma > output.after_physical_slots ||
                !signed_delta(after_mma, before_mma, &output.mma_delta) ||
                !signed_delta(after_residual, before_residual,
                              &output.residual_delta)) {
                output.flags |= exact_census_arithmetic_overflow;
            } else {
                output.after_padding_slots =
                        output.after_physical_slots - after_mma;
            }
        }
        results[proposal] = output;
    }
}

}  // namespace

exact_census_status launch_exact_census_v1(
        const exact_census_problem_v1& device_problem,
        exact_census_result_v1* device_results,
        std::uint64_t result_capacity,
        void* caller_stream) noexcept {
    if ((device_problem.proposal_count != 0 &&
         (device_problem.proposals == nullptr || device_results == nullptr)) ||
        (device_problem.change_count != 0 && device_problem.changes == nullptr)) {
        return exact_census_status::invalid_argument;
    }
    if (result_capacity < device_problem.proposal_count) {
        return exact_census_status::insufficient_capacity;
    }
    if (device_problem.proposal_count == 0) return exact_census_status::success;
    constexpr std::uint32_t threads = 128;
    const auto requested_blocks = device_problem.proposal_count / threads +
            (device_problem.proposal_count % threads == 0 ? 0U : 1U);
    const auto blocks = static_cast<std::uint32_t>(
            requested_blocks > 65535U ? 65535U : requested_blocks);
    exact_census_kernel<<<blocks, threads, 0,
            static_cast<cudaStream_t>(caller_stream)>>>(
                    device_problem, device_results);
    return cudaPeekAtLastError() == cudaSuccess
            ? exact_census_status::success
            : exact_census_status::launch_failed;
}

}  // namespace cellerator::geometry::optimizer::device
