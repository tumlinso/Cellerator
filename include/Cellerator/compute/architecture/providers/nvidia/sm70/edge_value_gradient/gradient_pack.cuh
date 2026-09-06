#pragma once
#include <Cellerator/compute/architecture/providers/nvidia/sm70/contract/contract_portfolio_v1.cuh>
namespace cellerator::compute::architecture::providers::nvidia::sm70::edge_value_gradient {
namespace contract = cellerator::compute::architecture::providers::nvidia::sm70::contract;
struct pack_request {
    const float *input=nullptr;
    std::uint32_t input_rows=0, input_stride=16;
    std::uint64_t input_capacity=0;
    __half *output=nullptr;
    std::uint32_t output_rows=0, output_stride=16;
    std::uint64_t output_capacity=0;
    // Optional immutable device IDs validated in cold cover preparation.
    // Each is an input row or UINT32_MAX for a zero-padded row.
    const std::uint32_t *gather_ids=nullptr;
    std::uint64_t gather_capacity=0;
    bool half_rounded=false;
    cudaStream_t stream=nullptr;
};
// N16 only. Refreshes every invocation; no pointer/version cache. Caller owns
// aligned output and keeps input/maps immutable through stream completion.
// IEEE binary16 RNE including signed zero/subnormal/overflow/nonfinite behavior.
contract::status_v1 enqueue_gradient_pack(const pack_request&) noexcept;
}
