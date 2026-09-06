#pragma once
#include <Cellerator/compute/architecture/providers/nvidia/sm70/contract/contract_portfolio_v1.cuh>
namespace cellerator::compute::architecture::providers::nvidia::sm70::edge_value_gradient {
namespace contract = cellerator::compute::architecture::providers::nvidia::sm70::contract;
struct relation_gradient_request {
    contract::support_view_v1 support{}; // cold-validated physical-order edges
    const float* source = nullptr;
    const float* cotangent = nullptr;
    float* output = nullptr; // one slot per physical edge
    std::uint64_t source_capacity = 0, cotangent_capacity = 0, output_capacity = 0;
    bool half_rounded = false;
    __half* source_scratch = nullptr;
    __half* cotangent_scratch = nullptr;
    std::uint64_t source_scratch_capacity = 0, cotangent_scratch_capacity = 0;
    cudaStream_t stream = nullptr;
};
// N16 full-f32 dot, or explicit fresh RNE packs plus retained half sparse dot.
// All maps/scratch are caller-owned and prepared; no hot topology work.
contract::status_v1 enqueue_relation_gradient(const relation_gradient_request&) noexcept;
}
