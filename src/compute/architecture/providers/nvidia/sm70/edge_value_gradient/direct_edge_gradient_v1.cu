#include <Cellerator/compute/architecture/providers/nvidia/sm70/contract/edge_gradient_v1.cuh>

namespace cellerator::compute::architecture::providers::nvidia::sm70::contract {

status_v1 validate_edge_gradient_binding_v1(
    const edge_gradient_binding_v1 &binding) noexcept {
    if (binding.structure_id == 0u || binding.structure_epoch == 0u
        || binding.value_generation == 0u
        || binding.source_activation == nullptr
        || binding.destination_gradient == nullptr
        || binding.dense_width == 0u || binding.edge_gradient == nullptr)
        return status_v1::invalid_argument;
    launch_request_v1 request{};
    request.support = binding.support;
    request.dense = {binding.source_activation, binding.destination_gradient,
        binding.dense_width};
    request.output_order = binding.gradient_order;
    request.output = binding.edge_gradient;
    return validate_launch_v1(request);
}

status_v1 enqueue_direct_edge_gradient_v1(
    const edge_gradient_binding_v1 &binding,
    sparse_candidate_v1 candidate) noexcept {
    const status_v1 validation = validate_edge_gradient_binding_v1(binding);
    if (validation != status_v1::success) return validation;
    launch_request_v1 request{};
    request.support = binding.support;
    request.dense = {binding.source_activation, binding.destination_gradient,
        binding.dense_width};
    request.candidate = candidate;
    request.output_order = binding.gradient_order;
    request.output = binding.edge_gradient;
    request.profiler_correlation_id = binding.profiler_correlation_id;
    request.stream = binding.stream;
    return enqueue_sparse_v1(request);
}

} // namespace cellerator::compute::architecture::providers::nvidia::sm70::contract
