#include <Cellerator/execution/object_binding/multi_extent_v1.hh>

#include <cassert>

namespace binding = cellerator::execution::object_binding;

binding::binding_status_v1 launch_stub(const void *,
    const binding::multi_extent_physical_binding_v1 &,
    void *, std::uint64_t, void *) noexcept {
    return {};
}

int main() {
    alignas(16) float values[4]{};
    const binding::physical_extent_binding_v1 extents[] = {
        {{1u, 1u}, values, 2u * sizeof(float), 2u, sizeof(float), 16u, 3u,
            binding::extent_residency_v1::device, {}},
        {{2u, 1u}, values + 2, 2u * sizeof(float), 2u, sizeof(float), 16u, 3u,
            binding::extent_residency_v1::device, {}},
    };
    const binding::multi_extent_physical_binding_v1 input{
        {3u, 1u}, extents, 2u};
    binding::direct_multi_extent_candidate_v1 candidate{};
    candidate.candidate_identity = {4u, 1u};
    candidate.requirements.maximum_extent_count = 4u;
    candidate.requirements.minimum_alignment_bytes = 16u;
    candidate.requirements.element_stride_bytes = sizeof(float);
    candidate.requirements.accepted_residencies = binding::device_extent_v1;
    candidate.launch = launch_stub;
    assert(binding::validate_direct_multi_extent_candidate_v1(candidate, input));

    candidate.requirements.maximum_extent_count = 1u;
    assert(binding::validate_direct_multi_extent_candidate_v1(
               candidate, input).code ==
        binding::binding_status_code_v1::incompatible_requirement);
    candidate.requirements.maximum_extent_count = 4u;
    candidate.requirements.accepted_residencies = binding::host_extent_v1;
    assert(binding::validate_direct_multi_extent_candidate_v1(
               candidate, input).code ==
        binding::binding_status_code_v1::incompatible_requirement);
}
