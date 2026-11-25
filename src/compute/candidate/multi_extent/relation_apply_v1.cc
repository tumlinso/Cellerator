#include <Cellerator/execution/object_binding/relation_apply_candidate_v1.hh>

namespace cellerator::compute::candidate::jbc_multi_extent {

namespace binding = execution::object_binding;

binding::binding_status_v1 launch_relation_apply_host_v1(
    const void *prepared_state,
    const binding::multi_extent_physical_binding_v1 &input,
    void *output, std::uint64_t output_bytes, void *caller_stream) noexcept {
    if (prepared_state == nullptr || output == nullptr ||
        output_bytes < sizeof(float) || caller_stream != nullptr) {
        return {binding::binding_status_code_v1::invalid_argument};
    }
    const auto &state = *static_cast<const relation_apply_state_v1 *>(
        prepared_state);
    if (state.extent_count != input.extent_count ||
        (state.extent_count != 0u && state.extent_scales == nullptr)) {
        return {binding::binding_status_code_v1::incompatible_requirement};
    }
    float result = state.bias;
    for (std::uint64_t extent_index = 0u;
         extent_index < input.extent_count; ++extent_index) {
        const auto &extent = input.extents[extent_index];
        if (extent.data == nullptr ||
            extent.element_stride_bytes != sizeof(float) ||
            (extent.residency != binding::extent_residency_v1::host &&
             extent.residency != binding::extent_residency_v1::managed)) {
            return {binding::binding_status_code_v1::incompatible_requirement,
                extent_index};
        }
        const auto *values = static_cast<const float *>(extent.data);
        float partial = 0.0f;
        for (std::uint64_t element = 0u;
             element < extent.element_count; ++element) {
            partial += values[element];
        }
        result += state.extent_scales[extent_index] * partial;
    }
    *static_cast<float *>(output) = result;
    return {};
}

binding::direct_multi_extent_candidate_v1
make_experimental_relation_apply_candidate_v1(
    const relation_apply_state_v1 *state,
    binding::stable_identity_v1 candidate_identity) noexcept {
    binding::direct_multi_extent_candidate_v1 candidate{};
    candidate.candidate_identity = candidate_identity;
    candidate.requirements.maximum_extent_count =
        state == nullptr ? 0u : state->extent_count;
    candidate.requirements.minimum_alignment_bytes = alignof(float);
    candidate.requirements.element_stride_bytes = sizeof(float);
    candidate.requirements.accepted_residencies =
        binding::host_extent_v1 | binding::managed_extent_v1;
    candidate.requirements.accepts_mixed_value_generations = true;
    candidate.requirements.preserves_logical_order = true;
    candidate.prepared_state = state;
    candidate.launch = launch_relation_apply_host_v1;
    return candidate;
}

}  // namespace cellerator::compute::candidate::jbc_multi_extent
