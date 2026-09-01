#include <Cellerator/execution/object_binding/relation_apply_candidate_v1.hh>

#include <cassert>

namespace binding = cellerator::execution::object_binding;
namespace candidate = cellerator::compute::candidate::jbc_multi_extent;

int main() {
    const float left[] = {1.0f, 2.0f};
    const float right[] = {3.0f, 4.0f};
    const binding::physical_extent_binding_v1 extents[] = {
        {{1u, 1u}, left, sizeof(left), 2u, sizeof(float), alignof(float), 1u,
            binding::extent_residency_v1::host, {}},
        {{2u, 1u}, right, sizeof(right), 2u, sizeof(float), alignof(float), 2u,
            binding::extent_residency_v1::host, {}},
    };
    const binding::multi_extent_physical_binding_v1 input{
        {3u, 1u}, extents, 2u};
    const float scales[] = {2.0f, -1.0f};
    const candidate::relation_apply_state_v1 state{scales, 2u, 0.5f};
    const auto direct = candidate::make_experimental_relation_apply_candidate_v1(
        &state, {4u, 1u});
    assert(binding::validate_direct_multi_extent_candidate_v1(direct, input));

    float output = 0.0f;
    assert(direct.launch(direct.prepared_state, input, &output,
        sizeof(output), nullptr));
    assert(output == -0.5f);
    assert(direct.launch(direct.prepared_state, input, &output,
               sizeof(output) - 1u, nullptr).code ==
        binding::binding_status_code_v1::invalid_argument);
}
