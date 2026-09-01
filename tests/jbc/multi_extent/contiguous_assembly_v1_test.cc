#include <Cellerator/execution/object_binding/multi_extent_v1.hh>

#include <cassert>

namespace binding = cellerator::execution::object_binding;

int main() {
    const float left[] = {1.0f, 2.0f};
    const float right[] = {3.0f, 4.0f, 5.0f};
    const binding::physical_extent_binding_v1 extents[] = {
        {{1u, 1u}, left, sizeof(left), 2u, sizeof(float), alignof(float), 1u,
            binding::extent_residency_v1::host, {}},
        {{2u, 1u}, right, sizeof(right), 3u, sizeof(float), alignof(float), 1u,
            binding::extent_residency_v1::host, {}},
    };
    const binding::multi_extent_physical_binding_v1 physical{
        {3u, 1u}, extents, 2u};

    binding::contiguous_assembly_segment_v1 segments[2]{};
    binding::contiguous_assembly_plan_v1 plan{};
    assert(binding::compile_contiguous_assembly_v1(
        physical, alignof(float), segments, 2u, &plan));
    assert(plan.destination_bytes == sizeof(left) + sizeof(right));
    assert(segments[1].destination_offset_bytes == sizeof(left));

    float destination[5]{};
    assert(binding::execute_contiguous_assembly_v1(
        plan, destination, sizeof(destination)));
    for (unsigned index = 0u; index < 5u; ++index) {
        assert(destination[index] == static_cast<float>(index + 1u));
    }

    binding::contiguous_assembly_plan_v1 insufficient{};
    const auto status = binding::compile_contiguous_assembly_v1(
        physical, alignof(float), segments, 1u, &insufficient);
    assert(status.code == binding::binding_status_code_v1::insufficient_capacity);
    assert(status.required_capacity == 2u);
}
