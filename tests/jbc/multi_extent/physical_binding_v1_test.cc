#include <Cellerator/execution/object_binding/multi_extent_v1.hh>

#include <cassert>
#include <cstdint>

namespace binding = cellerator::execution::object_binding;

int main() {
    alignas(16) float left[4]{};
    alignas(16) float right[3]{};
    const binding::physical_extent_binding_v1 extents[] = {
        {{1u, 1u}, left, sizeof(left), 4u, sizeof(float), 16u, 2u,
            binding::extent_residency_v1::device, {}},
        {{2u, 1u}, right, sizeof(right), 3u, sizeof(float), 16u, 4u,
            binding::extent_residency_v1::device, {}},
    };
    const binding::multi_extent_physical_binding_v1 port{
        {3u, 1u}, extents, 2u};
    const binding::multi_extent_physical_binding_list_v1 list{&port, 1u};
    assert(binding::validate_multi_extent_physical_bindings_v1(list));

    auto invalid = extents[0];
    invalid.alignment_bytes = 3u;
    const binding::multi_extent_physical_binding_v1 invalid_port{
        {3u, 1u}, &invalid, 1u};
    assert(binding::validate_multi_extent_physical_bindings_v1(
               {&invalid_port, 1u}).code ==
        binding::binding_status_code_v1::invalid_extent);

    const binding::physical_extent_binding_v1 duplicates[] = {
        extents[0], extents[0]};
    const binding::multi_extent_physical_binding_v1 duplicate_port{
        {3u, 1u}, duplicates, 2u};
    assert(binding::validate_multi_extent_physical_bindings_v1(
               {&duplicate_port, 1u}).code ==
        binding::binding_status_code_v1::duplicate_atom);
}
