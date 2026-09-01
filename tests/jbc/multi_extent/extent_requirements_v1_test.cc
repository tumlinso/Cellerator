#include <Cellerator/execution/object_binding/multi_extent_v1.hh>

#include <cassert>

namespace binding = cellerator::execution::object_binding;

int main() {
    alignas(16) float values[5]{};
    const binding::atom_port_binding_v1 atoms[] = {
        {{1u, 1u}, 0u, 2u}, {{2u, 1u}, 2u, 3u}};
    const binding::multi_atom_port_binding_v1 logical{
        {3u, 1u}, {4u, 1u}, {5u, 1u}, atoms, 2u,
        binding::port_access_v1::read_only, {}};
    const binding::physical_extent_binding_v1 extents[] = {
        {{1u, 1u}, values, 2u * sizeof(float), 2u, sizeof(float), 16u, 7u,
            binding::extent_residency_v1::device, {}},
        {{2u, 1u}, values + 2, 3u * sizeof(float), 3u, sizeof(float), 16u,
            7u, binding::extent_residency_v1::device, {}},
    };
    const binding::multi_extent_physical_binding_v1 physical{
        {3u, 1u}, extents, 2u};
    const binding::port_extent_requirement_v1 direct_requirement{
        {3u, 1u}, 16u, 2u,
        binding::contiguity_requirement_v1::multi_extent_allowed, true, {}};
    binding::port_extent_query_result_v1 result{};
    assert(binding::query_port_extent_requirements_v1(
        logical, physical, direct_requirement, &result));
    assert(result.directly_compatible);
    assert(!result.assembly_required);
    assert(result.logical_element_count == 5u);
    assert(result.contiguous_bytes == sizeof(values));

    auto contiguous = direct_requirement;
    contiguous.contiguity =
        binding::contiguity_requirement_v1::contiguous_required;
    assert(binding::query_port_extent_requirements_v1(
        logical, physical, contiguous, &result));
    assert(!result.directly_compatible && result.assembly_required);

    binding::physical_extent_binding_v1 mismatched_extents[] = {
        extents[0], extents[1]};
    mismatched_extents[1].value_generation = 8u;
    const binding::multi_extent_physical_binding_v1 mismatched{
        {3u, 1u}, mismatched_extents, 2u};
    assert(binding::query_port_extent_requirements_v1(
        logical, mismatched, direct_requirement, &result));
    assert(!result.directly_compatible);
}
