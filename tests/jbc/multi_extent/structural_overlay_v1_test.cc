#include <Cellerator/execution/object_binding/multi_extent_v1.hh>

#include <cassert>
#include <cstdint>

namespace binding = cellerator::execution::object_binding;

int main() {
    const std::uint32_t topology[] = {0u, 2u, 3u};
    float values[] = {1.0f, 2.0f, 3.0f};
    const binding::structural_atom_v1 structure{
        {1u, 1u}, {2u, 1u}, {3u, 1u}, 7u,
        topology, sizeof(topology), 3u};
    binding::mutable_value_overlay_v1 overlay{
        {1u, 1u}, 11u, values, 3u, sizeof(float)};
    binding::bound_structural_atom_v1 bound{};
    assert(binding::bind_structural_atom_overlay_v1(
        structure, overlay, &bound));
    assert(bound.structure == &structure && bound.overlay == &overlay);

    overlay.value_generation = 12u;
    assert(binding::bind_structural_atom_overlay_v1(
        structure, overlay, &bound));
    assert(bound.structure->structure_epoch == 7u);
    assert(bound.overlay->value_generation == 12u);

    overlay.value_count = 2u;
    assert(binding::bind_structural_atom_overlay_v1(
               structure, overlay, &bound).code ==
        binding::binding_status_code_v1::incompatible_requirement);
}
