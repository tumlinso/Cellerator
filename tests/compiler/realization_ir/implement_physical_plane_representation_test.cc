#include <Cellerator/compiler/ir/realization/implement_physical_plane_representation_v1.hh>

#include <cassert>

using namespace cellerator::compiler::ir::realization::v1;

int main() {
    const stable_identity_v1 structure{1u, 1u};
    physical_plane_set_v1 planes;
    planes.identity = {2u, 1u};
    planes.planes = {
        {{3u, 1u}, {4u, 1u}, structure, 7u, 0u,
            physical_plane_kind_v1::structure, plane_lifetime_v1::structure_epoch,
            memory_device_global_v1, false},
        {{3u, 2u}, {4u, 2u}, structure, 7u, 11u,
            physical_plane_kind_v1::values, plane_lifetime_v1::value_generation,
            memory_device_global_v1, true},
        {{3u, 3u}, {4u, 3u}, structure, 7u, 11u,
            physical_plane_kind_v1::gradients, plane_lifetime_v1::value_generation,
            memory_device_global_v1, true},
        {{3u, 4u}, {4u, 4u}, structure, 7u, 0u,
            physical_plane_kind_v1::workspace, plane_lifetime_v1::invocation,
            memory_device_global_v1, false},
    };
    assert(validate_physical_plane_set_v1(planes) == physical_plane_status_v1::valid);

    physical_plane_set_v1 advanced;
    assert(advance_value_generation_v1(planes, 12u, &advanced) ==
        physical_plane_status_v1::valid);
    assert(advanced.planes[0].structure_identity == planes.planes[0].structure_identity);
    assert(advanced.planes[0].structure_epoch == planes.planes[0].structure_epoch);
    assert(advanced.planes[0].artifact_identity == planes.planes[0].artifact_identity);
    assert(advanced.planes[0].value_generation == 0u);
    assert(advanced.planes[1].value_generation == 12u);
    assert(advanced.planes[2].value_generation == 12u);

    auto invalid = planes;
    invalid.planes[1].structure_epoch = 8u;
    assert(validate_physical_plane_set_v1(invalid) ==
        physical_plane_status_v1::structure_mismatch);
}
