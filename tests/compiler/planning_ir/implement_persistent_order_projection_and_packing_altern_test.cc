#include <Cellerator/compiler/ir/planning/implement_persistent_order_projection_and_packing_altern_v1.hh>

#include <cassert>

int main() {
    using namespace cellerator::compiler::ir::planning::v1;
    persistent_projection_alternative_v1 value{{1u, 1u}, {2u, 1u}, {2u, 2u},
        {2u, 3u}, {2u, 4u}, {3u, 1u}, {3u, 2u}, {3u, 3u}, {3u, 4u},
        7u, 9u, 7u, 0u, persistent_projection_source_v1::csg1, {}};
    assert(validate_persistent_projection_alternative_v1(value) ==
           persistent_projection_status_v1::ok);
    value.source = persistent_projection_source_v1::cpe2;
    value.packing_value_generation = 9u;
    assert(validate_persistent_projection_alternative_v1(value) ==
           persistent_projection_status_v1::ok);
    value.packing_structure_epoch = 8u;
    assert(validate_persistent_projection_alternative_v1(value) ==
           persistent_projection_status_v1::stale_structure);
    value.packing_structure_epoch = 7u;
    value.packing_value_generation = 10u;
    assert(validate_persistent_projection_alternative_v1(value) ==
           persistent_projection_status_v1::stale_values);
}
