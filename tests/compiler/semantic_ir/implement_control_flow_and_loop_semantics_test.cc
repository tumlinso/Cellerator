#include <Cellerator/compiler/ir/semantic/implement_control_flow_and_loop_semantics_v1.hh>

#include <cassert>
#include <iostream>

using namespace Cellerator::compiler::ir::semantic;

int main() {
    const std::vector<control_region_ir_v1> regions{
        {1, control_region_kind_ir_v1::sequence, {}, 0, control_effect_reads_v1, true},
        {2, control_region_kind_ir_v1::sequence, {}, 0, control_effect_writes_v1, true},
        {3, control_region_kind_ir_v1::branch, {1, 2}, 0, control_effect_none_v1, true},
        {4, control_region_kind_ir_v1::loop, {3}, 16, control_effect_writes_v1, true},
        {5, control_region_kind_ir_v1::opaque_cxx_control, {}, 0,
         control_effect_opaque_barrier_v1, false},
    };
    assert(validate_control_regions_ir_v1(regions) == control_flow_status_ir_v1::success);

    control_dataflow_state_ir_v1 left;
    left.profiles = {{{10, 11}, 0.7}};
    left.values = {{{20, 21}, 4, control_effect_reads_v1}};
    left.effects = control_effect_reads_v1;
    control_dataflow_state_ir_v1 right;
    right.profiles = {{{12, 13}, 0.3}};
    right.values = {{{20, 21}, 5, control_effect_writes_v1}};
    right.effects = control_effect_writes_v1;
    control_dataflow_state_ir_v1 joined;
    assert(join_control_dataflow_ir_v1(left, right, 2, &joined) ==
           control_flow_status_ir_v1::success);
    assert(joined.profiles.size() == 2);
    assert(joined.values.front().generation == 0);
    assert((joined.values.front().effects & control_effect_reads_v1) != 0);
    assert((joined.values.front().effects & control_effect_writes_v1) != 0);
    assert(join_control_dataflow_ir_v1(left, right, 1, &joined) ==
           control_flow_status_ir_v1::profile_alternative_limit);

    auto invalid_loop = regions;
    invalid_loop[3].bounded_trip_count = 0;
    assert(validate_control_regions_ir_v1(invalid_loop) ==
           control_flow_status_ir_v1::invalid_structure);

    std::cout << "regions=5 profile_join=2 divergent_generation=unknown\n";
}
