#include <Cellerator/compiler/ir/planning/implement_atom_requirement_and_affordance_nodes_v1.hh>

#include <cassert>

int main() {
    namespace planning = cellerator::compiler::ir::planning::v1;
    namespace joint = cellerator::execution::joint_compiler;
    namespace execution = cellerator::execution;
    const joint::persistent_identity_v1 species[] = {{1u, 1u}};
    const joint::persistent_identity_v1 planes[] = {{2u, 1u}};
    const joint::persistent_identity_v1 transforms[] = {{3u, 1u}};
    joint::atom_requirement_v1 requirement{};
    requirement.requirement_identity = {4u, 1u};
    requirement.exact_coverage_identity = {4u, 2u};
    requirement.accepted_atom_species = species;
    requirement.accepted_atom_species_count = 1u;
    requirement.required_planes = planes;
    requirement.required_plane_count = 1u;
    requirement.numeric = {execution::numeric_type::f16, execution::numeric_type::f32,
                           execution::numeric_type::f32, 0u};
    requirement.required_order = {5u, 1u};
    requirement.transform_paths = transforms;
    requirement.transform_path_count = 1u;

    planning::atom_requirement_node_v1 requirement_node{};
    assert(planning::import_atom_requirement_v1(requirement, {10u, 11u}, {12u, 13u},
                                                 &requirement_node) ==
           planning::atom_contract_import_status_v1::ok);
    assert(requirement_node.requirement.required_planes == requirement.required_planes);
    assert(requirement_node.requirement.numeric.storage == requirement.numeric.storage);
    assert(requirement_node.requirement.required_order.low == requirement.required_order.low);
    assert(requirement_node.requirement.required_order.high == requirement.required_order.high);
    assert(requirement_node.requirement.transform_paths == requirement.transform_paths);

    const joint::atom_plane_affordance_v1 available_planes[] = {
        {{2u, 1u}, {5u, 1u}, execution::numeric_type::f16,
         execution::numeric_type::f32, joint::mutability_requirement_v1::immutable, 0u, {1u}}};
    joint::atom_affordance_v1 affordance{};
    affordance.affordance_identity = {20u, 1u};
    affordance.atom_species = {20u, 2u};
    affordance.exact_coverage_identity = {4u, 2u};
    affordance.physical_encoding = {20u, 3u};
    affordance.local_projection_abi = {20u, 4u};
    affordance.planes = available_planes;
    affordance.plane_count = 1u;
    affordance.flags = joint::graph_stable_address_available_v1;
    affordance.fused_transforms = transforms;
    affordance.fused_transform_count = 1u;

    planning::atom_affordance_node_v1 affordance_node{};
    assert(planning::import_atom_affordance_v1(affordance, {30u, 31u}, {12u, 13u},
                                               &affordance_node) ==
           planning::atom_contract_import_status_v1::ok);
    assert(affordance_node.affordance.planes == affordance.planes);
    assert(affordance_node.affordance.exact_coverage_identity.local_identity ==
           affordance.exact_coverage_identity.local_identity);
    assert(affordance_node.affordance.local_projection_abi.local_identity ==
           affordance.local_projection_abi.local_identity);
    assert(affordance_node.affordance.fused_transforms == affordance.fused_transforms);

    requirement.minimum_alignment = 3u;
    assert(planning::import_atom_requirement_v1(requirement, {10u, 11u}, {12u, 13u},
                                                 &requirement_node) ==
           planning::atom_contract_import_status_v1::invalid_requirement);
}
