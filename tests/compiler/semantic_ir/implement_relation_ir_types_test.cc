#include <Cellerator/compiler/ir/semantic/implement_relation_ir_types_v1.hh>

#include <cassert>
#include <iostream>

using namespace Cellerator::compiler::ir::semantic;

namespace {

axis_ir_type_v1 axis(semantic_identity_v1 identity, semantic_identity_v1 domain,
                     semantic_identity_v1 order, semantic_identity_v1 geometry,
                     semantic_identity_v1 partition, const char* tag) {
    axis_ir_type_v1 result;
    result.identity = identity;
    result.domain = {domain, tag};
    result.order = {order, domain, false};
    result.geometry = {geometry, domain};
    result.partition = {partition, domain, {90, 91}};
    result.extent = {extent_knowledge_kind_v1::exact, 16, 16};
    result.recovery = {axis_identity_space_v1::global,
                       identity_recovery_kind_v1::identity, 16, 0, {}};
    return result;
}

cellerator::execution::persistent_axis_identity abi_axis(
    semantic_identity_v1 domain, semantic_identity_v1 order,
    semantic_identity_v1 geometry, semantic_identity_v1 partition) {
    using namespace cellerator::execution;
    return {{biological_abi_version, serialized_record_kind::persistent_axis_identity,
             sizeof(persistent_axis_identity)},
            {domain.low, domain.high}, {order.low, order.high},
            {geometry.low, geometry.high}, {partition.low, partition.high}};
}

}  // namespace

int main() {
    const auto source = axis({1, 2}, {10, 11}, {12, 13}, {14, 15}, {16, 17}, "gene");
    const auto destination = axis({3, 4}, {20, 21}, {22, 23}, {24, 25}, {26, 27}, "cell");
    cellerator::compute::operation::v2::typed_relation fixture;
    fixture.structure = {30, 31};
    fixture.epoch = {7};
    fixture.source_axis = abi_axis({10, 11}, {12, 13}, {14, 15}, {16, 17});
    fixture.destination_axis = abi_axis({20, 21}, {22, 23}, {24, 25}, {26, 27});
    fixture.logical_edge_order = {32, 33};
    fixture.logical_edge_count = 4096;

    relation_ir_binding_v1 binding;
    binding.logical_edge_identity = {34, 35};
    binding.support_identity = {36, 37};
    binding.value_plane_identity = {38, 39};
    binding.value_generation = 8;
    binding.active_support_generation = 9;
    binding.orientation = relation_orientation_ir_v1::transpose;

    const auto semantic = relation_ir_from_typed_relation_v1(
        fixture, source, destination, binding);
    assert(semantic);
    const auto round_trip = typed_relation_from_relation_ir_v1(*semantic);
    assert(round_trip);
    assert(round_trip->structure.low == fixture.structure.low);
    assert(round_trip->structure.high == fixture.structure.high);
    assert(round_trip->epoch.value == fixture.epoch.value);
    assert(round_trip->logical_edge_order.low == fixture.logical_edge_order.low);
    assert(round_trip->logical_edge_order.high == fixture.logical_edge_order.high);
    assert(round_trip->logical_edge_count == fixture.logical_edge_count);
    assert(round_trip->source_axis.domain.low == fixture.source_axis.domain.low);
    assert(round_trip->destination_axis.partition.high == fixture.destination_axis.partition.high);

    auto mismatched_source = source;
    mismatched_source.order.identity = {99, 13};
    assert(!relation_ir_from_typed_relation_v1(
        fixture, mismatched_source, destination, binding));

    std::cout << "typed_relation_edges=4096 round_trip=lossless orientation=transpose\n";
}
