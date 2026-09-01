#include <Cellerator/compute/projection_family/segment_and_gate_v1.hh>

#include <array>
#include <cassert>
#include <cstdint>

namespace family = cellerator::compute::projection_family;
namespace execution = cellerator::execution;

namespace {

execution::persistent_axis_identity axis(std::uint64_t base) {
    return {{execution::biological_abi_version,
             execution::serialized_record_kind::persistent_axis_identity,
             sizeof(execution::persistent_axis_identity)},
            {base + 1, base + 2}, {base + 3, base + 4},
            {base + 5, base + 6}, {base + 7, base + 8}};
}

family::support_family_descriptor_v1 descriptor() {
    family::support_family_descriptor_v1 value{};
    value.identity.family_identity = {1, 2};
    value.identity.exact_support_identity = {3, 4};
    value.identity.structure_identity = {5, 6};
    value.identity.structure_epoch = {7};
    value.identity.source_axis = axis(10);
    value.identity.destination_axis = axis(30);
    value.identity.logical_edge_order = {50, 51};
    value.identity.logical_edge_count = 5;
    value.supported_operations = family::support_segment_reduce_v1
        | family::support_segment_normalize_v1
        | family::support_edge_map_or_gate_v1;
    return value;
}

} // namespace

int main() {
    const auto support = descriptor();
    const std::array<family::logical_edge_segment_v1, 5> assignments{{
        {0, 1}, {0, 4}, {2, 0}, {2, 2}, {3, 3}}};
    std::array<std::uint64_t, 5> offsets{};
    std::array<std::uint64_t, 5> edge_ids{};
    std::array<std::uint8_t, 5> marks{};
    family::segment_physical_view_v1 segment{};
    assert(family::build_segment_physical_view_v1(
               support, family::support_segment_normalize_v1,
               {100, 101}, {102, 103}, 4, assignments.data(),
               assignments.size(),
               {offsets.data(), offsets.size(), edge_ids.data(),
                edge_ids.size(), marks.data(), marks.size()},
               &segment)
               .built());
    assert((offsets == std::array<std::uint64_t, 5>{0, 2, 2, 4, 5}));
    assert((edge_ids == std::array<std::uint64_t, 5>{1, 4, 0, 2, 3}));
    assert(segment.operation == family::support_segment_normalize_v1);

    auto unordered = assignments;
    unordered[1].logical_edge_id = 0;
    assert(family::build_segment_physical_view_v1(
               support, family::support_segment_reduce_v1,
               {100, 101}, {102, 103}, 4, unordered.data(), unordered.size(),
               {offsets.data(), offsets.size(), edge_ids.data(),
                edge_ids.size(), marks.data(), marks.size()},
               &segment)
               .code == family::segment_physical_code_v1::unordered_assignment);

    const std::array<std::uint64_t, 5> physical_edges{{4, 2, 0, 3, 1}};
    const std::array<std::uint64_t, 5> gates{{2, 1, 2, 0, 1}};
    family::gate_physical_view_v1 gate{};
    assert(family::build_gate_physical_view_v1(
               support, {200, 201}, {202, 203}, 3,
               physical_edges.data(), gates.data(), physical_edges.size(),
               marks.data(), marks.size(), &gate)
               .built());
    assert(gate.logical_edge_ids == physical_edges.data());
    assert(gate.gate_indices == gates.data());

    auto bad_gates = gates;
    bad_gates[3] = 3;
    assert(family::build_gate_physical_view_v1(
               support, {200, 201}, {202, 203}, 3,
               physical_edges.data(), bad_gates.data(), physical_edges.size(),
               marks.data(), marks.size(), &gate)
               .code == family::gate_physical_code_v1::gate_index_out_of_range);

    auto duplicate_edges = physical_edges;
    duplicate_edges[4] = duplicate_edges[0];
    assert(family::build_gate_physical_view_v1(
               support, {200, 201}, {202, 203}, 3,
               duplicate_edges.data(), gates.data(), duplicate_edges.size(),
               marks.data(), marks.size(), &gate)
               .code == family::gate_physical_code_v1::duplicate_logical_edge);
}
