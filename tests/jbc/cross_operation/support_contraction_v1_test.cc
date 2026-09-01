#include <Cellerator/compute/projection_family/support_contraction_v1.hh>

#include <array>
#include <cassert>
#include <cstdint>

namespace family = cellerator::compute::projection_family;
namespace execution = cellerator::execution;
namespace geometry = cellerator::geometry;

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
    value.identity.logical_edge_count = 6;
    value.supported_operations = family::support_contract_on_support_v1;
    return value;
}

} // namespace

int main() {
    const auto support = descriptor();
    const std::array<geometry::semantic_component_v1, 3> components{{
        {3, geometry::semantic_component_kind::hierarchical, {}, 0, 2},
        {1, geometry::semantic_component_kind::rectangular, {}, 2, 3},
        {2, geometry::semantic_component_kind::unstructured, {}, 5, 1}}};
    const std::array<std::uint64_t, 6> logical_ids{{
        4, 0, 5, 2, 1, 3}};
    geometry::relation_cover_view_v1 cover{};
    cover.logical_edge_count = logical_ids.size();
    cover.component_count = components.size();
    cover.components = components.data();
    cover.logical_edge_ids = logical_ids.data();

    std::array<std::uint8_t, 6> edge_marks{};
    std::array<std::uint8_t, 3> component_marks{};
    family::support_contraction_view_v1 view{};
    const auto built = family::build_support_contraction_view_v1(
        support, {100, 101}, {102, 103}, cover,
        {edge_marks.data(), edge_marks.size(), component_marks.data(),
         component_marks.size()},
        &view);
    assert(built.built());
    assert(view.components == components.data());
    assert(view.logical_edge_ids == logical_ids.data());
    assert(view.component_count == 3);
    assert(view.logical_edge_count == 6);

    auto duplicate_components = components;
    duplicate_components[2].component_id = 1;
    auto malformed = cover;
    malformed.components = duplicate_components.data();
    assert(family::build_support_contraction_view_v1(
               support, {100, 101}, {102, 103}, malformed,
               {edge_marks.data(), edge_marks.size(), component_marks.data(),
                component_marks.size()},
               &view)
               .code
           == family::support_contraction_code_v1::duplicate_component_id);

    auto duplicate_edges = logical_ids;
    duplicate_edges[5] = duplicate_edges[0];
    malformed = cover;
    malformed.logical_edge_ids = duplicate_edges.data();
    assert(family::build_support_contraction_view_v1(
               support, {100, 101}, {102, 103}, malformed,
               {edge_marks.data(), edge_marks.size(), component_marks.data(),
                component_marks.size()},
               &view)
               .code
           == family::support_contraction_code_v1::duplicate_logical_edge);

    auto gap_components = components;
    gap_components[1].logical_edge_offset = 3;
    malformed = cover;
    malformed.components = gap_components.data();
    assert(family::build_support_contraction_view_v1(
               support, {100, 101}, {102, 103}, malformed,
               {edge_marks.data(), edge_marks.size(), component_marks.data(),
                component_marks.size()},
               &view)
               .code
           == family::support_contraction_code_v1::
                  component_offset_mismatch);

    auto stale = support;
    stale.identity.logical_edge_count = (std::uint64_t{1} << 32u) + 6u;
    assert(family::build_support_contraction_view_v1(
               stale, {100, 101}, {102, 103}, cover,
               {edge_marks.data(), edge_marks.size(), component_marks.data(),
                component_marks.size()},
               &view)
               .code
           == family::support_contraction_code_v1::edge_count_mismatch);
}
