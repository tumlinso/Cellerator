#include <Cellerator/geometry/relation_cover.hh>

#include <cassert>

namespace {

namespace geo = cellerator::geometry;
namespace ex = cellerator::execution;

constexpr ex::axis_identity make_axis(std::uint32_t seed) noexcept {
    return {
        {seed + 1u, 1u},
        {seed + 2u, 1u},
        {seed + 3u, 1u},
        {seed + 4u, 1u}
    };
}

geo::relation_cover_view_v1 make_cover(
    const geo::semantic_component_v1 *components,
    const std::uint64_t *logical_edge_ids) noexcept {
    geo::relation_cover_view_v1 cover{};
    cover.structure = {11u, 1u};
    cover.structure_epoch = {7u};
    cover.source_axis = make_axis(10u);
    cover.destination_axis = make_axis(20u);
    cover.logical_edge_count = 6u;
    cover.component_count = 3u;
    cover.components = components;
    cover.logical_edge_ids = logical_edge_ids;
    return cover;
}

void test_exact_disjoint_semantic_cover() {
    const geo::semantic_component_v1 components[] = {
        {1u, geo::semantic_component_kind::rectangular, {}, 0u, 3u},
        {2u, geo::semantic_component_kind::hierarchical, {}, 3u, 1u},
        {3u, geo::semantic_component_kind::unstructured, {}, 4u, 2u}
    };
    const std::uint64_t edge_ids[] = {4u, 0u, 3u, 5u, 2u, 1u};
    std::uint8_t marks[6]{};
    const geo::relation_cover_view_v1 cover = make_cover(components, edge_ids);
    assert(geo::validate_relation_cover(cover, {marks, 6u}));
    for (std::uint8_t mark : marks)
        assert(mark == 1u);
}

void test_duplicate_and_out_of_range_edges_are_rejected() {
    const geo::semantic_component_v1 components[] = {
        {1u, geo::semantic_component_kind::rectangular, {}, 0u, 3u},
        {2u, geo::semantic_component_kind::hierarchical, {}, 3u, 1u},
        {3u, geo::semantic_component_kind::unstructured, {}, 4u, 2u}
    };
    std::uint64_t edge_ids[] = {4u, 0u, 3u, 5u, 2u, 2u};
    std::uint8_t marks[6]{};
    geo::relation_cover_view_v1 cover = make_cover(components, edge_ids);
    auto result = geo::validate_relation_cover(cover, {marks, 6u});
    assert(result.code
        == geo::relation_cover_validation_code::duplicate_logical_edge);
    assert(result.component_index == 2u);
    assert(result.logical_edge_id == 2u);

    edge_ids[5] = 6u;
    result = geo::validate_relation_cover(cover, {marks, 6u});
    assert(result.code
        == geo::relation_cover_validation_code::logical_edge_out_of_bounds);
    assert(result.logical_edge_id == 6u);
}

void test_component_partition_is_exact() {
    geo::semantic_component_v1 components[] = {
        {1u, geo::semantic_component_kind::rectangular, {}, 0u, 3u},
        {2u, geo::semantic_component_kind::hierarchical, {}, 3u, 1u},
        {3u, geo::semantic_component_kind::unstructured, {}, 4u, 2u}
    };
    const std::uint64_t edge_ids[] = {4u, 0u, 3u, 5u, 2u, 1u};
    std::uint8_t marks[6]{};
    geo::relation_cover_view_v1 cover = make_cover(components, edge_ids);

    components[1].logical_edge_offset = 2u;
    assert(geo::validate_relation_cover(cover, {marks, 6u}).code
        == geo::relation_cover_validation_code::component_offset_mismatch);

    components[1].logical_edge_offset = 3u;
    components[2].logical_edge_count = 1u;
    assert(geo::validate_relation_cover(cover, {marks, 6u}).code
        == geo::relation_cover_validation_code::incomplete_component_partition);

    components[2].logical_edge_count = 2u;
    components[2].component_id = 2u;
    assert(geo::validate_relation_cover(cover, {marks, 6u}).code
        == geo::relation_cover_validation_code::duplicate_component_id);
}

void test_identity_and_workspace_are_mandatory() {
    const geo::semantic_component_v1 components[] = {
        {1u, geo::semantic_component_kind::rectangular, {}, 0u, 3u},
        {2u, geo::semantic_component_kind::hierarchical, {}, 3u, 1u},
        {3u, geo::semantic_component_kind::unstructured, {}, 4u, 2u}
    };
    const std::uint64_t edge_ids[] = {4u, 0u, 3u, 5u, 2u, 1u};
    std::uint8_t marks[6]{};
    geo::relation_cover_view_v1 cover = make_cover(components, edge_ids);

    assert(geo::validate_relation_cover(cover, {nullptr, 6u}).code
        == geo::relation_cover_validation_code::missing_workspace);
    assert(geo::validate_relation_cover(cover, {marks, 5u}).code
        == geo::relation_cover_validation_code::insufficient_workspace);

    cover.structure = {};
    assert(geo::validate_relation_cover(cover, {marks, 6u}).code
        == geo::relation_cover_validation_code::invalid_structure);

    cover = make_cover(components, edge_ids);
    cover.destination_axis.order = {};
    assert(geo::validate_relation_cover(cover, {marks, 6u}).code
        == geo::relation_cover_validation_code::invalid_destination_axis);
}

void test_empty_relation_has_empty_cover() {
    geo::relation_cover_view_v1 cover{};
    cover.structure = {11u, 1u};
    cover.structure_epoch = {7u};
    cover.source_axis = make_axis(10u);
    cover.destination_axis = make_axis(20u);
    assert(geo::validate_relation_cover(cover, {}));
}

} // namespace

int main() {
    test_exact_disjoint_semantic_cover();
    test_duplicate_and_out_of_range_edges_are_rejected();
    test_component_partition_is_exact();
    test_identity_and_workspace_are_mandatory();
    test_empty_relation_has_empty_cover();
    return 0;
}
