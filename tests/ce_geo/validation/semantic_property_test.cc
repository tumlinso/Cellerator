#include <Cellerator/geometry/relation_cover.hh>
#include <Cellerator/geometry/work_layout.hh>
#include <Cellerator/geometry/work_window.hh>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

namespace execution = cellerator::execution;
namespace geometry = cellerator::geometry;

namespace {

execution::axis_identity axis(std::uint32_t seed) {
    return {{seed + 1u, 1u}, {seed + 2u, 1u},
        {seed + 3u, 1u}, {seed + 4u, 1u}};
}

geometry::work_window_view_v1 window(
    const std::vector<std::uint32_t> &members,
    std::uint32_t extent,
    execution::axis_identity bound_axis,
    std::uint64_t identity) {
    geometry::work_window_view_v1 result{};
    result.identity = {identity, identity + 1u};
    result.axis = bound_axis;
    result.axis_extent = extent;
    result.member_count = static_cast<std::uint32_t>(members.size());
    result.members = members.data();
    return result;
}

geometry::relation_cover_view_v1 cover(
    const std::vector<geometry::semantic_component_v1> &components,
    const std::vector<std::uint64_t> &edge_ids,
    execution::axis_identity source,
    execution::axis_identity destination) {
    geometry::relation_cover_view_v1 result{};
    result.structure = {71u, 1u};
    result.structure_epoch = {9u};
    result.source_axis = source;
    result.destination_axis = destination;
    result.logical_edge_count = edge_ids.size();
    result.component_count = static_cast<std::uint32_t>(components.size());
    result.components = components.data();
    result.logical_edge_ids = edge_ids.data();
    return result;
}

void verify_window_and_layout_properties(std::mt19937 &random,
    std::uint32_t iteration) {
    std::uniform_int_distribution<std::uint32_t> extent_distribution(1u, 96u);
    const std::uint32_t extent = extent_distribution(random);
    std::uniform_int_distribution<std::uint32_t> count_distribution(1u, extent);
    const std::uint32_t count = count_distribution(random);

    std::vector<std::uint32_t> axis_positions(extent);
    std::iota(axis_positions.begin(), axis_positions.end(), 0u);
    std::shuffle(axis_positions.begin(), axis_positions.end(), random);
    axis_positions.resize(count);
    const execution::axis_identity bound_axis = axis(iteration * 8u + 1u);
    const geometry::work_window_view_v1 valid_window = window(
        axis_positions, extent, bound_axis, 1000u + iteration * 2u);
    assert(geometry::validate_work_window(valid_window));

    std::vector<std::uint8_t> selected(extent, 0u);
    for (std::uint32_t member : axis_positions) {
        assert(member < extent);
        assert(selected[member] == 0u);
        selected[member] = 1u;
    }
    for (std::uint32_t position = 0u; position < extent; ++position)
        assert(geometry::work_window_contains(valid_window, position)
            == (selected[position] != 0u));
    assert(!geometry::work_window_contains(valid_window, extent));

    std::vector<std::uint32_t> duplicate_members = axis_positions;
    if (duplicate_members.size() == 1u)
        duplicate_members.push_back(duplicate_members[0]);
    else
        duplicate_members.back() = duplicate_members.front();
    const geometry::work_window_view_v1 duplicate_window = window(
        duplicate_members, std::max(extent, 2u), bound_axis,
        2000u + iteration * 2u);
    assert(geometry::validate_work_window(duplicate_window).code
        == geometry::work_window_validation_code::duplicate_member);

    std::vector<std::uint32_t> out_of_bounds_members = axis_positions;
    out_of_bounds_members.back() = extent;
    const geometry::work_window_view_v1 out_of_bounds_window = window(
        out_of_bounds_members, extent, bound_axis, 3000u + iteration * 2u);
    assert(geometry::validate_work_window(out_of_bounds_window).code
        == geometry::work_window_validation_code::member_out_of_bounds);

    geometry::work_window_view_v1 invalid_axis_window = valid_window;
    invalid_axis_window.axis.order = {};
    assert(geometry::validate_work_window(invalid_axis_window).code
        == geometry::work_window_validation_code::invalid_axis);

    std::vector<std::uint32_t> execution_to_window(count);
    std::iota(execution_to_window.begin(), execution_to_window.end(), 0u);
    std::shuffle(execution_to_window.begin(), execution_to_window.end(), random);
    std::vector<std::uint32_t> window_to_execution(
        count, geometry::invalid_work_item);
    geometry::work_layout_view_v1 layout{};
    assert(geometry::build_work_layout(valid_window,
        execution_to_window.data(), count, window_to_execution.data(), count,
        &layout));
    assert(geometry::validate_work_layout(valid_window, layout));

    std::vector<std::uint8_t> seen_window_indices(count, 0u);
    for (std::uint32_t execution_position = 0u;
         execution_position < count; ++execution_position) {
        const std::uint32_t window_index =
            execution_to_window[execution_position];
        assert(window_index < count);
        assert(seen_window_indices[window_index] == 0u);
        seen_window_indices[window_index] = 1u;
        assert(window_to_execution[window_index] == execution_position);
        assert(execution_to_window[window_to_execution[window_index]]
            == window_index);
        assert(geometry::work_layout_axis_position(valid_window, layout,
                   execution_position)
            == axis_positions[window_index]);
    }
    assert(std::all_of(seen_window_indices.begin(), seen_window_indices.end(),
        [](std::uint8_t value) { return value == 1u; }));
    assert(geometry::work_layout_axis_position(valid_window, layout, count)
        == geometry::invalid_work_item);

    std::vector<std::uint32_t> duplicate_permutation = execution_to_window;
    if (count == 1u)
        duplicate_permutation[0] = 1u;
    else
        duplicate_permutation.back() = duplicate_permutation.front();
    std::vector<std::uint32_t> rejected_inverse(count);
    geometry::work_layout_view_v1 rejected_layout{};
    const geometry::work_layout_build_result rejected_build =
        geometry::build_work_layout(valid_window, duplicate_permutation.data(),
            count, rejected_inverse.data(), count, &rejected_layout);
    assert(rejected_build.code == (count == 1u
            ? geometry::work_layout_build_code::work_item_out_of_bounds
            : geometry::work_layout_build_code::duplicate_work_item));

    geometry::work_layout_view_v1 wrong_axis_layout = layout;
    wrong_axis_layout.axis.geometry.slot += 1u;
    assert(geometry::validate_work_layout(valid_window, wrong_axis_layout).code
        == geometry::work_layout_validation_code::axis_mismatch);
    std::vector<std::uint32_t> corrupt_inverse = window_to_execution;
    corrupt_inverse[execution_to_window[0]] =
        count == 1u ? count : (window_to_execution[execution_to_window[0]] + 1u)
            % count;
    geometry::work_layout_view_v1 corrupt_layout = layout;
    corrupt_layout.window_to_execution = corrupt_inverse.data();
    const auto corrupt_result =
        geometry::validate_work_layout(valid_window, corrupt_layout);
    assert(corrupt_result.code
        == geometry::work_layout_validation_code::inverse_mismatch);
}

void verify_semantic_cover_properties(std::mt19937 &random,
    std::uint32_t iteration) {
    std::uniform_int_distribution<std::uint32_t> edge_distribution(1u, 128u);
    const std::uint32_t edge_count = edge_distribution(random);
    const std::uint32_t maximum_components = std::min(edge_count, 12u);
    std::uniform_int_distribution<std::uint32_t> component_distribution(
        1u, maximum_components);
    const std::uint32_t component_count = component_distribution(random);

    std::vector<std::uint32_t> cuts;
    for (std::uint32_t value = 1u; value < edge_count; ++value)
        cuts.push_back(value);
    std::shuffle(cuts.begin(), cuts.end(), random);
    cuts.resize(component_count - 1u);
    std::sort(cuts.begin(), cuts.end());

    std::vector<geometry::semantic_component_v1> components(component_count);
    std::uint64_t begin = 0u;
    for (std::uint32_t index = 0u; index < component_count; ++index) {
        const std::uint64_t end = index + 1u == component_count
            ? edge_count
            : cuts[index];
        components[index].component_id = 100u + index;
        components[index].kind = static_cast<geometry::semantic_component_kind>(
            1u + (random() % 3u));
        components[index].logical_edge_offset = begin;
        components[index].logical_edge_count = end - begin;
        begin = end;
    }

    std::vector<std::uint64_t> edge_ids(edge_count);
    std::iota(edge_ids.begin(), edge_ids.end(), 0u);
    std::shuffle(edge_ids.begin(), edge_ids.end(), random);
    const execution::axis_identity source = axis(iteration * 8u + 10001u);
    const execution::axis_identity destination =
        axis(iteration * 8u + 10005u);
    const geometry::relation_cover_view_v1 valid_cover =
        cover(components, edge_ids, source, destination);
    std::vector<std::uint8_t> validator_marks(edge_count, 0xffu);
    assert(geometry::validate_relation_cover(valid_cover,
        {validator_marks.data(), validator_marks.size()}));

    std::vector<std::uint32_t> independent_owner(edge_count, 0u);
    std::uint64_t expected_offset = 0u;
    for (std::uint32_t component_index = 0u;
         component_index < component_count; ++component_index) {
        const auto &component = components[component_index];
        assert(component.logical_edge_offset == expected_offset);
        assert(component.logical_edge_count != 0u);
        const std::uint64_t end = expected_offset + component.logical_edge_count;
        assert(end <= edge_count);
        for (std::uint64_t position = expected_offset; position < end;
             ++position) {
            const std::uint64_t edge = edge_ids[position];
            assert(edge < edge_count);
            assert(independent_owner[edge] == 0u);
            independent_owner[edge] = component.component_id;
        }
        expected_offset = end;
    }
    assert(expected_offset == edge_count);
    assert(std::all_of(independent_owner.begin(), independent_owner.end(),
        [](std::uint32_t owner) { return owner != 0u; }));

    std::vector<std::uint64_t> duplicate_edges = edge_ids;
    if (edge_count == 1u)
        duplicate_edges[0] = edge_count;
    else
        duplicate_edges.back() = duplicate_edges.front();
    const auto duplicate_cover =
        cover(components, duplicate_edges, source, destination);
    std::fill(validator_marks.begin(), validator_marks.end(), 0xffu);
    const auto duplicate_result = geometry::validate_relation_cover(
        duplicate_cover, {validator_marks.data(), validator_marks.size()});
    assert(duplicate_result.code == (edge_count == 1u
            ? geometry::relation_cover_validation_code::logical_edge_out_of_bounds
            : geometry::relation_cover_validation_code::duplicate_logical_edge));

    std::vector<geometry::semantic_component_v1> invalid_components = components;
    if (component_count == 1u)
        invalid_components[0].component_id = 0u;
    else
        invalid_components.back().component_id =
            invalid_components.front().component_id;
    const auto invalid_component_cover =
        cover(invalid_components, edge_ids, source, destination);
    const auto component_result = geometry::validate_relation_cover(
        invalid_component_cover, {validator_marks.data(), validator_marks.size()});
    assert(component_result.code == (component_count == 1u
            ? geometry::relation_cover_validation_code::invalid_component_id
            : geometry::relation_cover_validation_code::duplicate_component_id));

    geometry::relation_cover_view_v1 wrong_axis_cover = valid_cover;
    wrong_axis_cover.destination_axis.partition = {};
    assert(geometry::validate_relation_cover(wrong_axis_cover,
               {validator_marks.data(), validator_marks.size()}).code
        == geometry::relation_cover_validation_code::invalid_destination_axis);

    std::vector<geometry::semantic_component_v1> gapped_components = components;
    gapped_components[0].logical_edge_offset = 1u;
    const auto gapped_cover =
        cover(gapped_components, edge_ids, source, destination);
    assert(geometry::validate_relation_cover(gapped_cover,
               {validator_marks.data(), validator_marks.size()}).code
        == geometry::relation_cover_validation_code::component_offset_mismatch);
}

} // namespace

int main() {
    constexpr std::uint32_t seed = 0x5e4a102u;
    constexpr std::uint32_t trials = 512u;
    std::mt19937 random(seed);
    for (std::uint32_t iteration = 0u; iteration < trials; ++iteration) {
        verify_window_and_layout_properties(random, iteration);
        verify_semantic_cover_properties(random, iteration);
    }
    std::cout << "semantic_property_test passed seed=" << seed
              << " trials=" << trials
              << " invertible_layouts=" << trials
              << " exact_semantic_covers=" << trials << '\n';
    return 0;
}
