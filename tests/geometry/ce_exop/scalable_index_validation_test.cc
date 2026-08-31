#include <Cellerator/geometry/validation/scalable_validation_v2.hh>

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <vector>

namespace geometry = cellerator::geometry;
namespace execution = cellerator::execution;

namespace {

void require(bool value) {
    if (!value) {
        std::abort();
    }
}

geometry::support_component_view_v2 component(
    std::uint64_t id, std::uint64_t begin,
    const std::uint32_t *offsets, std::uint64_t destinations,
    const std::uint32_t *sources, std::uint64_t edges,
    const std::uint64_t *aggregate, const std::uint64_t *global) {
    geometry::support_component_view_v2 result{};
    result.component_identity = id;
    result.source_space.global_extent = (std::uint64_t{1} << 32u) + 4096u;
    result.source_space.local_extent = 4u;
    result.source_space.local_width = execution::local_index_width_v1::u32;
    result.destination_space.global_extent = (std::uint64_t{1} << 32u) + 8192u;
    result.destination_space.local_extent = destinations;
    result.destination_space.local_width = execution::local_index_width_v1::u32;
    result.destination_offsets = offsets;
    result.source_indices = sources;
    result.destination_count = destinations;
    result.local_edge_count = edges;
    result.edge_map = {id, begin, edges, aggregate, global};
    return result;
}

void global_identity_and_two_cover_regression() {
    const std::uint32_t offsets0[] = {0u, 2u, 3u};
    const std::uint32_t sources0[] = {0u, 3u, 1u};
    const std::uint64_t aggregate0[] = {0u, 1u, 2u};
    const std::uint64_t global0[] = {
        (std::uint64_t{1} << 32u) + 11u,
        (std::uint64_t{1} << 32u) + 12u,
        (std::uint64_t{1} << 32u) + 13u};
    const std::uint32_t offsets1[] = {0u, 1u, 2u};
    const std::uint32_t sources1[] = {2u, 0u};
    const std::uint64_t aggregate1[] = {3u, 4u};
    const std::uint64_t global1[] = {
        (std::uint64_t{1} << 40u) + 21u,
        (std::uint64_t{1} << 40u) + 22u};
    const geometry::support_component_view_v2 components[] = {
        component(100u, 0u, offsets0, 2u, sources0, 3u, aggregate0, global0),
        component(200u, 3u, offsets1, 2u, sources1, 2u, aggregate1, global1)};
    const geometry::scalable_support_view_v2 support{77u, 5u, components, 2u};
    const geometry::scale_validation_result_v2 support_result =
        geometry::validate_scalable_support_v2(support);
    require(support_result.code == geometry::scale_validation_code_v2::valid);
    require(support_result.operations <= 2u + 5u + 6u);
    require(components[0].source_width == execution::local_index_width_v1::u32);
    require(components[1].edge_map.global_identities[0] > (std::uint64_t{1} << 32u));

    const std::uint16_t semantic0[] = {0u, 1u, 2u};
    const std::uint16_t semantic1[] = {0u, 1u};
    const geometry::cover_work_item_view_v2 semantic_items[] = {
        {1u, 100u, {semantic0, 3u, execution::local_index_width_v1::u16, {}}},
        {2u, 200u, {semantic1, 2u, execution::local_index_width_v1::u16, {}}}};
    const std::uint16_t physical0_a[] = {2u};
    const std::uint16_t physical0_b[] = {1u, 0u};
    const std::uint16_t physical1[] = {1u, 0u};
    const geometry::cover_work_item_view_v2 physical_items[] = {
        {3u, 100u, {physical0_a, 1u, execution::local_index_width_v1::u16, {}}},
        {4u, 100u, {physical0_b, 2u, execution::local_index_width_v1::u16, {}}},
        {5u, 200u, {physical1, 2u, execution::local_index_width_v1::u16, {}}}};
    std::uint64_t marks[3]{};
    geometry::cover_validation_workspace_v2 workspace{marks, 3u, 1u};
    const geometry::scalable_cover_view_v2 semantic{
        10u, 77u, 5u, semantic_items, 2u, geometry::cover_domain_v2::semantic, {}};
    const geometry::scalable_cover_view_v2 physical{
        11u, 77u, 5u, physical_items, 3u, geometry::cover_domain_v2::physical, {}};
    require(geometry::validate_exact_cover_v2(semantic, support, workspace).code
            == geometry::scale_validation_code_v2::valid);
    // The workspace generation is caller-owned; advance past both semantic
    // components before reusing marks for the independent physical cover.
    workspace.generation = 3u;
    const geometry::scale_validation_result_v2 physical_result =
        geometry::validate_exact_cover_v2(physical, support, workspace);
    require(physical_result.code == geometry::scale_validation_code_v2::valid);
}

void linear_scan_and_adversarial_bounds() {
    constexpr std::uint64_t count = 100000u;
    std::vector<std::uint64_t> counts(count, 3u);
    std::vector<std::uint64_t> offsets(count + 1u);
    std::uint64_t operations = 0u;
    require(geometry::exclusive_scan_counts_v2(
                counts.data(), count, offsets.data(), offsets.size(), &operations)
            == geometry::scale_validation_code_v2::valid);
    require(operations == count);
    require(offsets.back() == count * 3u);
    require(geometry::exclusive_scan_counts_v2(
                counts.data(), count, offsets.data(), count, nullptr)
            == geometry::scale_validation_code_v2::workspace_too_small);
    counts.back() = std::numeric_limits<std::uint64_t>::max();
    require(geometry::exclusive_scan_counts_v2(
                counts.data(), count, offsets.data(), offsets.size(), nullptr)
            == geometry::scale_validation_code_v2::arithmetic_overflow);
    require(!geometry::local_width_can_represent_v2(
        execution::local_index_width_v1::u32, (std::uint64_t{1} << 32u) + 1u));
}

}  // namespace

int main() {
    global_identity_and_two_cover_regression();
    linear_scan_and_adversarial_bounds();
    return 0;
}
