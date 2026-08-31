#include <Cellerator/compute/architecture/providers/nvidia/sm70/transpose/transpose_value_map_v1.hh>

#include <cstdlib>

using namespace cellerator::compute::architecture::providers::nvidia::sm70::transpose;

namespace { void require(bool value) { if (!value) std::abort(); } }

int main() {
    constexpr std::uint64_t high = (std::uint64_t{1u} << 32u) + 5u;
    transpose_edge_placement_v1 original[]{
        {9u, high + 2u, high + 12u, 0u, invalid_local_index_v1, 0u},
        {4u, high + 2u, high + 10u, 0u, invalid_local_index_v1, 1u},
        {7u, high + 3u, high + 11u, 1u, invalid_local_index_v1, 2u}};
    source_owner_schedule_v1 owners[]{
        {high + 2u, 0u, 2u, 0u, 0u}, {high + 3u, 2u, 1u, 1u, 0u}};
    transpose_cover_view_v1 cover{transpose_cover_schema_v1, 0u, 1u, 2u,
        original, 3u, owners, 2u};
    std::uint64_t destinations[]{high + 10u, high + 11u, high + 12u};
    std::uint64_t identity_order[]{1u, 2u, 0u};
    transpose_edge_placement_v1 bound[3]{};
    projection_gradient_position_v1 projection_order[3]{};
    std::uint64_t logical_to_projection[3]{};
    transpose_cover_view_v1 bound_cover{};
    direct_gradient_order_v1 gradient_order{};
    require(bind_transpose_local_maps_v1(
        {cover, {destinations, 3u}, identity_order, 8u, 13u, 21u},
        {bound, 3u, projection_order, 3u, logical_to_projection, 3u},
        &bound_cover, &gradient_order) == transpose_status_v1::success);
    require(bound[0].local_destination_index == 2u
        && bound[1].local_destination_index == 0u
        && logical_to_projection[0] == 1u
        && gradient_order.value_generation == 13u);

    destinations[1] = destinations[0];
    require(bind_transpose_local_maps_v1(
        {cover, {destinations, 3u}, identity_order, 8u, 13u, 21u},
        {bound, 3u, projection_order, 3u, logical_to_projection, 3u},
        &bound_cover, &gradient_order) == transpose_status_v1::invalid_argument);
    return 0;
}
