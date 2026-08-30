#include "../../../src/compute/architecture/providers/nvidia/sm70/transpose_cover.cc"

#include <cassert>
#include <cstdint>

namespace projection = cellerator::compute::projection;
namespace sm70 = cellerator::compute::architecture::providers::nvidia::sm70;

int main() {
    const sm70::logical_relation_edge_v1 edges[] = {
        {101u, 2u, 3u},
        {205u, 18u, 7u},
        {309u, 15u, 33u},
        {411u, 19u, 34u}};
    sm70::target_edge_placement_v1 forward[4]{};
    sm70::target_edge_placement_v1 transpose[4]{};
    sm70::transpose_cover_request_v1 request{};
    request.logical_edges = edges;
    request.logical_edge_count = 4u;
    request.source_count = 20u;
    request.destination_count = 35u;
    request.forward = forward;
    request.forward_capacity = 4u;
    request.transpose = transpose;
    request.transpose_capacity = 4u;
    assert(sm70::build_transpose_cover_v1(request)
        == sm70::transpose_cover_status_v1::success);

    for (std::uint32_t index = 0u; index < 4u; ++index) {
        assert(forward[index].logical_edge_id.value
            == edges[index].logical_edge_id);
        assert(transpose[index].logical_edge_id.value
            == edges[index].logical_edge_id);
        assert(forward[index].logical_edge_id.width
            == projection::logical_edge_id_width_v1::u32);
        assert(transpose[index].logical_edge_id.width
            == projection::logical_edge_id_width_v1::u32);
        assert(forward[index].source_group
            == transpose[index].destination_group);
        assert(forward[index].destination_group
            == transpose[index].source_group);
        assert(forward[index].row == transpose[index].column);
        assert(forward[index].column == transpose[index].row);
        assert(forward[index].region_kind == transpose[index].region_kind);
    }
    assert(forward[0].region_kind == projection::physical_region_kind_v1::mma);
    assert(forward[1].region_kind
        == projection::physical_region_kind_v1::residual);
    assert(forward[2].region_kind
        == projection::physical_region_kind_v1::residual);
    assert(forward[0].source_group == 0u);
    assert(forward[0].destination_group == 0u);
    assert(forward[0].row == 3u && forward[0].column == 2u);
    assert(transpose[0].row == 2u && transpose[0].column == 3u);
    assert(forward[3].source_group == 1u);
    assert(forward[3].destination_group == 2u);

    sm70::transpose_cover_request_v1 invalid = request;
    invalid.forward_capacity = 3u;
    assert(sm70::build_transpose_cover_v1(invalid)
        == sm70::transpose_cover_status_v1::insufficient_capacity);
    invalid = request;
    invalid.transpose = invalid.forward;
    assert(sm70::build_transpose_cover_v1(invalid)
        == sm70::transpose_cover_status_v1::invalid_argument);
    invalid = request;
    invalid.source_count = 19u;
    assert(sm70::build_transpose_cover_v1(invalid)
        == sm70::transpose_cover_status_v1::invalid_argument);

    sm70::logical_relation_edge_v1 duplicates[] = {edges[0], edges[1]};
    duplicates[1].logical_edge_id = duplicates[0].logical_edge_id;
    invalid = request;
    invalid.logical_edges = duplicates;
    invalid.logical_edge_count = 2u;
    assert(sm70::build_transpose_cover_v1(invalid)
        == sm70::transpose_cover_status_v1::duplicate_logical_edge);

    sm70::logical_relation_edge_v1 wide_edge[] = {
        {0x100000001ull, 2u, 3u}};
    invalid = request;
    invalid.logical_edges = wide_edge;
    invalid.logical_edge_count = 1u;
    assert(sm70::build_transpose_cover_v1(invalid)
        == sm70::transpose_cover_status_v1::invalid_argument);
    invalid.logical_edge_id_width = projection::logical_edge_id_width_v1::u64;
    assert(sm70::build_transpose_cover_v1(invalid)
        == sm70::transpose_cover_status_v1::success);
    assert(forward[0].logical_edge_id.value == 0x100000001ull);
    assert(transpose[0].logical_edge_id.value == 0x100000001ull);
    return 0;
}
