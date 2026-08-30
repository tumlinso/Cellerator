#include "../../../src/compute/architecture/providers/nvidia/sm70/contract_on_support_projection.cc"

#include <cassert>
#include <cstdint>

namespace projection = cellerator::compute::projection;
namespace sm70 = cellerator::compute::architecture::providers::nvidia::sm70;

int main() {
    sm70::support_logical_edge_v1 edges[5]{};
    projection::projection_value_map_v1 map[5]{};
    for (std::uint32_t index = 0u; index < 5u; ++index) {
        edges[index].logical_edge_id.value = 100u + index * 7u;
        edges[index].source_index = index % 3u;
        edges[index].destination_index = index % 4u;
        map[index].logical_edge_id = edges[index].logical_edge_id;
        map[index].region_kind = index % 2u == 0u
            ? projection::physical_region_kind_v1::mma
            : projection::physical_region_kind_v1::residual;
        map[index].region_index = index / 2u;
        map[index].projection_slot = index * 11u;
    }
    const std::uint8_t source_support[] = {1u, 0u, 1u};
    const std::uint8_t destination_support[] = {1u, 1u, 0u, 1u};
    sm70::support_projection_edge_v1 selected[5]{};
    sm70::contract_projection_request_v1 request{};
    request.logical_edges = edges;
    request.physical_value_map = map;
    request.logical_edge_count = 5u;
    request.source_support = source_support;
    request.source_count = 3u;
    request.destination_support = destination_support;
    request.destination_count = 4u;
    request.selected_edges = selected;
    request.selected_capacity = 5u;
    sm70::contract_projection_result_v1 result{};
    assert(sm70::prepare_contract_projection_v1(request, &result)
        == sm70::contract_projection_status_v1::success);
    assert(result.selected_edge_count == 2u);
    assert(result.mma_edge_count == 1u);
    assert(result.residual_edge_count == 1u);
    assert(selected[0].logical_edge_id.value == edges[0].logical_edge_id.value);
    assert(selected[0].stable_output_index == 0u);
    assert(selected[0].region_kind == map[0].region_kind);
    assert(selected[0].projection_slot == map[0].projection_slot);
    assert(selected[1].logical_edge_id.value == edges[3].logical_edge_id.value);
    assert(selected[1].stable_output_index == 3u);
    assert(selected[1].region_kind == map[3].region_kind);

    sm70::contract_projection_request_v1 invalid = request;
    invalid.selected_capacity = 1u;
    selected[0].stable_output_index = 77u;
    assert(sm70::prepare_contract_projection_v1(invalid, &result)
        == sm70::contract_projection_status_v1::insufficient_capacity);
    assert(selected[0].stable_output_index == 77u);
    invalid = request;
    map[2].logical_edge_id.value = 999u;
    assert(sm70::prepare_contract_projection_v1(invalid, &result)
        == sm70::contract_projection_status_v1::invalid_argument);
    map[2].logical_edge_id = edges[2].logical_edge_id;
    edges[4].logical_edge_id = edges[0].logical_edge_id;
    map[4].logical_edge_id = edges[4].logical_edge_id;
    assert(sm70::prepare_contract_projection_v1(invalid, &result)
        == sm70::contract_projection_status_v1::invalid_argument);
    return 0;
}
