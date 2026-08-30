#include <Cellerator/compute/projection/physical_mma_hybrid.hh>

#include <cstdint>

namespace cellerator::compute::architecture::providers::nvidia::sm70 {

enum class contract_projection_status_v1 : std::uint8_t {
    success = 0u,
    invalid_argument = 1u,
    insufficient_capacity = 2u
};

struct support_logical_edge_v1 {
    projection::width_tagged_logical_edge_id_v1 logical_edge_id{};
    std::uint32_t source_index = 0u;
    std::uint32_t destination_index = 0u;
};

struct support_projection_edge_v1 {
    projection::width_tagged_logical_edge_id_v1 logical_edge_id{};
    projection::physical_region_kind_v1 region_kind =
        projection::physical_region_kind_v1::residual;
    std::uint8_t reserved0[3]{};
    std::uint32_t region_index = projection::invalid_physical_index_v1;
    std::uint32_t projection_slot = projection::invalid_physical_index_v1;
    std::uint32_t stable_output_index = projection::invalid_physical_index_v1;
};

struct contract_projection_request_v1 {
    const support_logical_edge_v1 *logical_edges = nullptr;
    const projection::projection_value_map_v1 *physical_value_map = nullptr;
    std::uint64_t logical_edge_count = 0u;
    const std::uint8_t *source_support = nullptr;
    std::uint32_t source_count = 0u;
    const std::uint8_t *destination_support = nullptr;
    std::uint32_t destination_count = 0u;
    support_projection_edge_v1 *selected_edges = nullptr;
    std::uint64_t selected_capacity = 0u;
};

struct contract_projection_result_v1 {
    std::uint64_t selected_edge_count = 0u;
    std::uint64_t mma_edge_count = 0u;
    std::uint64_t residual_edge_count = 0u;
};

namespace {

bool valid_map(const support_logical_edge_v1 &edge,
    const projection::projection_value_map_v1 &map) noexcept {
    return edge.logical_edge_id.value != 0u
        && edge.logical_edge_id.value == map.logical_edge_id.value
        && edge.logical_edge_id.width == map.logical_edge_id.width
        && projection::valid_logical_edge_id_width_v1(
            edge.logical_edge_id.width)
        && projection::valid_physical_region_kind_v1(map.region_kind)
        && map.region_index != projection::invalid_physical_index_v1
        && map.projection_slot != projection::invalid_physical_index_v1;
}

} // namespace

contract_projection_status_v1 prepare_contract_projection_v1(
    const contract_projection_request_v1 &request,
    contract_projection_result_v1 *result) noexcept {
    if (result == nullptr || request.logical_edges == nullptr
        || request.physical_value_map == nullptr
        || request.logical_edge_count == 0u || request.source_support == nullptr
        || request.source_count == 0u
        || request.destination_support == nullptr
        || request.destination_count == 0u || request.selected_edges == nullptr)
        return contract_projection_status_v1::invalid_argument;

    std::uint64_t selected = 0u;
    for (std::uint64_t index = 0u; index < request.logical_edge_count;
        ++index) {
        const support_logical_edge_v1 &edge = request.logical_edges[index];
        if (edge.source_index >= request.source_count
            || edge.destination_index >= request.destination_count
            || request.source_support[edge.source_index] > 1u
            || request.destination_support[edge.destination_index] > 1u
            || !valid_map(edge, request.physical_value_map[index]))
            return contract_projection_status_v1::invalid_argument;
        for (std::uint64_t prior = 0u; prior < index; ++prior)
            if (request.logical_edges[prior].logical_edge_id.value
                == edge.logical_edge_id.value)
                return contract_projection_status_v1::invalid_argument;
        if (request.source_support[edge.source_index] != 0u
            && request.destination_support[edge.destination_index] != 0u)
            ++selected;
    }
    if (selected > request.selected_capacity)
        return contract_projection_status_v1::insufficient_capacity;

    contract_projection_result_v1 prepared{};
    for (std::uint64_t index = 0u; index < request.logical_edge_count;
        ++index) {
        const support_logical_edge_v1 &edge = request.logical_edges[index];
        if (request.source_support[edge.source_index] == 0u
            || request.destination_support[edge.destination_index] == 0u)
            continue;
        const projection::projection_value_map_v1 &map =
            request.physical_value_map[index];
        support_projection_edge_v1 &output =
            request.selected_edges[prepared.selected_edge_count];
        output.logical_edge_id = edge.logical_edge_id;
        output.region_kind = map.region_kind;
        output.region_index = map.region_index;
        output.projection_slot = map.projection_slot;
        output.stable_output_index = static_cast<std::uint32_t>(index);
        ++prepared.selected_edge_count;
        prepared.mma_edge_count +=
            map.region_kind == projection::physical_region_kind_v1::mma;
        prepared.residual_edge_count +=
            map.region_kind == projection::physical_region_kind_v1::residual;
    }
    *result = prepared;
    return contract_projection_status_v1::success;
}

} // namespace cellerator::compute::architecture::providers::nvidia::sm70
