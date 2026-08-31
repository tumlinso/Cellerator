#pragma once

#include <Cellerator/geometry/optimizer/overlap/overlap_contract.hh>

#include <cstdint>

namespace cellerator::geometry::optimizer::overlap {

struct logical_value_location {
    logical_contribution_id logical = 0;
    std::uint64_t physical_index = 0;
};

struct logical_value_map_view {
    const logical_value_location *locations = nullptr;
    std::uint64_t location_count = 0;
    std::uint64_t logical_count = 0;
    std::uint64_t physical_capacity = 0;
};

struct source_replica_location {
    source_id logical_source = 0;
    source_group_id group = 0;
    std::uint64_t physical_index = 0;
    bool canonical_owner = false;
};

struct source_replica_map_view {
    const source_replica_location *locations = nullptr;
    std::uint64_t location_count = 0;
    std::uint64_t logical_source_count = 0;
    std::uint64_t group_count = 0;
    std::uint64_t physical_capacity = 0;
};

struct logical_mapping_workspace_view {
    std::uint8_t *physical_seen = nullptr;
    std::uint64_t physical_capacity = 0;
    std::uint8_t *logical_seen = nullptr;
    std::uint64_t logical_capacity = 0;
};

contract_status validate_logical_value_map(
    logical_value_map_view map,
    logical_mapping_workspace_view workspace) noexcept;

contract_status validate_source_replica_map(
    source_replica_map_view map,
    logical_mapping_workspace_view workspace) noexcept;

template <class T>
contract_status pack_logical_values(
    logical_value_map_view map,
    const T *logical_values,
    std::uint64_t logical_count,
    T *physical_values,
    std::uint64_t physical_count) noexcept {
    if ((logical_count != 0 && logical_values == nullptr)
        || (physical_count != 0 && physical_values == nullptr)) {
        return {contract_error::null_pointer, 0};
    }
    if (logical_count < map.logical_count || physical_count < map.physical_capacity) {
        return {contract_error::insufficient_workspace, map.location_count};
    }
    for (std::uint64_t index = 0; index < map.location_count; ++index) {
        const logical_value_location location = map.locations[index];
        physical_values[location.physical_index] = logical_values[location.logical];
    }
    return {};
}

template <class T>
contract_status gather_logical_gradients(
    logical_value_map_view map,
    const T *physical_gradients,
    std::uint64_t physical_count,
    T *logical_gradients,
    std::uint64_t logical_count) noexcept {
    if ((physical_count != 0 && physical_gradients == nullptr)
        || (logical_count != 0 && logical_gradients == nullptr)) {
        return {contract_error::null_pointer, 0};
    }
    if (logical_count < map.logical_count || physical_count < map.physical_capacity) {
        return {contract_error::insufficient_workspace, map.location_count};
    }
    for (std::uint64_t index = 0; index < map.location_count; ++index) {
        const logical_value_location location = map.locations[index];
        logical_gradients[location.logical] = physical_gradients[location.physical_index];
    }
    return {};
}

template <class T>
contract_status reconcile_source_gradients(
    source_replica_map_view map,
    const T *physical_gradients,
    std::uint64_t physical_count,
    T *logical_gradients,
    std::uint64_t logical_count) noexcept {
    if ((physical_count != 0 && physical_gradients == nullptr)
        || (logical_count != 0 && logical_gradients == nullptr)) {
        return {contract_error::null_pointer, 0};
    }
    if (logical_count < map.logical_source_count || physical_count < map.physical_capacity) {
        return {contract_error::insufficient_workspace, map.location_count};
    }
    for (std::uint64_t source = 0; source < map.logical_source_count; ++source) {
        logical_gradients[source] = T{};
    }
    for (std::uint64_t index = 0; index < map.location_count; ++index) {
        const source_replica_location location = map.locations[index];
        logical_gradients[location.logical_source]
            += physical_gradients[location.physical_index];
    }
    return {};
}

}  // namespace cellerator::geometry::optimizer::overlap
