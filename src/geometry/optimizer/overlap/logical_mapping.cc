#include <Cellerator/geometry/optimizer/overlap/logical_mapping.hh>

namespace cellerator::geometry::optimizer::overlap {
namespace {

contract_status require_workspace(
    logical_mapping_workspace_view workspace,
    std::uint64_t physical_count,
    std::uint64_t logical_count) noexcept {
    if (workspace.physical_capacity < physical_count
        || workspace.logical_capacity < logical_count
        || (physical_count != 0 && workspace.physical_seen == nullptr)
        || (logical_count != 0 && workspace.logical_seen == nullptr)) {
        return {contract_error::insufficient_workspace, physical_count};
    }
    for (std::uint64_t index = 0; index < physical_count; ++index) {
        workspace.physical_seen[index] = 0;
    }
    for (std::uint64_t index = 0; index < logical_count; ++index) {
        workspace.logical_seen[index] = 0;
    }
    return {};
}

}  // namespace

contract_status validate_logical_value_map(
    logical_value_map_view map,
    logical_mapping_workspace_view workspace) noexcept {
    if (map.location_count != 0 && map.locations == nullptr) {
        return {contract_error::null_pointer, 0};
    }
    if (map.location_count != map.logical_count) {
        return {contract_error::missing_contribution_owner, map.location_count};
    }
    const contract_status workspace_status =
        require_workspace(workspace, map.physical_capacity, map.logical_count);
    if (!workspace_status) {
        return workspace_status;
    }
    for (std::uint64_t index = 0; index < map.location_count; ++index) {
        const logical_value_location location = map.locations[index];
        if (location.logical >= map.logical_count) {
            return {contract_error::contribution_out_of_range, index};
        }
        if (location.physical_index >= map.physical_capacity) {
            return {contract_error::physical_index_out_of_range, index};
        }
        if (workspace.logical_seen[location.logical] != 0) {
            return {contract_error::duplicate_contribution_owner, index};
        }
        if (workspace.physical_seen[location.physical_index] != 0) {
            return {contract_error::duplicate_physical_index, index};
        }
        workspace.logical_seen[location.logical] = 1;
        workspace.physical_seen[location.physical_index] = 1;
    }
    for (std::uint64_t logical = 0; logical < map.logical_count; ++logical) {
        if (workspace.logical_seen[logical] == 0) {
            return {contract_error::missing_contribution_owner, logical};
        }
    }
    return {};
}

contract_status validate_source_replica_map(
    source_replica_map_view map,
    logical_mapping_workspace_view workspace) noexcept {
    if (map.location_count != 0 && map.locations == nullptr) {
        return {contract_error::null_pointer, 0};
    }
    const contract_status workspace_status =
        require_workspace(workspace, map.physical_capacity, map.logical_source_count);
    if (!workspace_status) {
        return workspace_status;
    }
    for (std::uint64_t index = 0; index < map.location_count; ++index) {
        const source_replica_location location = map.locations[index];
        if (location.logical_source >= map.logical_source_count) {
            return {contract_error::source_out_of_range, index};
        }
        if (location.group >= map.group_count) {
            return {contract_error::owner_group_out_of_range, index};
        }
        if (location.physical_index >= map.physical_capacity) {
            return {contract_error::physical_index_out_of_range, index};
        }
        if (workspace.physical_seen[location.physical_index] != 0) {
            return {contract_error::duplicate_physical_index, index};
        }
        workspace.physical_seen[location.physical_index] = 1;
        if (location.canonical_owner) {
            if (workspace.logical_seen[location.logical_source] != 0) {
                return {contract_error::duplicate_source_owner, index};
            }
            workspace.logical_seen[location.logical_source] = 1;
        }
    }
    for (std::uint64_t source = 0; source < map.logical_source_count; ++source) {
        if (workspace.logical_seen[source] == 0) {
            return {contract_error::missing_source_owner, source};
        }
    }
    return {};
}

}  // namespace cellerator::geometry::optimizer::overlap
