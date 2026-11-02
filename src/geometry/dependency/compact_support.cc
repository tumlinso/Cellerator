#include <Cellerator/geometry/dependency/compact_support.hh>

namespace cellerator::geometry::dependency::compact {

ValidationStatusV1 validate(const ViewV1& support) noexcept {
    if (support.schema_version != schema_version_v1)
        return ValidationStatusV1::unsupported_schema;
    if (support.reserved != 0u) return ValidationStatusV1::invalid_reserved;
    if (support.relation_id == 0u || support.structure_id == 0u
        || support.structure_epoch == 0u)
        return ValidationStatusV1::invalid_identity;
    if ((support.edge_count != 0u && support.edges == nullptr)
        || (support.participant_count != 0u && support.participants == nullptr))
        return ValidationStatusV1::invalid_argument;

    std::uint64_t previous_id = 0u;
    for (std::uint64_t i = 0u; i < support.edge_count; ++i) {
        const auto& edge = support.edges[i];
        if (edge.reserved != 0u) return ValidationStatusV1::invalid_reserved;
        if (edge.logical_edge_id == 0u
            || (i != 0u && edge.logical_edge_id <= previous_id))
            return ValidationStatusV1::invalid_edge_order;
        if (edge.participant_count == 0u)
            return ValidationStatusV1::empty_participant_list;
        if (edge.participant_begin > support.participant_count
            || edge.participant_count
                > support.participant_count - edge.participant_begin)
            return ValidationStatusV1::invalid_participant_range;
        previous_id = edge.logical_edge_id;
    }
    for (std::uint64_t i = 0u; i < support.participant_count; ++i)
        if (support.participants[i].reserved != 0u)
            return ValidationStatusV1::invalid_reserved;
    return ValidationStatusV1::success;
}

bool find_edge_ordinal(const ViewV1& support, std::uint64_t logical_edge_id,
                       std::uint64_t* ordinal) noexcept {
    if (ordinal == nullptr || logical_edge_id == 0u
        || validate(support) != ValidationStatusV1::success)
        return false;
    std::uint64_t first = 0u;
    std::uint64_t last = support.edge_count;
    while (first < last) {
        const std::uint64_t middle = first + (last - first) / 2u;
        if (support.edges[middle].logical_edge_id < logical_edge_id)
            first = middle + 1u;
        else
            last = middle;
    }
    if (first == support.edge_count
        || support.edges[first].logical_edge_id != logical_edge_id)
        return false;
    *ordinal = first;
    return true;
}

SnapshotStatusV1 validate_snapshot(const SnapshotPairV1& snapshot) noexcept {
    if (snapshot.base == nullptr || snapshot.snapshot_structure_epoch == 0u)
        return SnapshotStatusV1::invalid_argument;
    if (validate(*snapshot.base) != ValidationStatusV1::success)
        return SnapshotStatusV1::invalid_base;
    if (snapshot.base->structure_epoch != snapshot.snapshot_structure_epoch)
        return SnapshotStatusV1::epoch_mismatch;
    if (snapshot.delta == nullptr) return SnapshotStatusV1::success;
    if (validate(*snapshot.delta) != ValidationStatusV1::success)
        return SnapshotStatusV1::invalid_delta;
    if (snapshot.delta->structure_epoch != snapshot.snapshot_structure_epoch)
        return SnapshotStatusV1::epoch_mismatch;
    if (snapshot.delta->structure_id != snapshot.base->structure_id)
        return SnapshotStatusV1::structure_mismatch;

    std::uint64_t base_index = 0u;
    std::uint64_t delta_index = 0u;
    while (base_index < snapshot.base->edge_count
        && delta_index < snapshot.delta->edge_count) {
        const auto base_id = snapshot.base->edges[base_index].logical_edge_id;
        const auto delta_id = snapshot.delta->edges[delta_index].logical_edge_id;
        if (base_id == delta_id)
            return SnapshotStatusV1::overlapping_logical_edge;
        if (base_id < delta_id) ++base_index;
        else ++delta_index;
    }
    return SnapshotStatusV1::success;
}

} // namespace cellerator::geometry::dependency::compact
