#include <Cellerator/compute/support/logical_support.hh>

#include <limits>

namespace cellerator::compute::support {

logical_support_status_v1 validate_logical_support_v1(
    const logical_support_view_v1 &support) noexcept {
    if (support.schema_version != logical_support_schema_version_v1)
        return logical_support_status_v1::unsupported_schema;
    if (support.reserved != 0u) return logical_support_status_v1::invalid_reserved;
    if (support.relation_id == 0u || support.structure_id == 0u
        || support.structure_epoch == 0u)
        return logical_support_status_v1::invalid_identity;
    if ((support.edge_count != 0u && support.edges == nullptr)
        || (support.participant_count != 0u && support.participants == nullptr))
        return logical_support_status_v1::invalid_argument;

    std::uint64_t previous_id = 0u;
    for (std::uint64_t i = 0u; i < support.edge_count; ++i) {
        const auto &edge = support.edges[i];
        if (edge.reserved != 0u) return logical_support_status_v1::invalid_reserved;
        if (edge.logical_edge_id == 0u
            || (i != 0u && edge.logical_edge_id <= previous_id))
            return logical_support_status_v1::invalid_edge_order;
        if (edge.participant_count == 0u)
            return logical_support_status_v1::empty_participant_list;
        if (edge.participant_begin > support.participant_count
            || edge.participant_count
                > support.participant_count - edge.participant_begin)
            return logical_support_status_v1::invalid_participant_range;
        previous_id = edge.logical_edge_id;
    }
    for (std::uint64_t i = 0u; i < support.participant_count; ++i) {
        if (support.participants[i].reserved != 0u)
            return logical_support_status_v1::invalid_reserved;
    }
    return logical_support_status_v1::success;
}

bool find_logical_edge_ordinal_v1(const logical_support_view_v1 &support,
                                  std::uint64_t logical_edge_id,
                                  std::uint64_t *ordinal) noexcept {
    if (ordinal == nullptr || logical_edge_id == 0u
        || validate_logical_support_v1(support)
            != logical_support_status_v1::success)
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

snapshot_lifecycle_status_v1 snapshot_lifecycle_v1::pin_forward_tape(
    std::uint64_t expected_epoch) noexcept {
    if (expected_epoch == 0u || current_structure_epoch == 0u)
        return snapshot_lifecycle_status_v1::invalid_epoch;
    if (expected_epoch != current_structure_epoch)
        return snapshot_lifecycle_status_v1::stale_epoch;
    if (live_forward_tapes == std::numeric_limits<std::uint64_t>::max())
        return snapshot_lifecycle_status_v1::invalid_epoch;
    ++live_forward_tapes;
    return snapshot_lifecycle_status_v1::success;
}

snapshot_lifecycle_status_v1 snapshot_lifecycle_v1::retire_forward_tape(
    std::uint64_t expected_epoch) noexcept {
    if (expected_epoch == 0u || current_structure_epoch == 0u)
        return snapshot_lifecycle_status_v1::invalid_epoch;
    if (expected_epoch != current_structure_epoch)
        return snapshot_lifecycle_status_v1::stale_epoch;
    if (live_forward_tapes == 0u)
        return snapshot_lifecycle_status_v1::no_live_forward_tapes;
    --live_forward_tapes;
    return snapshot_lifecycle_status_v1::success;
}

snapshot_lifecycle_status_v1 snapshot_lifecycle_v1::publish_structure(
    std::uint64_t next_epoch) noexcept {
    if (next_epoch == 0u || current_structure_epoch == 0u)
        return snapshot_lifecycle_status_v1::invalid_epoch;
    if (live_forward_tapes != 0u)
        return snapshot_lifecycle_status_v1::live_forward_tapes;
    if (next_epoch <= current_structure_epoch)
        return snapshot_lifecycle_status_v1::stale_epoch;
    current_structure_epoch = next_epoch;
    return snapshot_lifecycle_status_v1::success;
}

support_snapshot_status_v1 validate_support_snapshot_pair_v1(
    const support_snapshot_pair_v1 &snapshot) noexcept {
    if (snapshot.base == nullptr || snapshot.snapshot_structure_epoch == 0u)
        return support_snapshot_status_v1::invalid_argument;
    if (validate_logical_support_v1(*snapshot.base)
        != logical_support_status_v1::success)
        return support_snapshot_status_v1::invalid_base;
    if (snapshot.base->structure_epoch != snapshot.snapshot_structure_epoch)
        return support_snapshot_status_v1::epoch_mismatch;
    if (snapshot.delta == nullptr) return support_snapshot_status_v1::success;
    if (validate_logical_support_v1(*snapshot.delta)
        != logical_support_status_v1::success)
        return support_snapshot_status_v1::invalid_delta;
    if (snapshot.delta->structure_epoch != snapshot.snapshot_structure_epoch)
        return support_snapshot_status_v1::epoch_mismatch;
    if (snapshot.delta->structure_id != snapshot.base->structure_id)
        return support_snapshot_status_v1::structure_mismatch;

    std::uint64_t base_index = 0u;
    std::uint64_t delta_index = 0u;
    while (base_index < snapshot.base->edge_count
        && delta_index < snapshot.delta->edge_count) {
        const auto base_id = snapshot.base->edges[base_index].logical_edge_id;
        const auto delta_id = snapshot.delta->edges[delta_index].logical_edge_id;
        if (base_id == delta_id)
            return support_snapshot_status_v1::overlapping_logical_edge;
        if (base_id < delta_id) ++base_index;
        else ++delta_index;
    }
    return support_snapshot_status_v1::success;
}

} // namespace cellerator::compute::support
