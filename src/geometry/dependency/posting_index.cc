#include <Cellerator/geometry/dependency/posting_index.hh>

#include <limits>

namespace cellerator::geometry::dependency {
namespace {

bool incidence_count(const compact::ViewV1& support,
                     std::uint64_t* count) noexcept {
    *count = 0u;
    const auto maximum = std::numeric_limits<std::uint64_t>::max();
    for (std::uint64_t i = 0u; i < support.edge_count; ++i) {
        const auto size = support.edges[i].participant_count;
        if (*count > maximum - size) return false;
        *count += size;
    }
    return true;
}

bool snapshot_sizes_fit(const compact::SnapshotPairV1& snapshot,
                        std::uint64_t* record_count,
                        std::uint64_t* edge_count) noexcept {
    std::uint64_t base_records = 0u;
    if (!incidence_count(*snapshot.base, &base_records)) return false;
    *record_count = base_records;
    *edge_count = snapshot.base->edge_count;
    if (snapshot.delta == nullptr) return true;
    const auto maximum = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t delta_records = 0u;
    if (!incidence_count(*snapshot.delta, &delta_records)
        || base_records > maximum - delta_records
        || snapshot.base->edge_count > maximum - snapshot.delta->edge_count)
        return false;
    *record_count += delta_records;
    *edge_count += snapshot.delta->edge_count;
    return true;
}

} // namespace

containers::PostingRequirements posting_requirements(
    const compact::SnapshotPairV1& snapshot) noexcept {
    if (compact::validate_snapshot(snapshot) != compact::SnapshotStatusV1::success)
        return {};
    containers::PostingRequirements requirements{};
    if (!snapshot_sizes_fit(snapshot, &requirements.record_capacity,
                            &requirements.maximum_work_items))
        return {};
    requirements.maximum_unique_entries = requirements.record_capacity;
    return requirements;
}

PostingBuildStatus build_posting_index(
    const compact::SnapshotPairV1& snapshot,
    containers::PostingRecord* record_scratch,
    std::uint64_t record_capacity,
    containers::PostingRecord* entries,
    std::uint64_t entry_capacity,
    containers::PostingIndexView* out) noexcept {
    if (out == nullptr) return PostingBuildStatus::invalid_argument;
    if (compact::validate_snapshot(snapshot) != compact::SnapshotStatusV1::success)
        return PostingBuildStatus::invalid_snapshot;
    std::uint64_t required_records = 0u;
    std::uint64_t edge_count = 0u;
    if (!snapshot_sizes_fit(snapshot, &required_records, &edge_count))
        return PostingBuildStatus::invalid_snapshot;
    if (required_records != 0u && record_scratch == nullptr)
        return PostingBuildStatus::invalid_argument;
    if (record_capacity < required_records)
        return PostingBuildStatus::insufficient_record_workspace;

    std::uint64_t record_count = 0u;
    const auto append_relation = [&](const compact::ViewV1& relation,
                                     std::uint64_t edge_ordinal_begin) {
        for (std::uint64_t edge_ordinal = 0u;
             edge_ordinal < relation.edge_count; ++edge_ordinal) {
            const auto& edge = relation.edges[edge_ordinal];
            for (std::uint32_t slot = 0u; slot < edge.participant_count; ++slot) {
                const auto& participant = relation.participants[
                    edge.participant_begin + slot];
                record_scratch[record_count++] = {
                    participant.role, 0u, participant.entity_id,
                    edge.logical_edge_id, edge_ordinal_begin + edge_ordinal};
            }
        }
    };
    append_relation(*snapshot.base, 0u);
    if (snapshot.delta != nullptr)
        append_relation(*snapshot.delta, snapshot.base->edge_count);

    const containers::PostingIndexView metadata{
        snapshot.base->relation_id, snapshot.base->structure_id,
        snapshot.snapshot_structure_epoch, nullptr, 0u};
    const auto status = containers::canonicalize_postings(
        metadata, record_scratch, record_count, entries, entry_capacity, out);
    switch (status) {
    case containers::PostingStatus::success:
        return PostingBuildStatus::success;
    case containers::PostingStatus::insufficient_output_capacity:
        return PostingBuildStatus::insufficient_output_capacity;
    case containers::PostingStatus::invalid_argument:
        return PostingBuildStatus::invalid_argument;
    default:
        return PostingBuildStatus::invalid_snapshot;
    }
}

} // namespace cellerator::geometry::dependency
