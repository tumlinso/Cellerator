#pragma once

#include <Cellerator/structure/indexing/posting_index.hh>
#include <Cellerator/geometry/dependency/compact_support.hh>

namespace cellerator::geometry::dependency {

enum class PostingBuildStatus : std::uint8_t {
    success = 0u,
    invalid_argument,
    invalid_snapshot,
    insufficient_record_workspace,
    insufficient_output_capacity,
};

// Upper bounds from compact support incidences. Duplicate participants may
// make the actual unique posting count smaller than record_capacity.
::cellerator::structure::indexing::PostingRequirements posting_requirements(
    const compact::SnapshotPairV1& snapshot) noexcept;

// Geometry-specific support adapter; generic posting storage and queries live
// in cellerator::structure::indexing.
PostingBuildStatus build_posting_index(
    const compact::SnapshotPairV1& snapshot,
    ::cellerator::structure::indexing::PostingRecord* record_scratch,
    std::uint64_t record_capacity,
    ::cellerator::structure::indexing::PostingRecord* entries,
    std::uint64_t entry_capacity,
    ::cellerator::structure::indexing::PostingIndexView* out) noexcept;

} // namespace cellerator::geometry::dependency
