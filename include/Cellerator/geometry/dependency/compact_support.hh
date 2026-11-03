#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellerator::geometry::dependency::compact {

inline constexpr std::uint32_t schema_version_v1 = 1u;

// These records encode ordered logical incidences. Repeated participants are
// meaningful and retained; physical positions and mutable values are separate.
struct ParticipantV1 {
    std::uint32_t role = 0u;
    std::uint32_t reserved = 0u;
    std::uint64_t entity_id = 0u;
};

struct EdgeV1 {
    std::uint64_t logical_edge_id = 0u;
    std::uint64_t participant_begin = 0u;
    std::uint32_t participant_count = 0u;
    std::uint32_t reserved = 0u;
};

// Borrowed compact view. Edges are strictly ordered by stable ID; incidence
// slots are ordered and may repeat the same role/entity pair.
struct ViewV1 {
    std::uint32_t schema_version = schema_version_v1;
    std::uint32_t reserved = 0u;
    std::uint64_t relation_id = 0u;
    std::uint64_t structure_id = 0u;
    std::uint64_t structure_epoch = 0u;
    const EdgeV1* edges = nullptr;
    std::uint64_t edge_count = 0u;
    const ParticipantV1* participants = nullptr;
    std::uint64_t participant_count = 0u;
};

enum class ValidationStatusV1 : std::uint8_t {
    success = 0u,
    invalid_argument,
    unsupported_schema,
    invalid_identity,
    invalid_edge_order,
    invalid_participant_range,
    empty_participant_list,
    invalid_reserved,
};

ValidationStatusV1 validate(const ViewV1& support) noexcept;

// Binary search a validated view's stable logical edge ID.
bool find_edge_ordinal(const ViewV1& support, std::uint64_t logical_edge_id,
                       std::uint64_t* ordinal) noexcept;

struct SnapshotPairV1 {
    const ViewV1* base = nullptr;
    const ViewV1* delta = nullptr;
    std::uint64_t snapshot_structure_epoch = 0u;
};

enum class SnapshotStatusV1 : std::uint8_t {
    success = 0u,
    invalid_argument,
    invalid_base,
    invalid_delta,
    epoch_mismatch,
    structure_mismatch,
    overlapping_logical_edge,
};

// Base and delta share one structure epoch and cannot own the same stable ID.
SnapshotStatusV1 validate_snapshot(const SnapshotPairV1& snapshot) noexcept;

static_assert(std::is_trivially_copyable<ViewV1>::value,
              "compact support views are borrowed metadata");
static_assert(std::is_trivially_copyable<SnapshotPairV1>::value,
              "compact support snapshots are borrowed metadata");

} // namespace cellerator::geometry::dependency::compact
