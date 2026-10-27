#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellerator::compute::support {

inline constexpr std::uint32_t logical_support_schema_version_v1 = 1u;

// Participant roles are relation-local labels.  In particular, source and
// destination are not inferred from array position; higher-order consumers may
// declare additional roles while preserving the ordered participant slots.
struct logical_support_participant_v1 {
    std::uint32_t role = 0u;
    std::uint32_t reserved = 0u;
    std::uint64_t entity_id = 0u;
};

struct logical_support_edge_v1 {
    std::uint64_t logical_edge_id = 0u;
    std::uint64_t participant_begin = 0u;
    std::uint32_t participant_count = 0u;
    std::uint32_t reserved = 0u;
};

// Canonical logical support. Edges are strictly ordered by logical_edge_id;
// participants are ordered incidences and may repeat. Physical projection
// positions, gates, and mutable values live in separate views.
struct logical_support_view_v1 {
    std::uint32_t schema_version = logical_support_schema_version_v1;
    std::uint32_t reserved = 0u;
    std::uint64_t relation_id = 0u;
    std::uint64_t structure_id = 0u;
    std::uint64_t structure_epoch = 0u;
    const logical_support_edge_v1 *edges = nullptr;
    std::uint64_t edge_count = 0u;
    const logical_support_participant_v1 *participants = nullptr;
    std::uint64_t participant_count = 0u;
};

enum class logical_support_status_v1 : std::uint8_t {
    success = 0u,
    invalid_argument,
    unsupported_schema,
    invalid_identity,
    invalid_edge_order,
    invalid_participant_range,
    empty_participant_list,
    invalid_reserved,
};

logical_support_status_v1 validate_logical_support_v1(
    const logical_support_view_v1 &support) noexcept;

// Binary search maps a stable logical edge ID to its canonical edge ordinal.
// The input view must pass validate_logical_support_v1.
bool find_logical_edge_ordinal_v1(const logical_support_view_v1 &support,
                                  std::uint64_t logical_edge_id,
                                  std::uint64_t *ordinal) noexcept;

enum class snapshot_lifecycle_status_v1 : std::uint8_t {
    success = 0u,
    invalid_epoch,
    stale_epoch,
    live_forward_tapes,
    no_live_forward_tapes,
};

// Host-serialized publication gate for immutable support/projection snapshots.
// A training tape pins one structure epoch. A replacement epoch can publish only
// after every tape using the current epoch has retired.
struct snapshot_lifecycle_v1 {
    std::uint64_t current_structure_epoch = 0u;
    std::uint64_t live_forward_tapes = 0u;

    snapshot_lifecycle_status_v1 pin_forward_tape(
        std::uint64_t expected_epoch) noexcept;
    snapshot_lifecycle_status_v1 retire_forward_tape(
        std::uint64_t expected_epoch) noexcept;
    snapshot_lifecycle_status_v1 publish_structure(
        std::uint64_t next_epoch) noexcept;
};

struct support_snapshot_pair_v1 {
    const logical_support_view_v1 *base = nullptr;
    const logical_support_view_v1 *delta = nullptr;
    std::uint64_t snapshot_structure_epoch = 0u;
};

// Mutable execution generations are separate from immutable topology and
// invalidate only the cache domains that consume each generation.
struct support_runtime_generations_v1 {
    std::uint64_t structure_epoch = 0u;
    std::uint64_t state_generation = 0u;
    std::uint64_t parameter_generation = 0u;
};

enum class support_snapshot_status_v1 : std::uint8_t {
    success = 0u,
    invalid_argument,
    invalid_base,
    invalid_delta,
    epoch_mismatch,
    structure_mismatch,
    overlapping_logical_edge,
};

// The base and adaptive delta may have different relation IDs, but share one
// published structure epoch and cannot own the same logical edge ID.
support_snapshot_status_v1 validate_support_snapshot_pair_v1(
    const support_snapshot_pair_v1 &snapshot) noexcept;

static_assert(std::is_trivially_copyable<logical_support_view_v1>::value,
              "support views are borrowed metadata");
static_assert(std::is_trivially_copyable<support_snapshot_pair_v1>::value,
              "support snapshot pairs are borrowed metadata");
static_assert(std::is_trivially_copyable<support_runtime_generations_v1>::value,
              "runtime generation stamps are compact values");

} // namespace cellerator::compute::support
