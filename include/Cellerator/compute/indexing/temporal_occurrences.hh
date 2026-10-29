#pragma once

#include <cstdint>
#include <type_traits>

namespace cellerator::compute::indexing {

inline constexpr std::uint32_t occurrence_has_timestamp_v1 = 1u;

struct temporal_occurrence_v1 {
    std::uint64_t cell_id = 0u;
    std::uint64_t entity_id = 0u;
    std::uint64_t occurrence_id = 0u;
    std::uint64_t ordinal = 0u;
    std::int64_t timestamp = 0;
    std::uint32_t flags = 0u;
    std::uint32_t reserved = 0u;
};

struct temporal_event_ring_v1 {
    std::uint64_t cell_id = 0u;
    std::uint64_t entity_id = 0u;
    temporal_occurrence_v1 *events = nullptr;
    std::uint64_t capacity = 0u;
    std::uint64_t size = 0u;
    std::uint64_t next_write = 0u;
    std::uint64_t evicted_count = 0u;
    std::uint64_t generation = 0u;
    std::uint64_t last_ordinal = 0u;
};

enum class occurrence_status_v1 : std::uint8_t {
    success = 0u,
    invalid_argument,
    invalid_occurrence,
    stale_generation,
    stale_ordinal,
    generation_overflow,
    ring_capacity_exhausted,
    invalid_window,
    missing_timestamp,
    output_overflow,
};

occurrence_status_v1 initialize_temporal_event_ring_v1(
    temporal_event_ring_v1 *ring, std::uint64_t cell_id,
    std::uint64_t entity_id, temporal_occurrence_v1 *storage,
    std::uint64_t capacity) noexcept;

struct occurrence_window_v1 {
    bool use_ordinal_window = false;
    bool use_timestamp_window = false;
    std::uint8_t reserved[6]{};
    std::uint64_t minimum_ordinal = 0u;
    std::uint64_t maximum_ordinal = UINT64_MAX;
    std::int64_t minimum_timestamp = INT64_MIN;
    std::int64_t maximum_timestamp = INT64_MAX;
};

struct occurrence_query_result_v1 {
    std::uint64_t written = 0u;
    std::uint64_t required = 0u;
    bool history_truncated = false;
    std::uint8_t reserved[7]{};
};

// One fixed-capacity ring is owned by a caller for one (cell, entity) stream.
// Appending at capacity evicts the oldest occurrence and reports its ID.
occurrence_status_v1 append_temporal_occurrence_v1(
    temporal_event_ring_v1 *ring,
    const temporal_occurrence_v1 &occurrence,
    std::uint64_t expected_generation,
    std::uint64_t next_generation,
    std::uint64_t *evicted_occurrence_id) noexcept;

// Queries only this entity's retained occurrences and applies ordinal/time
// filters before returning candidates for compatibility scoring. Output is in
// event order and is not partially written on capacity or timestamp errors.
occurrence_status_v1 query_temporal_occurrences_v1(
    const temporal_event_ring_v1 &ring,
    const occurrence_window_v1 &window,
    temporal_occurrence_v1 *output,
    std::uint64_t output_capacity,
    occurrence_query_result_v1 *result) noexcept;

static_assert(std::is_trivially_copyable<temporal_occurrence_v1>::value,
              "event occurrences are compact borrowed records");
static_assert(std::is_trivially_copyable<temporal_event_ring_v1>::value,
              "event rings are caller-owned views");

} // namespace cellerator::compute::indexing
