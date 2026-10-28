#include <Cellerator/compute/indexing/temporal_occurrences.hh>

#include <limits>

namespace cellerator::compute::indexing {
namespace {

std::uint64_t oldest_index(const temporal_event_ring_v1 &ring) noexcept {
    return ring.size == ring.capacity ? ring.next_write : 0u;
}

const temporal_occurrence_v1 &event_at(
    const temporal_event_ring_v1 &ring, std::uint64_t chronological_index) noexcept {
    return ring.events[(oldest_index(ring) + chronological_index) % ring.capacity];
}

bool occurrence_in_window(const temporal_occurrence_v1 &event,
                          const occurrence_window_v1 &window) noexcept {
    if (window.use_ordinal_window
        && (event.ordinal < window.minimum_ordinal
            || event.ordinal > window.maximum_ordinal))
        return false;
    if (window.use_timestamp_window
        && (event.timestamp < window.minimum_timestamp
            || event.timestamp > window.maximum_timestamp))
        return false;
    return true;
}

bool occurrence_in_ordinal_window(const temporal_occurrence_v1 &event,
                                  const occurrence_window_v1 &window) noexcept {
    return !window.use_ordinal_window
        || (event.ordinal >= window.minimum_ordinal
            && event.ordinal <= window.maximum_ordinal);
}

} // namespace

occurrence_status_v1 initialize_temporal_event_ring_v1(
    temporal_event_ring_v1 *ring, std::uint64_t cell_id,
    std::uint64_t entity_id, temporal_occurrence_v1 *storage,
    std::uint64_t capacity) noexcept {
    if (ring == nullptr || cell_id == 0u || entity_id == 0u
        || (capacity != 0u && storage == nullptr))
        return occurrence_status_v1::invalid_argument;
    *ring = {cell_id, entity_id, storage, capacity, 0u, 0u, 0u, 1u, 0u};
    return occurrence_status_v1::success;
}

occurrence_status_v1 append_temporal_occurrence_v1(
    temporal_event_ring_v1 *ring,
    const temporal_occurrence_v1 &occurrence,
    std::uint64_t expected_generation,
    std::uint64_t next_generation,
    std::uint64_t *evicted_occurrence_id) noexcept {
    if (ring == nullptr || evicted_occurrence_id == nullptr
        || ring->cell_id == 0u || ring->entity_id == 0u
        || (ring->capacity != 0u && ring->events == nullptr)
        || ring->size > ring->capacity || ring->next_write >= (ring->capacity == 0u ? 1u : ring->capacity)
        || occurrence.cell_id != ring->cell_id
        || occurrence.entity_id != ring->entity_id
        || occurrence.occurrence_id == 0u
        || occurrence.reserved != 0u
        || (occurrence.flags & ~occurrence_has_timestamp_v1) != 0u)
        return occurrence_status_v1::invalid_occurrence;
    if (ring->generation != expected_generation)
        return occurrence_status_v1::stale_generation;
    if (ring->generation == std::numeric_limits<std::uint64_t>::max()
        || next_generation != ring->generation + 1u)
        return occurrence_status_v1::generation_overflow;
    if (ring->size != 0u && occurrence.ordinal <= ring->last_ordinal)
        return occurrence_status_v1::stale_ordinal;
    if (ring->capacity == 0u)
        return occurrence_status_v1::ring_capacity_exhausted;

    *evicted_occurrence_id = 0u;
    if (ring->size == ring->capacity) {
        *evicted_occurrence_id = ring->events[ring->next_write].occurrence_id;
        ++ring->evicted_count;
    } else {
        ++ring->size;
    }
    ring->events[ring->next_write] = occurrence;
    ring->next_write = (ring->next_write + 1u) % ring->capacity;
    ring->last_ordinal = occurrence.ordinal;
    ring->generation = next_generation;
    return occurrence_status_v1::success;
}

occurrence_status_v1 query_temporal_occurrences_v1(
    const temporal_event_ring_v1 &ring,
    const occurrence_window_v1 &window,
    temporal_occurrence_v1 *output,
    std::uint64_t output_capacity,
    occurrence_query_result_v1 *result) noexcept {
    if (result == nullptr || ring.cell_id == 0u || ring.entity_id == 0u
        || ring.size > ring.capacity
        || (ring.capacity != 0u && ring.events == nullptr)
        || (ring.capacity == 0u && (ring.events != nullptr || ring.next_write != 0u))
        || (ring.capacity != 0u && ring.next_write >= ring.capacity)
        || (ring.size != 0u && ring.generation == 0u)
        || (output_capacity != 0u && output == nullptr))
        return occurrence_status_v1::invalid_argument;
    *result = {};
    for (const auto byte : window.reserved)
        if (byte != 0u) return occurrence_status_v1::invalid_window;
    if ((window.use_ordinal_window
            && window.minimum_ordinal > window.maximum_ordinal)
        || (window.use_timestamp_window
            && window.minimum_timestamp > window.maximum_timestamp))
        return occurrence_status_v1::invalid_window;

    for (std::uint64_t i = 0u; i < ring.size; ++i) {
        const auto &event = event_at(ring, i);
        if (!occurrence_in_ordinal_window(event, window)) continue;
        if (window.use_timestamp_window
            && (event.flags & occurrence_has_timestamp_v1) == 0u)
            return occurrence_status_v1::missing_timestamp;
        if (occurrence_in_window(event, window)) ++result->required;
    }
    result->history_truncated = ring.evicted_count != 0u;
    if (result->required > output_capacity)
        return occurrence_status_v1::output_overflow;
    for (std::uint64_t i = 0u; i < ring.size; ++i) {
        const auto &event = event_at(ring, i);
        if (occurrence_in_window(event, window))
            output[result->written++] = event;
    }
    return occurrence_status_v1::success;
}

} // namespace cellerator::compute::indexing
