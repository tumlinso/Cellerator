#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(__CUDACC__)
#define CELLERATOR_SEQUENCE_EVENT_HD __host__ __device__
#else
#define CELLERATOR_SEQUENCE_EVENT_HD
#endif

namespace cellerator::ops::sequence {

// The enclosing source binding carries identity, chunk origin, and halo ownership.
struct sequence_event {
    uint32_t local_position;
    uint16_t predicate_id;
    uint8_t payload, flags;
};

inline constexpr uint8_t sequence_event_forward_strand = 0x1u;
inline constexpr uint8_t sequence_event_reverse_strand = 0x2u;

enum class sequence_output_mode : uint8_t {
    count = 1u, mask = 2u, stable_emit = 3u, unordered_emit = 4u,
};

enum class sequence_event_ordering : uint8_t {
    not_applicable = 0u, coordinate_stable = 1u, unordered = 2u,
};

enum class sequence_buffer_residency : uint8_t {
    host = 1u, device = 2u, managed = 3u, peer_device = 4u,
};

// Keep the atomic counters in the native type of CUDA's 64-bit atomicAdd.
struct sequence_emit_counts {
    unsigned long long total_matches, stored_records, dropped_records, required_capacity;
};

// All pointers refer to one declared residency. Counters are caller-owned in
// every mode. Emission permits a null record pointer at zero capacity.
struct sequence_output_descriptor {
    sequence_event *records;
    uint32_t *mask_words;
    sequence_emit_counts *counts;
    uint64_t capacity, mask_word_count;
    int32_t device_ordinal;
    sequence_output_mode mode;
    sequence_event_ordering ordering;
    sequence_buffer_residency residency;
    uint8_t reserved;
};

CELLERATOR_SEQUENCE_EVENT_HD constexpr bool sequence_buffer_location_valid(
    sequence_buffer_residency residency, int32_t device_ordinal) noexcept {
    using R = sequence_buffer_residency;
    return residency == R::host ? device_ordinal == -1
        : device_ordinal >= 0 && (residency == R::device || residency == R::managed
            || residency == R::peer_device);
}

CELLERATOR_SEQUENCE_EVENT_HD constexpr bool sequence_emit_counts_valid(
    sequence_output_mode mode, uint64_t capacity, const sequence_emit_counts &counts) noexcept {
    using M = sequence_output_mode;
    if (counts.required_capacity != counts.total_matches) return false;
    if (mode == M::count || mode == M::mask)
        return !counts.stored_records && !counts.dropped_records;
    return (mode == M::stable_emit || mode == M::unordered_emit)
        && counts.stored_records <= capacity && counts.stored_records <= counts.total_matches
        && counts.dropped_records == counts.total_matches - counts.stored_records;
}

CELLERATOR_SEQUENCE_EVENT_HD constexpr bool sequence_output_descriptor_valid(
    const sequence_output_descriptor &out) noexcept {
    using M = sequence_output_mode;
    using O = sequence_event_ordering;
    if (!out.counts || !sequence_buffer_location_valid(out.residency, out.device_ordinal))
        return false;
    switch (out.mode) {
    case M::count:
        return !out.records && !out.mask_words && !out.capacity && !out.mask_word_count
            && out.ordering == O::not_applicable;
    case M::mask:
        return !out.records && !out.capacity && (!out.mask_word_count || out.mask_words)
            && out.ordering == O::not_applicable;
    case M::stable_emit:
    case M::unordered_emit:
        return !out.mask_words && !out.mask_word_count && (!out.capacity || out.records)
            && out.ordering == (out.mode == M::stable_emit ? O::coordinate_stable : O::unordered);
    default:
        return false;
    }
}

static_assert(sizeof(sequence_event) == 8u && alignof(sequence_event) == 4u);
static_assert(offsetof(sequence_event, local_position) == 0u);
static_assert(offsetof(sequence_event, predicate_id) == 4u);
static_assert(offsetof(sequence_event, payload) == 6u && offsetof(sequence_event, flags) == 7u);
static_assert(sizeof(unsigned long long) == 8u && sizeof(sequence_emit_counts) == 32u);
static_assert(__is_trivially_copyable(sequence_event));
static_assert(__is_trivially_copyable(sequence_emit_counts));
static_assert(__is_trivially_copyable(sequence_output_descriptor));

} // namespace cellerator::ops::sequence

#undef CELLERATOR_SEQUENCE_EVENT_HD
