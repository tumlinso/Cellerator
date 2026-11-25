#pragma once

#include <Cellerator/memory/status.hh>

#include <stddef.h>

namespace cellerator::execution::streaming {

struct PacketPlacement {
    size_t offset = 0, early_offset = 0, early_bytes = 0, deferred_bytes = 0;
};

// Both packets are aligned and placed from opposite ends of an aligned arena.
// The caller supplies the last-reader dependency for the deferred bytes.
inline memory::status place_packet(size_t arena_bytes, size_t live_bytes, size_t packet_bytes,
                                   bool high, size_t minimum_early, PacketPlacement *out) noexcept {
    using memory::status;
    if (!out || !packet_bytes || live_bytes > arena_bytes) return status::invalid_argument;
    if (packet_bytes > arena_bytes) return status::capacity_exceeded;
    const size_t free_bytes = arena_bytes-live_bytes;
    const size_t early = packet_bytes <= free_bytes ? packet_bytes : free_bytes >= minimum_early ? free_bytes : 0;
    const size_t deferred = packet_bytes-early;
    *out = {high ? arena_bytes-packet_bytes : 0, high && early ? deferred : 0, early, deferred};
    return status::success;
}

} // namespace cellerator::execution::streaming
