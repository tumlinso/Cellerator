#pragma once

#include <Cellerator/memory/status.hh>

#include <stddef.h>

namespace cellerator::execution::planning {

struct PacketCut {
    size_t left_slices = 0, left_bytes = 0, right_bytes = 0;
};

struct PacketCutChoice {
    PacketCut cut{};
    size_t wait_before = 0, wait_between = 0, wait_after = 0;
};

// The consumer permits splitting this logical batch. The caller supplies
// candidate whole-slice cuts and exact aligned sizes from the packet layouts.
// A zero neighbor size omits that handoff. Minimize deferred bytes, then
// distance from the byte target, then take the earliest cut in slice order.
inline memory::status choose_packet_cut(size_t slices, size_t arena, size_t target, size_t previous, size_t next,
                                        const PacketCut *cuts, size_t count, PacketCutChoice *out) noexcept {
    using memory::status;
    if (!out || !cuts || !count || slices < 2 || !target || target > arena || previous > arena || next > arena)
        return status::invalid_argument;
    auto wait = [arena](size_t a, size_t b) noexcept { return b > arena-a ? b-(arena-a) : size_t(0); };
    auto distance = [target](size_t n) noexcept { return n > target ? n-target : target-n; };
    PacketCutChoice best{};
    size_t best_wait = 0, best_distance = 0;
    for (size_t i = 0; i < count; ++i) {
        const auto cut = cuts[i];
        if (!cut.left_slices || cut.left_slices >= slices || !cut.left_bytes || !cut.right_bytes)
            return status::invalid_argument;
        if (cut.left_bytes > arena || cut.right_bytes > arena) continue;
        const PacketCutChoice candidate{cut, wait(previous,cut.left_bytes),
            wait(cut.left_bytes,cut.right_bytes), wait(cut.right_bytes,next)};
        size_t total, delta;
        if (__builtin_add_overflow(candidate.wait_before, candidate.wait_between, &total) ||
            __builtin_add_overflow(total, candidate.wait_after, &total) ||
            __builtin_add_overflow(distance(cut.left_bytes), distance(cut.right_bytes), &delta))
            return status::arithmetic_overflow;
        if (!best.cut.left_slices || total < best_wait ||
            (total == best_wait && (delta < best_distance ||
            (delta == best_distance && cut.left_slices < best.cut.left_slices)))) {
            best = candidate; best_wait = total; best_distance = delta;
        }
    }
    if (!best.cut.left_slices) return status::capacity_exceeded;
    *out = best;
    return status::success;
}

} // namespace cellerator::execution::planning
