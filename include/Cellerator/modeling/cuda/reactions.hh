#pragma once
#include <Cellerator/modeling/cuda/runtime.hh>
#include <cstdint>

namespace cellerator::modeling::cuda {

// Immutable shared CSR bank. Each reaction row must have strictly increasing,
// unique species slots and nonzero int64 deltas. Biological IDs are mapped by
// the caller; physical species slots are not interpreted as identities.
struct ReactionBank {
    ConstDeviceView offsets;       // uint64[reaction_count+1]
    ConstDeviceView species_slots; // uint64[change_count]
    ConstDeviceView deltas;        // int64[change_count]
    std::uint64_t reaction_count = 0;
    std::uint64_t change_count = 0;
    std::uint64_t species_count = 0;
};

// Each cell owns [cell_offsets[c],cell_offsets[c+1]) in sequence order.
// If event_indices is empty, position indexes reaction_ids directly. Otherwise
// event_indices[position] indexes an event-major reaction_ids array. The caller
// supplies grouping and ordering; this operation does not sort event times.
struct ReactionEventWorklist {
    ConstDeviceView cell_offsets;  // uint64[cell_count+1]
    ConstDeviceView reaction_ids;  // uint64[event_count]
    ConstDeviceView event_indices; // optional uint64[reference_count]
    std::uint64_t event_count = 0;
    std::uint64_t reference_count = 0;
};

struct ReactionCountBatch {
    // Initial logical counts must be nonnegative. Only event-touched slots are
    // checked by the sparse kernel; padding and unrelated slots are not scanned.
    DeviceView counts;             // int64[cell_count*cell_stride]
    std::uint64_t cell_count = 0;
    std::uint64_t cell_stride = 0;  // >= bank.species_count
};

enum ReactionApplyFlag : std::uint32_t {
    reaction_apply_success = 0,
    reaction_invalid_worklist = 1,
    reaction_invalid_channel = 2,
    reaction_invalid_stoichiometry = 4,
    reaction_negative_count = 8,
    reaction_count_overflow = 16,
    reaction_frontier_capacity = 32,
};

struct ReactionCellStatus {
    std::uint64_t events_applied = 0;
    std::uint64_t failed_position = UINT64_MAX;
    std::uint64_t changed_written = 0; // complete log for committed prefix
    std::uint64_t changed_required = 0; // prefix + failed event on capacity failure
    std::uint32_t flags = 0;
    std::uint32_t reserved = 0;
};

// Optional per-cell compact changed-source logs. Repeated species across events
// are retained; build_changed_frontier performs union/deduplication. An explicit
// map translates (cell,species_slot) to that frontier's source ordinal. With an
// empty source_ordinals view output is LOCAL species slots, requiring caller
// mapping before using an identity-sensitive frontier. Output cells are disjoint.
struct ReactionChangeOutput {
    DeviceView sources;                  // uint64[cell_count*capacity_per_cell]
    ConstDeviceView source_ordinals;      // optional uint64[cell_count*species_count]
    std::uint64_t capacity_per_cell = 0;
    bool enabled = false;
};

// Asynchronous, allocation-free launch on stream.device(). All borrowed views
// must remain valid and disjoint until completion; immutable bank/worklist may
// be shared by calls, mutable counts/status/change outputs may not. No pointer
// ownership transfer or implicit synchronization. Host extent/device errors
// throw before launch. Device content errors are reported per cell. A cell
// stops on first failed event: prior events remain committed, failing event has
// no count/log writes, and later events are skipped. Retry only its unapplied
// suffix after resolving the cause. statuses: ReactionCellStatus[cell_count].
// This applies supplied events; it computes no propensities or SSA sampling.
void apply_reaction_events(ReactionBank bank, ReactionEventWorklist events,
                           ReactionCountBatch batch, DeviceView statuses,
                           ReactionChangeOutput changed, const Stream& stream);

} // namespace cellerator::modeling::cuda
