#include <Cellerator/modeling/cuda/reactions.hh>
#include <cuda_runtime.h>
#include <algorithm>
#include <limits>

namespace cellerator::modeling::cuda {
namespace {
__global__ void apply_events_kernel(ReactionBank bank, ReactionEventWorklist events,
                                   ReactionCountBatch batch, ReactionCellStatus* statuses,
                                   ReactionChangeOutput changed) {
    const auto* offsets = static_cast<const std::uint64_t*>(bank.offsets.data);
    const auto* species = static_cast<const std::uint64_t*>(bank.species_slots.data);
    const auto* deltas = static_cast<const std::int64_t*>(bank.deltas.data);
    const auto* cell_offsets = static_cast<const std::uint64_t*>(events.cell_offsets.data);
    const auto* reaction_ids = static_cast<const std::uint64_t*>(events.reaction_ids.data);
    const auto* indices = static_cast<const std::uint64_t*>(events.event_indices.data);
    const auto* source_map = static_cast<const std::uint64_t*>(changed.source_ordinals.data);
    auto* output = static_cast<std::uint64_t*>(changed.sources.data);
    auto* counts = static_cast<std::int64_t*>(batch.counts.data);
    const std::uint64_t stride = std::uint64_t(gridDim.x) * blockDim.x;
    for (std::uint64_t cell = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         cell < batch.cell_count; cell += stride) {
        ReactionCellStatus status{};
        status.failed_position = UINT64_MAX;
        const auto begin = cell_offsets[cell], end = cell_offsets[cell + 1];
        if (begin > end || end > events.reference_count) {
            status.flags = reaction_invalid_worklist;
            statuses[cell] = status;
            continue;
        }
        auto* row = counts ? counts + cell * batch.cell_stride : nullptr;
        for (auto position = begin; position < end; ++position) {
            const auto event = indices ? indices[position] : position;
            if (event >= events.event_count) {
                status.flags = reaction_invalid_worklist;
                status.failed_position = position;
                break;
            }
            const auto reaction = reaction_ids[event];
            if (reaction >= bank.reaction_count) {
                status.flags = reaction_invalid_channel;
                status.failed_position = position;
                break;
            }
            const auto first = offsets[reaction], last = offsets[reaction + 1];
            if (first > last || last > bank.change_count) {
                status.flags = reaction_invalid_stoichiometry;
                status.failed_position = position;
                break;
            }
            // Every entry is checked before any count is modified. Sorted unique
            // slots ensure duplicate deltas cannot bypass transactional checks.
            for (auto entry = first; entry < last; ++entry) {
                const auto slot = species[entry];
                const auto delta = deltas[entry];
                if (slot >= bank.species_count || !delta || delta == INT64_MIN
                    || (entry > first && species[entry - 1] >= slot)) {
                    status.flags = reaction_invalid_stoichiometry;
                    break;
                }
                const auto count = row[slot];
                if (count < 0 || (delta < 0 && count < -delta)) {
                    status.flags = reaction_negative_count;
                    break;
                }
                if (delta > 0 && count > INT64_MAX - delta) {
                    status.flags = reaction_count_overflow;
                    break;
                }
            }
            if (status.flags) { status.failed_position = position; break; }
            if (changed.enabled) {
                const auto length = last - first;
                // Host bounds make total addressable log size <= SIZE_MAX/8;
                // avoid addition overflow for adversarial event streams.
                if (length > changed.capacity_per_cell - status.changed_written) {
                    status.flags = reaction_frontier_capacity;
                    status.failed_position = position;
                    status.changed_required = status.changed_written > UINT64_MAX - length
                        ? UINT64_MAX : status.changed_written + length;
                    break;
                }
            }
            for (auto entry = first; entry < last; ++entry) {
                const auto slot = species[entry];
                row[slot] += deltas[entry];
                if (changed.enabled) {
                    const auto ordinal = source_map ? source_map[cell * bank.species_count + slot] : slot;
                    output[cell * changed.capacity_per_cell + status.changed_written++] = ordinal;
                }
            }
            ++status.events_applied;
            status.changed_required = status.changed_written;
        }
        statuses[cell] = status;
    }
}

std::size_t extent(std::uint64_t count, std::size_t size) {
    if (count > std::numeric_limits<std::size_t>::max())
        throw std::overflow_error("reaction extent overflows size_t");
    return checked_bytes(static_cast<std::size_t>(count), size);
}
std::uint64_t product(std::uint64_t a, std::uint64_t b) {
    if (b && a > UINT64_MAX / b) throw std::overflow_error("reaction shape overflow");
    return a * b;
}
void require(ConstDeviceView view, std::uint64_t count, std::size_t size, int device, const char* what) {
    require_view(view, extent(count, size), device, what);
    if (count && reinterpret_cast<std::uintptr_t>(view.data) % size != 0)
        throw std::invalid_argument("misaligned reaction array");
}
} // namespace

void apply_reaction_events(ReactionBank bank, ReactionEventWorklist events,
                           ReactionCountBatch batch, DeviceView statuses,
                           ReactionChangeOutput changed, const Stream& stream) {
    const auto device = stream.device();
    if (bank.reaction_count == UINT64_MAX || batch.cell_count == UINT64_MAX)
        throw std::overflow_error("reaction offset shape overflow");
    if (batch.cell_stride < bank.species_count)
        throw std::invalid_argument("reaction count rows overlap");
    require(bank.offsets, bank.reaction_count + 1, sizeof(std::uint64_t), device, "reaction offsets");
    require(bank.species_slots, bank.change_count, sizeof(std::uint64_t), device, "reaction species");
    require(bank.deltas, bank.change_count, sizeof(std::int64_t), device, "reaction deltas");
    require(events.cell_offsets, batch.cell_count + 1, sizeof(std::uint64_t), device, "cell event offsets");
    require(events.reaction_ids, events.event_count, sizeof(std::uint64_t), device, "reaction event ids");
    if (events.event_indices.data)
        require(events.event_indices, events.reference_count, sizeof(std::uint64_t), device, "reaction event indices");
    else if (events.reference_count > events.event_count)
        throw std::invalid_argument("direct event references exceed event array");
    require(batch.counts, product(batch.cell_count, batch.cell_stride), sizeof(std::int64_t), device, "reaction counts");
    require_view(statuses, extent(batch.cell_count, sizeof(ReactionCellStatus)), device, "reaction statuses");
    if (batch.cell_count && reinterpret_cast<std::uintptr_t>(statuses.data) % alignof(ReactionCellStatus))
        throw std::invalid_argument("misaligned reaction status");
    if (changed.enabled) {
        require(changed.sources, product(batch.cell_count, changed.capacity_per_cell), sizeof(std::uint64_t), device, "reaction changed sources");
        if (changed.source_ordinals.data)
            require(changed.source_ordinals, product(batch.cell_count, bank.species_count), sizeof(std::uint64_t), device, "reaction source ordinal map");
    }
    if (!batch.cell_count) return;
    DeviceGuard guard(device);
    constexpr unsigned threads = 128;
    const auto blocks = static_cast<unsigned>(std::min<std::uint64_t>((batch.cell_count - 1) / threads + 1, 65535));
    apply_events_kernel<<<blocks, threads, 0, stream.get()>>>(bank, events, batch,
        static_cast<ReactionCellStatus*>(statuses.data), changed);
    check(cudaGetLastError(), "apply reaction events");
}
} // namespace cellerator::modeling::cuda
