#ifdef NDEBUG
#undef NDEBUG
#endif
#include <Cellerator/modeling/cuda/reactions.hh>
#include <cassert>
#include <cstdint>
#include <limits>
#include <vector>
using namespace cellerator::modeling::cuda;

template<class T>
DeviceBuffer upload(const std::vector<T>& values, const Stream& stream) {
    DeviceBuffer out(values.size() * sizeof(T), stream.device());
    out.upload_async(values.data(), values.size() * sizeof(T), stream);
    // Keep caller-owned ordinary host storage alive until transfer completion.
    stream.synchronize();
    return out;
}
template<class T>
std::vector<T> download(const DeviceBuffer& buffer, std::size_t count, const Stream& stream) {
    std::vector<T> result(count);
    buffer.download_async(result.data(), count * sizeof(T), stream);
    stream.synchronize();
    return result;
}

int main() {
    Stream stream(0);
    auto offsets = upload<std::uint64_t>({0, 2, 4, 5}, stream);
    auto slots = upload<std::uint64_t>({0, 1, 0, 1, 2}, stream);
    auto deltas = upload<std::int64_t>({-1, 1, 1, -1, 1}, stream);
    ReactionBank bank{offsets.view(), slots.view(), deltas.view(), 3, 5, 3};
    auto cell_offsets = upload<std::uint64_t>({0, 3, 5, 6}, stream);
    // Event-major source is deliberately indexed in a different physical order.
    auto event_ids = upload<std::uint64_t>({2, 1, 0}, stream);
    auto indices = upload<std::uint64_t>({2, 2, 1, 2, 1, 0}, stream);
    ReactionEventWorklist events{cell_offsets.view(), event_ids.view(), indices.view(), 3, 6};
    const std::vector<std::int64_t> initial{4, 0, 0, 0, 1, 0, 0, 0, INT64_MAX};
    auto counts = upload(initial, stream);
    auto ordinal_map = upload<std::uint64_t>({100, 101, 102, 200, 201, 202, 300, 301, 302}, stream);
    DeviceBuffer statuses(3 * sizeof(ReactionCellStatus));
    DeviceBuffer changed(18 * sizeof(std::uint64_t));
    ReactionCountBatch batch{counts.view(), 3, 3};
    ReactionChangeOutput frontier{changed.view(), ordinal_map.view(), 6, true};
    apply_reaction_events(bank, events, batch, statuses.view(), frontier, stream);
    auto values = download<std::int64_t>(counts, 9, stream);
    auto status = download<ReactionCellStatus>(statuses, 3, stream);
    auto sources = download<std::uint64_t>(changed, 18, stream);
    assert((values == std::vector<std::int64_t>{3, 1, 0, 0, 1, 0, 0, 0, INT64_MAX}));
    assert(status[0].flags == reaction_apply_success && status[0].events_applied == 3);
    assert(status[0].changed_written == 6 && sources[0] == 100 && sources[1] == 101);
    assert(status[1].flags == reaction_negative_count && status[1].failed_position == 3);
    assert(status[1].events_applied == 0 && status[1].changed_written == 0);
    assert(status[2].flags == reaction_count_overflow && status[2].events_applied == 0);

    // Capacity failure must preserve complete invalidations for the accepted
    // prefix and must not apply even part of the next event.
    counts.upload_async(initial.data(), initial.size() * sizeof(std::int64_t), stream);
    frontier.capacity_per_cell = 2;
    apply_reaction_events(bank, events, batch, statuses.view(), frontier, stream);
    values = download<std::int64_t>(counts, 9, stream);
    status = download<ReactionCellStatus>(statuses, 3, stream);
    assert(values[0] == 3 && values[1] == 1);
    assert(status[0].flags == reaction_frontier_capacity && status[0].events_applied == 1);
    assert(status[0].changed_written == 2 && status[0].changed_required == 4);

    // Duplicate sparse entries are rejected before any state change.
    auto duplicate_slots = upload<std::uint64_t>({0, 0, 0, 1, 2}, stream);
    bank.species_slots = duplicate_slots.view();
    counts.upload_async(initial.data(), initial.size() * sizeof(std::int64_t), stream);
    apply_reaction_events(bank, events, batch, statuses.view(), {}, stream);
    values = download<std::int64_t>(counts, 9, stream);
    status = download<ReactionCellStatus>(statuses, 3, stream);
    assert(values[0] == 4 && values[1] == 0);
    assert(status[0].flags == reaction_invalid_stoichiometry);

    // Host shape failures happen before launch.
    bool rejected = false;
    try {
        apply_reaction_events(bank, events, {counts.view(), 3, 2}, statuses.view(), {}, stream);
    } catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
}
