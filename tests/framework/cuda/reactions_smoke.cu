#ifdef NDEBUG
#undef NDEBUG
#endif
#include <Cellerator/modeling/cuda/reactions.hh>
#include <cassert>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>
using namespace cellerator::modeling::cuda;

struct ReactionFixture {
    std::vector<std::uint64_t> offsets, slots, cell_offsets, reactions, source_map;
    std::vector<std::int64_t> deltas, initial;
    std::vector<std::uint64_t> expected_sources;
    std::vector<ReactionCellStatus> expected_status;
    std::vector<std::int64_t> expected_counts;
    std::uint64_t cells = 13, stride = 9, species = 7, capacity = 64;
};

template<class T>
DeviceBuffer upload(const std::vector<T>& values, const Stream& stream);
template<class T>
std::vector<T> download(const DeviceBuffer& buffer, std::size_t count, const Stream& stream);

// Deliberately simple CPU interpreter: it walks the event sequence and applies
// each stoichiometric row in order. Generated counts ensure valid events here;
// adversarial failure prefixes are covered by the focused cases below.
ReactionFixture make_fixture() {
    ReactionFixture x;
    x.offsets = {0, 0, 2, 5, 6};
    x.slots = {0, 3, 1, 2, 6, 4};
    x.deltas = {-2, 5, 7, -3, 4, -1};
    x.cell_offsets.push_back(0);
    x.initial.assign(x.cells * x.stride, 1000);
    for (std::uint64_t c = 0; c < x.cells; ++c) {
        x.initial[c * x.stride + 7] = -999; // padding is not a species slot
        for (std::uint64_t s = 0; s < x.species; ++s)
            x.source_map.push_back(10000 + c * 100 + s);
        const std::uint64_t count = (c * 7) % 21;
        for (std::uint64_t e = 0; e < count; ++e)
            x.reactions.push_back((e * 3 + c) % 4);
        x.cell_offsets.push_back(x.reactions.size());
    }
    x.expected_counts = x.initial;
    x.expected_status.resize(x.cells);
    x.expected_sources.assign(x.cells * x.capacity, UINT64_MAX);
    for (std::uint64_t c = 0; c < x.cells; ++c) {
        auto& status = x.expected_status[c];
        status.failed_position = UINT64_MAX;
        for (std::uint64_t p = x.cell_offsets[c]; p < x.cell_offsets[c + 1]; ++p) {
            const auto r = x.reactions[p];
            for (auto q = x.offsets[r]; q < x.offsets[r + 1]; ++q) {
                const auto slot = x.slots[q];
                x.expected_counts[c * x.stride + slot] += x.deltas[q];
                x.expected_sources[c * x.capacity + status.changed_written++] =
                    x.source_map[c * x.species + slot];
            }
            ++status.events_applied;
        }
        status.changed_required = status.changed_written;
    }
    return x;
}

std::vector<std::uint64_t> download_changed_prefixes(
    const DeviceBuffer& buffer, const std::vector<ReactionCellStatus>& statuses,
    std::uint64_t capacity_per_cell, const Stream& stream) {
    std::vector<std::uint64_t> result(statuses.size() * capacity_per_cell, UINT64_MAX);
    for (std::size_t c = 0; c < statuses.size(); ++c) {
        const auto written = statuses[c].changed_written;
        assert(written <= capacity_per_cell);
        if (written)
            buffer.download_async(result.data() + c * capacity_per_cell,
                                  written * sizeof(std::uint64_t), stream,
                                  c * capacity_per_cell * sizeof(std::uint64_t));
    }
    stream.synchronize();
    return result;
}

void check_device_fixture(const Stream& stream) {
    auto x = make_fixture();
    auto offsets = upload(x.offsets, stream);
    auto slots = upload(x.slots, stream);
    auto deltas = upload(x.deltas, stream);
    auto cell_offsets = upload(x.cell_offsets, stream);
    auto reaction_ids = upload(x.reactions, stream);
    auto counts = upload(x.initial, stream);
    auto source_map = upload(x.source_map, stream);
    DeviceBuffer status(x.cells * sizeof(ReactionCellStatus));
    DeviceBuffer sources(x.cells * x.capacity * sizeof(std::uint64_t));
    ReactionBank bank{offsets.view(), slots.view(), deltas.view(), 4, x.slots.size(), x.species};
    ReactionEventWorklist work{cell_offsets.view(), reaction_ids.view(), {},
                               x.reactions.size(), x.reactions.size()};
    ReactionChangeOutput changed{sources.view(), source_map.view(), x.capacity, true};
    apply_reaction_events(bank, work, {counts.view(), x.cells, x.stride}, status.view(), changed, stream);
    const auto actual_counts = download<std::int64_t>(counts, x.expected_counts.size(), stream);
    const auto actual_status = download<ReactionCellStatus>(status, x.cells, stream);
    // The changed-source contract initializes only each cell's written prefix.
    // Downloading capacity tails reads uninitialized device memory and correctly
    // fails Compute Sanitizer initcheck.
    const auto actual_sources = download_changed_prefixes(sources, actual_status, x.capacity, stream);
    assert(actual_counts == x.expected_counts);
    for (std::size_t c = 0; c < x.cells; ++c) {
        assert(actual_status[c].flags == x.expected_status[c].flags);
        assert(actual_status[c].events_applied == x.expected_status[c].events_applied);
        assert(actual_status[c].failed_position == x.expected_status[c].failed_position);
        assert(actual_status[c].changed_written == x.expected_status[c].changed_written);
        assert(actual_status[c].changed_required == x.expected_status[c].changed_required);
    }
    for (std::size_t c = 0; c < x.cells; ++c)
        for (std::uint64_t j = 0; j < x.expected_status[c].changed_written; ++j)
            assert(actual_sources[c * x.capacity + j] == x.expected_sources[c * x.capacity + j]);
}

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

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--host-only") {
        const auto fixture = make_fixture();
        assert(fixture.expected_status.size() == fixture.cells);
        assert(fixture.expected_counts[7] == -999);
        return 0;
    }
    int device = 0;
    if (argc == 3 && std::string(argv[1]) == "--device") device = std::atoi(argv[2]);
    else assert(argc == 1);
    Stream stream(device);
    check_device_fixture(stream);
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
    auto sources = download_changed_prefixes(changed, status, frontier.capacity_per_cell, stream);
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
