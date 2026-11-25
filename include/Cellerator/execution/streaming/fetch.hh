#pragma once

#include <Cellerator/execution/streaming/source.hh>
#include <Cellerator/layout/sliced_ell_pack.hh>

namespace cellerator::execution::streaming {

enum class FetchFormat : uint8_t { native, sliced_ell };
enum class FetchState : uint8_t { stale, pending, ready, failed };

struct FetchQueue;
struct FetchTicket {
    const FetchQueue *owner = nullptr;
    uint64_t generation = 0;
    unsigned slot = 0;
};

using FetchJob = void (*)(void *) noexcept;
struct FetchWorker {
    void *context = nullptr;
    // Acceptance publishes the job argument and completes exactly once,
    // even during drain. The worker owns the cross-thread dispatch ordering.
    // Rejection means the submitter did not start and will not run the job.
    bool (*submit)(void *, FetchJob, void *) noexcept = nullptr;
};

struct FetchClock {
    void *context = nullptr;
    // Optional monotonic host clock. It measures the I/O worker phases.
    uint64_t (*now_ns)(void *) noexcept = nullptr;
};

struct FetchRun {
    CellRange range;
    size_t logical_row, first_entry, entries;
};

struct FetchWorkspace {
    SparseBuffers scratch{};
    FetchRun *runs = nullptr;
    size_t run_capacity = 0;
    unsigned *lengths = nullptr, *order = nullptr;
    uint64_t *source_lengths = nullptr;
    layout::SlicedEllRow *logical_rows = nullptr;
    size_t row_capacity = 0;
    layout::SlicedEllSlice *slices = nullptr;
    size_t slice_capacity = 0;
};

struct FetchSlotStorage {
    SparseBuffers native{};
    void *packet_data = nullptr;
    size_t packet_bytes = 0;
};

// Ranges express logical occurrence order, including repeated cells. The
// caller keeps them live until completion and supplies disjoint row-map storage
// that remains live through its own final consumer, including after release().
struct FetchRequest {
    const CellRange *ranges = nullptr;
    size_t range_count = 0;
    uint64_t batch = 0, logical_first = 0;
    layout::SlicedEllRow *row_map = nullptr;
    size_t row_map_capacity = 0;
};

struct FetchOptions {
    FetchFormat format = FetchFormat::native;
    unsigned slots = 1;
    layout::SlicedEllPackOptions packing{};
};

struct FetchRequirements {
    size_t rows = 0, entries = 0, runs = 0, slices = 0;
    size_t value_bytes = 0, index_bytes = 0, offset_bytes = 0, packet_bytes = 0;
};

struct FetchTimes {
    uint64_t planning_ns = 0, read_ns = 0, packing_ns = 0;
};

// Native buffers are cell-major, with rebased offsets and the exact types in
// source. Only the operand of the requested format is populated. Data views
// borrow the slot until release; the row map borrows the caller's request.
struct FetchedBatch {
    SourceInfo source{};
    uint64_t batch = 0;
    FetchRequirements required{};
    SparseBuffers native{};
    const void *packet_data = nullptr;
    tensor::base::device::SlicedEllLayout packet_layout{};
    const layout::SlicedEllRow *row_map = nullptr;
    ReadStats read{};
    FetchTimes time{};
};

struct FetchReport {
    FetchState state = FetchState::stale;
    memory::status status = memory::status::invalid_argument;
    FetchedBatch batch{};
};

// One coordinator and at most one outstanding worker job. Capacity and all
// user memory are supplied at prepare. Keep this object, its worker, source,
// and backing spans live until drain succeeds. No implicit synchronization.
struct FetchQueue {
    FetchQueue() noexcept = default;
    FetchQueue(const FetchQueue &) = delete;
    FetchQueue &operator=(const FetchQueue &) = delete;

    memory::status prepare(FetchSource source, FetchWorker worker, FetchClock clock,
                           FetchOptions options, FetchWorkspace workspace,
                           const FetchSlotStorage *slots, unsigned slot_count) noexcept;
    // Apply physical scheduling choices only to future submissions. Row
    // permutation is allowed only under the consumer's logical-order contract.
    memory::status configure(unsigned lookahead, bool reorder_rows) noexcept;
    memory::status fetch(FetchRequest request, FetchTicket *ticket) noexcept;
    FetchReport poll(FetchTicket ticket) noexcept;
    memory::status release(FetchTicket ticket) noexcept;

    // Stop accepting work, cancel unsubmitted jobs, and poll until the worker
    // and all leased results are released. Existing ready views stay live.
    bool drain() noexcept;

private:
    enum class Phase : uint8_t { free, queued, running, finished };
    struct Slot {
        FetchQueue *queue = nullptr;
        FetchSlotStorage storage{};
        FetchRequest request{};
        FetchOptions options{};
        uint64_t generation = 0;
        Phase phase = Phase::free;
        unsigned complete = 0;
        FetchReport report{};
    };

    static void read_job(void *) noexcept;
    memory::status read_batch(Slot &slot) noexcept;
    void advance() noexcept;
    Slot *lookup(FetchTicket ticket) noexcept;

    FetchSource source_{};
    FetchWorker worker_{};
    FetchClock clock_{};
    FetchOptions options_{};
    FetchWorkspace workspace_{};
    Slot slots_[2]{};
    uint64_t generation_ = 0;
    unsigned slot_count_ = 0, running_ = 2;
    bool draining_ = false;
};

} // namespace cellerator::execution::streaming
