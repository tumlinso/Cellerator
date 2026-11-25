#include <Cellerator/execution/streaming/fetch.hh>

namespace cellerator::execution::streaming {
namespace {

using memory::status;

bool integral(ScalarType type) noexcept {
    return type >= ScalarType::i8 && type <= ScalarType::u64;
}

uint64_t offset_max(ScalarType type) noexcept {
    switch (type) {
    case ScalarType::i8: return INT8_MAX;
    case ScalarType::u8: return UINT8_MAX;
    case ScalarType::i16: return INT16_MAX;
    case ScalarType::u16: return UINT16_MAX;
    case ScalarType::i32: return INT32_MAX;
    case ScalarType::u32: return UINT32_MAX;
    case ScalarType::i64: return INT64_MAX;
    default: return UINT64_MAX;
    }
}

struct Span { const void *data; size_t bytes; };

bool supported_span(Span span, size_t alignment) noexcept {
    return !span.bytes || (span.data && uintptr_t(span.data)%alignment == 0 &&
                           span.bytes <= UINTPTR_MAX - uintptr_t(span.data));
}

bool disjoint(const Span *spans, size_t n) noexcept {
    for (size_t i = 0; i < n; ++i) {
        if (!spans[i].bytes) continue;
        const auto a = uintptr_t(spans[i].data);
        for (size_t j = 0; j < i; ++j) {
            if (!spans[j].bytes) continue;
            const auto b = uintptr_t(spans[j].data);
            if (a < b + spans[j].bytes && b < a + spans[i].bytes) return false;
        }
    }
    return true;
}

uint64_t now(FetchClock clock) noexcept {
    return clock.now_ns ? clock.now_ns(clock.context) : 0;
}

status size_sparse(SourceInfo info, FetchRequirements &need) noexcept {
    size_t offsets;
    if (__builtin_add_overflow(need.rows, size_t(1), &offsets) ||
        __builtin_mul_overflow(need.entries, scalar_bytes(info.value), &need.value_bytes) ||
        __builtin_mul_overflow(need.entries, scalar_bytes(info.index), &need.index_bytes) ||
        __builtin_mul_overflow(offsets, scalar_bytes(info.offset), &need.offset_bytes) ||
        need.entries > offset_max(info.offset)) return status::arithmetic_overflow;
    return status::success;
}

bool before(const FetchRun &a, const FetchRun &b) noexcept {
    return a.range.first < b.range.first || (a.range.first == b.range.first && a.logical_row < b.logical_row);
}

void sift_runs(FetchRun *runs, size_t root, size_t n) noexcept {
    const FetchRun saved = runs[root];
    while (root < n/2) {
        size_t child = 2*root+1;
        if (child+1 < n && before(runs[child], runs[child+1])) ++child;
        if (!before(saved, runs[child])) break;
        runs[root] = runs[child];
        root = child;
    }
    runs[root] = saved;
}

void sort_runs(FetchRun *runs, size_t n) noexcept {
    for (size_t root = n/2; root != 0; --root) sift_runs(runs, root-1, n);
    for (size_t end = n; end > 1;) {
        --end;
        const auto run = runs[end];
        runs[end] = runs[0]; runs[0] = run;
        sift_runs(runs, 0, end);
    }
}

void *at(void *data, size_t offset) noexcept {
    return data ? static_cast<unsigned char *>(data)+offset : nullptr;
}

template<class O>
status rebase(void *data, const FetchRun &run, unsigned *lengths, bool compare) noexcept {
    auto *offsets = static_cast<O *>(data);
    if (offsets[0] != 0 || uint64_t(offsets[run.range.count]) != run.entries)
        return status::invalid_argument;
    uint64_t prior = 0;
    for (size_t row = 0; row < run.range.count; ++row) {
        const uint64_t next = uint64_t(offsets[row+1]);
        if (next < prior || next > run.entries) return status::invalid_argument;
        if (lengths) {
            if (next-prior > UINT32_MAX) return status::arithmetic_overflow;
            if (compare && lengths[row] != next-prior) return status::invalid_argument;
            lengths[row] = unsigned(next-prior);
        }
        offsets[row] = O(prior + run.first_entry);
        prior = next;
    }
    offsets[run.range.count] = O(prior + run.first_entry);
    return status::success;
}

template<class F>
status with_unsigned_width(size_t bytes, F call) noexcept {
    switch (bytes) {
    case 1: return call.template operator()<uint8_t>();
    case 2: return call.template operator()<uint16_t>();
    case 4: return call.template operator()<uint32_t>();
    case 8: return call.template operator()<uint64_t>();
    default: return status::invalid_argument;
    }
}

template<class F>
status with_index(ScalarType type, F call) noexcept {
    switch (type) {
    case ScalarType::i8: return call.template operator()<int8_t>();
    case ScalarType::i16: return call.template operator()<int16_t>();
    case ScalarType::i32: return call.template operator()<int32_t>();
    case ScalarType::i64: return call.template operator()<int64_t>();
    default: return with_unsigned_width(scalar_bytes(type), call);
    }
}

template<class F>
status with_value(ScalarType type, F call) noexcept {
    switch (type) {
    case ScalarType::boolean:
    case ScalarType::u8: return call.template operator()<uint8_t>();
    case ScalarType::i8: return call.template operator()<int8_t>();
    case ScalarType::i16: return call.template operator()<int16_t>();
    case ScalarType::u16: return call.template operator()<uint16_t>();
    case ScalarType::i32: return call.template operator()<int32_t>();
    case ScalarType::u32: return call.template operator()<uint32_t>();
    case ScalarType::i64: return call.template operator()<int64_t>();
    case ScalarType::u64: return call.template operator()<uint64_t>();
    case ScalarType::f16: return call.template operator()<real::f16_t>();
    case ScalarType::f32: return call.template operator()<float>();
    case ScalarType::f64: return call.template operator()<double>();
    case ScalarType::invalid: return status::invalid_argument;
    }
    return status::invalid_argument;
}

status pack(SourceInfo info, const layout::SlicedEllPackPlan &plan, SparseBuffers sparse,
            const layout::SlicedEllRow *rows, const FetchSlotStorage &storage,
            const FetchRequest &request) noexcept {
    return with_unsigned_width(scalar_bytes(info.offset), [&]<class O>() noexcept {
        return with_index(info.index, [&]<class I>() noexcept {
            return with_value(info.value, [&]<class V>() noexcept {
                using tensor::base::CompressedAxis;
                using tensor::base::CompressedView;
                const CompressedView<I, const V, O> source{
                    {plan.shape.rows, info.features, plan.shape.nnz, CompressedAxis::row,
                     static_cast<const O *>(sparse.offsets), static_cast<const I *>(sparse.indices)},
                    static_cast<const V *>(sparse.values)};
                return layout::pack_sliced_ell(plan, source, rows, storage.packet_data, storage.packet_bytes,
                                               request.row_map, request.row_map_capacity);
            });
        });
    });
}

} // namespace

status FetchQueue::prepare(FetchSource source, FetchWorker worker, FetchClock clock, FetchOptions options,
                           FetchWorkspace workspace, const FetchSlotStorage *storage, unsigned count) noexcept {
    for (const auto &slot : slots_) if (slot.phase != Phase::free) return status::invalid_state;
    if (!storage || !count || count > 2 || !options.slots || options.slots > count || !worker.submit ||
        !source.requirements || !source.read_into || !scalar_bytes(source.info.value) ||
        !integral(source.info.index) || !integral(source.info.offset) ||
        (options.format != FetchFormat::native && options.format != FetchFormat::sliced_ell))
        return status::invalid_argument;
    const bool sell = options.format == FetchFormat::sliced_ell;
    if (sell && (!options.packing.slice_rows || (options.packing.reorder_rows && !options.packing.sort_window) ||
                 (source.info.value != ScalarType::f16 && !options.packing.allow_fp16_conversion)))
        return status::invalid_argument;
    if (workspace.run_capacity > SIZE_MAX/sizeof(FetchRun) ||
        workspace.row_capacity > SIZE_MAX/sizeof(layout::SlicedEllRow) ||
        workspace.slice_capacity > SIZE_MAX/sizeof(layout::SlicedEllSlice))
        return status::arithmetic_overflow;

    Span spans[12]{};
    size_t n = 0;
    auto add = [&](const void *data, size_t bytes, size_t alignment) noexcept {
        const Span span{data, bytes};
        if (!supported_span(span, alignment)) return false;
        spans[n++] = span;
        return true;
    };
    auto add_sparse = [&](SparseBuffers raw) noexcept {
        return add(raw.offsets, raw.offset_bytes, scalar_bytes(source.info.offset)) &&
               add(raw.indices, raw.index_bytes, scalar_bytes(source.info.index)) &&
               add(raw.values, raw.value_bytes, scalar_bytes(source.info.value));
    };
    auto add_array = [&]<class T>(T *p, size_t capacity) noexcept {
        return add(p, capacity*sizeof(T), alignof(T));
    };
    if (!add_array(workspace.runs, workspace.run_capacity)) return status::invalid_alignment;
    if (sell && (!add_sparse(workspace.scratch) || !add_array(workspace.lengths, workspace.row_capacity) ||
                 !add_array(workspace.order, workspace.row_capacity) ||
                 !add_array(workspace.logical_rows, workspace.row_capacity) ||
                 !add_array(workspace.slices, workspace.slice_capacity) ||
                 (source.cell_lengths && !add_array(workspace.source_lengths, workspace.row_capacity))))
        return status::invalid_alignment;
    for (unsigned i = 0; i < count; ++i) {
        const bool valid = sell ? add(storage[i].packet_data, storage[i].packet_bytes,
                                      tensor::base::device::SlicedEllLayout::alignment) : add_sparse(storage[i].native);
        if (!valid) return status::invalid_alignment;
    }
    if (!disjoint(spans, n)) return status::invalid_argument;

    source_ = source; worker_ = worker; clock_ = clock; options_ = options; workspace_ = workspace;
    for (unsigned i = 0; i < count; ++i) {
        slots_[i].queue = this;
        slots_[i].storage = storage[i];
    }
    slot_count_ = count; running_ = 2; draining_ = false;
    return status::success;
}

status FetchQueue::configure(unsigned lookahead, bool reorder_rows) noexcept {
    if (!slot_count_ || draining_) return status::invalid_state;
    if (!lookahead || lookahead > slot_count_ ||
        (reorder_rows && (options_.format != FetchFormat::sliced_ell || !options_.packing.sort_window)))
        return status::invalid_argument;
    options_.slots = lookahead;
    options_.packing.reorder_rows = reorder_rows;
    return status::success;
}

status FetchQueue::fetch(FetchRequest request, FetchTicket *ticket) noexcept {
    if (!ticket || (request.range_count && !request.ranges)) return status::invalid_argument;
    if (!slot_count_ || draining_) return status::invalid_state;
    if (generation_ == UINT64_MAX) return status::arithmetic_overflow;
    advance();
    unsigned free = 2, active = 0;
    for (unsigned i = 0; i < slot_count_; ++i) {
        if (slots_[i].phase == Phase::free) free = i;
        else ++active;
    }
    if (active >= options_.slots || free == 2) return status::capacity_exceeded;
    auto &slot = slots_[free];
    slot.generation = ++generation_; slot.request = request; slot.options = options_; slot.complete = 0;
    slot.report = {}; slot.phase = Phase::queued;
    *ticket = {this, slot.generation, free};
    advance();
    return status::success;
}

void FetchQueue::advance() noexcept {
    if (running_ != 2) {
        auto &slot = slots_[running_];
        if (!__atomic_load_n(&slot.complete, __ATOMIC_ACQUIRE)) return;
        slot.phase = Phase::finished;
        running_ = 2;
    }
    if (draining_) return;
    for (unsigned i = 0; i < slot_count_; ++i) {
        const auto &slot = slots_[i];
        if (slot.phase == Phase::queued && (running_ == 2 || slot.generation < slots_[running_].generation))
            running_ = i;
    }
    if (running_ == 2) return;
    auto &slot = slots_[running_];
    slot.phase = Phase::running;
    if (!worker_.submit(worker_.context, read_job, &slot)) {
        slot.phase = Phase::queued;
        running_ = 2;
    }
}

void FetchQueue::read_job(void *context) noexcept {
    auto &slot = *static_cast<Slot *>(context);
    slot.report.status = slot.queue->read_batch(slot);
    slot.report.state = slot.report.status == status::success ? FetchState::ready : FetchState::failed;
    __atomic_store_n(&slot.complete, 1, __ATOMIC_RELEASE);
}

FetchQueue::Slot *FetchQueue::lookup(FetchTicket ticket) noexcept {
    if (ticket.owner != this || ticket.slot >= slot_count_ || !ticket.generation) return nullptr;
    auto &slot = slots_[ticket.slot];
    return slot.phase != Phase::free && slot.generation == ticket.generation ? &slot : nullptr;
}

FetchReport FetchQueue::poll(FetchTicket ticket) noexcept {
    advance();
    const auto *slot = lookup(ticket);
    if (!slot) return {};
    if (slot->phase != Phase::finished) return {FetchState::pending, status::success, {}};
    return slot->report;
}

status FetchQueue::release(FetchTicket ticket) noexcept {
    advance();
    auto *slot = lookup(ticket);
    if (!slot) return status::invalid_argument;
    if (slot->phase != Phase::finished) return status::invalid_state;
    slot->phase = Phase::free;
    return status::success;
}

bool FetchQueue::drain() noexcept {
    draining_ = true;
    for (auto &slot : slots_) if (slot.phase == Phase::queued) slot.phase = Phase::free;
    advance();
    for (const auto &slot : slots_) if (slot.phase != Phase::free) return false;
    return true;
}

status FetchQueue::read_batch(Slot &slot) noexcept {
    const auto &request = slot.request;
    const auto &options = slot.options;
    auto &batch = slot.report.batch;
    auto &need = batch.required;
    const auto info = source_.info;
    const bool sell = options.format == FetchFormat::sliced_ell;
    const bool early_plan = sell && source_.cell_lengths;
    batch.source = info; batch.batch = request.batch;
    const uint64_t planning_start = now(clock_);
    bool ordered = true;
    auto *runs = workspace_.runs;
    need.runs = request.range_count;
    if (need.runs > workspace_.run_capacity) return status::capacity_exceeded;
    size_t active_runs = 0;
    for (size_t i = 0; i < request.range_count; ++i) {
        const auto range = request.ranges[i];
        if (range.first > info.cells || range.count > info.cells-range.first) return status::invalid_argument;
        if (!range.count) continue;
        uint64_t entries = 0;
        const auto sized = source_.requirements(source_.context, range, &entries);
        if (sized != status::success) return sized;
        if (range.count > SIZE_MAX-need.rows || entries > SIZE_MAX-need.entries)
            return status::arithmetic_overflow;
        auto &run = runs[active_runs];
        run = {range, need.rows, need.entries, size_t(entries)};
        if (active_runs && before(run, runs[active_runs-1])) ordered = false;
        ++active_runs; need.rows += size_t(range.count); need.entries += size_t(entries);
    }
    auto result = size_sparse(info, need);
    if (result != status::success) return result;
    const size_t m = need.rows;
    if ((m && m-1 > UINT64_MAX-request.logical_first) || m > SIZE_MAX/sizeof(layout::SlicedEllRow))
        return status::arithmetic_overflow;
    if (sell && (m > UINT32_MAX || info.features > UINT32_MAX || need.entries > UINT32_MAX))
        return status::arithmetic_overflow;
    if (sell) need.slices = m/options.packing.slice_rows + (m%options.packing.slice_rows != 0);
    if (m > request.row_map_capacity ||
        (sell && (m > workspace_.row_capacity || need.slices > workspace_.slice_capacity)))
        return status::capacity_exceeded;
    if (!supported_span({request.row_map, m*sizeof(layout::SlicedEllRow)}, alignof(layout::SlicedEllRow)))
        return status::invalid_alignment;
    auto raw = sell ? workspace_.scratch : slot.storage.native;
    if (need.value_bytes > raw.value_bytes || need.index_bytes > raw.index_bytes ||
        need.offset_bytes > raw.offset_bytes) return status::capacity_exceeded;
    raw.value_bytes = need.value_bytes; raw.index_bytes = need.index_bytes; raw.offset_bytes = need.offset_bytes;

    if (!ordered) sort_runs(runs, active_runs);
    auto *logical = sell ? workspace_.logical_rows : request.row_map;
    for (size_t i = 0; i < active_runs; ++i) {
        const auto &run = runs[i];
        for (size_t row = 0; row < run.range.count; ++row)
            logical[run.logical_row+row] = {request.logical_first+run.logical_row+row, run.range.first+row};
    }
    layout::SlicedEllPackPlan plan{};
    auto plan_packet = [&]() noexcept {
        const auto planned = layout::plan_sliced_ell(workspace_.lengths, m, info.features, options.packing,
            workspace_.order, workspace_.row_capacity, workspace_.slices, workspace_.slice_capacity, &plan);
        if (planned != status::success) return planned;
        need.packet_bytes = plan.layout.bytes();
        if (plan.shape.nnz != need.entries) return status::invalid_argument;
        return need.packet_bytes <= slot.storage.packet_bytes ? status::success : status::capacity_exceeded;
    };
    if (early_plan) {
        for (size_t i = 0; i < active_runs; ++i) {
            const auto &run = runs[i];
            auto *lengths = workspace_.source_lengths+run.logical_row;
            result = source_.cell_lengths(source_.context, run.range, lengths, size_t(run.range.count));
            if (result != status::success) return result;
            for (size_t row = 0; row < run.range.count; ++row) {
                if (lengths[row] > UINT32_MAX) return status::arithmetic_overflow;
                workspace_.lengths[run.logical_row+row] = unsigned(lengths[row]);
            }
        }
        result = plan_packet();
        if (result != status::success) return result;
    }

    const uint64_t read_start = now(clock_);
    batch.time.planning_ns = read_start-planning_start;
    const size_t vb = scalar_bytes(info.value), ib = scalar_bytes(info.index), ob = scalar_bytes(info.offset);
    if (!active_runs) __builtin_memset(raw.offsets, 0, ob);
    for (size_t i = 0; i < active_runs; ++i) {
        const auto &run = runs[i];
        SparseBuffers dst{at(raw.values, run.first_entry*vb), at(raw.indices, run.first_entry*ib),
                          at(raw.offsets, run.logical_row*ob), run.entries*vb, run.entries*ib,
                          (size_t(run.range.count)+1)*ob};
        ReadStats stats{};
        result = source_.read_into(source_.context, run.range, dst, &stats);
        if (result != status::success) return result;
        batch.read.compressed_bytes += stats.compressed_bytes;
        batch.read.decoded_bytes += stats.decoded_bytes;
        result = with_unsigned_width(ob, [&]<class O>() noexcept {
            return rebase<O>(dst.offsets, run, sell ? workspace_.lengths+run.logical_row : nullptr, early_plan);
        });
        if (result != status::success) return result;
    }
    const uint64_t read_stop = now(clock_);
    batch.time.read_ns = read_stop-read_start;

    if (sell) {
        if (!early_plan) {
            result = plan_packet();
            if (result != status::success) return result;
        }
        const uint64_t packing_start = now(clock_);
        batch.time.planning_ns += packing_start-read_stop;
        result = pack(info, plan, raw, logical, slot.storage, request);
        if (result != status::success) return result;
        batch.time.packing_ns = now(clock_)-packing_start;
        batch.packet_data = slot.storage.packet_data; batch.packet_layout = plan.layout;
    } else batch.native = raw;
    batch.row_map = request.row_map;
    return status::success;
}

} // namespace cellerator::execution::streaming
