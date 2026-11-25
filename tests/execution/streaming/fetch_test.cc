#include <Cellerator/execution/planning/packet_cut.hh>
#include <Cellerator/execution/streaming/fetch.hh>
#include <Cellerator/execution/streaming/placement.hh>
#include <Cellerator/execution/tuning/fetch_tuning.hh>

#include <cstdlib>
#include <cstdio>

using namespace cellerator::execution::streaming;
using cellerator::memory::status;
using cellerator::layout::SlicedEllRow;
using cellerator::layout::SlicedEllSlice;

namespace {

void require(bool ok, const char *label) {
    if (ok) return;
    std::fprintf(stderr, "fetch test: %s\n", label);
    std::exit(1);
}

constexpr uint32_t offsets[]{0,1,4,6,6};
constexpr int32_t indices[]{0, 0,1,2, 2,2};
constexpr double values[]{7, 1,2,3, 0,2};

struct Source {
    unsigned order[16]{};
    unsigned reads = 0, probes = 0;
    bool fail = false, overflow = false, negative_index = false;

    static status requirements(void *, CellRange r, uint64_t *n) noexcept {
        *n = offsets[r.first+r.count]-offsets[r.first];
        return status::success;
    }
    static status lengths(void *context, CellRange r, uint64_t *length, size_t cap) noexcept {
        if (cap < r.count) return status::capacity_exceeded;
        auto &s = *static_cast<Source *>(context);
        ++s.probes;
        for (size_t i = 0; i < r.count; ++i) length[i] = offsets[r.first+i+1]-offsets[r.first+i];
        return status::success;
    }
    static status read(void *context, CellRange r, SparseBuffers buf, ReadStats *stats) noexcept {
        auto &s = *static_cast<Source *>(context);
        s.order[s.reads++] = unsigned(r.first);
        const size_t start = offsets[r.first], count = offsets[r.first+r.count]-start;
        auto *off = static_cast<uint32_t *>(buf.offsets);
        for (size_t i = 0; i <= r.count; ++i) off[i] = offsets[r.first+i]-uint32_t(start);
        if (count) {
            __builtin_memcpy(buf.indices, indices+start, count*sizeof(int32_t));
            __builtin_memcpy(buf.values, values+start, count*sizeof(double));
            if (s.overflow) static_cast<double *>(buf.values)[0] = 1e9;
            if (s.negative_index) static_cast<int32_t *>(buf.indices)[0] = -2;
        }
        *stats = {1, buf.offset_bytes+buf.index_bytes+buf.value_bytes};
        return s.fail ? status::invalid_state : status::success;
    }
    FetchSource adapter(bool early) noexcept {
        return {this, {4,3,9,ScalarType::f64,ScalarType::i32,ScalarType::u32},
                requirements, read, early ? lengths : nullptr};
    }
};

struct Worker {
    FetchJob job = nullptr;
    void *arg = nullptr;
    bool blocked = false;

    static bool submit(void *context, FetchJob job, void *arg) noexcept {
        auto &self = *static_cast<Worker *>(context);
        if (self.job || self.blocked) return false;
        self.job = job; self.arg = arg;
        return true;
    }
    void finish() {
        require(job, "worker has pending work");
        job(arg); job = nullptr;
    }
    FetchWorker worker() noexcept { return {this, submit}; }
};

struct Scratch {
    double data[16]{};
    int32_t index[16]{};
    uint32_t offset[8]{};
    SparseBuffers buffers() noexcept { return {data,index,offset,sizeof(data),sizeof(index),sizeof(offset)}; }
};

void native_and_drain() {
    FetchQueue queue;
    Worker worker;
    Source source;
    Scratch scratch[2];
    FetchRun runs[4];
    const FetchSlotStorage storage[2]{{scratch[0].buffers()}, {scratch[1].buffers()}};
    require(queue.prepare(source.adapter(false), worker.worker(), {}, {FetchFormat::native,2}, {{},runs,4}, storage, 2)
        == status::success, "native preparation");
    const CellRange shuffled[]{{2,1},{0,1},{2,1},{3,1}}, second{1,1};
    SlicedEllRow map[4], next_map[1];
    FetchTicket first{}, next{}, rejected{};
    require(queue.fetch({shuffled,4,7,12,map,4}, &first) == status::success, "enqueue first");
    require(queue.fetch({&second,1,8,16,next_map,1}, &next) == status::success, "enqueue second");
    require(queue.fetch({&second,1,9,17,next_map,1}, &rejected) == status::capacity_exceeded, "backpressure");
    require(queue.poll(first).state == FetchState::pending, "not ready without worker");
    require(queue.release(first) == status::invalid_state, "cannot release in-flight");
    worker.finish();
    const auto report = queue.poll(first);
    require(report.state == FetchState::ready && report.batch.batch == 7, "publication");
    require(report.batch.source.feature_axis == 9 && report.batch.source.value == ScalarType::f64,
            "source metadata");
    require(source.reads == 4 && source.order[0] == 0 && source.order[1] == 2 &&
            source.order[2] == 2 && source.order[3] == 3, "reads follow source order");
    const auto *off = static_cast<const uint32_t *>(report.batch.native.offsets);
    const auto *pv = static_cast<const double *>(report.batch.native.values);
    require(off[0] == 0 && off[1] == 2 && off[2] == 3 && off[3] == 5 && off[4] == 5 &&
            pv[0] == 0 && pv[1] == 2 && pv[2] == 7 && pv[3] == 0 && pv[4] == 2,
            "offsets and payload follow requested order");
    require(map[0].logical_position == 12 && map[2].logical_position == 14 &&
            map[0].source_cell == 2 && map[2].source_cell == 2, "duplicate cell occurrences stay distinct");
    require(queue.release(first) == status::success && queue.poll(first).state == FetchState::stale, "stale token");
    require(!queue.drain(), "drain waits for accepted worker");
    worker.finish();
    require(queue.poll(next).state == FetchState::ready && !queue.drain(), "drain keeps borrowed input");
    require(queue.release(next) == status::success && queue.drain(), "release and drain");
    require(queue.fetch({&second,1,9,17,next_map,1}, &next) == status::invalid_state, "drain closes submissions");
}

void sell_plan_and_failure() {
    FetchQueue queue;
    Worker worker;
    Source source;
    Scratch raw;
    FetchRun runs[4];
    unsigned lengths[4]{}, order[4]{};
    uint64_t early_lengths[4]{};
    SlicedEllSlice slices[2]{};
    SlicedEllRow map[4]{}, logical[4]{};
    alignas(32) unsigned char packet[160]{};
    const CellRange ranges[]{{3,1},{1,1},{2,1}};
    const FetchRequest req{ranges,3,0,0,map,4};
    FetchTicket ticket{};
    FetchSlotStorage slot{{},packet,32};
    FetchOptions policy{FetchFormat::sliced_ell,1,{2,4,true,true}};
    const FetchWorkspace work{raw.buffers(),runs,4,lengths,order,early_lengths,logical,4,slices,2};
    uint64_t time = 0;
    const FetchClock clock{&time, [](void *value) noexcept { return ++*static_cast<uint64_t *>(value); }};
    require(queue.prepare(source.adapter(true), worker.worker(), clock, policy, work, &slot, 1) ==
            status::success, "SELL preparation");
    worker.blocked = true;
    require(queue.fetch(req,&ticket) == status::success && queue.poll(ticket).state == FetchState::pending,
            "executor rejection can retry");
    worker.blocked = false;
    queue.poll(ticket);
    worker.finish();
    auto result = queue.poll(ticket);
    require(result.state == FetchState::failed && result.status == status::capacity_exceeded &&
            result.batch.required.packet_bytes == 96 && !result.batch.packet_data &&
            source.reads == 0 && source.probes == 3, "early exact packet size before decoding");
    require(queue.release(ticket) == status::success && queue.drain(), "release sizing failure");

    slot.packet_bytes = sizeof(packet);
    require(queue.prepare(source.adapter(false), worker.worker(), clock, policy, work, &slot, 1) ==
            status::success, "reprepare using decoded offsets");
    require(queue.fetch(req,&ticket) == status::success, "SELL submission");
    require(queue.configure(1,false) == status::success, "change only future submissions");
    worker.finish();
    result = queue.poll(ticket);
    require(result.state == FetchState::ready && result.batch.packet_layout.bytes() == 96 && result.batch.time.packing_ns &&
            result.batch.time.read_ns && result.batch.time.planning_ns, "packed and timed packet");
    cellerator::tensor::base::device::SlicedEllView view{};
    require(result.batch.packet_layout.bind(packet,sizeof(packet),&view) == status::success, "packet binding");
    require(view.rows == 3 && view.nnz == 5 && view.slice_widths[0] == 3 && view.slice_widths[1] == 0 &&
            view.col_idx[3] == 2 && view.col_idx[4] == 2 && __builtin_bit_cast(uint16_t,view.val[3]) == 0 &&
            view.col_idx[5] == UINT32_MAX && __builtin_bit_cast(uint16_t,view.val[5]) == 0 &&
            map[0].source_cell == 1 && map[1].source_cell == 2 && map[2].source_cell == 3, "row order, zeros and padding");
    require(queue.release(ticket) == status::success, "release packed slot");
    source.fail = true;
    require(queue.fetch(req,&ticket) == status::success, "failing read admitted");
    worker.finish();
    result = queue.poll(ticket);
    require(result.state == FetchState::failed && !result.batch.packet_data && !result.batch.row_map,
            "partial read unpublished");
    require(queue.release(ticket) == status::success, "release partial read");
    source.fail = false; source.overflow = true;
    require(queue.fetch(req,&ticket) == status::success, "overflow read admitted");
    worker.finish();
    result = queue.poll(ticket);
    require(result.state == FetchState::failed && result.status == status::arithmetic_overflow &&
            !result.batch.packet_data && !result.batch.row_map, "rounding overflow unpublished");
    require(queue.release(ticket) == status::success && queue.drain(), "clean final drain");

    source.overflow = false; source.negative_index = true;
    auto wide = source.adapter(false);
    wide.info.features = UINT32_MAX;
    require(queue.prepare(wide, worker.worker(), clock, policy, work, &slot, 1) == status::success,
            "wide extent preparation");
    require(queue.fetch(req,&ticket) == status::success, "negative index read admitted");
    worker.finish();
    result = queue.poll(ticket);
    require(result.state == FetchState::failed && result.status == status::invalid_argument &&
            !result.batch.packet_data, "signed negative index cannot become a valid unsigned column");
    require(queue.release(ticket) == status::success && queue.drain(), "release invalid column");
}

void placement() {
    PacketPlacement p{};
    require(place_packet(100,60,40,true,10,&p) == status::success &&
            p.offset == 60 && p.early_offset == 0 && p.early_bytes == 40 && p.deferred_bytes == 0,
            "complementary high placement");
    require(place_packet(100,60,60,true,10,&p) == status::success &&
            p.offset == 40 && p.early_offset == 20 && p.early_bytes == 40 && p.deferred_bytes == 20,
            "high copy suffix");
    require(place_packet(100,60,60,false,10,&p) == status::success &&
            p.offset == 0 && p.early_offset == 0 && p.early_bytes == 40 && p.deferred_bytes == 20,
            "low copy prefix");
    require(place_packet(100,60,60,true,41,&p) == status::success &&
            p.early_bytes == 0 && p.deferred_bytes == 60, "defer for small gains");
}

void packet_cuts() {
    using namespace cellerator::execution::planning;
    PacketCutChoice choice{};
    const PacketCut cuts[]{{1,30,70},{2,40,60},{3,50,50},{4,105,25}};
    require(choose_packet_cut(5,100,50,60,40,cuts,4,&choice) == status::success &&
            choice.cut.left_slices == 2 && !choice.wait_before && !choice.wait_between && !choice.wait_after,
            "complementary whole-slice cut");
    require(choose_packet_cut(5,100,50,75,75,cuts,4,&choice) == status::success &&
            choice.cut.left_slices == 3 && choice.wait_before == 25 && choice.wait_after == 25,
            "equal deferred bytes prefer byte target");
    const PacketCut ties[]{{3,50,50},{2,50,50},{1,60,60}};
    require(choose_packet_cut(5,100,50,0,0,ties,3,&choice) == status::success &&
            choice.cut.left_slices == 2, "internal handoff and deterministic tie");
    require(choose_packet_cut(5,100,50,0,0,cuts+3,1,&choice) == status::capacity_exceeded &&
            choice.cut.left_slices == 2, "no legal cut does not publish a replacement");
}

void tune_useful_windows() {
    using namespace cellerator::execution::tuning;
    FetchTuner tuner;
    require(tuner.prepare({}, {}) && tuner.observe({tuner.revision(),1000,16,4096,32768}) == FetchTuneDecision::unchanged,
            "fixed default");
    FetchTuneOptions opt;
    opt.adaptive = true; opt.may_reorder = true; opt.split_threshold = 128;
    require(tuner.prepare({},opt), "enable trials");
    auto observe = [&](uint64_t ns, uint64_t entries = 32768) {
        return tuner.observe({tuner.revision(),ns,16,4096,entries});
    };
    auto original = tuner.revision();
    require(observe(1000) == FetchTuneDecision::trial && tuner.schedule().lookahead == 2 &&
            !tuner.schedule().reorder_rows && tuner.schedule().minimum_early_bytes == SIZE_MAX, "lookahead only");
    require(tuner.observe({original,500,16,4096,32768}) == FetchTuneDecision::unchanged, "ignore stale window");
    require(observe(850) == FetchTuneDecision::accepted && tuner.schedule().lookahead == 2, "retain full-window gain");
    require(observe(855) == FetchTuneDecision::trial && tuner.schedule().reorder_rows, "try permitted row ordering");
    require(observe(600,15000) == FetchTuneDecision::incomparable && !tuner.schedule().reorder_rows,
            "reject faster but lighter window");
    require(observe(855) == FetchTuneDecision::trial && tuner.schedule().minimum_early_bytes == 128,
            "try split threshold");
    require(observe(850) == FetchTuneDecision::restored && tuner.schedule().minimum_early_bytes == SIZE_MAX &&
            tuner.schedule().lookahead == 2, "restore small or noisy change");
    require(observe(1000) == FetchTuneDecision::unchanged, "bounded trials");
}

} // namespace

int main() {
    native_and_drain();
    sell_plan_and_failure();
    placement();
    packet_cuts();
    tune_useful_windows();
}
