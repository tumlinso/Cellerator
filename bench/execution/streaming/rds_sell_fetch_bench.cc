#include "../../benchmark_mutex.hh"

#include <Cellerator/common/real/fp16.hh>
#include <Cellerator/compute/operators/sparse/column_moments.cuh>
#include <Cellerator/execution/streaming/cuda_staging.cuh>
#include <Cellerator/memory/view.hh>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <vector>

import cellerator.rds;

namespace {
namespace rds = cellerator::rds;
namespace real = cellerator::real;
namespace memory = cellerator::memory;
namespace gpu = memory::cuda;
namespace sparse = cellerator::compute::sparse;
namespace layout = cellerator::layout;
using namespace cellerator::execution::streaming;
using memory::status;
using Clock = std::chrono::steady_clock;

constexpr size_t max_cells = 32768, max_packets = 8;
constexpr size_t host_budget = size_t(1) << 30, device_budget = size_t(1) << 29;
constexpr size_t split_threshold = 256 * 1024;
constexpr layout::SlicedEllPackOptions packing{16, 256, true, true};

void check(bool ok, const char *operation) {
    if (!ok) throw std::runtime_error(operation);
}

void check(status result, const char *operation) {
    if (result != status::success)
        throw std::runtime_error(std::string(operation) + " failed with status " + std::to_string(int(result)));
}

void check(rds::Status result, const char *operation) {
    if (result != rds::Status::ok)
        throw std::runtime_error(std::string(operation) + " failed with status " + std::to_string(int(result)));
}

uint64_t now_ns(void *) noexcept {
    return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count());
}

size_t argument(const char *text) {
    const std::string_view token(text);
    size_t n = 0;
    const auto result = std::from_chars(token.data(), token.data()+token.size(), n);
    check(result.ec == std::errc{} && result.ptr == token.data()+token.size(), "invalid numeric argument");
    return n;
}

// The benchmark supplies one real dispatch thread. A single mailbox suffices
// because FetchQueue admits at most one source job at a time.
struct Worker {
    std::mutex mutex;
    std::condition_variable cv;
    FetchJob pending = nullptr;
    void *argument = nullptr;
    bool stopping = false;
    std::jthread thread{[this] { run(); }};

    ~Worker() { stop(); }

    void stop() {
        {
            std::lock_guard lock(mutex);
            stopping = true;
        }
        cv.notify_one();
        if (thread.joinable()) thread.join();
    }

    void run() {
        std::unique_lock lock(mutex);
        for (;;) {
            cv.wait(lock, [&] { return pending || stopping; });
            if (!pending) return;
            const auto job = pending;
            void *arg = argument;
            pending = nullptr;
            lock.unlock();
            job(arg);
            lock.lock();
        }
    }

    static bool submit(void *data, FetchJob job, void *arg) noexcept {
        auto &worker = *static_cast<Worker *>(data);
        {
            std::lock_guard lock(worker.mutex);
            if (worker.pending || worker.stopping) return false;
            worker.pending = job;
            worker.argument = arg;
        }
        worker.cv.notify_one();
        return true;
    }
};

struct Batch {
    CellRange range{};
    size_t stored = 0, padding = 0, bytes = 0;
};

struct Samples {
    uint64_t elapsed_ns = 0, compressed = 0, decoded = 0;
    uint64_t packed = 0, early = 0, deferred = 0, stored = 0, padding = 0;
    FetchTimes worker{};
};

struct Buffers {
    std::vector<double> values;
    std::vector<int32_t> indices, offsets;
    std::vector<uint64_t> source_lengths;
    std::vector<unsigned> lengths, order;
    std::vector<layout::SlicedEllSlice> slices;
    std::vector<layout::SlicedEllRow> logical, maps;
    FetchRun run{};
    gpu::PinnedBuffer pinned[2];
    gpu::PinnedBuffer host_output;
    gpu::DeviceBuffer output;
    memory::StripedBuffer arena;

    Buffers(size_t entries, size_t rows, size_t total_rows, size_t packet, size_t output_count)
        : values(entries), indices(entries), offsets(rows+1), source_lengths(rows), lengths(rows), order(rows),
          slices((rows+packing.slice_rows-1)/packing.slice_rows), logical(rows), maps(total_rows),
          pinned{gpu::PinnedBuffer(packet),gpu::PinnedBuffer(packet)},
          host_output(output_count*sizeof(float)), output(host_output.bytes()), arena(packet+packet/64*32) {}

    SparseBuffers raw() noexcept {
        return {values.data(),indices.data(),offsets.data(),values.size()*sizeof(double),
                indices.size()*sizeof(int32_t),offsets.size()*sizeof(int32_t)};
    }

    size_t raw_bytes() const noexcept {
        return values.capacity()*sizeof(double) + (indices.capacity()+offsets.capacity())*sizeof(int32_t);
    }

    size_t planning_bytes() const noexcept {
        return sizeof(run) + source_lengths.capacity()*sizeof(uint64_t) +
            (lengths.capacity()+order.capacity())*sizeof(unsigned) +
            slices.capacity()*sizeof(layout::SlicedEllSlice) +
            (logical.capacity()+maps.capacity())*sizeof(layout::SlicedEllRow);
    }
};

std::vector<double> reference(const rds::RdsFile &file, const std::vector<Batch> &batches,
                              Buffers &buffers, unsigned feature_count) {
    rds::ReadContext context;
    check(context.prepare_transfer(file,rds::TransferMode::numeric), "prepare scalar reference");
    rds::FetchReader reader(file,context,1);
    const auto source = reader.source();
    std::vector<double> expected(3*size_t(feature_count)+1);
    for (const auto &batch : batches) {
        ReadStats ignored{};
        check(source.read_into(source.context,batch.range,buffers.raw(),&ignored), "read scalar reference");
        for (size_t i = 0; i < batch.stored; ++i) {
            const auto col = size_t(buffers.indices[i]);
            const double v = real::load_f16(real::round_f16(buffers.values[i]));
            check(col < feature_count && std::isfinite(v), "out-of-domain value in scalar reference");
            if (v == 0) continue;
            expected[col] += v;
            expected[feature_count+col] += v*v;
            ++expected[2*feature_count+col];
        }
        expected[3*size_t(feature_count)] += double(batch.range.count);
    }
    return expected;
}

void verify(const std::vector<Batch> &batches, const Buffers &buffers, const std::vector<double> &expected,
            unsigned features) {
    const auto *actual = static_cast<const float *>(buffers.host_output.data());
    for (size_t i = 0; i < expected.size(); ++i) {
        const double tolerance = i < 2*size_t(features) ? 1e-5 + 2e-4*std::abs(expected[i]) : 0;
        check(std::abs(double(actual[i])-expected[i]) <= tolerance, "column moments differ from scalar reference");
    }

    std::vector<unsigned char> seen(buffers.maps.size());
    size_t first = 0;
    for (const auto &batch : batches) {
        for (size_t row = 0; row < batch.range.count; ++row) {
            const auto &map = buffers.maps[first+row];
            check(map.logical_position >= first && map.logical_position-first < batch.range.count &&
                  map.source_cell == batch.range.first+map.logical_position-first && !seen[map.logical_position],
                  "packed row map lost an occurrence");
            seen[map.logical_position] = 1;
        }
        first += size_t(batch.range.count);
    }
    check(first == seen.size(), "row count mismatch");
}

Samples run(bool eager, FetchSource source, const std::vector<Batch> &batches, Buffers &buf, Worker &worker,
            const gpu::Stream &copy, const gpu::Stream &compute, const StageEvents (&events)[2]) {
    FetchQueue queue;
    CudaStaging staging;
    const FetchWorkspace workspace{buf.raw(),&buf.run,1,buf.lengths.data(),buf.order.data(),
        buf.source_lengths.data(),buf.logical.data(),buf.lengths.size(),buf.slices.data(),buf.slices.size()};
    const FetchSlotStorage slots[2]{{{},buf.pinned[0].data(),buf.pinned[0].bytes()},
                                    {{},buf.pinned[1].data(),buf.pinned[1].bytes()}};
    check(queue.prepare(source,{&worker,Worker::submit},{nullptr,now_ns},
          {FetchFormat::sliced_ell,eager ? 2u : 1u,packing},workspace,slots,2), "prepare fetch");
    check(staging.prepare(queue,buf.arena,copy,events,split_threshold), "prepare device staging");
    const auto features = unsigned(source.info.features);
    auto *p = static_cast<float *>(buf.output.data());
    sparse::column_moments_view moments{features,p,p+features,p+2*features,p+3*features};
    check(sparse::zero_column_moments(&moments,compute.get()), "clear moments");
    compute.synchronize();
    FetchTicket host[max_packets]{};
    DeviceTicket device[max_packets]{};
    Samples samples;
    const uint64_t start = now_ns(nullptr);
    try {
        size_t submitted = 0, staged = 0, launched = 0, first = 0;
        while (launched < batches.size()) {
            if (launched) staging.poll(device[launched-1]);
            bool progress = false;
            while (submitted < batches.size() && (eager || submitted == launched)) {
                const auto &batch = batches[submitted];
                const auto result = queue.fetch({&batch.range,1,submitted,first,buf.maps.data()+first,
                                                 size_t(batch.range.count)}, &host[submitted]);
                if (result == status::capacity_exceeded) break;
                check(result,"submit read");
                first += size_t(batch.range.count);
                ++submitted;
                progress = true;
            }
            while (staged < submitted) {
                const auto ready = queue.poll(host[staged]);
                if (ready.state == FetchState::pending) break;
                check(ready.status,"read and pack");
                check(ready.state == FetchState::ready, "unexpected fetch state");
                const auto result = staging.stage(host[staged],&device[staged]);
                if (result == status::capacity_exceeded) break;
                check(result,"stage SELL");
                const auto &batch = batches[staged];
                check(ready.batch.required.rows == batch.range.count && ready.batch.required.entries == batch.stored &&
                      ready.batch.required.packet_bytes == batch.bytes, "unexpected SELL shape");
                samples.packed += ready.batch.required.packet_bytes;
                samples.compressed += ready.batch.read.compressed_bytes;
                samples.decoded += ready.batch.read.decoded_bytes;
                samples.worker.planning_ns += ready.batch.time.planning_ns;
                samples.worker.read_ns += ready.batch.time.read_ns;
                samples.worker.packing_ns += ready.batch.time.packing_ns;
                samples.stored += batch.stored;
                samples.padding += batch.padding;
                ++staged;
                progress = true;
            }
            while (launched < staged) {
                const auto ready = staging.poll(device[launched]);
                if (ready.state == FetchState::pending) break;
                check(ready.status,"wait for SELL packet");
                check(ready.state == FetchState::ready && ready.batch.batch == launched,"unexpected device batch");
                samples.early += ready.batch.placement.early_bytes;
                samples.deferred += ready.batch.placement.deferred_bytes;
                compute.wait(*ready.batch.ready);
                check(sparse::accumulate_column_moments(&ready.batch.view,nullptr,&moments,compute.get()),"launch column moments");
                check(staging.release(device[launched],&compute),"record last reader");
                if (!eager) {
                    compute.synchronize();
                    staging.poll(device[launched]); // Reclaim this packet before submitting the next read.
                }
                ++launched;
                progress = true;
            }
            if (!progress) {
                check(now_ns(nullptr)-start < 120ull*1000*1000*1000, "timed out waiting for fetch progress");
                std::this_thread::yield();
            }
        }
        compute.synchronize();
        check(staging.drain(),"drain copies");
        check(queue.drain(), "drain fetch queue");
        samples.elapsed_ns = now_ns(nullptr)-start;
    } catch (...) {
        worker.stop();
        cudaStreamSynchronize(copy.get());
        cudaStreamSynchronize(compute.get());
        throw;
    }
    buf.output.download_async(buf.host_output.data(),buf.host_output.bytes(),compute);
    compute.synchronize();
    return samples;
}

void print(const char *mode, Samples s, const Buffers &buf, size_t packets, size_t first, size_t rows,
           size_t features) {
    std::printf("{\"schedule\":\"%s\",\"cache_policy\":\"same_range_warmups\","
                "\"first_cell\":%zu,\"batches\":%zu,\"checked_rows\":%zu,\"checked_features\":%zu,"
                "\"elapsed_ns\":%llu,\"planning_ns\":%llu,\"read_ns\":%llu,\"packing_ns\":%llu,"
                "\"compressed_bytes\":%llu,\"decoded_bytes\":%llu,\"packed_bytes\":%llu,"
                "\"early_bytes\":%llu,\"deferred_bytes\":%llu,\"stored_entries\":%llu,\"padding_entries\":%llu,"
                "\"raw_scratch_bytes\":%zu,\"planning_bytes\":%zu,\"pinned_packet_bytes\":%zu,"
                "\"device_arena_bytes\":%zu,\"moments_bytes_per_side\":%zu,\"checked\":true}\n",
                mode,first,packets,rows,features,(unsigned long long)s.elapsed_ns,
                (unsigned long long)s.worker.planning_ns,(unsigned long long)s.worker.read_ns,
                (unsigned long long)s.worker.packing_ns,(unsigned long long)s.compressed,(unsigned long long)s.decoded,
                (unsigned long long)s.packed,(unsigned long long)s.early,(unsigned long long)s.deferred,
                (unsigned long long)s.stored,(unsigned long long)s.padding,
                buf.raw_bytes(),buf.planning_bytes(),
                buf.pinned[0].bytes()+buf.pinned[1].bytes(),buf.arena.bytes(),buf.output.bytes());
}
} // namespace

int main(int argc, char **argv) try {
    if (argc < 3 || argc > 6) {
        std::fprintf(stderr, "usage: %s SOURCE.rds INDEX [PACKETS=4] [CELLS_PER_PACKET=4096] [FIRST_CELL=0]\n",argv[0]);
        return 2;
    }
    const size_t packets = argc > 3 ? argument(argv[3]) : 4;
    const size_t cells = argc > 4 ? argument(argv[4]) : 4096;
    const size_t first = argc > 5 ? argument(argv[5]) : 0;
    check(packets && packets <= max_packets && cells && cells <= max_cells/packets, "probe exceeds packet or cell limit");
    rds::RdsFile file;
    check(file.open(argv[1],argv[2]),"open indexed RDS");
    const auto info = file.info();
    check(info.rows && first <= info.columns && packets*cells <= info.columns-first, "requested rows outside RDS");

    std::vector<Batch> batches(packets);
    size_t max_packet = 0, max_entries = 0;
    {
        std::vector<uint64_t> lengths64(cells);
        std::vector<unsigned> lengths(cells), order(cells);
        std::vector<layout::SlicedEllSlice> slices((cells+packing.slice_rows-1)/packing.slice_rows);
        for (size_t i = 0; i < packets; ++i) {
            const size_t start = first+i*cells;
            check(file.cell_lengths({uint32_t(start),uint32_t(cells)},
                  {lengths64.data(),cells,{memory::domain::host}}), "estimate packet bytes");
            for (size_t row = 0; row < cells; ++row) lengths[row] = unsigned(lengths64[row]);
            layout::SlicedEllPackPlan plan{};
            check(layout::plan_sliced_ell(lengths.data(),cells,info.rows,packing,order.data(),order.size(),
                  slices.data(),slices.size(),&plan), "plan indexed cell lengths");
            batches[i] = {{start,cells},plan.shape.nnz,plan.shape.slots-plan.shape.nnz,plan.layout.bytes()};
            max_packet = std::max(max_packet,batches[i].bytes);
            max_entries = std::max(max_entries,batches[i].stored);
        }
    }
    const size_t output_count = 3*size_t(info.rows)+1, output_bytes = output_count*sizeof(float);
    check(2*max_packet + max_entries*(sizeof(double)+sizeof(int32_t)) + output_bytes <= host_budget &&
          max_packet+max_packet/2 + output_bytes <= device_budget, "probe exceeds payload memory budget");
    cellerator::bench::benchmark_mutex_guard mutex("rds-sell-fetch");
    Buffers buffers(max_entries,cells,packets*cells,max_packet,output_count);
    const auto expected = reference(file,batches,buffers,info.rows);
    rds::ReadContext decode;
    check(decode.prepare_transfer(file,rds::TransferMode::numeric),"prepare worker reader");
    rds::FetchReader reader(file,decode,1);
    const auto source = reader.source();
    Worker worker;
    gpu::Stream copy, compute;
    gpu::Event ready[2], last[2];
    const StageEvents events[2]{{&ready[0],&last[0]},{&ready[1],&last[1]}};
    for (const bool eager : {false,true}) {
        run(eager,source,batches,buffers,worker,copy,compute,events);
        verify(batches,buffers,expected,info.rows);
    }
    for (const bool eager : {false,true}) {
        const auto samples = run(eager,source,batches,buffers,worker,copy,compute,events);
        verify(batches,buffers,expected,info.rows);
        print(eager ? "fixed_eager" : "serial",samples,buffers,packets,first,packets*cells,info.rows);
    }
} catch (const std::exception &error) {
    std::fprintf(stderr,"rds-sell-fetch: %s\n",error.what());
    return 1;
}
