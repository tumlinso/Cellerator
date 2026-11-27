#include <Cellerator/ops/matrix/sparse/column_moments.cuh>
#include <Cellerator/execution/streaming/cuda_staging.cuh>

#include <cstdio>
#include <cstdlib>

namespace memory = cellerator::memory;
namespace gpu = memory::cuda;
namespace sparse = cellerator::ops::matrix::sparse;
using namespace cellerator::execution::streaming;
using memory::status;

namespace {

void require(bool ok, const char *label) {
    if (ok) return;
    std::fprintf(stderr, "fetch CUDA staging: %s\n", label);
    std::exit(1);
}

struct Source {
    unsigned second_entries;

    static status requirements(void *context, CellRange range, uint64_t *n) noexcept {
        auto &source = *static_cast<Source *>(context);
        *n = range.first ? source.second_entries : 16;
        return status::success;
    }
    static status read(void *context, CellRange range, SparseBuffers raw, ReadStats *) noexcept {
        const auto &source = *static_cast<Source *>(context);
        const auto n = range.first ? source.second_entries : 16;
        auto *off = static_cast<uint32_t *>(raw.offsets);
        auto *idx = static_cast<uint32_t *>(raw.indices);
        auto *val = static_cast<uint16_t *>(raw.values);
        off[0] = 0; off[1] = n;
        for (unsigned i = 0; i < n; ++i) {
            idx[i] = i;
            val[i] = range.first ? 0x4000 : 0x3c00;
        }
        return status::success;
    }
    FetchSource adapter() noexcept {
        return {this,{2,16,55,ScalarType::f16,ScalarType::u32,ScalarType::u32},requirements,read,nullptr};
    }
};

void scenario(bool split) {
    Source source{split ? 16u : 8u};
    FetchQueue queue;
    uint16_t values[16]{};
    uint32_t offsets[2]{}, indices[16]{};
    const SparseBuffers scratch{values,indices,offsets,sizeof(values),sizeof(indices),sizeof(offsets)};
    FetchRun run{};
    unsigned length = 0, order = 0;
    cellerator::layout::SlicedEllRow logical{}, maps[2]{};
    cellerator::layout::SlicedEllSlice slice{};
    const FetchWorkspace work{scratch,&run,1,&length,&order,nullptr,&logical,1,&slice,1};
    gpu::PinnedBuffer host[2]{gpu::PinnedBuffer(128),gpu::PinnedBuffer(128)};
    const FetchSlotStorage slots[2]{{{},host[0].data(),host[0].bytes()},{{},host[1].data(),host[1].bytes()}};
    const FetchWorker inline_worker{nullptr, [](void *, FetchJob job, void *arg) noexcept { job(arg); return true; }};
    require(queue.prepare(source.adapter(),inline_worker,{}, {FetchFormat::sliced_ell,2,{}},work,slots,2)
            == status::success, "prepare host fetch");
    const CellRange ranges[2]{{0,1},{1,1}};
    FetchTicket packets[2]{};
    require(queue.fetch({&ranges[0],1,0,0,&maps[0],1}, &packets[0]) == status::success &&
            queue.fetch({&ranges[1],1,1,0,&maps[1],1}, &packets[1]) == status::success, "fetch both inputs");

    memory::StripedBuffer arena(split ? 192 : 224);
    gpu::Stream copy, compute;
    gpu::Event ready[2], last[2];
    gpu::Event early_begin[2]{gpu::Event(0,true),gpu::Event(0,true)};
    gpu::Event early_end[2]{gpu::Event(0,true),gpu::Event(0,true)};
    gpu::Event tail_begin[2]{gpu::Event(0,true),gpu::Event(0,true)};
    gpu::Event tail_end[2]{gpu::Event(0,true),gpu::Event(0,true)};
    const StageEvents events[2]{{&ready[0],&last[0],&early_begin[0],&early_end[0],&tail_begin[0],&tail_end[0]},
                               {&ready[1],&last[1],&early_begin[1],&early_end[1],&tail_begin[1],&tail_end[1]}};
    CudaStaging staging;
    require(staging.prepare(queue,arena,copy,events,32) == status::success, "prepare staging");
    DeviceTicket a{}, b{};
    require(staging.stage(packets[0],&a) == status::success &&
            staging.stage(packets[1],&b) == status::success, "submit early copies");
    const auto first = staging.poll(a);
    require(first.state == FetchState::ready && first.batch.placement.offset == 0 &&
            first.batch.placement.early_bytes == 128, "first packet layout");
    require(staging.poll(b).state == (split ? FetchState::pending : FetchState::ready), "copy dependency");
    copy.synchronize();
    require(staging.poll(a).state == FetchState::ready && queue.poll(packets[0]).state == FetchState::stale &&
            maps[0].source_cell == 0, "copied host bytes recycle independently of row map and device range");
    require(staging.drain() == status::invalid_state, "a leased device view remains live");

    constexpr size_t n = 16, span = n*3+1;
    gpu::DeviceBuffer output(2*span*sizeof(float));
    auto moments = [&](size_t offset) {
        auto *p = static_cast<float *>(output.data())+offset;
        return sparse::column_moments_view{n,p,p+n,p+2*n,p+3*n};
    };
    auto consume = [&](const StageReport &report, DeviceTicket ticket, size_t offset) {
        auto view = moments(offset);
        compute.wait(*report.batch.ready);
        require(sparse::zero_column_moments(&view,compute.get()) &&
                sparse::accumulate_column_moments(&report.batch.view,nullptr,&view,compute.get()),
                "existing SELL consumer");
        require(staging.release(ticket,&compute) == status::success, "record final reader");
    };
    consume(first,a,0);
    const auto second = staging.poll(b);
    require(second.state == FetchState::ready && second.batch.row_map[0].source_cell == 1 &&
            second.batch.placement.early_bytes == (split ? 64u : 96u) &&
            second.batch.placement.deferred_bytes == (split ? 64u : 0u) &&
            second.batch.placement.offset == (split ? 64u : 128u), "second packet placement");
    consume(second,b,span);
    gpu::PinnedBuffer out(output.bytes());
    output.download_async(out.data(),out.bytes(),compute);
    compute.synchronize();
    require(early_begin[b.slot].elapsed_ms(early_end[b.slot]) >= 0, "early copy timing");
    if (split) require(tail_begin[b.slot].elapsed_ms(tail_end[b.slot]) >= 0, "deferred copy timing");
    require(staging.drain() == status::success && queue.drain(), "drain both queues");
    auto *host_moments = static_cast<const float *>(out.data());
    for (size_t i = 0; i < n; ++i)
        require(host_moments[i] == 1 && host_moments[span+i] == (i < source.second_entries ? 2 : 0),
                "current and successor values");
}

} // namespace

int main() try {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    scenario(false);
    scenario(true);
} catch (const std::exception &error) {
    std::fprintf(stderr, "fetch CUDA staging: %s\n", error.what());
    return 1;
}
