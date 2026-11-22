#include "../../tests/tensor/feature_major_tiles_fixture.hh"

#include "csr_cusparse.cuh"
#include "csr_multiply.cuh"

#include <algorithm>
#include <bit>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <string_view>

namespace {

using namespace cellerator::test::feature_tiles;
namespace vendor = matrix::cusparse;
namespace tensor = cellerator::tensor;
using Clock = std::chrono::steady_clock;
constexpr uint32_t threads = 256;

// Dense rows are features and columns are cells. The structural matrix is a
// single shared target-by-TF relation, with one FP32 master per supported edge.
struct Dataset {
    uint32_t cells, genes;
    Support relation;
    std::vector<float> initial, expression;
    std::vector<uint32_t> targets, tfs, order;

    size_t batches() const { return order.size() / width; }
    uint32_t active(size_t batch) const {
        return uint32_t(std::min(size_t(width), size_t(cells) - batch * width));
    }
};

template<class T>
void read(std::ifstream& in, std::vector<T>& values) {
    in.read(reinterpret_cast<char*>(values.data()), std::streamsize(values.size() * sizeof(T)));
    require(bool(in), "truncated dataset fixture");
}

Dataset load(const char* file) {
    static_assert(std::endian::native == std::endian::little);
    std::ifstream in(file, std::ios::binary | std::ios::ate);
    const auto bytes = in.tellg();
    in.seekg(0);
    uint32_t h[8]{};
    in.read(reinterpret_cast<char*>(h), sizeof(h));
    require(bool(in) && h[0] == 0x31445443 && h[1] && h[2] && h[3] && h[4] && h[5]
            && h[5] <= INT32_MAX && h[6] == 0 && h[7] == 0, "invalid dataset header");
    const size_t matrix_words = size_t(h[1]) * h[2];
    const size_t other_words = 2 * size_t(h[3]) + h[4] + 2 * size_t(h[5]) + 1;
    require(bytes >= std::streamoff(sizeof(h)) && h[3] <= h[2] && h[4] <= h[2]
            && (bytes - std::streamoff(sizeof(h))) % 4 == 0
            && matrix_words <= size_t((bytes - std::streamoff(sizeof(h))) / 4)
            && other_words == size_t((bytes - std::streamoff(sizeof(h))) / 4) - matrix_words,
            "dataset size does not match header");

    Dataset d{h[1], h[2], {h[3], h[4], std::vector<uint32_t>(size_t(h[3]) + 1),
                std::vector<uint32_t>(h[5])}, std::vector<float>(h[5]),
              std::vector<float>(matrix_words), std::vector<uint32_t>(h[3]),
              std::vector<uint32_t>(h[4]),
              std::vector<uint32_t>((size_t(h[1]) + width - 1) / width * width, UINT32_MAX)};
    auto& a = d.relation;
    read(in, a.offsets);
    read(in, a.columns);
    read(in, d.initial);
    read(in, d.targets);
    read(in, d.tfs);
    read(in, d.expression);
    require(a.offsets.front() == 0 && a.offsets.back() == h[5], "invalid CSR endpoints");
    for (uint32_t row = 0; row < a.rows; ++row) {
        uint32_t begin = a.offsets[row], end = a.offsets[row + 1];
        require(begin <= end && end <= h[5], "invalid CSR interval");
        for (uint32_t e = begin; e < end; ++e)
            require(a.columns[e] < a.features && (e == begin || a.columns[e - 1] < a.columns[e]),
                    "expected sorted unique CSR columns");
    }
    for (auto g : d.targets) require(g < d.genes, "invalid target feature map");
    for (auto g : d.tfs) require(g < d.genes, "invalid TF feature map");
    for (float v : d.initial)
        require(std::isfinite(__half2float(__float2half_rn(v))), "invalid initial sparse weight");
    for (float v : d.expression) require(std::isfinite(v), "nonfinite observation");
    auto end = d.order.begin() + d.cells;
    std::iota(d.order.begin(), end, 0u);
    std::mt19937 rng(613);
    std::shuffle(d.order.begin(), end, rng);
    return d;
}

__global__ void gather_examples(const float* expression, uint32_t genes,
    const uint32_t* cells, const uint32_t* tfs, uint32_t n_tf, const uint32_t* targets,
    uint32_t n_target, float* x, float* target) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t row = i / width;
    if (row >= size_t(n_tf) + n_target) return;
    const uint32_t cell = cells[i % width];
    if (row < n_tf) {
        x[i] = cell != UINT32_MAX ? expression[size_t(cell) * genes + tfs[row]] : 0.0f;
    } else {
        const size_t t = row - n_tf;
        target[t * width + i % width] =
            cell != UINT32_MAX ? expression[size_t(cell) * genes + targets[t]] : 0.0f;
    }
}

__global__ void loss_cotangent(const float* y, const float* target, float* g, size_t count,
                               uint32_t active, float scale) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count)
        g[i] = i % width < active ? (y[i] - target[i]) * scale : 0.0f;
}

__global__ void csr_samples(const uint32_t* offsets, const uint32_t* columns,
                            const float* g, const float* x, float* da) {
    const uint32_t row = blockIdx.x;
    const uint32_t lane = threadIdx.x % width;
    const uint32_t group = threadIdx.x / width;
    const uint32_t end = offsets[row + 1];
    for (uint32_t e = offsets[row] + group; e < end; e += blockDim.x / width) {
        float sum = g[size_t(row) * width + lane] * x[size_t(columns[e]) * width + lane];
        const uint32_t mask = (threadIdx.x & 16) ? 0xffff0000u : 0xffffu;
        for (uint32_t delta = 8; delta; delta >>= 1)
            sum += __shfl_down_sync(mask, sum, delta, width);
        if (lane == 0) da[e] = sum;
    }
}

__global__ void csr_update(uint32_t nnz, float* master, __half* values, float* expanded,
                           const float* da) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnz) return;
    if (da) master[i] -= learning_rate * da[i];
    const __half a = __float2half_rn(master[i]);
    if (values) values[i] = a;
    if (expanded) expanded[i] = __half2float(a);
}

__device__ uint32_t record_tile(FeatureMajorTileStructureView s, uint32_t record) {
    uint32_t first = 0, last = s.tile_count;
    while (last - first > 1) {
        const uint32_t mid = first + (last - first) / 2;
        if (s.tile_offsets[mid] <= record) first = mid;
        else last = mid;
    }
    return first;
}

// Keep the native packed values. Each half warp combines the same cotangent
// and active row with the input and learned edge value for both gradients.
__global__ void fused_tile_gradients(FeatureMajorTilesView<const __half> a,
    const uint32_t* logical_rows, const float* g, const float* x, float* dx, float* da) {
    constexpr uint32_t warp_mask = 0xffffffffu;
    constexpr uint32_t records_per_warp = 4;
    const uint32_t lane = threadIdx.x % 32, column = lane % width, half = lane / width;
    const uint32_t warp = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const uint32_t begin = records_per_warp * warp;
    const auto s = a.structure;
    if (begin >= s.record_count) return;
    const uint32_t end = begin + min(records_per_warp, s.record_count - begin);
    uint32_t tile = lane == 0 ? record_tile(s, begin) : 0;
    tile = __shfl_sync(warp_mask, tile, 0);
    for (uint32_t r = begin; r < end; ++r) {
        if (r == s.tile_offsets[tile + 1]) {
            if (lane == 0) tile = record_tile(s, r);
            tile = __shfl_sync(warp_mask, tile, 0);
        }
        const uint32_t feature = s.feature_ids[r];
        const size_t feature_slot = size_t(feature) * width + column;
        const float input = x[feature_slot];
        uint32_t mask = s.row_masks[r], p = s.value_offsets[r];
        float input_gradient = 0;
        while (mask) {
            const uint32_t remaining = mask & (mask - 1);
            const uint32_t active = half ? remaining : mask;
            float cotangent = 0, edge = 0;
            if (active) {
                const uint32_t row = tile * s.tile_row_width + uint32_t(__ffs(int(active)) - 1);
                const uint32_t logical = logical_rows ? logical_rows[row] : row;
                cotangent = g[size_t(logical) * width + column];
                if (column == 0) edge = __half2float(a.values[p + half]);
            }
            edge = __shfl_sync(warp_mask, edge, 0, width);
            input_gradient += edge * cotangent;
            float dot = input * cotangent;
            for (uint32_t delta = width / 2; delta; delta /= 2)
                dot += __shfl_down_sync(warp_mask, dot, delta, width);
            if (column == 0 && active) da[p + half] = dot;
            p += 2;
            mask = remaining & (remaining - 1);
        }
        input_gradient += __shfl_xor_sync(warp_mask, input_gradient, width);
        if (half == 0) atomicAdd(dx + feature_slot, input_gradient);
    }
}

struct Common {
    const Dataset& data;
    const Stream stream;
    DeviceArray<float> expression, initial, x, target, y, g, dx, da, master;
    DeviceArray<uint32_t> tfs, targets, order;

    explicit Common(const Dataset& d)
        : data(d), expression(d.expression.size()), initial(d.initial.size()),
          x(d.tfs.size() * width), target(d.targets.size() * width), y(target.count()),
          g(target.count()), dx(x.count()), da(d.initial.size()), master(d.initial.size()),
          tfs(d.tfs.size()), targets(d.targets.size()), order(d.order.size()) {
        expression.upload(d.expression, stream);
        initial.upload(d.initial, stream);
        tfs.upload(d.tfs, stream);
        targets.upload(d.targets, stream);
        order.upload(d.order, stream);
        stream.sync();
    }

    void gather(size_t batch) {
        const size_t n = x.count() + target.count();
        gather_examples<<<(n + threads - 1) / threads, threads, 0, stream>>>(
            expression.get(), data.genes, order.get() + batch * width, tfs.get(),
            uint32_t(tfs.count()), targets.get(), uint32_t(targets.count()), x.get(), target.get());
        check(cudaGetLastError(), "gather cell examples");
    }

    void loss(size_t batch) {
        const uint32_t active = data.active(batch);
        const size_t n = y.count();
        loss_cotangent<<<(n + threads - 1) / threads, threads, 0, stream>>>(
            y.get(), target.get(), g.get(), n, active, 1.0f / float(active * data.targets.size()));
        check(cudaGetLastError(), "form squared-loss cotangent");
    }

    size_t bytes() const {
        return expression.bytes() + initial.bytes() + x.bytes() + target.bytes() + y.bytes()
            + g.bytes() + dx.bytes() + da.bytes() + master.bytes() + tfs.bytes()
            + targets.bytes() + order.bytes();
    }
};

struct Tiled {
    const HostTiles host;
    DeviceTiles packed;
    DeviceArray<__half> values;
    DeviceArray<float> scratch;
    FeatureMajorTilesView<__half> write;
    FeatureMajorTilesView<const __half> read;
    const bool fused;

    explicit Tiled(Common& p, bool use_fused)
        : host(p.data.relation, FeatureMajorTileGrouping::contiguous),
          packed(host, p.stream), values(p.da.count()),
          scratch(use_fused ? 0 : matrix::feature_major_tiles_transpose_partial_count(packed.index)),
          write{packed.structure, values.get()}, read{packed.structure, values.get()}, fused(use_fused) {}

    void reset(Common& p) {
        check(matrix::feature_major_tiles_initialize_master(write, packed.index,
              p.initial.get(), p.master.get(), p.stream), "gather initial packed weights");
    }

    void forward(Common& p) {
        check(matrix::feature_major_tiles_spmm(read, packed.index, p.x.get(), p.y.get(),
              p.stream), "tiled forward");
    }

    void backward(Common& p) {
        if (fused) {
            constexpr unsigned lanes = 128, records_per_block = lanes / 32 * 4;
            check(cudaMemsetAsync(p.dx.get(), 0, p.dx.bytes(), p.stream), "clear fused input gradient");
            const uint32_t records = packed.structure.record_count;
            fused_tile_gradients<<<(records + records_per_block - 1) / records_per_block,
                lanes, 0, p.stream>>>(read, packed.index.logical_rows, p.g.get(), p.x.get(),
                                     p.dx.get(), p.da.get());
            check(cudaGetLastError(), "fused tiled input and edge gradients");
            return;
        }
        check(matrix::feature_major_tiles_spmm_transpose(read, packed.index, p.g.get(),
              p.dx.get(), scratch.get(), scratch.count(), p.stream), "tiled input gradient");
        check(matrix::feature_major_tiles_sampled_gradient(packed.structure, packed.index,
              p.g.get(), p.x.get(), p.da.get(), p.stream), "tiled edge gradient");
    }

    void update(Common& p) {
        check(matrix::feature_major_tiles_sparse_sgd(write, p.master.get(), p.da.get(),
              learning_rate, p.stream), "tiled sparse SGD");
    }

    std::vector<float> csr_order(Common& p, const DeviceArray<float>& input) {
        DeviceArray<float> output(input.count());
        check(matrix::feature_major_tiles_export_gradient(packed.structure, packed.index,
              input.get(), output.get(), p.stream), "export canonical edge values");
        return output.read(p.stream);
    }

    size_t bytes() const { return packed.bytes() + values.bytes() + scratch.bytes(); }
};

void check_sparse(cusparseStatus_t s, const char* what) {
    if (s != CUSPARSE_STATUS_SUCCESS)
        throw std::runtime_error(std::string(what) + ": " + cusparseGetErrorString(s));
}

struct SparseHandle {
    cusparseHandle_t value = nullptr;
    ~SparseHandle() { if (value) cusparseDestroy(value); }
};

struct Csr {
    const uint32_t rows, features, nnz;
    const bool library;
    DeviceArray<uint32_t> offsets, columns;
    DeviceArray<__half> values;
    DeviceArray<float> expanded;
    SparseHandle sparse;
    std::unique_ptr<DeviceArray<uint8_t>> fwd_work, reverse_work, sample_work;
    vendor::spmm_plan fwd, reverse;
    vendor::sddmm_plan sample;

    Csr(Common& p, bool use_library)
        : rows(p.data.relation.rows), features(p.data.relation.features),
          nnz(uint32_t(p.da.count())), library(use_library),
          offsets(p.data.relation.offsets.size()), columns(nnz), values(library ? 0 : nnz),
          expanded(library ? nnz : 0) {
        offsets.upload(p.data.relation.offsets, p.stream);
        columns.upload(p.data.relation.columns, p.stream);
        if (!library) return;
        check_sparse(cusparseCreate(&sparse.value), "create cuSPARSE handle");
        check_sparse(cusparseSetStream(sparse.value, p.stream), "set cuSPARSE stream");
        const tensor::base::CompressedStructureView<int32_t> s{
            rows, features, nnz, tensor::base::CompressedAxis::row,
            reinterpret_cast<const int32_t*>(offsets.get()),
            reinterpret_cast<const int32_t*>(columns.get())};
        const auto alg = CUSPARSE_SPMM_CSR_ALG2;
        check_sparse(fwd.provision(sparse.value, CUSPARSE_OPERATION_NON_TRANSPOSE,
            {s, expanded.get()}, CUDA_R_32F, {p.x.get(), features, width, width},
            {p.y.get(), rows, width, width}, alg), "prepare cuSPARSE forward");
        check_sparse(reverse.provision(sparse.value, CUSPARSE_OPERATION_TRANSPOSE,
            {s, expanded.get()}, CUDA_R_32F, {p.g.get(), rows, width, width},
            {p.dx.get(), features, width, width}, alg), "prepare cuSPARSE transpose");
        check_sparse(sample.provision(sparse.value, {s, p.da.get()},
            {p.g.get(), rows, width, width}, {p.x.get(), features, width, width}),
            "prepare cuSPARSE sampled gradient");
        fwd_work = std::make_unique<DeviceArray<uint8_t>>(fwd.workspace_bytes());
        reverse_work = std::make_unique<DeviceArray<uint8_t>>(reverse.workspace_bytes());
        sample_work = std::make_unique<DeviceArray<uint8_t>>(sample.workspace_bytes());
        check_sparse(fwd.bind_workspace(fwd_work->get(), fwd_work->bytes()), "bind forward workspace");
        check_sparse(reverse.bind_workspace(reverse_work->get(), reverse_work->bytes()),
                     "bind transpose workspace");
        check_sparse(sample.bind_workspace(sample_work->get(), sample_work->bytes()),
                     "bind sampled-gradient workspace");
    }

    matrix::Csr<const __half> view() const {
        return {{rows, features, nnz, tensor::base::CompressedAxis::row, offsets.get(),
                 columns.get()}, values.get()};
    }

    void cast(Common& p, const float* da) {
        csr_update<<<(nnz + threads - 1) / threads, threads, 0, p.stream>>>(
            nnz, p.master.get(), values.get(), expanded.get(), da);
        check(cudaGetLastError(), "CSR sparse SGD and cast");
    }

    void reset(Common& p) {
        check(cudaMemcpyAsync(p.master.get(), p.initial.get(), p.master.bytes(),
              cudaMemcpyDeviceToDevice, p.stream), "copy initial CSR master");
        cast(p, nullptr);
    }

    void forward(Common& p) {
        if (library) check_sparse(fwd.enqueue(), "cuSPARSE forward");
        else check(matrix::csr_spmm(view(), p.x.get(), width, width, p.y.get(), width,
                                    p.stream), "CSR forward");
    }

    void backward(Common& p) {
        if (library) {
            check_sparse(reverse.enqueue(), "cuSPARSE input gradient");
            check_sparse(sample.enqueue(), "cuSPARSE edge gradient");
        } else {
            check(matrix::csr_spmm_transpose(view(), p.g.get(), width, width, p.dx.get(),
                  width, p.stream), "CSR input gradient");
            csr_samples<<<rows, 128, 0, p.stream>>>(offsets.get(), columns.get(), p.g.get(),
                                                   p.x.get(), p.da.get());
            check(cudaGetLastError(), "CSR sampled gradient");
        }
    }

    void update(Common& p) { cast(p, p.da.get()); }

    std::vector<float> csr_order(Common& p, const DeviceArray<float>& input) {
        return input.read(p.stream);
    }

    size_t bytes() const {
        return offsets.bytes() + columns.bytes() + values.bytes() + expanded.bytes() +
            (library ? fwd_work->bytes() + reverse_work->bytes() + sample_work->bytes() : 0);
    }
};

template<class Route>
void oracle(Route& route, Common& p, size_t batch, std::vector<float>& master) {
    const auto& d = p.data;
    const auto& a = d.relation;
    const uint32_t active = d.active(batch);
    std::vector<double> x(p.x.count()), target(p.target.count()), y(p.y.count()), g(p.g.count()),
                        dx(p.dx.count()), da(p.da.count());
    for (uint32_t c = 0; c < active; ++c) {
        const uint32_t cell = d.order[batch * width + c];
        const auto* values = d.expression.data() + size_t(cell) * d.genes;
        for (uint32_t f = 0; f < a.features; ++f) x[size_t(f) * width + c] = values[d.tfs[f]];
        for (uint32_t r = 0; r < a.rows; ++r) target[size_t(r) * width + c] = values[d.targets[r]];
    }
    for (uint32_t r = 0; r < a.rows; ++r)
        for (uint32_t e = a.offsets[r]; e < a.offsets[r + 1]; ++e) {
            const double weight = __half2float(__float2half_rn(master[e]));
            for (uint32_t c = 0; c < active; ++c)
                y[size_t(r) * width + c] += weight * x[size_t(a.columns[e]) * width + c];
        }
    for (uint32_t r = 0; r < a.rows; ++r)
        for (uint32_t c = 0; c < active; ++c) {
            const size_t i = size_t(r) * width + c;
            g[i] = (y[i] - target[i]) / (double(active) * a.rows);
        }
    for (uint32_t r = 0; r < a.rows; ++r)
        for (uint32_t e = a.offsets[r]; e < a.offsets[r + 1]; ++e) {
            const double weight = __half2float(__float2half_rn(master[e]));
            for (uint32_t c = 0; c < active; ++c) {
                const size_t i = size_t(a.columns[e]) * width + c;
                const double cotangent = g[size_t(r) * width + c];
                dx[i] += weight * cotangent;
                da[e] += cotangent * x[i];
            }
        }

    p.gather(batch);
    route.forward(p);
    p.loss(batch);
    route.backward(p);
    compare(p.x.read(p.stream), x, "gathered TF state", 0, 0);
    compare(p.target.read(p.stream), target, "gathered measured targets", 0, 0);
    compare(p.y.read(p.stream), y, "target prediction", 4e-6, 1e-4);
    compare(p.g.read(p.stream), g, "target cotangent");
    compare(p.dx.read(p.stream), dx, "TF input gradient");
    compare(route.csr_order(p, p.da), da, "shared edge gradient");
    route.update(p);
    for (size_t e = 0; e < master.size(); ++e)
        master[e] -= learning_rate * float(da[e]);
    compare(route.csr_order(p, p.master), master, "updated shared edge weights", 2e-7, 1e-5);
}

class Event {
public:
    Event() { check(cudaEventCreate(&event), "create timing event"); }
    ~Event() { cudaEventDestroy(event); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;

    void record(cudaStream_t stream) const { check(cudaEventRecord(event, stream), "record event"); }
    double since(const Event& begin) const {
        float elapsed = 0;
        check(cudaEventElapsedTime(&elapsed, begin.event, event), "read event interval");
        return elapsed;
    }

private:
    cudaEvent_t event = nullptr;
};

struct PhaseTimes {
    Event begin, gathered, forwarded, differentiated, updated;
};

template<bool update_weights = true, class Route>
void epoch(Route& route, Common& p, PhaseTimes* phases = nullptr) {
    for (size_t b = 0; b < p.data.batches(); ++b) {
        PhaseTimes* times = phases ? phases + b : nullptr;
        if (times) times->begin.record(p.stream);
        p.gather(b);
        if (times) times->gathered.record(p.stream);
        route.forward(p);
        if (times) times->forwarded.record(p.stream);
        p.loss(b);
        route.backward(p);
        if (times) times->differentiated.record(p.stream);
        if constexpr (update_weights) route.update(p);
        if (times) times->updated.record(p.stream);
    }
}

void samples(const char* label, const std::vector<double>& ms) {
    auto sorted = ms;
    std::sort(sorted.begin(), sorted.end());
    const size_t mid = sorted.size() / 2;
    const double median = sorted.size() % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
    std::cout << ",\"" << label << "\":{\"median\":" << median << ",\"samples\":[";
    for (size_t i = 0; i < ms.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << ms[i];
    }
    std::cout << "]}";
}

template<class Route>
void run(Route& route, Common& p, Clock::time_point start, double upload_ms,
         const cudaDeviceProp& device, std::string_view backend, uint32_t epochs, uint32_t repeats) {
    route.reset(p);
    p.stream.sync();
    const double setup_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    auto canonical_master = p.data.initial;
    oracle(route, p, 0, canonical_master);
    oracle(route, p, p.data.batches() - 1, canonical_master);

    std::vector<double> wall(repeats), gpu(repeats), requested_gpu(repeats), requested_wall(repeats);
    const double steps = double(epochs) * double(p.data.batches());
    Event start_gpu, stop_gpu;
    auto measure = [&](bool update_weights, double& wall_ms, double& gpu_ms) {
        route.reset(p);
        p.stream.sync();
        auto start_wall = Clock::now();
        start_gpu.record(p.stream);
        if (update_weights) {
            for (uint32_t e = 0; e < epochs; ++e) epoch(route, p);
        } else {
            for (uint32_t e = 0; e < epochs; ++e) epoch<false>(route, p);
        }
        stop_gpu.record(p.stream);
        p.stream.sync();
        wall_ms = std::chrono::duration<double, std::milli>(Clock::now() - start_wall).count() / steps;
        gpu_ms = stop_gpu.since(start_gpu) / steps;
    };
    for (uint32_t repeat = 0; repeat < repeats; ++repeat) {
        route.reset(p);
        epoch(route, p);
        measure(false, requested_wall[repeat], requested_gpu[repeat]);
        measure(true, wall[repeat], gpu[repeat]);
    }

    const size_t batches = p.data.batches();
    auto phases = std::make_unique<PhaseTimes[]>(batches);
    route.reset(p);
    epoch(route, p, phases.get());
    p.stream.sync();
    double batch_gather = 0, forward = 0, backward = 0, update = 0, requested = 0;
    for (size_t i = 0; i < batches; ++i) {
        const auto& times = phases[i];
        batch_gather += times.gathered.since(times.begin);
        forward += times.forwarded.since(times.gathered);
        backward += times.differentiated.since(times.forwarded);
        update += times.updated.since(times.differentiated);
        requested += times.differentiated.since(times.begin);
    }

    std::cout << std::setprecision(10) << "{\"route\":\"" << backend
              << "\",\"oracle\":\"pass\",\"setup_ms\":" << setup_ms
              << ",\"dataset_upload_ms\":" << upload_ms
              << ",\"layout_and_route_setup_ms\":" << setup_ms - upload_ms
              << ",\"live_bytes\":" << route.bytes() + p.bytes() << ",\"gpu_uuid\":\"";
    constexpr char hex[] = "0123456789abcdef";
    for (auto raw : device.uuid.bytes) {
        const auto byte = static_cast<uint8_t>(raw);
        std::cout << hex[byte >> 4] << hex[byte & 15];
    }
    std::cout << "\",\"compute_capability\":" << device.major * 10 + device.minor
              << ",\"cells\":" << p.data.cells << ",\"genes\":" << p.data.genes
              << ",\"targets\":" << p.data.targets.size() << ",\"tfs\":" << p.data.tfs.size()
              << ",\"edges\":" << p.data.initial.size() << ",\"batches_per_epoch\":" << batches
              << ",\"epochs\":" << epochs << ",\"shuffle_seed\":613"
              << ",\"diagnostic_ms_per_batch\":{\"gather\":" << batch_gather / batches
              << ",\"forward\":" << forward / batches << ",\"backward_with_loss\":"
              << backward / batches << ",\"sgd_and_cast\":" << update / batches
              << ",\"gather_forward_backward\":" << requested / batches << "}";
    samples("training_host_ms_per_batch", wall);
    samples("training_gpu_ms_per_batch", gpu);
    samples("gather_forward_backward_gpu_ms_per_batch", requested_gpu);
    samples("gather_forward_backward_host_ms_per_batch", requested_wall);
    std::cout << "}\n";
}

uint32_t bounded(const char* value, unsigned limit) {
    size_t end = 0;
    const auto n = std::stoul(value, &end);
    require(n > 0 && n <= limit && value[end] == '\0', "invalid bounded run count");
    return uint32_t(n);
}

} // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 5, "usage: dataset-training-bench fixture tiled|tiled-fused|csr|cusparse epochs repeats");
        const std::string_view backend = argv[2];
        const bool fused = backend == "tiled-fused";
        require(backend == "tiled" || fused || backend == "csr" || backend == "cusparse",
                "invalid route");
        const uint32_t epochs = bounded(argv[3], 1000), repeats = bounded(argv[4], 32);
        const Dataset dataset = load(argv[1]);
        check(cudaFree(nullptr), "initialize CUDA context");
        int ordinal = 0;
        cudaDeviceProp device{};
        check(cudaGetDevice(&ordinal), "current CUDA device");
        check(cudaGetDeviceProperties(&device, ordinal), "get GPU identity");
        auto start = Clock::now();
        Common problem(dataset);
        const double upload_ms =
            std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        if (backend == "tiled" || fused) {
            Tiled route(problem, fused);
            run(route, problem, start, upload_ms, device, backend, epochs, repeats);
        } else {
            Csr route(problem, backend == "cusparse");
            run(route, problem, start, upload_ms, device, backend, epochs, repeats);
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
