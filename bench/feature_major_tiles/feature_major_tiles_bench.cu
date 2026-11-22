#include "../../tests/tensor/feature_major_tiles_fixture.hh"

#include "csr_cusparse.cuh"
#include "csr_multiply.cuh"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string_view>

namespace {

using namespace cellerator::test::feature_tiles;
namespace vendor = matrix::cusparse;
namespace tensor = cellerator::tensor;
using Clock = std::chrono::steady_clock;

double milliseconds(Clock::time_point begin, Clock::time_point end = Clock::now()) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

void check_sparse(cusparseStatus_t status, const char* what) {
    if (status != CUSPARSE_STATUS_SUCCESS)
        throw std::runtime_error(std::string(what) + ": " + cusparseGetErrorString(status));
}

Support read_support(const char* path) {
    std::ifstream file(path, std::ios::binary);
    uint32_t header[3]{};
    file.read(reinterpret_cast<char*>(header), sizeof(header));
    require(bool(file) && header[0] && header[1] && header[2], "invalid benchmark CSR header");
    Support support{header[0], header[1], std::vector<uint32_t>(size_t(header[0]) + 1),
                    std::vector<uint32_t>(header[2])};
    file.read(reinterpret_cast<char*>(support.offsets.data()),
              std::streamsize(support.offsets.size() * sizeof(uint32_t)));
    file.read(reinterpret_cast<char*>(support.columns.data()),
              std::streamsize(support.columns.size() * sizeof(uint32_t)));
    require(bool(file) && file.peek() == std::ifstream::traits_type::eof(),
            "unexpected CSR payload length");
    require(support.offsets.front() == 0 && support.offsets.back() == header[2],
            "invalid CSR offset bounds");
    for (uint32_t row = 0; row < support.rows; ++row) {
        uint32_t begin = support.offsets[row], end = support.offsets[row + 1];
        require(begin <= end && end <= header[2], "invalid CSR row offset");
        for (uint32_t e = begin; e < end; ++e)
            require(support.columns[e] < support.features &&
                    (e == begin || support.columns[e - 1] < support.columns[e]),
                    "benchmark requires sorted unique CSR columns");
    }
    return support;
}

enum class Mode { learn, refresh, frozen };

__global__ void csr_samples(const uint32_t* offsets, const uint32_t* columns,
                            const float* g, const float* w, float* dx) {
    const uint32_t row = blockIdx.x;
    const uint32_t lane = threadIdx.x % width;
    const uint32_t group = threadIdx.x / width;
    const uint32_t end = offsets[row + 1];
    for (uint32_t e = offsets[row] + group; e < end; e += blockDim.x / width) {
        float sum = g[size_t(row) * width + lane] * w[size_t(columns[e]) * width + lane];
        const uint32_t mask = (threadIdx.x & 16) ? 0xffff0000u : 0xffffu;
        for (uint32_t delta = 8; delta; delta >>= 1)
            sum += __shfl_down_sync(mask, sum, delta, width);
        if (lane == 0) dx[e] = sum;
    }
}

__global__ void per_entry_transpose(uint32_t features, const uint32_t* offsets,
                                    const uint32_t* rows, const uint32_t* positions,
                                    const __half* x, const float* g, float* dw) {
    const uint32_t feature = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const uint32_t lane = threadIdx.x % width;
    if (feature >= features) return;
    const uint32_t end = offsets[feature + 1];
    float sum = 0.0f;
    for (uint32_t e = offsets[feature] + ((threadIdx.x & 16) != 0); e < end; e += 2)
        sum += __half2float(x[positions[e]]) * g[size_t(rows[e]) * width + lane];
    sum += __shfl_xor_sync(0xffffffffu, sum, 16);
    if ((threadIdx.x & 16) == 0) dw[size_t(feature) * width + lane] = sum;
}

__global__ void csr_update(uint32_t nnz, float* master, __half* values, float* expanded,
                           const float* dx, const __half* external) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnz) return;
    if (dx) master[i] -= learning_rate * dx[i];
    const __half v = external ? external[i] : __float2half_rn(master[i]);
    if (values) values[i] = v;
    if (expanded) expanded[i] = __half2float(v);
}

struct Common {
    const Stream stream;
    DeviceArray<float> w, dw, y, target, g, dx, master, initial_x;
    DeviceArray<__half> external_a, external_b;
    uint32_t nnz;
    bool use_b = false;

    Common(const Parameters& p, Mode mode)
        : w(p.w.size()), dw(p.w.size()), y(p.target.size()), target(p.target.size()),
          g(p.target.size()), dx(p.x.size()), master(p.x.size()), initial_x(p.x.size()),
          external_a(mode == Mode::refresh ? p.x.size() : 0),
          external_b(external_a.count()), nnz(uint32_t(p.x.size())) {
        initial_x.upload(p.x, stream);
        target.upload(p.target, stream);
        std::vector<__half> source(external_a.count());
        for (size_t i = 0; i < source.size(); ++i) source[i] = refreshed(i, false);
        external_a.upload(source, stream);
        stream.sync();
        for (size_t i = 0; i < source.size(); ++i) source[i] = refreshed(i, true);
        external_b.upload(source, stream);
        stream.sync();
    }

    static __half refreshed(size_t i, bool b) {
        return __float2half_rn(i ? parameter(i, b ? 11 : 23, 31, 0.004f) : 0.0f);
    }

    const __half* next_external() {
        const auto* source = use_b ? external_b.get() : external_a.get();
        use_b = !use_b;
        return source;
    }

    size_t bytes() const {
        return w.bytes() + dw.bytes() + y.bytes() + target.bytes() + g.bytes() + dx.bytes()
            + master.bytes() + initial_x.bytes() + external_a.bytes() + external_b.bytes();
    }
};

struct Tiled {
    HostTiles host;
    DeviceTiles tiles;
    DeviceArray<__half> values;
    DeviceArray<float> partials;
    FeatureMajorTilesView<__half> write;
    FeatureMajorTilesView<const __half> read;

    Tiled(const Support& support, const Stream& stream, FeatureMajorTileGrouping grouping)
        : host(support, grouping), tiles(host, stream), values(support.columns.size()),
          partials(matrix::feature_major_tiles_transpose_partial_count(tiles.index)),
          write{tiles.structure, values.get()},
          read{tiles.structure, values.get()} {}

    void reset(Common& p) {
        check(matrix::feature_major_tiles_initialize_master(write, tiles.index, p.initial_x.get(),
              p.master.get(), p.stream), "tile initialize");
    }

    void forward(Common& p) {
        check(matrix::feature_major_tiles_spmm(read, tiles.index, p.w.get(), p.y.get(),
              p.stream), "tile forward");
    }

    void weight_gradient(Common& p) {
        check(matrix::feature_major_tiles_spmm_transpose(read, tiles.index, p.g.get(), p.dw.get(),
              partials.get(), partials.count(), p.stream), "tile weight gradient");
    }

    void sparse_gradient(Common& p) {
        check(matrix::feature_major_tiles_sampled_gradient(tiles.structure, tiles.index, p.g.get(),
              p.w.get(), p.dx.get(), p.stream), "tile sampled gradient");
    }

    void update(Common& p, Mode mode) {
        if (mode == Mode::learn)
            check(matrix::feature_major_tiles_sparse_sgd(write, p.master.get(), p.dx.get(),
                  learning_rate, p.stream), "tile SGD");
        if (mode == Mode::refresh)
            check(matrix::feature_major_tiles_refresh(write, tiles.index, p.next_external(),
                  p.stream), "tile refresh");
    }

    std::vector<float> csr_order(Common& p, const float* packed) {
        DeviceArray<float> exported(p.nnz);
        check(matrix::feature_major_tiles_export_gradient(tiles.structure, tiles.index,
              packed, exported.get(), p.stream), "tile export gradient or master");
        return exported.read(p.stream);
    }

    bool has_value_update(Mode mode) const { return mode != Mode::frozen; }

    uint32_t feature_records() const { return tiles.structure.record_count; }

    size_t bytes() const { return tiles.bytes() + values.bytes() + partials.bytes(); }
};

struct SparseHandle {
    cusparseHandle_t handle = nullptr;
    ~SparseHandle() { if (handle) cusparseDestroy(handle); }
};

struct Csr {
    const uint32_t rows, features, nnz;
    const bool library;
    DeviceArray<uint32_t> offsets, columns, reverse_offsets, reverse_rows, reverse_positions;
    DeviceArray<__half> values;
    DeviceArray<float> expanded;
    const __half* current_values = nullptr;
    SparseHandle sparse;
    std::unique_ptr<DeviceArray<uint8_t>> w_forward, w_reverse, w_sampled;
    vendor::spmm_plan forward_plan, reverse_plan;
    vendor::sddmm_plan sampled_plan;

    Csr(const Support& support, Common& p, bool prepared_transpose, bool library,
        bool alternate_library_alg)
        : rows(support.rows), features(support.features), nnz(uint32_t(support.columns.size())),
          library(library), offsets(support.offsets.size()), columns(nnz),
          reverse_offsets(prepared_transpose ? size_t(features) + 1 : 0),
          reverse_rows(prepared_transpose ? nnz : 0), reverse_positions(prepared_transpose ? nnz : 0),
          values(library ? 0 : nnz), expanded(library ? nnz : 0) {
        offsets.upload(support.offsets, p.stream);
        columns.upload(support.columns, p.stream);
        if (prepared_transpose) {
            std::vector<uint32_t> starts(size_t(features) + 1), rs(nnz), ps(nnz);
            for (uint32_t feature : support.columns) ++starts[feature + 1];
            for (uint32_t f = 0; f < features; ++f) starts[f + 1] += starts[f];
            auto ends = starts;
            for (uint32_t r = 0; r < rows; ++r)
                for (uint32_t e = support.offsets[r]; e < support.offsets[r + 1]; ++e) {
                    const uint32_t position = ends[support.columns[e]]++;
                    rs[position] = r;
                    ps[position] = e;
                }
            reverse_offsets.upload(starts, p.stream);
            reverse_rows.upload(rs, p.stream);
            reverse_positions.upload(ps, p.stream);
            p.stream.sync();
        }
        if (library) prepare_library(p, alternate_library_alg);
    }

    void prepare_library(Common& p, bool alternate) {
        check_sparse(cusparseCreate(&sparse.handle), "create cuSPARSE handle");
        check_sparse(cusparseSetStream(sparse.handle, p.stream), "set cuSPARSE stream");
        const tensor::base::CompressedStructureView<int32_t> structure{
            rows, features, nnz, tensor::base::CompressedAxis::row,
            reinterpret_cast<const int32_t*>(offsets.get()),
            reinterpret_cast<const int32_t*>(columns.get())};
        const auto alg = alternate ? CUSPARSE_SPMM_CSR_ALG2 : CUSPARSE_SPMM_CSR_ALG1;
        check_sparse(forward_plan.provision(sparse.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            {structure, expanded.get()}, CUDA_R_32F, {p.w.get(), features, width, width},
            {p.y.get(), rows, width, width}, alg), "prepare cuSPARSE forward");
        check_sparse(reverse_plan.provision(sparse.handle, CUSPARSE_OPERATION_TRANSPOSE,
            {structure, expanded.get()}, CUDA_R_32F, {p.g.get(), rows, width, width},
            {p.dw.get(), features, width, width}, alg), "prepare cuSPARSE transpose");
        check_sparse(sampled_plan.provision(sparse.handle, {structure, p.dx.get()},
            {p.g.get(), rows, width, width}, {p.w.get(), features, width, width}),
            "prepare cuSPARSE sampled gradient");
        w_forward = std::make_unique<DeviceArray<uint8_t>>(forward_plan.workspace_bytes());
        w_reverse = std::make_unique<DeviceArray<uint8_t>>(reverse_plan.workspace_bytes());
        w_sampled = std::make_unique<DeviceArray<uint8_t>>(sampled_plan.workspace_bytes());
        check_sparse(forward_plan.bind_workspace(w_forward->get(), w_forward->bytes(),
            !alternate), "bind cuSPARSE forward workspace");
        check_sparse(reverse_plan.bind_workspace(w_reverse->get(), w_reverse->bytes(),
            !alternate), "bind cuSPARSE transpose workspace");
        check_sparse(sampled_plan.bind_workspace(w_sampled->get(), w_sampled->bytes()),
                     "bind cuSPARSE sampled-gradient workspace");
    }

    matrix::Csr<const __half> view() const {
        return {{rows, features, nnz, tensor::base::CompressedAxis::row, offsets.get(),
                 columns.get()}, current_values};
    }

    void refresh(Common& p, const float* dx, const __half* source) {
        csr_update<<<(nnz + 255) / 256, 256, 0, p.stream>>>(nnz, p.master.get(), values.get(),
                     expanded.get(), dx, source);
        check(cudaGetLastError(), "CSR update and cast");
    }

    void reset(Common& p) {
        check(cudaMemcpyAsync(p.master.get(), p.initial_x.get(), p.master.bytes(),
              cudaMemcpyDeviceToDevice, p.stream), "set CSR master");
        current_values = values.get();
        refresh(p, nullptr, nullptr);
    }

    void forward(Common& p) {
        if (library) check_sparse(forward_plan.enqueue(), "cuSPARSE forward");
        else check(matrix::csr_spmm(view(), p.w.get(), width, width, p.y.get(), width,
                                    p.stream), "native CSR forward");
    }

    void weight_gradient(Common& p) {
        if (library) {
            check_sparse(reverse_plan.enqueue(), "cuSPARSE transpose");
            return;
        }
        if (reverse_offsets.count()) {
            per_entry_transpose<<<(features + 3) / 4, 128, 0, p.stream>>>(
                features, reverse_offsets.get(), reverse_rows.get(), reverse_positions.get(),
                current_values, p.g.get(), p.dw.get());
            check(cudaGetLastError(), "prepared CSR transpose");
        } else {
            check(matrix::csr_spmm_transpose(view(), p.g.get(), width, width, p.dw.get(),
                  width, p.stream), "native CSR transpose");
        }
    }

    void sparse_gradient(Common& p) {
        if (library) {
            check_sparse(sampled_plan.enqueue(), "cuSPARSE sampled gradient");
            return;
        }
        csr_samples<<<rows, 128, 0, p.stream>>>(offsets.get(), columns.get(), p.g.get(),
                                               p.w.get(), p.dx.get());
        check(cudaGetLastError(), "native CSR sampled gradient");
    }

    void update(Common& p, Mode mode) {
        if (mode == Mode::learn) refresh(p, p.dx.get(), nullptr);
        if (mode == Mode::refresh) {
            const auto* external = p.next_external();
            if (library) refresh(p, nullptr, external);
            else current_values = external;
        }
    }

    std::vector<float> csr_order(Common& p, const float* data) {
        return (data == p.dx.get() ? p.dx : p.master).read(p.stream);
    }

    bool has_value_update(Mode mode) const {
        return mode == Mode::learn || (mode == Mode::refresh && library);
    }

    uint32_t feature_records() const { return 0; }

    size_t bytes() const {
        return offsets.bytes() + columns.bytes() + reverse_offsets.bytes() + reverse_rows.bytes()
            + reverse_positions.bytes() + values.bytes() + expanded.bytes()
            + (library ? w_forward->bytes() + w_reverse->bytes() + w_sampled->bytes() : 0);
    }
};

struct Event {
    cudaEvent_t event = nullptr;
    Event() { check(cudaEventCreate(&event), "create timing event"); }
    ~Event() { cudaEventDestroy(event); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};

struct Times { double gpu, wall; };

template<class Candidate, class Work>
Times time(Candidate& route, Common& problem, Work work, uint32_t steps) {
    Event start, stop;
    const auto begin = Clock::now();
    check(cudaEventRecord(start.event, problem.stream), "record timer start");
    for (uint32_t step = 0; step < steps; ++step) work(route, problem);
    check(cudaEventRecord(stop.event, problem.stream), "record timer stop");
    check(cudaEventSynchronize(stop.event), "wait timer stop");
    const double wall = milliseconds(begin);
    float elapsed = 0;
    check(cudaEventElapsedTime(&elapsed, start.event, stop.event), "read elapsed time");
    return {elapsed / steps, wall / steps};
}

template<class Candidate>
void reset(Candidate& route, Common& p, const Parameters& initial) {
    p.w.upload(initial.w, p.stream);
    p.use_b = false;
    route.reset(p);
    p.stream.sync();
}

template<class Candidate>
void train(Candidate& route, Common& p, Mode mode) {
    route.forward(p);
    loss(p.y, p.target, p.g, p.stream);
    route.weight_gradient(p);
    route.sparse_gradient(p);
    route.update(p, mode);
    update_w(p.w, p.dw, p.stream);
}

template<class Candidate>
void run(Candidate& route, Common& p, const Support& support, const Parameters& initial,
         Mode mode, uint32_t reuse, uint32_t warmups, uint32_t repeats,
         Clock::time_point before_setup, const cudaDeviceProp& gpu) {
    reset(route, p, initial);
    const double setup_ms = milliseconds(before_setup);
    auto expected = initial;
    for (int validation_step = 0; validation_step < 3; ++validation_step) {
        const auto ref = reference(support, expected);
        train(route, p, mode);
        compare(p.y.read(p.stream), ref.y, "forward", 1e-5);
        compare(p.g.read(p.stream), ref.g, "loss gradient", 1e-5);
        compare(p.dw.read(p.stream), ref.dw, "dense weight gradient", 1e-5);
        compare(route.csr_order(p, p.dx.get()), ref.dx, "sampled sparse gradient", 1e-5);
        for (size_t i = 0; i < expected.w.size(); ++i)
            expected.w[i] -= learning_rate * float(ref.dw[i]);
        if (mode == Mode::learn)
            for (size_t i = 0; i < expected.x.size(); ++i)
                expected.x[i] -= learning_rate * float(ref.dx[i]);
        if (mode == Mode::refresh)
            for (size_t i = 0; i < expected.x.size(); ++i)
                expected.x[i] = __half2float(Common::refreshed(i, (validation_step % 2) != 0));
        compare(p.w.read(p.stream), expected.w, "weight update", 1e-6);
        compare(route.csr_order(p, p.master.get()), mode == Mode::learn ? expected.x : initial.x,
            "sparse master update", 1e-6);
    }

    reset(route, p, initial);
    const auto step = [mode](Candidate& r, Common& common) { train(r, common, mode); };
    for (uint32_t i = 0; i < warmups; ++i) step(route, p);
    p.stream.sync();

    std::cout << std::setprecision(10) << "{\"setup_ms\":" << setup_ms
              << ",\"compute_capability\":" << (gpu.major * 10 + gpu.minor)
              << ",\"gpu_uuid_bytes\":\"";
    for (char byte : gpu.uuid.bytes)
        std::cout << std::hex << std::setw(2) << std::setfill('0') << uint32_t(uint8_t(byte));
    std::cout << std::dec << std::setfill(' ') << "\""
              << ",\"feature_records\":" << route.feature_records()
              << ",\"live_device_array_bytes\":" << (route.bytes() + p.bytes())
              << ",\"horizons\":{";
    const uint32_t horizons[3] = {1, 8, 64};
    bool first = true;
    for (uint32_t horizon : horizons) {
        if (reuse && reuse != horizon) continue;
        if (!first) std::cout << ',';
        first = false;
        std::cout << '"' << horizon << "\":[";
        for (uint32_t i = 0; i < repeats; ++i) {
            reset(route, p, initial);
            const auto [device, wall] = time(route, p, step, horizon);
            if (i) std::cout << ',';
            std::cout << "{\"gpu_step_ms\":" << device << ",\"wall_step_ms\":" << wall
                      << ",\"amortized_step_ms\":" << (wall + setup_ms / horizon) << '}';
        }
        std::cout << ']';
    }

    reset(route, p, initial);
    route.forward(p);
    loss(p.y, p.target, p.g, p.stream);
    route.weight_gradient(p);
    route.sparse_gradient(p);
    p.stream.sync();
    const auto forward = [](Candidate& r, Common& common) { r.forward(common); };
    const auto backward = [](Candidate& r, Common& common) {
        r.weight_gradient(common);
        r.sparse_gradient(common);
    };
    const auto weight_gradient = [](Candidate& r, Common& common) { r.weight_gradient(common); };
    const auto sparse_gradient = [](Candidate& r, Common& common) { r.sparse_gradient(common); };
    const auto value_update = [mode](Candidate& r, Common& common) { r.update(common, mode); };
    const double forward_ms = time(route, p, forward, 16).gpu;
    const double backward_ms = time(route, p, backward, 16).gpu;
    const double weight_gradient_ms = time(route, p, weight_gradient, 16).gpu;
    const double sparse_gradient_ms = time(route, p, sparse_gradient, 16).gpu;
    const double refresh_ms = route.has_value_update(mode) ? time(route, p, value_update, 16).gpu : 0;
    std::cout << "},\"forward_ms\":" << forward_ms << ",\"backward_ms\":" << backward_ms
              << ",\"weight_gradient_ms\":" << weight_gradient_ms
              << ",\"sparse_gradient_ms\":" << sparse_gradient_ms
              << ",\"sparse_update_ms\":" << refresh_ms << ",\"oracle\":\"pass\"}\n";
}

uint32_t parse_count(const char* text, uint32_t min, uint32_t max) {
    size_t end = 0;
    const auto count = std::stoul(text, &end);
    require(text[end] == '\0' && count >= min && count <= max, "invalid benchmark count");
    return static_cast<uint32_t>(count);
}

} // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 7, "usage: feature-major-tiles-bench csr-file route mode reuse warmups repeats");
        const std::string_view route = argv[2], raw_mode = argv[3];
        require(raw_mode == "learn" || raw_mode == "refresh" || raw_mode == "frozen", "invalid update mode");
        const Mode mode = raw_mode == "learn" ? Mode::learn : raw_mode == "refresh" ? Mode::refresh : Mode::frozen;
        const uint32_t reuse = parse_count(argv[4], 0, 64), warmups = parse_count(argv[5], 0, 100);
        require(reuse == 0 || reuse == 1 || reuse == 8 || reuse == 64, "unsupported reuse horizon");
        const uint32_t repeats = parse_count(argv[6], 1, 31);
        auto support = read_support(argv[1]);
        auto initial = parameters(support);
        check(cudaFree(nullptr), "initialize CUDA context");
        int device = 0;
        cudaDeviceProp gpu{};
        check(cudaGetDevice(&device), "current device");
        check(cudaGetDeviceProperties(&gpu, device), "read current GPU identity");
        const auto before_setup = Clock::now();
        Common p(initial, mode);
        if (route == "contiguous" || route == "grouped") {
            Tiled candidate(support, p.stream, route == "contiguous" ? FeatureMajorTileGrouping::contiguous
                : FeatureMajorTileGrouping::local_feature_hash);
            run(candidate, p, support, initial, mode, reuse, warmups, repeats, before_setup, gpu);
        } else {
            const bool library = route == "cusparse" || route == "cusparse-alg2";
            require(library || route == "csr" || route == "csr-transpose", "invalid benchmark route");
            Csr candidate(support, p, route == "csr-transpose", library, route == "cusparse-alg2");
            run(candidate, p, support, initial, mode, reuse, warmups, repeats, before_setup, gpu);
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
