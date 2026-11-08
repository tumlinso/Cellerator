#include <cuda_runtime_api.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

import velocity.cuda;

namespace {
namespace ac = anndata::cuda;
namespace vc = velocity::cuda;
using ac::ScalarType;

int failures = 0;

void expect(bool ok, const char* message) {
    if (ok) return;
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

void cuda_check(cudaError_t error, const char* message) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(error));
}

template<class T> constexpr ScalarType scalar_type();
template<> constexpr ScalarType scalar_type<float>() { return ScalarType::float32; }
template<> constexpr ScalarType scalar_type<anndata::Float16>() { return ScalarType::float16; }

template<class T> T store(float value) {
    if constexpr (std::is_same_v<T, anndata::Float16>)
        return anndata::Float16::from_double(value);
    else
        return value;
}

struct DeviceDense {
    ac::DeviceBuffer owner;
    ac::DenseView view;
};

template<class T>
DeviceDense upload_dense(const std::vector<T>& values, std::size_t rows,
                         std::size_t columns, ac::Stream stream) {
    DeviceDense out;
    out.owner = ac::DeviceBuffer(values.size() * sizeof(T), stream.device);
    out.view = {out.owner.span(), 0, rows, columns,
                static_cast<std::int64_t>(columns), 1, scalar_type<T>()};
    if (!values.empty())
        ac::copy_to_device(out.owner.span(), 0, values.data(), values.size() * sizeof(T), stream);
    return out;
}

template<class T>
DeviceDense upload_float_dense(const std::vector<float>& values, std::size_t rows,
                               std::size_t columns, ac::Stream stream) {
    std::vector<T> converted;
    converted.reserve(values.size());
    for (float value : values) converted.push_back(store<T>(value));
    auto out = upload_dense(converted, rows, columns, stream);
    // Keep the conversion buffer alive until its asynchronous upload is done.
    ac::synchronize(stream);
    return out;
}

struct DeviceOutput {
    ac::DeviceBuffer owner;
    ac::DenseView view;

    DeviceOutput(std::size_t rows, std::size_t columns, ScalarType type,
                 ac::Stream stream, std::int64_t row_stride = -1)
        : owner(rows * columns * ac::scalar_size(type), stream.device),
          view{owner.span(), 0, rows, columns,
               row_stride < 0 ? static_cast<std::int64_t>(columns) : row_stride,
               1, type} {}
};

std::vector<double> read_device(ac::DenseView source, ac::Stream stream) {
    const auto count = source.rows * source.columns;
    if (!count) return {};
    DeviceOutput converted(source.rows, source.columns, ScalarType::float64, stream);
    ac::cast(converted.view, source, stream);
    std::vector<double> result(count);
    ac::copy_to_host(result.data(), converted.owner.span(), 0,
                     count * sizeof(double), stream);
    ac::synchronize(stream);
    return result;
}

std::vector<double> read_host(anndata::ArrayView source) {
    const auto rows = source.extent(0);
    const auto columns = source.rank() == 2 ? source.extent(1) : 1;
    std::vector<double> result;
    result.reserve(rows * columns);
    const auto bytes = source.bytes();
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < columns; ++j) {
            const auto offset = std::ptrdiff_t(i) * bytes.stride_bytes(0) +
                (source.rank() == 2 ? std::ptrdiff_t(j) * bytes.stride_bytes(1) : 0);
            const auto* ptr = bytes.data() + offset;
            if (source.dtype() == anndata::DType::float64) {
                double value;
                std::memcpy(&value, ptr, sizeof(value));
                result.push_back(value);
            } else if (source.dtype() == anndata::DType::float32) {
                float value;
                std::memcpy(&value, ptr, sizeof(value));
                result.push_back(value);
            } else if (source.dtype() == anndata::DType::float16) {
                anndata::Float16 value;
                std::memcpy(&value, ptr, sizeof(value));
                result.push_back(value.to_float());
            } else {
                std::uint8_t value;
                std::memcpy(&value, ptr, sizeof(value));
                result.push_back(value);
            }
        }
    }
    return result;
}

void close(const std::vector<double>& actual, const std::vector<double>& expected,
           double tolerance, const char* message) {
    if (actual.size() != expected.size()) {
        expect(false, message);
        return;
    }
    double scale = 1;
    for (double value : expected) scale = std::max(scale, std::abs(value));
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) ||
            std::abs(actual[i] - expected[i]) > tolerance * (scale + std::abs(expected[i]))) {
            std::cerr << message << " index=" << i << " actual=" << actual[i]
                      << " expected=" << expected[i] << '\n';
            expect(false, message);
            return;
        }
    }
}

template<class T>
anndata::Array borrow_dense(const std::vector<T>& values, std::size_t rows,
                            std::size_t columns, std::ptrdiff_t row_stride = -1,
                            std::ptrdiff_t column_stride = 1) {
    return anndata::Array::borrow_dense(const_cast<T*>(values.data()), {rows, columns},
        {row_stride < 0 ? static_cast<std::ptrdiff_t>(columns) : row_stride, column_stride});
}

struct Fixture {
    std::size_t rows = 0, source_genes = 0;
    std::vector<float> weights, spliced, unspliced;
};

Fixture fixture(std::size_t rows, std::size_t genes) {
    Fixture out{rows, genes, std::vector<float>(rows * rows),
                std::vector<float>(rows * genes), std::vector<float>(rows * genes)};
    for (std::size_t i = 0; i < rows; ++i) {
        out.weights[i * rows + i] = .65f;
        out.weights[i * rows + (i + 1) % rows] += .35f;
        for (std::size_t j = 0; j < genes; ++j) {
            const float s = 1.f + .23f * float(i) + .17f * float(j) +
                .027f * float(i * j) + .009f * float(i * i + j * j);
            out.spliced[i * genes + j] = s;
            out.unspliced[i * genes + j] = (.43f + .045f * float(j)) * s +
                .11f * float((i + 2 * j) % 5) + .018f * float(i * i);
        }
    }
    return out;
}

velocity::FitOptions exact_fit() {
    velocity::FitOptions options;
    options.trimming = velocity::Trimming::none;
    options.min_r2 = -100;
    options.min_ratio = -100;
    options.low_signal = velocity::LowSignal::keep_filters;
    return options;
}

velocity::StochasticOptions exact_stochastic_fit() {
    velocity::StochasticOptions options;
    options.initialization = exact_fit();
    options.fit_second_intercept = true;
    options.refit_all_if_none = true;
    return options;
}

// The single host staging pair is reused only after its preceding H2D
// copy completes. Device buffers are reused on the same nondefault stream;
// stream order puts each new copy after the previous block's consumers.
template<class T>
struct ReusedBlockSource {
    std::size_t rows, genes, capacity;
    const std::vector<float>* spliced;
    const std::vector<float>* unspliced;
    ac::DeviceBuffer device_spliced, device_unspliced;
    std::vector<T> host_spliced, host_unspliced;
    cudaEvent_t staging_ready = nullptr;
    bool copy_pending = false;
    bool stream_ok = true, capacity_ok = true;
    std::size_t calls = 0;
    std::vector<std::uint64_t> requested;
    void* stream_identity = nullptr;
    const void* reused_spliced = nullptr;
    const void* reused_unspliced = nullptr;
    bool same_stream = true, same_device_buffers = true;

    ReusedBlockSource(std::size_t n, std::size_t source_g, std::size_t block,
                      const std::vector<float>& s, const std::vector<float>& u,
                      ac::Stream stream)
        : rows(n), genes(source_g), capacity(block), spliced(&s), unspliced(&u),
          device_spliced(n * block * sizeof(T), stream.device),
          device_unspliced(n * block * sizeof(T), stream.device),
          host_spliced(n * block), host_unspliced(n * block) {
        cuda_check(cudaEventCreateWithFlags(&staging_ready, cudaEventDisableTiming),
                   "create staging event");
    }

    ~ReusedBlockSource() {
        if (copy_pending) cudaEventSynchronize(staging_ready);
        if (staging_ready) cudaEventDestroy(staging_ready);
    }

    std::size_t staging_bytes() const {
        return 4 * rows * capacity * sizeof(T); // two host and two device matrices
    }

    vc::BlockSource source() {
        return {this, &ReusedBlockSource::acquire, staging_bytes()};
    }

    static void acquire(void* opaque, const std::uint64_t* source_genes,
                        std::size_t count, ac::Stream stream, vc::SourceBlock& out) {
        auto& self = *static_cast<ReusedBlockSource*>(opaque);
        ++self.calls;
        self.stream_ok &= stream.native != nullptr && stream.device == self.device_spliced.device();
        if (!self.stream_identity) self.stream_identity = stream.native;
        else self.same_stream &= self.stream_identity == stream.native;
        self.capacity_ok &= count > 0 && count <= self.capacity;
        if (!count || count > self.capacity)
            throw std::runtime_error("block provider request exceeds reusable capacity");
        self.requested.insert(self.requested.end(), source_genes, source_genes + count);
        if (self.copy_pending) {
            cuda_check(cudaEventSynchronize(self.staging_ready), "wait for host staging copy");
            self.copy_pending = false;
        }
        for (std::size_t j = 0; j < count; ++j) {
            if (source_genes[j] >= self.genes) {
                self.capacity_ok = false;
                continue;
            }
            for (std::size_t i = 0; i < self.rows; ++i) {
                const auto source = std::size_t(source_genes[j]);
                self.host_spliced[j * self.rows + i] = store<T>((*self.spliced)[i * self.genes + source]);
                self.host_unspliced[j * self.rows + i] = store<T>((*self.unspliced)[i * self.genes + source]);
            }
        }
        const auto bytes = self.rows * count * sizeof(T);
        auto native = reinterpret_cast<cudaStream_t>(stream.native);
        if (bytes) {
            cuda_check(cudaMemcpyAsync(self.device_spliced.data(), self.host_spliced.data(), bytes,
                                       cudaMemcpyHostToDevice, native), "copy spliced block");
            cuda_check(cudaMemcpyAsync(self.device_unspliced.data(), self.host_unspliced.data(), bytes,
                                       cudaMemcpyHostToDevice, native), "copy unspliced block");
            cuda_check(cudaEventRecord(self.staging_ready, native), "record staging event");
            self.copy_pending = true;
        }
        const auto type = scalar_type<T>();
        out.spliced = {self.device_spliced.span(), 0, self.rows, count, 1,
                       static_cast<std::int64_t>(self.rows), type};
        out.unspliced = {self.device_unspliced.span(), 0, self.rows, count, 1,
                         static_cast<std::int64_t>(self.rows), type};
        if (!self.reused_spliced) self.reused_spliced = out.spliced.values.data;
        else self.same_device_buffers &= self.reused_spliced == out.spliced.values.data;
        if (!self.reused_unspliced) self.reused_unspliced = out.unspliced.values.data;
        else self.same_device_buffers &= self.reused_unspliced == out.unspliced.values.data;
        out.available = nullptr;
    }
};

struct SinkCapture {
    std::size_t rows, genes, calls = 0;
    std::vector<float> velocity, second_residual;
    std::vector<std::size_t> first, count;
    const void* reusable_velocity = nullptr;
    bool same_velocity_buffer = true;

    SinkCapture(std::size_t n, std::size_t g, bool stochastic)
        : rows(n), genes(g), velocity(n * g),
          second_residual(stochastic ? n * g : 0) {}

    static void copy_columns(std::vector<float>& destination, std::size_t rows,
                             std::size_t genes, const ac::DenseView& view,
                             std::size_t first, std::size_t count, ac::Stream stream) {
        if (!view.values.data || !count) return;
        const auto width = ac::scalar_size(view.dtype);
        for (std::size_t i = 0; i < rows; ++i) {
            const auto byte_offset = std::ptrdiff_t(view.origin) +
                std::ptrdiff_t(i) * view.row_stride * std::ptrdiff_t(width);
            ac::copy_to_host(destination.data() + i * genes + first, view.values,
                             static_cast<std::size_t>(byte_offset), count * width, stream);
        }
    }

    static void write(void* opaque, const vc::OutputBlock& block, ac::Stream stream) {
        auto& self = *static_cast<SinkCapture*>(opaque);
        ++self.calls;
        self.first.push_back(block.first);
        self.count.push_back(block.count);
        if (!self.reusable_velocity) self.reusable_velocity = block.outputs.velocity.values.data;
        else self.same_velocity_buffer &= self.reusable_velocity == block.outputs.velocity.values.data;
        copy_columns(self.velocity, self.rows, self.genes, block.outputs.velocity,
                     block.first, block.count, stream);
        if (!self.second_residual.empty())
            copy_columns(self.second_residual, self.rows, self.genes,
                         block.outputs.second_residual, block.first, block.count, stream);
    }
};

void compare_diagnostics(vc::ResultView actual, vc::ResultView expected,
                         bool stochastic, ac::Stream stream) {
    close(read_device(actual.selected, stream), read_device(expected.selected, stream),
          0, "block-source selected genes match resident input");
    close(read_device(actual.status, stream), read_device(expected.status, stream),
          0, "block-source status matches resident input");
    close(read_device(actual.initial_gamma, stream), read_device(expected.initial_gamma, stream),
          2e-5, "block-source initial gamma matches resident input");
    close(read_device(actual.initial_offset, stream), read_device(expected.initial_offset, stream),
          2e-5, "block-source initial offset matches resident input");
    close(read_device(actual.r2, stream), read_device(expected.r2, stream),
          2e-5, "block-source R2 matches resident input");
    if (!stochastic) return;
    close(read_device(actual.gamma, stream), read_device(expected.gamma, stream),
          2e-5, "block-source stochastic gamma matches resident input");
    close(read_device(actual.offset, stream), read_device(expected.offset, stream),
          2e-5, "block-source stochastic offset matches resident input");
    close(read_device(actual.offset2, stream), read_device(expected.offset2, stream),
          2e-5, "block-source second offset matches resident input");
    close(read_device(actual.first_scale, stream), read_device(expected.first_scale, stream),
          2e-5, "block-source first scale matches resident input");
    close(read_device(actual.second_scale, stream), read_device(expected.second_scale, stream),
          2e-5, "block-source second scale matches resident input");
    close(read_device(actual.objective, stream), read_device(expected.objective, stream),
          2e-5, "block-source objective matches resident input");
}

template<class T>
void block_source_case(ac::Stream stream, bool stochastic, std::size_t block_size) {
    constexpr std::size_t n = 6, source_g = 7;
    const std::vector<std::size_t> selected{5, 1, 4, 0, 3};
    const auto data = fixture(n, source_g);
    const auto d_weights = upload_float_dense<float>(data.weights, n, n, stream);
    const auto d_spliced = upload_float_dense<T>(data.spliced, n, source_g, stream);
    const auto d_unspliced = upload_float_dense<T>(data.unspliced, n, source_g, stream);
    const auto genes = anndata::Axis::gather(selected.data(), selected.size());

    vc::Input resident_input{d_weights.view, d_spliced.view, d_unspliced.view, genes};
    vc::PreparedInput resident(resident_input, stream);
    const auto dtype = scalar_type<T>();
    if (resident.n_genes() != selected.size()) expect(false, "resident gather length is retained");
    for (std::size_t j = 0; j < selected.size(); ++j)
        expect(resident.gene_index(j) == selected[j], "resident gather order is retained");

    velocity::RunOptions run;
    velocity::StochasticRunOptions stochastic_run;
    run.compute_precision = velocity::ComputePrecision::float32;
    run.block_genes = block_size;
    stochastic_run.compute_precision = velocity::ComputePrecision::float32;
    stochastic_run.block_genes = block_size;
    stochastic_run.workspace_bytes = 1u << 20;
    run.workspace_bytes = 1u << 20;

    std::vector<double> resident_velocity, resident_second;
    std::optional<vc::FitResult> resident_result;
    if (stochastic) {
        vc::Workspace workspace(resident, stochastic_run);
        DeviceOutput velocity(n, selected.size(), ScalarType::float32, stream);
        DeviceOutput second(n, selected.size(), ScalarType::float32, stream);
        resident_result.emplace(vc::reference::stochastic(resident, workspace,
            exact_stochastic_fit(), {.velocity = velocity.view, .second_residual = second.view}, stream));
        ac::synchronize(stream);
        resident_velocity = read_device(velocity.view, stream);
        resident_second = read_device(second.view, stream);
    } else {
        vc::Workspace workspace(resident, run);
        DeviceOutput velocity(n, selected.size(), ScalarType::float32, stream);
        resident_result.emplace(vc::reference::steady_state(resident, workspace, exact_fit(),
            {.velocity = velocity.view}, stream));
        ac::synchronize(stream);
        resident_velocity = read_device(velocity.view, stream);
    }

    ReusedBlockSource<T> provider(n, source_g, block_size, data.spliced, data.unspliced, stream);
    vc::Input source_input;
    source_input.weights = d_weights.view;
    source_input.genes = genes;
    source_input.source = provider.source();
    source_input.source_genes = source_g;
    source_input.source_spliced_dtype = dtype;
    source_input.source_unspliced_dtype = dtype;
    vc::PreparedInput streamed(source_input, stream);
    if (streamed.n_genes() != selected.size() || streamed.source_genes() != source_g)
        expect(false, "block source retains the selected axis and declared source extent");
    for (std::size_t j = 0; j < selected.size(); ++j)
        expect(streamed.gene_index(j) == selected[j], "block-source gather order is retained");

    SinkCapture capture(n, selected.size(), stochastic);
    vc::BlockSink sink{&capture, &SinkCapture::write};
    DeviceOutput block_velocity(n, block_size, ScalarType::float32, stream,
                                static_cast<std::int64_t>(block_size));
    DeviceOutput block_second(n, stochastic ? block_size : 0, ScalarType::float32, stream,
                              static_cast<std::int64_t>(stochastic ? block_size : 0));
    vc::Outputs outputs;
    outputs.velocity = block_velocity.view;
    outputs.block_local = true;
    if (stochastic) outputs.second_residual = block_second.view;

    std::optional<vc::FitResult> streamed_result;
    std::unique_ptr<vc::Workspace> streamed_workspace;
    if (stochastic) {
        streamed_workspace = std::make_unique<vc::Workspace>(streamed, stochastic_run);
        expect(streamed_workspace->block_genes() == block_size, "stochastic provider workspace keeps requested block size");
        expect(streamed_workspace->staging_bytes() == provider.staging_bytes(), "provider staging is charged to workspace");
        streamed_result.emplace(vc::reference::stochastic(streamed, *streamed_workspace,
            exact_stochastic_fit(), outputs, stream, sink));
    } else {
        streamed_workspace = std::make_unique<vc::Workspace>(streamed, run);
        expect(streamed_workspace->block_genes() == block_size, "deterministic provider workspace keeps requested block size");
        expect(streamed_workspace->staging_bytes() == provider.staging_bytes(), "provider staging is charged to workspace");
        streamed_result.emplace(vc::reference::steady_state(streamed, *streamed_workspace,
            exact_fit(), outputs, stream, sink));
    }
    ac::synchronize(stream);

    const auto batches_per_pass = (selected.size() + block_size - 1) / block_size;
    const auto passes = stochastic ? 2u : 1u;
    expect(provider.calls == batches_per_pass * passes, "provider is called once per requested batch and pass");
    expect(provider.stream_ok, "provider receives the same nondefault stream and device");
    expect(provider.same_stream, "provider receives one stable stream across batches and passes");
    expect(provider.same_device_buffers, "provider reuses its device block buffers");
    expect(provider.capacity_ok, "provider returns selected batches within reusable buffer capacity");
    std::vector<std::uint64_t> expected_requests;
    for (std::size_t pass = 0; pass < passes; ++pass)
        for (auto gene : selected) expected_requests.push_back(gene);
    expect(provider.requested == expected_requests,
           "provider receives selected and reordered source genes on both stochastic passes");

    expect(capture.calls == batches_per_pass && capture.same_velocity_buffer,
           "block-local sink receives every block through the reused output buffer");
    for (std::size_t k = 0; k < capture.calls; ++k) {
        const auto first = k * block_size;
        const auto count = std::min(block_size, selected.size() - first);
        expect(capture.first[k] == first && capture.count[k] == count,
               "sink receives the actual first/count for each output block");
    }
    close(std::vector<double>(capture.velocity.begin(), capture.velocity.end()),
          resident_velocity, 2e-5, "block-local sink velocity roundtrip matches resident output");
    if (stochastic)
        close(std::vector<double>(capture.second_residual.begin(), capture.second_residual.end()),
              resident_second, 2e-5, "block-local sink residual roundtrip matches resident output");
    compare_diagnostics(streamed_result->view(), resident_result->view(), stochastic, stream);
}

void signed_stride_case(ac::Stream stream) {
    constexpr std::size_t n = 5, g = 4;
    auto data = fixture(n, g);
    std::vector<float> reverse_s(n * g), reverse_u(n * g);
    for (std::size_t physical = 0; physical < n; ++physical) {
        const auto logical = n - 1 - physical;
        std::copy_n(data.spliced.data() + logical * g, g, reverse_s.data() + physical * g);
        std::copy_n(data.unspliced.data() + logical * g, g, reverse_u.data() + physical * g);
    }
    auto hs = anndata::Array::borrow_dense(reverse_s.data() + (n - 1) * g, {n, g},
        {-static_cast<std::ptrdiff_t>(g), 1});
    auto hu = anndata::Array::borrow_dense(reverse_u.data() + (n - 1) * g, {n, g},
        {-static_cast<std::ptrdiff_t>(g), 1});
    auto hw = borrow_dense(data.weights, n, n);
    velocity::PreparedInput cpu({hs.view(), hu.view(), hw.view()});

    auto d_weights = upload_dense(data.weights, n, n, stream);
    auto d_reverse_s = upload_dense(reverse_s, n, g, stream);
    auto d_reverse_u = upload_dense(reverse_u, n, g, stream);
    d_reverse_s.view.origin = (n - 1) * g * sizeof(float);
    d_reverse_u.view.origin = (n - 1) * g * sizeof(float);
    d_reverse_s.view.row_stride = -static_cast<std::int64_t>(g);
    d_reverse_u.view.row_stride = -static_cast<std::int64_t>(g);
    vc::PreparedInput gpu({d_weights.view, d_reverse_s.view, d_reverse_u.view}, stream);

    velocity::RunOptions cpu_run;
    cpu_run.compute_precision = velocity::ComputePrecision::float32;
    cpu_run.block_genes = 2;
    velocity::RunOptions gpu_run;
    gpu_run.compute_precision = velocity::ComputePrecision::float32;
    gpu_run.block_genes = 2;
    velocity::Workspace cpu_workspace(cpu, cpu_run);
    vc::Workspace gpu_workspace(gpu, gpu_run);
    auto cpu_velocity = anndata::Array::empty<float>({n, g});
    DeviceOutput gpu_velocity(n, g, ScalarType::float32, stream);
    auto cpu_result = velocity::reference::steady_state_into(cpu, cpu_workspace,
        cpu_velocity.edit(), exact_fit());
    auto gpu_result = vc::reference::steady_state(gpu, gpu_workspace, exact_fit(),
        {.velocity = gpu_velocity.view}, stream);
    ac::synchronize(stream);
    close(read_device(gpu_velocity.view, stream), read_host(cpu_velocity.view()), 5e-4,
          "zero-copy negative-row-stride CUDA input matches signed-stride CPU input");
    close(read_device(gpu_result.view().gamma, stream), read_host(cpu_result.gamma.view()), 5e-4,
          "negative-stride FP32 gamma matches CPU reference");
    close(read_device(gpu_result.view().status, stream), read_host(cpu_result.status.view()), 0,
          "negative-stride FP32 status matches CPU reference");
}

void mutation_reuse_case(ac::Stream stream) {
    constexpr std::size_t n = 6, g = 5;
    auto data = fixture(n, g);
    auto hs = borrow_dense(data.spliced, n, g);
    auto hu = borrow_dense(data.unspliced, n, g);
    auto hw = borrow_dense(data.weights, n, n);
    velocity::PreparedInput cpu({hs.view(), hu.view(), hw.view()});
    auto d_weights = upload_dense(data.weights, n, n, stream);
    auto d_spliced = upload_dense(data.spliced, n, g, stream);
    auto d_unspliced = upload_dense(data.unspliced, n, g, stream);
    vc::PreparedInput gpu({d_weights.view, d_spliced.view, d_unspliced.view}, stream);

    velocity::RunOptions cpu_run;
    cpu_run.compute_precision = velocity::ComputePrecision::float32;
    cpu_run.block_genes = 2;
    velocity::RunOptions gpu_run;
    gpu_run.compute_precision = velocity::ComputePrecision::float32;
    gpu_run.block_genes = 2;
    velocity::Workspace cpu_workspace(cpu, cpu_run);
    vc::Workspace gpu_workspace(gpu, gpu_run);

    auto first_cpu_velocity = anndata::Array::empty<float>({n, g});
    DeviceOutput first_gpu_velocity(n, g, ScalarType::float32, stream);
    auto first_cpu = velocity::reference::steady_state_into(cpu, cpu_workspace,
        first_cpu_velocity.edit(), exact_fit());
    auto first_gpu = vc::reference::steady_state(gpu, gpu_workspace, exact_fit(),
        {.velocity = first_gpu_velocity.view}, stream);
    ac::synchronize(stream);
    auto first_output = read_device(first_gpu_velocity.view, stream);
    close(first_output, read_host(first_cpu_velocity.view()), 5e-4,
          "reusable FP32 CUDA baseline matches CPU before numeric updates");

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < g; ++j) {
            const auto k = i * g + j;
            data.spliced[k] = data.spliced[k] * (.72f + .06f * float(j)) + .14f * float(i + 1);
            data.unspliced[k] = data.unspliced[k] + .31f * float((2 * i + j) % 4) + .07f * data.spliced[k];
        }
    }
    ac::copy_to_device(d_spliced.owner.span(), 0, data.spliced.data(),
                       data.spliced.size() * sizeof(float), stream);
    ac::copy_to_device(d_unspliced.owner.span(), 0, data.unspliced.data(),
                       data.unspliced.size() * sizeof(float), stream);
    velocity::PreparedInput fresh_cpu({hs.view(), hu.view(), hw.view()});
    velocity::Workspace fresh_cpu_workspace(fresh_cpu, cpu_run);
    auto second_cpu_velocity = anndata::Array::empty<float>({n, g});
    DeviceOutput second_gpu_velocity(n, g, ScalarType::float32, stream);
    auto second_cpu = velocity::reference::steady_state_into(fresh_cpu, fresh_cpu_workspace,
        second_cpu_velocity.edit(), exact_fit());
    auto second_gpu = vc::reference::steady_state(gpu, gpu_workspace, exact_fit(),
        {.velocity = second_gpu_velocity.view}, stream);
    ac::synchronize(stream);
    auto second_output = read_device(second_gpu_velocity.view, stream);
    close(second_output, read_host(second_cpu_velocity.view()), 5e-4,
          "same prepared CUDA input/workspace observes updated S/U and matches fresh CPU result");
    close(read_device(second_gpu.view().gamma, stream), read_host(second_cpu.gamma.view()), 5e-4,
          "updated numeric input yields fresh matching gamma diagnostics");
    double change = 0;
    for (std::size_t i = 0; i < first_output.size(); ++i)
        change = std::max(change, std::abs(second_output[i] - first_output[i]));
    expect(change > 1e-4, "numeric S/U mutation changes a result without structural re-preparation");
    (void)first_cpu;
    (void)first_gpu;
}

void empty_shape_case(ac::Stream stream, std::size_t n, std::size_t g,
                      const char* label) {
    std::vector<float> s(n * g), u(n * g), w(n * n);
    for (std::size_t i = 0; i < n; ++i) w[i * n + i] = 1;
    auto hs = borrow_dense(s, n, g);
    auto hu = borrow_dense(u, n, g);
    auto hw = borrow_dense(w, n, n);
    auto genes = g ? anndata::Axis::all() : anndata::Axis::gather({});
    velocity::PreparedInput cpu({hs.view(), hu.view(), hw.view(), genes});
    auto d_s = upload_dense(s, n, g, stream);
    auto d_u = upload_dense(u, n, g, stream);
    auto d_w = upload_dense(w, n, n, stream);
    vc::PreparedInput gpu({d_w.view, d_s.view, d_u.view, genes}, stream);
    velocity::RunOptions cpu_run;
    cpu_run.compute_precision = velocity::ComputePrecision::float32;
    cpu_run.block_genes = 2;
    velocity::RunOptions gpu_run;
    gpu_run.compute_precision = velocity::ComputePrecision::float32;
    gpu_run.block_genes = 2;
    velocity::Workspace cpu_workspace(cpu, cpu_run);
    vc::Workspace gpu_workspace(gpu, gpu_run);
    auto cpu_fit = velocity::reference::steady_state(cpu, cpu_workspace, exact_fit());
    auto gpu_fit = vc::reference::steady_state(gpu, gpu_workspace, exact_fit(), {}, stream);
    ac::synchronize(stream);
    close(read_device(gpu_fit.view().status, stream), read_host(cpu_fit.status.view()), 0,
          label);
    close(read_device(gpu_fit.view().selected, stream), read_host(cpu_fit.genes.view()), 0,
          label);
    expect((read_device(gpu_fit.view().low_signal, stream).front() != 0) == cpu_fit.low_signal,
           "empty deterministic low-signal flag matches CPU reference");

    velocity::StochasticRunOptions cpu_stochastic_run;
    cpu_stochastic_run.compute_precision = velocity::ComputePrecision::float32;
    cpu_stochastic_run.block_genes = 2;
    velocity::StochasticRunOptions gpu_stochastic_run;
    gpu_stochastic_run.compute_precision = velocity::ComputePrecision::float32;
    gpu_stochastic_run.block_genes = 2;
    velocity::StochasticWorkspace cpu_stochastic_workspace(cpu, cpu_stochastic_run);
    vc::Workspace gpu_stochastic_workspace(gpu, gpu_stochastic_run);
    auto cpu_stochastic = velocity::reference::stochastic(cpu, cpu_stochastic_workspace,
        exact_stochastic_fit());
    auto gpu_stochastic = vc::reference::stochastic(gpu, gpu_stochastic_workspace,
        exact_stochastic_fit(), {}, stream);
    ac::synchronize(stream);
    close(read_device(gpu_stochastic.view().status, stream),
          read_host(cpu_stochastic.status.view()), 0, label);
    close(read_device(gpu_stochastic.view().selected, stream),
          read_host(cpu_stochastic.initialization.genes.view()), 0, label);
    expect((read_device(gpu_stochastic.view().low_signal, stream).front() != 0) ==
           cpu_stochastic.initialization.low_signal,
           "empty stochastic low-signal flag matches CPU reference");
    expect((read_device(gpu_stochastic.view().refit_all, stream).front() != 0) ==
           cpu_stochastic.refit_all, "empty stochastic refit flag matches CPU reference");
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2 || std::string_view(argv[1]) != "--gpu") {
        std::cerr << "CUDA batch tests require admitted GPU execution; pass --gpu explicitly\n";
        return 77;
    }
    try {
        int device = 0;
        cuda_check(cudaSetDevice(device), "select scheduler-provided device");
        cudaStream_t raw = nullptr;
        cuda_check(cudaStreamCreateWithFlags(&raw, cudaStreamNonBlocking), "create nondefault stream");
        ac::Stream stream{raw, device};

        for (auto type_case : {0, 1}) {
            const bool half = type_case != 0;
            if (half) {
                block_source_case<anndata::Float16>(stream, false, 1);
                block_source_case<anndata::Float16>(stream, false, 2);
                block_source_case<anndata::Float16>(stream, true, 1);
                block_source_case<anndata::Float16>(stream, true, 2);
            } else {
                block_source_case<float>(stream, false, 1);
                block_source_case<float>(stream, false, 2);
                block_source_case<float>(stream, true, 1);
                block_source_case<float>(stream, true, 2);
            }
        }
        signed_stride_case(stream);
        mutation_reuse_case(stream);
        empty_shape_case(stream, 0, 3, "zero-row fit statuses match CPU reference");
        empty_shape_case(stream, 3, 0, "zero-gene fit statuses match CPU reference");
        ac::synchronize(stream);
        cuda_check(cudaStreamDestroy(raw), "destroy nondefault stream");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
    if (failures) return 1;
    std::cout << "CUDA batch, sink, signed-stride, empty-shape, and value-reuse contracts passed\n";
    return 0;
}
