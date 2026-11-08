#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <initializer_list>
#include <limits>
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

namespace vc = velocity::cuda;
namespace ac = anndata::cuda;

constexpr std::size_t sensitivity_rows = 12;
constexpr std::size_t sensitivity_genes = 2;
constexpr std::size_t promotion_rows = 12;
constexpr std::size_t promotion_genes = 3;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

struct StreamOwner {
    cudaStream_t native = nullptr;
    int device = -1;

    StreamOwner() {
        cuda_check(cudaGetDevice(&device), "cudaGetDevice");
        cuda_check(cudaStreamCreateWithFlags(&native, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    }
    ~StreamOwner() {
        if (native) {
            cudaStreamSynchronize(native);
            cudaStreamDestroy(native);
        }
    }
    StreamOwner(const StreamOwner&) = delete;
    StreamOwner& operator=(const StreamOwner&) = delete;

    ac::Stream view() const noexcept { return {reinterpret_cast<void*>(native), device}; }
};

struct DeviceDense {
    ac::DeviceBuffer owner;
    ac::DenseView descriptor{};

    DeviceDense() = default;
    DeviceDense(std::size_t rows, std::size_t columns, ac::ScalarType type, int device,
                bool column_major = true)
        : owner(rows * columns * ac::scalar_size(type), device) {
        descriptor = {owner.span(ac::Access::read_write), 0, rows, columns,
                      column_major ? 1 : std::ptrdiff_t(columns),
                      column_major ? std::ptrdiff_t(rows) : 1, type};
    }

    template<class T> void upload(const std::vector<T>& host, ac::Stream stream) {
        require(host.size() * sizeof(T) <= owner.size_bytes(), "device upload exceeds allocation");
        if (!host.empty())
            cuda_check(cudaMemcpyAsync(owner.data(), host.data(), host.size() * sizeof(T),
                                       cudaMemcpyHostToDevice,
                                       reinterpret_cast<cudaStream_t>(stream.native)),
                       "cudaMemcpyAsync host-to-device");
    }

    ac::DenseView view() const noexcept { return descriptor; }
};

template<class T> std::vector<T> download_column_major(ac::DenseView view,
                                                        std::size_t rows,
                                                        std::size_t columns,
                                                        ac::Stream stream) {
    std::vector<T> host(rows * columns);
    auto* base = static_cast<const std::byte*>(view.values.data) + view.origin;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream.native);
    for (std::size_t j = 0; j < columns; ++j) {
        for (std::size_t i = 0; i < rows; ++i) {
            const auto offset = (std::ptrdiff_t(i) * view.row_stride +
                                 std::ptrdiff_t(j) * view.column_stride) *
                                std::ptrdiff_t(sizeof(T));
            cuda_check(cudaMemcpyAsync(host.data() + j * rows + i, base + offset, sizeof(T),
                                       cudaMemcpyDeviceToHost, cuda_stream),
                       "cudaMemcpyAsync device-to-host");
        }
    }
    cuda_check(cudaStreamSynchronize(cuda_stream), "cudaStreamSynchronize download");
    return host;
}

template<class T> std::vector<T> download_rows(ac::DenseView view, std::size_t count,
                                                ac::Stream stream) {
    std::vector<T> host(count);
    auto* base = static_cast<const std::byte*>(view.values.data) + view.origin;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream.native);
    for (std::size_t i = 0; i < count; ++i) {
        const auto offset = std::ptrdiff_t(i) * view.row_stride * std::ptrdiff_t(sizeof(T));
        cuda_check(cudaMemcpyAsync(host.data() + i, base + offset, sizeof(T),
                                   cudaMemcpyDeviceToHost, cuda_stream),
                   "cudaMemcpyAsync result download");
    }
    cuda_check(cudaStreamSynchronize(cuda_stream), "cudaStreamSynchronize result download");
    return host;
}

template<class T> bool close(T actual, T expected, T tolerance) {
    return std::isfinite(actual) && std::isfinite(expected) &&
           std::abs(actual - expected) <= tolerance * (T(1) + std::abs(expected));
}

bool close_fp32_gate(float actual, float expected) {
    return std::isfinite(actual) && std::isfinite(expected) &&
           std::abs(actual - expected) <= 2e-5f + 2e-4f * std::abs(expected);
}

template<class T> ac::ScalarType scalar_type();
template<> ac::ScalarType scalar_type<float>() { return ac::ScalarType::float32; }
template<> ac::ScalarType scalar_type<double>() { return ac::ScalarType::float64; }

template<class T> velocity::ComputePrecision compute_precision();
template<> velocity::ComputePrecision compute_precision<float>() { return velocity::ComputePrecision::float32; }
template<> velocity::ComputePrecision compute_precision<double>() { return velocity::ComputePrecision::float64; }

template<class T> void make_sensitivity_data(std::vector<T>& spliced,
                                             std::vector<T>& unspliced,
                                             std::vector<T>& weights) {
    spliced.resize(sensitivity_rows * sensitivity_genes);
    unspliced.resize(sensitivity_rows * sensitivity_genes);
    weights.assign(sensitivity_rows * sensitivity_rows, T(0));
    for (std::size_t i = 0; i < sensitivity_rows; ++i) {
        weights[i + i * sensitivity_rows] = T(1);
        for (std::size_t j = 0; j < sensitivity_genes; ++j) {
            const T x = T(i + 1) * (T(.7) + T(.23) * T(j)) +
                        T(int((i * 7 + j * 3) % 5) - 2) * T(.017);
            const T noise = T(int((i * 11 + j * 5) % 9) - 4) * T(.035);
            spliced[j * sensitivity_rows + i] = x;
            unspliced[j * sensitivity_rows + i] =
                (T(1.35) + T(.16) * T(j)) * x + T(.18) * T(j) + noise;
        }
    }
}

velocity::FitOptions sensitivity_fit_options(bool trimming) {
    velocity::FitOptions options;
    options.trimming = trimming ? velocity::Trimming::upper : velocity::Trimming::none;
    options.percentile_low = 5;
    options.percentile_high = 70;
    options.fit_intercept = true;
    options.positive_intercept = false;
    options.min_r2 = 0;
    options.min_ratio = 0;
    options.low_signal = velocity::LowSignal::keep_filters;
    return options;
}

template<class T> void compare_report(const velocity::precision::BasicSensitivityReport<T>& cpu,
                                     ac::DenseView state, ac::DenseView low,
                                     ac::DenseView high, ac::DenseView delta,
                                     ac::DenseView prediction_change, ac::DenseView r2_change,
                                     ac::DenseView expanded_status, ac::DenseView contracted_status,
                                     ac::DenseView status_change, ac::DenseView selection_change,
                                     std::size_t gene, ac::Stream stream, T tolerance,
                                     bool compare_all, const char* label) {
    const auto gpu_state = download_rows<std::uint8_t>(state, sensitivity_genes, stream);
    require(gpu_state[gene] == static_cast<std::uint8_t>(cpu.state), label);
    if (!compare_all) return;
    const auto gpu_low = download_rows<T>(low, sensitivity_genes, stream);
    const auto gpu_high = download_rows<T>(high, sensitivity_genes, stream);
    const auto gpu_delta = download_rows<T>(delta, sensitivity_genes, stream);
    const auto gpu_prediction = download_rows<T>(prediction_change, sensitivity_genes, stream);
    const auto gpu_r2 = download_rows<T>(r2_change, sensitivity_genes, stream);
    const auto gpu_expanded = download_rows<std::uint8_t>(expanded_status, sensitivity_genes, stream);
    const auto gpu_contracted = download_rows<std::uint8_t>(contracted_status, sensitivity_genes, stream);
    const auto gpu_status_change = download_rows<std::uint8_t>(status_change, sensitivity_genes, stream);
    const auto gpu_selection_change = download_rows<std::uint8_t>(selection_change, sensitivity_genes, stream);
    require(close(gpu_low[gene], cpu.low, tolerance), "sensitivity low quantile differs from CPU");
    require(close(gpu_high[gene], cpu.high, tolerance), "sensitivity high quantile differs from CPU");
    require(close(gpu_delta[gene], cpu.delta, tolerance), "sensitivity perturbation width differs from CPU");
    require(close(gpu_prediction[gene], cpu.prediction_change, tolerance), "sensitivity prediction change differs from CPU");
    require(close(gpu_r2[gene], cpu.r2_change, tolerance), "sensitivity R2 change differs from CPU");
    require(gpu_expanded[gene] == static_cast<std::uint8_t>(cpu.expanded_status),
            "expanded sensitivity fit status differs from CPU");
    require(gpu_contracted[gene] == static_cast<std::uint8_t>(cpu.contracted_status),
            "contracted sensitivity fit status differs from CPU");
    require(bool(gpu_status_change[gene]) == cpu.status_change,
            "sensitivity status-change flag differs from CPU");
    require(bool(gpu_selection_change[gene]) == cpu.selection_change,
            "sensitivity selection-change flag differs from CPU");
}

template<class T> void sensitivity_case(bool trimmed, bool masked, bool partial_outputs,
                                       bool reversed_state_stride = false) {
    std::vector<T> host_s, host_u, host_w;
    make_sensitivity_data(host_s, host_u, host_w);
    StreamOwner stream_owner;
    const auto stream = stream_owner.view();
    require(stream.native != nullptr, "test must use an explicitly created non-default CUDA stream");

    DeviceDense d_s(sensitivity_rows, sensitivity_genes, scalar_type<T>(), stream.device);
    DeviceDense d_u(sensitivity_rows, sensitivity_genes, scalar_type<T>(), stream.device);
    DeviceDense d_w(sensitivity_rows, sensitivity_rows, scalar_type<T>(), stream.device);
    d_s.upload(host_s, stream);
    d_u.upload(host_u, stream);
    d_w.upload(host_w, stream);

    std::vector<std::uint8_t> fit_mask(sensitivity_rows, 1), gene_mask(sensitivity_genes, 1);
    if (masked) {
        fit_mask[1] = 0;
        fit_mask[8] = 0;
        fit_mask[10] = 0;
        gene_mask[1] = 0;
    }
    std::optional<DeviceDense> d_fit_mask, d_gene_mask;
    if (masked) {
        d_fit_mask.emplace(sensitivity_rows, 1, ac::ScalarType::boolean, stream.device);
        d_gene_mask.emplace(sensitivity_genes, 1, ac::ScalarType::boolean, stream.device);
        d_fit_mask->upload(fit_mask, stream);
        d_gene_mask->upload(gene_mask, stream);
    }

    vc::Input input;
    input.weights = vc::MatrixView(d_w.view());
    input.spliced = vc::MatrixView(d_s.view());
    input.unspliced = vc::MatrixView(d_u.view());
    if (masked) {
        input.fit_mask = d_fit_mask->view();
        input.gene_mask = d_gene_mask->view();
    }
    vc::PreparedInput prepared(std::move(input), stream);

    velocity::RunOptions run;
    run.compute_precision = compute_precision<T>();
    run.block_genes = sensitivity_genes;
    run.workspace_bytes = 1u << 20;
    vc::Workspace workspace(prepared, run);
    vc::FitResult result(prepared, run.compute_precision);
    const auto fit_options = sensitivity_fit_options(trimmed);
    vc::ReferenceStages stages(prepared, workspace, result.view(), fit_options, {}, stream);

    auto moments = stages.initial_moments(0, sensitivity_genes);
    auto fit = stages.initial_fit(moments);

    DeviceDense state(sensitivity_genes, 1, ac::ScalarType::uint8, stream.device);
    DeviceDense low(sensitivity_genes, 1, scalar_type<T>(), stream.device);
    DeviceDense high(sensitivity_genes, 1, scalar_type<T>(), stream.device);
    DeviceDense delta(sensitivity_genes, 1, scalar_type<T>(), stream.device);
    DeviceDense prediction(sensitivity_genes, 1, scalar_type<T>(), stream.device);
    DeviceDense r2(sensitivity_genes, 1, scalar_type<T>(), stream.device);
    DeviceDense expanded(sensitivity_genes, 1, ac::ScalarType::uint8, stream.device);
    DeviceDense contracted(sensitivity_genes, 1, ac::ScalarType::uint8, stream.device);
    DeviceDense status_change(sensitivity_genes, 1, ac::ScalarType::uint8, stream.device);
    DeviceDense selection_change(sensitivity_genes, 1, ac::ScalarType::uint8, stream.device);

    vc::SensitivityView report;
    report.state = state.view();
    if (reversed_state_stride) {
        report.state.origin = (sensitivity_genes - 1) * sizeof(std::uint8_t);
        report.state.row_stride = -1;
    }
    if (!partial_outputs) {
        report.low = low.view();
        report.high = high.view();
        report.delta = delta.view();
        report.prediction_change = prediction.view();
        report.r2_change = r2.view();
        report.expanded_status = expanded.view();
        report.contracted_status = contracted.view();
        report.status_change = status_change.view();
        report.selection_change = selection_change.view();
    } else {
        // Individual omitted diagnostic views are permitted; state remains required.
        report.delta = delta.view();
    }

    velocity::precision::SensitivityOptions diagnostic;
    if (trimmed) {
        diagnostic.band = 500;
        diagnostic.prediction_limit = partial_outputs ? 1.0e6 : 0;
        diagnostic.r2_limit = partial_outputs ? 1.0e6 : 0;
    } else {
        diagnostic.band = .73;
        diagnostic.prediction_limit = .025;
        diagnostic.r2_limit = .012;
    }
    stages.sensitivity(moments, fit, report, diagnostic);

    const auto moment_s = download_column_major<T>(moments.spliced, sensitivity_rows,
                                                   sensitivity_genes, stream);
    const auto moment_u = download_column_major<T>(moments.unspliced, sensitivity_rows,
                                                   sensitivity_genes, stream);
    const auto baseline_gamma = download_rows<T>(fit.result.initial_gamma, sensitivity_genes, stream);
    const auto baseline_offset = download_rows<T>(fit.result.initial_offset, sensitivity_genes, stream);
    const auto baseline_r2 = download_rows<T>(fit.result.r2, sensitivity_genes, stream);
    const auto baseline_selected = download_rows<std::uint8_t>(fit.result.selected, sensitivity_genes, stream);
    const auto baseline_status = download_rows<std::uint8_t>(fit.result.initial_status, sensitivity_genes, stream);

    auto cpu_mask = masked ? fit_mask.data() : nullptr;
    velocity::precision::SensitivityOptions cpu_diagnostic = diagnostic;
    std::vector<T> scratch(sensitivity_rows);
    for (std::size_t gene = 0; gene < sensitivity_genes; ++gene) {
        velocity::BasicGeneFit<T> baseline{baseline_gamma[gene], baseline_offset[gene],
            baseline_r2[gene], static_cast<velocity::FitStatus>(baseline_status[gene]),
            bool(baseline_selected[gene])};
        auto cpu = velocity::precision::sensitivity_gene(
            moment_s.data() + gene * sensitivity_rows,
            moment_u.data() + gene * sensitivity_rows,
            sensitivity_rows, 1, baseline, fit_options, scratch.data(), cpu_mask,
            !masked || gene_mask[gene] != 0, cpu_diagnostic);
        compare_report(cpu, report.state, report.low, report.high, report.delta,
                       report.prediction_change, report.r2_change, report.expanded_status,
                       report.contracted_status, report.status_change, report.selection_change,
                       gene, stream, std::is_same_v<T, float> ? T(2e-4) : T(2e-11),
                       !partial_outputs, "GPU sensitivity state differs from CPU");
        if (partial_outputs) {
            const auto gpu_delta = download_rows<T>(report.delta, sensitivity_genes, stream);
            require(close(gpu_delta[gene], cpu.delta,
                          std::is_same_v<T, float> ? T(2e-4) : T(2e-11)),
                    "optional sensitivity delta output differs from CPU");
        }
    }

    if (trimmed && !partial_outputs) {
        const auto states = download_rows<std::uint8_t>(report.state, sensitivity_genes, stream);
        const auto lows = download_rows<T>(report.prediction_change, sensitivity_genes, stream);
        const auto highs = download_rows<T>(report.r2_change, sensitivity_genes, stream);
        require(states[0] == static_cast<std::uint8_t>(velocity::precision::SensitivityState::sensitive) &&
                (lows[0] > T(0) || highs[0] > T(0)),
                "zero configurable sensitivity thresholds must report a positive change as sensitive");

        auto broad = diagnostic;
        broad.prediction_limit = 1.0e6;
        broad.r2_limit = 1.0e6;
        DeviceDense broad_state(sensitivity_genes, 1, ac::ScalarType::uint8, stream.device);
        vc::SensitivityView broad_report;
        broad_report.state = broad_state.view();
        stages.sensitivity(moments, fit, broad_report, broad);
        const auto broad_states = download_rows<std::uint8_t>(broad_report.state,
                                                               sensitivity_genes, stream);
        require(broad_states[0] == static_cast<std::uint8_t>(
                    velocity::precision::SensitivityState::stable),
                "broad configurable sensitivity thresholds must classify finite changes as stable");
        velocity::precision::SensitivityOptions cpu_broad = broad;
        velocity::BasicGeneFit<T> baseline{baseline_gamma[0], baseline_offset[0], baseline_r2[0],
            static_cast<velocity::FitStatus>(baseline_status[0]), bool(baseline_selected[0])};
        auto expected_broad = velocity::precision::sensitivity_gene(
            moment_s.data(), moment_u.data(), sensitivity_rows, 1, baseline, fit_options,
            scratch.data(), cpu_mask, !masked || gene_mask[0] != 0, cpu_broad);
        compare_report(expected_broad, broad_report.state, {}, {}, {}, {}, {}, {}, {}, {}, {},
                       0, stream, std::is_same_v<T, float> ? T(2e-4) : T(2e-11), false,
                       "broad-threshold sensitivity state differs from CPU");

        vc::SensitivityView aliased_outputs;
        aliased_outputs.state = report.state;
        aliased_outputs.low = low.view();
        aliased_outputs.high = low.view();
        bool output_alias_rejected = false;
        try {
            stages.sensitivity(moments, fit, aliased_outputs, diagnostic);
        } catch (const std::invalid_argument&) {
            output_alias_rejected = true;
        }
        require(output_alias_rejected,
                "host boundary must reject overlapping sensitivity output views");

        vc::SensitivityView aliases_moments;
        aliases_moments.state = report.state;
        aliases_moments.state.values = moments.spliced.values;
        aliases_moments.state.origin = moments.spliced.origin;
        aliases_moments.state.rows = sensitivity_genes;
        aliases_moments.state.columns = 1;
        aliases_moments.state.row_stride = 1;
        aliases_moments.state.column_stride = std::ptrdiff_t(sensitivity_genes);
        aliases_moments.state.dtype = ac::ScalarType::uint8;
        bool input_alias_rejected = false;
        try {
            stages.sensitivity(moments, fit, aliases_moments, diagnostic);
        } catch (const std::invalid_argument&) {
            input_alias_rejected = true;
        }
        require(input_alias_rejected,
                "host boundary must reject a sensitivity output overlapping moment inputs");
    }
    if (!trimmed) {
        const auto states = download_rows<std::uint8_t>(report.state, sensitivity_genes, stream);
        require(states[0] == static_cast<std::uint8_t>(velocity::precision::SensitivityState::no_trimming),
                "no-trimming sensitivity returns the no_trimming state");
    }
}

struct SourceBatch {
    std::vector<float> spliced, unspliced;
};

struct OriginalBlockSource {
    std::size_t rows = 0, genes = 0;
    const std::vector<float>* spliced = nullptr;
    const std::vector<float>* unspliced = nullptr;
    std::size_t unavailable_gene = std::numeric_limits<std::size_t>::max();
    DeviceDense device_spliced, device_unspliced;
    std::vector<std::uint8_t> available;
    std::vector<std::vector<std::uint64_t>> acquisitions;
    std::vector<std::unique_ptr<SourceBatch>> staging;
    bool invalid_gene = false;

    OriginalBlockSource(std::size_t n, std::size_t g, int device_id,
                        const std::vector<float>& s, const std::vector<float>& u,
                        std::size_t missing)
        : rows(n), genes(g), spliced(&s), unspliced(&u),
          unavailable_gene(missing),
          device_spliced(n, g, ac::ScalarType::float32, device_id),
          device_unspliced(n, g, ac::ScalarType::float32, device_id) {}

    static void acquire(void* opaque, const std::uint64_t* source_genes, std::size_t count,
                        ac::Stream stream, vc::SourceBlock& out) {
        auto& self = *static_cast<OriginalBlockSource*>(opaque);
        self.acquisitions.emplace_back();
        auto& requested = self.acquisitions.back();
        if (count) requested.assign(source_genes, source_genes + count);
        self.available.assign(count, 1);
        auto batch = std::make_unique<SourceBatch>();
        batch->spliced.assign(self.rows * count, 0.0f);
        batch->unspliced.assign(self.rows * count, 0.0f);
        for (std::size_t j = 0; j < count; ++j) {
            const auto gene = static_cast<std::size_t>(source_genes[j]);
            if (gene >= self.genes) {
                self.invalid_gene = true;
                self.available[j] = 0;
                continue;
            }
            const bool available = gene != self.unavailable_gene;
            self.available[j] = available ? 1 : 0;
            for (std::size_t i = 0; i < self.rows; ++i) {
                if (!available) continue;
                batch->spliced[j * self.rows + i] = (*self.spliced)[gene * self.rows + i];
                batch->unspliced[j * self.rows + i] = (*self.unspliced)[gene * self.rows + i];
            }
        }
        const auto bytes = self.rows * count * sizeof(float);
        auto cuda_stream = reinterpret_cast<cudaStream_t>(stream.native);
        if (bytes) {
            cuda_check(cudaMemcpyAsync(self.device_spliced.owner.data(), batch->spliced.data(), bytes,
                                       cudaMemcpyHostToDevice, cuda_stream),
                       "copy original spliced source block");
            cuda_check(cudaMemcpyAsync(self.device_unspliced.owner.data(), batch->unspliced.data(), bytes,
                                       cudaMemcpyHostToDevice, cuda_stream),
                       "copy original unspliced source block");
        }
        self.staging.push_back(std::move(batch));
        out.spliced = self.device_spliced.view();
        out.unspliced = self.device_unspliced.view();
        out.spliced.columns = count;
        out.unspliced.columns = count;
        out.available = self.available.empty() ? nullptr : self.available.data();
    }

    bool received_only(std::initializer_list<std::uint64_t> expected) const {
        if (invalid_gene) return false;
        const std::vector<std::uint64_t> exact(expected);
        return !acquisitions.empty() &&
            std::all_of(acquisitions.begin(), acquisitions.end(),
                [&](const auto& ids) { return ids == exact; });
    }

    bool received_only_from(std::initializer_list<std::uint64_t> allowed) const {
        if (invalid_gene || acquisitions.empty()) return false;
        const std::vector<std::uint64_t> permitted(allowed);
        for (const auto& ids : acquisitions)
            for (auto id : ids)
                if (std::find(permitted.begin(), permitted.end(), id) == permitted.end()) return false;
        return true;
    }

    std::size_t staging_bytes() const noexcept {
        // Retain two host columns across the initial pass plus one promoted
        // column on the stochastic revisit; device staging holds the full block.
        return 4 * rows * genes * sizeof(float);
    }
};

struct DisappearingBlockSource {
    OriginalBlockSource original;
    std::vector<std::uint8_t> unavailable;
    std::vector<std::vector<std::uint64_t>> requests;

    DisappearingBlockSource(std::size_t rows, std::size_t genes, int device,
                            const std::vector<float>& spliced,
                            const std::vector<float>& unspliced)
        : original(rows, genes, device, spliced, unspliced,
                   std::numeric_limits<std::size_t>::max()) {}

    static void acquire(void* opaque, const std::uint64_t* genes, std::size_t count,
                        ac::Stream stream, vc::SourceBlock& out) {
        auto& self = *static_cast<DisappearingBlockSource*>(opaque);
        self.requests.emplace_back();
        if (count) self.requests.back().assign(genes, genes + count);
        if (self.requests.size() == 1) {
            OriginalBlockSource::acquire(&self.original, genes, count, stream, out);
            return;
        }
        self.unavailable.assign(count, 0);
        out = {};
        out.available = self.unavailable.empty() ? nullptr : self.unavailable.data();
    }

    std::size_t staging_bytes() const noexcept { return original.staging_bytes(); }

    bool requested_only_gene_zero_twice() const {
        const std::vector<std::uint64_t> expected{0};
        return requests.size() == 2 && requests[0] == expected && requests[1] == expected;
    }
};

struct CopiedFit {
    std::vector<float> gamma, offset, r2, velocity;
    std::vector<std::uint8_t> selected, status;
    std::uint8_t low_signal = 0, refit_all = 0;
};

CopiedFit copy_cuda_result(vc::ResultView result, ac::DenseView velocity_view,
                           std::size_t rows, std::size_t genes, ac::Stream stream) {
    CopiedFit copy;
    copy.gamma = download_rows<float>(result.gamma, genes, stream);
    copy.offset = download_rows<float>(result.offset, genes, stream);
    copy.r2 = download_rows<float>(result.r2, genes, stream);
    copy.selected = download_rows<std::uint8_t>(result.selected, genes, stream);
    copy.status = download_rows<std::uint8_t>(result.status, genes, stream);
    copy.velocity = download_column_major<float>(velocity_view, rows, genes, stream);
    if (result.low_signal.values.data)
        copy.low_signal = download_rows<std::uint8_t>(result.low_signal, 1, stream)[0];
    if (result.refit_all.values.data)
        copy.refit_all = download_rows<std::uint8_t>(result.refit_all, 1, stream)[0];
    return copy;
}

struct CopiedStochasticFit {
    std::vector<float> gamma, offset, first_scale, second_scale, objective;
    std::vector<std::uint8_t> selected, initial_status, status;
    std::uint8_t low_signal = 0, refit_all = 0;
};

CopiedStochasticFit copy_cuda_stochastic_result(vc::ResultView result, std::size_t genes,
                                                ac::Stream stream) {
    CopiedStochasticFit copy;
    copy.gamma = download_rows<float>(result.gamma, genes, stream);
    copy.offset = download_rows<float>(result.offset, genes, stream);
    copy.first_scale = download_rows<float>(result.first_scale, genes, stream);
    copy.second_scale = download_rows<float>(result.second_scale, genes, stream);
    copy.objective = download_rows<float>(result.objective, genes, stream);
    copy.selected = download_rows<std::uint8_t>(result.selected, genes, stream);
    copy.initial_status = download_rows<std::uint8_t>(result.initial_status, genes, stream);
    copy.status = download_rows<std::uint8_t>(result.status, genes, stream);
    if (result.low_signal.values.data)
        copy.low_signal = download_rows<std::uint8_t>(result.low_signal, 1, stream)[0];
    if (result.refit_all.values.data)
        copy.refit_all = download_rows<std::uint8_t>(result.refit_all, 1, stream)[0];
    return copy;
}

void set_steady_options(velocity::RunOptions& run) {
    run.compute_precision = velocity::ComputePrecision::float32;
    run.output_dtype = anndata::DType::float32;
    run.block_genes = promotion_genes;
    run.workspace_bytes = 1u << 20;
}

velocity::FitOptions promotion_fit_options() {
    velocity::FitOptions fit;
    fit.trimming = velocity::Trimming::none;
    fit.fit_intercept = true;
    fit.positive_intercept = false;
    fit.min_r2 = 0;
    fit.min_ratio = 0;
    fit.low_signal = velocity::LowSignal::keep_filters;
    return fit;
}

CopiedFit run_gpu_uniform(vc::PreparedInput& prepared, ac::Stream stream,
                          const velocity::FitOptions& fit_options,
                          vc::ExpressionSource* expression_source = nullptr,
                          const std::uint8_t* source_mask = nullptr,
                          std::uint8_t* promotion_status = nullptr) {
    velocity::RunOptions run;
    set_steady_options(run);
    vc::Workspace workspace(prepared, run);
    vc::FitResult result(prepared, run.compute_precision);
    DeviceDense velocity_output(promotion_rows, promotion_genes,
                                ac::ScalarType::float32, stream.device);
    vc::Outputs outputs;
    outputs.velocity = velocity_output.view();
    vc::ReferenceStages stages(prepared, workspace, result.view(), fit_options, outputs, stream);
    auto moments = expression_source
        ? stages.moments_from(*expression_source, 0, promotion_genes, source_mask, promotion_status)
        : stages.moments(0, promotion_genes);
    auto fit = stages.fit(moments);
    stages.emit(moments, fit);
    auto final_view = stages.finish();
    cuda_check(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream.native)),
               "cudaStreamSynchronize steady state");
    return copy_cuda_result(final_view, outputs.velocity, promotion_rows, promotion_genes, stream);
}

velocity::StochasticOptions promotion_stochastic_options() {
    velocity::StochasticOptions options;
    options.initialization = promotion_fit_options();
    options.fit_second_intercept = true;
    options.refit_all_if_none = false;
    return options;
}

void set_stochastic_run_options(velocity::StochasticRunOptions& run) {
    run.compute_precision = velocity::ComputePrecision::float32;
    run.output_dtype = anndata::DType::float32;
    run.block_genes = promotion_genes;
    run.workspace_bytes = 1u << 20;
    run.velocity = true;
}

CopiedStochasticFit run_gpu_stochastic(vc::PreparedInput& prepared, ac::Stream stream,
                                      const velocity::StochasticOptions& options,
                                      vc::ExpressionSource* expression_source = nullptr,
                                      const std::uint8_t* source_mask = nullptr,
                                      std::uint8_t* promotion_status = nullptr) {
    velocity::StochasticRunOptions run;
    set_stochastic_run_options(run);
    vc::Workspace workspace(prepared, run);
    vc::FitResult result(prepared, run.compute_precision, true);
    vc::ReferenceStages stages(prepared, workspace, result.view(), options, {}, stream);
    auto initialization_moments = expression_source
        ? stages.initial_moments_from(*expression_source, 0, promotion_genes,
                                      source_mask, promotion_status)
        : stages.initial_moments(0, promotion_genes);
    auto initialization_fit = stages.initial_fit(initialization_moments);
    stages.accept_initialization(initialization_moments, initialization_fit);
    stages.resolve_selection();
    auto moments = stages.moments(0, promotion_genes);
    auto scales = stages.scales(moments);
    auto fit = stages.fit(moments, scales);
    stages.emit(moments, fit);
    auto final_view = stages.finish();
    cuda_check(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream.native)),
               "cudaStreamSynchronize stochastic stages");
    return copy_cuda_stochastic_result(final_view, promotion_genes, stream);
}

void stochastic_source_disappears_case(vc::PreparedInput& prepared, ac::Stream stream,
                                       const std::vector<float>& original_s,
                                       const std::vector<float>& original_u) {
    DisappearingBlockSource provider(promotion_rows, promotion_genes, stream.device,
                                     original_s, original_u);
    vc::ExpressionSource expression(prepared,
        vc::BlockSource{&provider, &DisappearingBlockSource::acquire,
                        provider.staging_bytes()});
    velocity::StochasticRunOptions run;
    set_stochastic_run_options(run);
    vc::Workspace workspace(prepared, run);
    vc::FitResult result(prepared, run.compute_precision, true);
    DeviceDense velocity_output(promotion_rows, promotion_genes,
                                ac::ScalarType::float32, stream.device);
    std::vector<float> sentinel(promotion_rows * promotion_genes, -73.0f);
    velocity_output.upload(sentinel, stream);
    vc::Outputs outputs;
    outputs.velocity = velocity_output.view();
    vc::ReferenceStages stages(prepared, workspace, result.view(),
                               promotion_stochastic_options(), outputs, stream);
    const std::array<std::uint8_t, promotion_genes> mask{1, 0, 0};
    std::array<std::uint8_t, promotion_genes> promotion_status{255, 255, 255};
    auto initialization = stages.initial_moments_from(expression, 0, promotion_genes,
                                                       mask.data(), promotion_status.data());
    auto initialization_fit = stages.initial_fit(initialization);
    stages.accept_initialization(initialization, initialization_fit);
    stages.resolve_selection();
    require(promotion_status[0] == static_cast<std::uint8_t>(
                velocity::precision::PromotionStatus::promoted),
            "disappearance fixture must first promote the requested gene");

    bool second_pass_returned = false, fit_called = false, emit_called = false;
    try {
        auto second = stages.moments(0, promotion_genes);
        second_pass_returned = true;
        fit_called = true;
        auto scales = stages.scales(second);
        auto fit = stages.fit(second, scales);
        emit_called = true;
        stages.emit(second, fit);
    } catch (const std::runtime_error&) {
    }
    require(!second_pass_returned,
            "a promoted original disappearing before stochastic moments must throw");
    require(!fit_called && !emit_called,
            "source disappearance must fail before stochastic fitting or output emission");
    require(provider.requested_only_gene_zero_twice(),
            "stochastic source reacquisition must request only the previously promoted gene");
    cuda_check(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream.native)),
               "cudaStreamSynchronize disappeared source");
    const auto output = download_column_major<float>(velocity_output.view(),
        promotion_rows, promotion_genes, stream);
    require(output == sentinel,
            "failed second pass must leave the caller velocity output untouched");
}

struct CpuCopiedFit {
    std::vector<float> gamma, offset, r2, velocity;
    std::vector<std::uint8_t> selected, status;
    bool low_signal = false;
};

CpuCopiedFit run_cpu_original(const std::vector<float>& weights,
                              std::vector<float>& spliced,
                              std::vector<float>& unspliced) {
    auto wa = anndata::Array::borrow_dense(weights.data(),
        {promotion_rows, promotion_rows}, {1, std::ptrdiff_t(promotion_rows)});
    auto sa = anndata::Array::borrow_dense(spliced.data(),
        {promotion_rows, promotion_genes}, {1, std::ptrdiff_t(promotion_rows)});
    auto ua = anndata::Array::borrow_dense(unspliced.data(),
        {promotion_rows, promotion_genes}, {1, std::ptrdiff_t(promotion_rows)});
    velocity::PreparedInput prepared({sa.view(), ua.view(), wa.view()});
    velocity::RunOptions run;
    set_steady_options(run);
    velocity::Workspace workspace(prepared, run);
    auto output = anndata::Array::empty<float>({promotion_rows, promotion_genes});
    auto result = velocity::reference::steady_state_into(prepared, workspace, output.edit(),
                                                         promotion_fit_options());
    auto gamma = result.gamma.view().dense<float>();
    auto offset = result.offset.view().dense<float>();
    auto r2 = result.r2.view().dense<float>();
    auto selected = result.genes.view().dense<bool>();
    auto status = result.status.view().dense<std::uint8_t>();
    auto velocity = output.view().dense<float>();
    CpuCopiedFit copy;
    copy.gamma.assign(gamma.data(), gamma.data() + promotion_genes);
    copy.offset.assign(offset.data(), offset.data() + promotion_genes);
    copy.r2.assign(r2.data(), r2.data() + promotion_genes);
    copy.selected.resize(promotion_genes);
    for (std::size_t j = 0; j < promotion_genes; ++j)
        copy.selected[j] = selected.data()[j] ? 1 : 0;
    copy.status.assign(status.data(), status.data() + promotion_genes);
    copy.velocity.resize(promotion_rows * promotion_genes);
    for (std::size_t i = 0; i < promotion_rows; ++i)
        for (std::size_t j = 0; j < promotion_genes; ++j)
            copy.velocity[i * promotion_genes + j] =
                velocity.data()[std::ptrdiff_t(i) * velocity.stride(0) +
                                std::ptrdiff_t(j) * velocity.stride(1)];
    copy.low_signal = result.low_signal ? 1 : 0;
    return copy;
}

struct CpuCopiedStochasticFit {
    std::vector<float> gamma, offset, first_scale, second_scale, objective;
    std::vector<std::uint8_t> selected, initial_status, status;
    std::uint8_t low_signal = 0;
    std::uint8_t refit_all = 0;
};

CpuCopiedStochasticFit run_cpu_stochastic_original(const std::vector<float>& weights,
                                                    std::vector<float>& spliced,
                                                    std::vector<float>& unspliced) {
    auto wa = anndata::Array::borrow_dense(weights.data(),
        {promotion_rows, promotion_rows}, {1, std::ptrdiff_t(promotion_rows)});
    auto sa = anndata::Array::borrow_dense(spliced.data(),
        {promotion_rows, promotion_genes}, {1, std::ptrdiff_t(promotion_rows)});
    auto ua = anndata::Array::borrow_dense(unspliced.data(),
        {promotion_rows, promotion_genes}, {1, std::ptrdiff_t(promotion_rows)});
    velocity::PreparedInput prepared({sa.view(), ua.view(), wa.view()});
    velocity::StochasticRunOptions run;
    set_stochastic_run_options(run);
    velocity::StochasticWorkspace workspace(prepared, run);
    auto result = velocity::reference::stochastic(prepared, workspace,
                                                   promotion_stochastic_options());
    auto gamma = result.gamma.view().dense<float>();
    auto offset = result.offset.view().dense<float>();
    auto first_scale = result.first_scale.view().dense<float>();
    auto second_scale = result.second_scale.view().dense<float>();
    auto objective = result.objective.view().dense<float>();
    auto selected = result.initialization.genes.view().dense<bool>();
    auto initial_status = result.initialization.status.view().dense<std::uint8_t>();
    auto status = result.status.view().dense<std::uint8_t>();
    CpuCopiedStochasticFit copy;
    copy.gamma.assign(gamma.data(), gamma.data() + promotion_genes);
    copy.offset.assign(offset.data(), offset.data() + promotion_genes);
    copy.first_scale.assign(first_scale.data(), first_scale.data() + promotion_genes);
    copy.second_scale.assign(second_scale.data(), second_scale.data() + promotion_genes);
    copy.objective.assign(objective.data(), objective.data() + promotion_genes);
    copy.selected.resize(promotion_genes);
    for (std::size_t j = 0; j < promotion_genes; ++j)
        copy.selected[j] = selected.data()[j] ? 1 : 0;
    copy.initial_status.assign(initial_status.data(), initial_status.data() + promotion_genes);
    copy.status.assign(status.data(), status.data() + promotion_genes);
    copy.low_signal = result.initialization.low_signal ? 1 : 0;
    copy.refit_all = result.refit_all ? 1 : 0;
    return copy;
}

void compare_stochastic_gene(const CopiedStochasticFit& actual,
                             const CopiedStochasticFit& expected,
                             std::size_t gene, float tolerance, const char* message) {
    require(close(actual.gamma[gene], expected.gamma[gene], tolerance), message);
    require(close(actual.offset[gene], expected.offset[gene], tolerance), message);
    require(close(actual.first_scale[gene], expected.first_scale[gene], tolerance), message);
    require(close(actual.second_scale[gene], expected.second_scale[gene], tolerance), message);
    require(close(actual.objective[gene], expected.objective[gene], tolerance), message);
    require(actual.selected[gene] == expected.selected[gene],
            "stochastic initialization selection differs");
    require(actual.initial_status[gene] == expected.initial_status[gene],
            "stochastic initialization status differs");
    require(actual.status[gene] == expected.status[gene], "stochastic fit status differs");
}

void compare_stochastic_gene(const CopiedStochasticFit& actual,
                             const CpuCopiedStochasticFit& expected,
                             std::size_t gene, const char* message) {
    require(close_fp32_gate(actual.gamma[gene], expected.gamma[gene]), message);
    require(close_fp32_gate(actual.offset[gene], expected.offset[gene]), message);
    require(close_fp32_gate(actual.first_scale[gene], expected.first_scale[gene]), message);
    require(close_fp32_gate(actual.second_scale[gene], expected.second_scale[gene]), message);
    require(close_fp32_gate(actual.objective[gene], expected.objective[gene]), message);
    require(actual.selected[gene] == expected.selected[gene],
            "stochastic initialization selection differs from CPU");
    require(actual.initial_status[gene] == expected.initial_status[gene],
            "stochastic initialization status differs from CPU");
    require(actual.status[gene] == expected.status[gene],
            "stochastic fit status differs from CPU");
}

void make_promotion_data(std::vector<float>& spliced,
                         std::vector<float>& unspliced,
                         std::vector<float>& weights) {
    spliced.resize(promotion_rows * promotion_genes);
    unspliced.resize(promotion_rows * promotion_genes);
    weights.assign(promotion_rows * promotion_rows, 0.0f);
    for (std::size_t i = 0; i < promotion_rows; ++i) {
        weights[i + i * promotion_rows] = 1.0f;
        for (std::size_t j = 0; j < promotion_genes; ++j) {
            const float x = (0.11f + 0.075f * float(j)) * float(i + 1) +
                            float(int((i * 7 + j * 3) % 7) - 3) * 0.0097f;
            const float residual = float(int((i * 5 + j * 2) % 9) - 4) * 0.025f;
            spliced[j * promotion_rows + i] = x;
            unspliced[j * promotion_rows + i] =
                (1.25f + .17f * float(j)) * x + .12f * float(j) + residual;
        }
    }
}

void compare_fit_gene(const CopiedFit& gpu, const CopiedFit& reference,
                      std::size_t gene, const char* label) {
    require(close_fp32_gate(gpu.gamma[gene], reference.gamma[gene]), label);
    require(close_fp32_gate(gpu.offset[gene], reference.offset[gene]), label);
    require(close_fp32_gate(gpu.r2[gene], reference.r2[gene]), label);
    require(gpu.selected[gene] == reference.selected[gene], "gene-selection result differs from GPU reference");
    require(gpu.status[gene] == reference.status[gene], "fit status differs from GPU reference");
    for (std::size_t row = 0; row < promotion_rows; ++row)
        require(close_fp32_gate(gpu.velocity[gene * promotion_rows + row],
                                reference.velocity[gene * promotion_rows + row]),
                "velocity output differs from GPU reference");
}

void compare_fit_gene(const CopiedFit& gpu, const CpuCopiedFit& cpu,
                      std::size_t gene, const char* label) {
    require(close_fp32_gate(gpu.gamma[gene], cpu.gamma[gene]), label);
    require(close_fp32_gate(gpu.offset[gene], cpu.offset[gene]), label);
    require(close_fp32_gate(gpu.r2[gene], cpu.r2[gene]), label);
    require(gpu.selected[gene] == cpu.selected[gene], "gene-selection result differs from CPU reference");
    require(gpu.status[gene] == cpu.status[gene], "fit status differs from CPU reference");
    for (std::size_t row = 0; row < promotion_rows; ++row)
        require(close_fp32_gate(gpu.velocity[gene * promotion_rows + row],
                                cpu.velocity[row * promotion_genes + gene]),
                "velocity output differs from CPU reference");
}

void original_provider_budget_case(vc::PreparedInput& prepared, OriginalBlockSource& source,
                                   ac::Stream stream,
                                   const std::array<std::uint8_t, promotion_genes>& mask,
                                   const CopiedFit& resident_half) {
    vc::BlockSource block_source{&source, &OriginalBlockSource::acquire, 2u << 20};
    vc::ExpressionSource expression(prepared, block_source);
    velocity::RunOptions run;
    set_steady_options(run);
    vc::Workspace workspace(prepared, run);
    vc::FitResult result(prepared, run.compute_precision);
    DeviceDense velocity_output(promotion_rows, promotion_genes,
                                ac::ScalarType::float32, stream.device);
    vc::Outputs outputs;
    outputs.velocity = velocity_output.view();
    vc::ReferenceStages stages(prepared, workspace, result.view(), promotion_fit_options(),
                               outputs, stream);

    std::array<std::uint8_t, promotion_genes> statuses{255, 255, 255};
    bool rejected = false;
    try {
        (void)stages.moments_from(expression, 0, promotion_genes, mask.data(), statuses.data());
    } catch (const std::length_error&) {
        rejected = true;
    }
    require(rejected, "over-budget original provider must reject an explicit promotion request");
    require(source.acquisitions.empty(),
            "over-budget original provider must reject before invoking its callback");

    // The same provider declaration is irrelevant when no promotion is requested.
    auto moments = stages.moments_from(expression, 0, promotion_genes, nullptr, statuses.data());
    auto fit = stages.fit(moments);
    stages.emit(moments, fit);
    const auto final_view = stages.finish();
    cuda_check(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream.native)),
               "cudaStreamSynchronize over-budget null-mask source");
    const auto actual = copy_cuda_result(final_view, outputs.velocity,
                                         promotion_rows, promotion_genes, stream);
    require(source.acquisitions.empty(),
            "null-mask request must bypass an over-budget original provider without acquisition");
    require(std::all_of(statuses.begin(), statuses.end(), [](auto status) {
                return status == static_cast<std::uint8_t>(
                    velocity::precision::PromotionStatus::not_requested);
            }), "null-mask request must set over-budget provider statuses to not_requested");
    for (std::size_t gene = 0; gene < promotion_genes; ++gene)
        compare_fit_gene(actual, resident_half, gene,
                         "over-budget null-mask result changed resident half precision");
}

void selective_expression_source_case() {
    static_assert(sizeof(anndata::Float16) == sizeof(std::uint16_t));
    std::vector<float> original_s, original_u, host_w;
    make_promotion_data(original_s, original_u, host_w);
    std::vector<anndata::Float16> half_s(original_s.size()), half_u(original_u.size());
    for (std::size_t i = 0; i < original_s.size(); ++i) {
        half_s[i] = anndata::Float16::from_float(original_s[i]);
        half_u[i] = anndata::Float16::from_float(original_u[i]);
    }

    StreamOwner stream_owner;
    const auto stream = stream_owner.view();
    require(stream.native != nullptr, "selective-source run must use a non-default stream");
    DeviceDense weights(promotion_rows, promotion_rows, ac::ScalarType::float32, stream.device);
    DeviceDense original_gpu_s(promotion_rows, promotion_genes, ac::ScalarType::float32, stream.device);
    DeviceDense original_gpu_u(promotion_rows, promotion_genes, ac::ScalarType::float32, stream.device);
    DeviceDense base_half_s(promotion_rows, promotion_genes, ac::ScalarType::float16, stream.device);
    DeviceDense base_half_u(promotion_rows, promotion_genes, ac::ScalarType::float16, stream.device);
    weights.upload(host_w, stream);
    original_gpu_s.upload(original_s, stream);
    original_gpu_u.upload(original_u, stream);
    base_half_s.upload(half_s, stream);
    base_half_u.upload(half_u, stream);

    vc::Input original_input;
    original_input.weights = vc::MatrixView(weights.view());
    original_input.spliced = vc::MatrixView(original_gpu_s.view());
    original_input.unspliced = vc::MatrixView(original_gpu_u.view());
    vc::PreparedInput original_prepared(std::move(original_input), stream);
    const auto fit_options = promotion_fit_options();
    auto gpu_original = run_gpu_uniform(original_prepared, stream, fit_options);
    auto cpu_original = run_cpu_original(host_w, original_s, original_u);
    require(cpu_original.low_signal == 0 && gpu_original.low_signal == cpu_original.low_signal,
            "all-original CPU/GPU reference low_signal flag differs");
    require(gpu_original.refit_all == 0,
            "steady-state result must leave the stochastic refit_all flag clear");

    vc::Input half_input;
    half_input.weights = vc::MatrixView(weights.view());
    half_input.spliced = vc::MatrixView(base_half_s.view());
    half_input.unspliced = vc::MatrixView(base_half_u.view());
    vc::PreparedInput half_prepared(std::move(half_input), stream);

    // Ordinary reference stages use the resident half values. Merely providing
    // an ExpressionSource object does not opt the run into original-source reads.
    OriginalBlockSource unused_source(promotion_rows, promotion_genes, stream.device,
                                      original_s, original_u,
                                      std::numeric_limits<std::size_t>::max());
    vc::ExpressionSource unused_expression(half_prepared,
        vc::BlockSource{&unused_source, &OriginalBlockSource::acquire,
                        unused_source.staging_bytes()});
    auto gpu_half = run_gpu_uniform(half_prepared, stream, fit_options);
    require(unused_source.acquisitions.empty(),
            "ordinary moments/fit path must not invoke the optional original source");
    const std::array<std::uint8_t, promotion_genes> promotion_mask{1, 0, 1};
    original_provider_budget_case(half_prepared, unused_source, stream, promotion_mask, gpu_half);

    std::array<std::uint8_t, promotion_genes> default_status{255, 255, 255};
    auto gpu_default_source = run_gpu_uniform(half_prepared, stream, fit_options,
                                             &unused_expression, nullptr,
                                             default_status.data());
    require(unused_source.acquisitions.empty(),
            "a null promotion mask must not invoke the original-source callback");
    require(std::all_of(default_status.begin(), default_status.end(), [](auto status) {
                return status == static_cast<std::uint8_t>(
                    velocity::precision::PromotionStatus::not_requested);
            }), "a null promotion mask must leave all promotion statuses not_requested");
    for (std::size_t gene = 0; gene < promotion_genes; ++gene)
        require(close(gpu_default_source.gamma[gene], gpu_half.gamma[gene], 0.0f) &&
                close(gpu_default_source.offset[gene], gpu_half.offset[gene], 0.0f) &&
                gpu_default_source.status[gene] == gpu_half.status[gene],
                "null-mask default source path must retain the resident half result");

    OriginalBlockSource source(promotion_rows, promotion_genes, stream.device,
                               original_s, original_u, 2);
    vc::ExpressionSource expression(half_prepared,
        vc::BlockSource{&source, &OriginalBlockSource::acquire, source.staging_bytes()});
    std::array<std::uint8_t, promotion_genes> promotion_status{255, 255, 255};
    auto gpu_selective = run_gpu_uniform(half_prepared, stream, fit_options,
                                         &expression, promotion_mask.data(), promotion_status.data());
    cuda_check(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream.native)),
               "cudaStreamSynchronize original-source callbacks");
    require(source.received_only({0, 2}),
            "original source callback must receive only explicitly requested source gene IDs");
    require(promotion_status[0] == static_cast<std::uint8_t>(velocity::precision::PromotionStatus::promoted) &&
            promotion_status[1] == static_cast<std::uint8_t>(velocity::precision::PromotionStatus::not_requested) &&
            promotion_status[2] == static_cast<std::uint8_t>(velocity::precision::PromotionStatus::unavailable),
            "selective promotion status must distinguish promoted, unrequested, and unavailable genes");

    compare_fit_gene(gpu_selective, cpu_original, 0,
                     "promoted FP16 source gene differs from the all-original CPU reference");
    compare_fit_gene(gpu_selective, gpu_original, 0,
                     "promoted FP16 source gene differs from the all-original GPU reference");
    for (std::size_t gene : {std::size_t(1), std::size_t(2)}) {
        require(close(gpu_selective.gamma[gene], gpu_half.gamma[gene], 0.0f) &&
                close(gpu_selective.offset[gene], gpu_half.offset[gene], 0.0f) &&
                close(gpu_selective.r2[gene], gpu_half.r2[gene], 0.0f) &&
                gpu_selective.selected[gene] == gpu_half.selected[gene] &&
                gpu_selective.status[gene] == gpu_half.status[gene],
                "unrequested or unavailable source gene must retain its resident half-precision fit");
        for (std::size_t row = 0; row < promotion_rows; ++row)
            require(close(gpu_selective.velocity[gene * promotion_rows + row],
                           gpu_half.velocity[gene * promotion_rows + row], 0.0f),
                    "unrequested or unavailable source gene must retain its resident half velocity");
    }
    require(gpu_selective.low_signal == gpu_half.low_signal,
            "selective promotion must preserve the global low_signal result for the same selected set");

    vc::ExpressionSource half_alternate(half_prepared,
        vc::MatrixView(base_half_s.view()), vc::MatrixView(base_half_u.view()), stream);
    std::array<std::uint8_t, promotion_genes> half_source_status{255, 255, 255};
    auto gpu_half_alternate = run_gpu_uniform(half_prepared, stream, fit_options,
        &half_alternate, promotion_mask.data(), half_source_status.data());
    require(half_source_status[0] == static_cast<std::uint8_t>(
                velocity::precision::PromotionStatus::unavailable) &&
            half_source_status[1] == static_cast<std::uint8_t>(
                velocity::precision::PromotionStatus::not_requested) &&
            half_source_status[2] == static_cast<std::uint8_t>(
                velocity::precision::PromotionStatus::unavailable),
            "an FP16 alternate expression source must be reported unavailable for promotion");
    for (std::size_t gene = 0; gene < promotion_genes; ++gene)
        require(close(gpu_half_alternate.gamma[gene], gpu_half.gamma[gene], 0.0f) &&
                close(gpu_half_alternate.offset[gene], gpu_half.offset[gene], 0.0f) &&
                gpu_half_alternate.status[gene] == gpu_half.status[gene],
                "unavailable FP16 alternate source must preserve the resident half result");

    const auto stochastic_options = promotion_stochastic_options();
    auto gpu_original_stochastic = run_gpu_stochastic(original_prepared, stream, stochastic_options);
    auto cpu_original_stochastic = run_cpu_stochastic_original(host_w, original_s, original_u);
    auto gpu_half_stochastic = run_gpu_stochastic(half_prepared, stream, stochastic_options);
    stochastic_source_disappears_case(half_prepared, stream, original_s, original_u);
    OriginalBlockSource stochastic_source(promotion_rows, promotion_genes, stream.device,
                                          original_s, original_u, 2);
    vc::ExpressionSource stochastic_expression(half_prepared,
        vc::BlockSource{&stochastic_source, &OriginalBlockSource::acquire,
                        stochastic_source.staging_bytes()});
    std::array<std::uint8_t, promotion_genes> stochastic_promotion_status{255, 255, 255};
    auto gpu_selective_stochastic = run_gpu_stochastic(
        half_prepared, stream, stochastic_options, &stochastic_expression,
        promotion_mask.data(), stochastic_promotion_status.data());
    cuda_check(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream.native)),
               "cudaStreamSynchronize stochastic source callbacks");

    require(stochastic_source.received_only_from({0, 2}),
            "stochastic passes may reacquire only explicitly promoted source genes");
    require(stochastic_promotion_status[0] == static_cast<std::uint8_t>(
                velocity::precision::PromotionStatus::promoted) &&
            stochastic_promotion_status[1] == static_cast<std::uint8_t>(
                velocity::precision::PromotionStatus::not_requested) &&
            stochastic_promotion_status[2] == static_cast<std::uint8_t>(
                velocity::precision::PromotionStatus::unavailable),
            "stochastic precision choice must remain pinned across both passes");
    require(gpu_original_stochastic.low_signal == cpu_original_stochastic.low_signal &&
            gpu_original_stochastic.refit_all == cpu_original_stochastic.refit_all,
            "all-original CPU/GPU stochastic global flags differ");
    require(gpu_selective_stochastic.low_signal == gpu_original_stochastic.low_signal &&
            gpu_selective_stochastic.refit_all == gpu_original_stochastic.refit_all &&
            gpu_selective_stochastic.low_signal == gpu_half_stochastic.low_signal &&
            gpu_selective_stochastic.refit_all == gpu_half_stochastic.refit_all,
            "selective stochastic promotion changed global low_signal/refit_all flags");
    require(gpu_selective_stochastic.selected[0] &&
            gpu_selective_stochastic.initial_status[0] ==
                static_cast<std::uint8_t>(velocity::FitStatus::ok),
            "stochastic test fixture must select the promoted gene for refinement");
    compare_stochastic_gene(gpu_original_stochastic, cpu_original_stochastic, 0,
        "all-original GPU stochastic result differs from the CPU reference");
    compare_stochastic_gene(gpu_selective_stochastic, gpu_original_stochastic, 0, 0.0f,
        "promoted stochastic gene differs from the all-original GPU reference");
    compare_stochastic_gene(gpu_selective_stochastic, cpu_original_stochastic, 0,
        "promoted stochastic gene differs from the all-original CPU reference");
    for (std::size_t gene : {std::size_t(1), std::size_t(2)})
        compare_stochastic_gene(gpu_selective_stochastic, gpu_half_stochastic, gene, 0.0f,
            "unrequested or unavailable stochastic gene changed from the resident half reference");
}

} // namespace

int main(int argc, char** argv) {
    bool run_gpu = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string_view(argv[i]) == "--gpu") run_gpu = true;
        else if (std::string_view(argv[i]) == "--help") {
            std::cout << "test_precision_cuda [--gpu]\n"
                         "Without --gpu, no CUDA runtime call is made.\n";
            return 0;
        } else {
            std::cerr << "unknown argument: " << argv[i] << '\n';
            return 2;
        }
    }
    if (!run_gpu) {
        std::cout << "CUDA precision tests skipped; pass --gpu under Device Scheduler.\n";
        return 0;
    }

    try {
        sensitivity_case<float>(false, false, true);
        sensitivity_case<float>(true, true, false, true);
        sensitivity_case<double>(false, false, true);
        sensitivity_case<double>(true, true, false, true);
        selective_expression_source_case();
        std::cout << "CUDA precision and sensitivity tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
