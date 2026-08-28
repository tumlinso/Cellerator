#include <Cellerator/geometry/gating_cuda.cuh>
#include <Cellerator/geometry/layout_selector.hh>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "benchmark_mutex.hh"

namespace {

using clock_type = std::chrono::steady_clock;

struct bench_options {
    cellpack::oracle_gating_scenario scenario = cellpack::oracle_gating_scenario::alternating_modules;
    std::string output_dir;
    cellpack::u32 rows = 4096u;
    cellpack::u32 features = 2048u;
    cellpack::u32 modules = 32u;
    int warmup = 1;
    int repeats = 5;
};

struct synthetic_fixture {
    std::vector<cellpack::u32> feature_modules;
    std::vector<cellpack::u32> signature_offsets;
    std::vector<cellpack::u32> signature_modules;
    std::vector<cellpack::u32> row_offsets;
    std::vector<cellpack::u32> feature_ids;
    std::vector<float> values;
};

struct phase_times {
    double generate_ms = 0.0;
    double plan_select_ms = 0.0;
    double compile_runtime_ms = 0.0;
    double forward_ms = 0.0;
    double backward_replay_ms = 0.0;
    double validate_ms = 0.0;
};

struct bench_result {
    bench_options options;
    phase_times phases;
    double no_gating_forward_ms = 0.0;
    double no_gating_backward_replay_ms = 0.0;
    double oracle_forward_ms = 0.0;
    double oracle_backward_replay_ms = 0.0;
    cellpack::u32 row_count = 0u;
    cellpack::u32 feature_count = 0u;
    cellpack::u32 nnz = 0u;
    cellpack::u32 region_count = 0u;
    cellpack::u32 no_gating_region_count = 0u;
    cellpack::u32 oracle_region_count = 0u;
    bool correctness_passed = false;
};

template <typename T>
class device_buffer {
public:
    device_buffer() = default;
    explicit device_buffer(std::size_t count) { reset(count); }
    ~device_buffer() { if (ptr_ != nullptr) cudaFree(ptr_); }

    device_buffer(const device_buffer &) = delete;
    device_buffer &operator=(const device_buffer &) = delete;

    void reset(std::size_t count) {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
        count_ = count;
        if (count_ != 0u) check_cuda(cudaMalloc(reinterpret_cast<void **>(&ptr_), count_ * sizeof(T)), "cudaMalloc");
    }

    void copy_from_host(const T *src, std::size_t count) {
        if (count > count_) throw std::runtime_error("device copy exceeds allocation");
        if (count != 0u) check_cuda(cudaMemcpy(ptr_, src, count * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy H2D");
    }

    std::vector<T> copy_to_host(std::size_t count) const {
        if (count > count_) throw std::runtime_error("device copy exceeds allocation");
        std::vector<T> out(count);
        if (count != 0u) check_cuda(cudaMemcpy(out.data(), ptr_, count * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
        return out;
    }

    void zero() {
        if (count_ != 0u) check_cuda(cudaMemset(ptr_, 0, count_ * sizeof(T)), "cudaMemset");
    }

    T *get() { return ptr_; }
    const T *get() const { return ptr_; }

private:
    static void check_cuda(cudaError_t status, const char *label) {
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string(label) + ": " + cudaGetErrorString(status));
        }
    }

    T *ptr_ = nullptr;
    std::size_t count_ = 0u;
};

double elapsed_ms(clock_type::time_point begin, clock_type::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

void check_cuda(cudaError_t status, const char *label) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(label) + ": " + cudaGetErrorString(status));
    }
}

cellpack::u32 parse_u32(const char *text, const char *name) {
    char *end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (end == text || *end != '\0' || value > 0xfffffffful) {
        throw std::runtime_error(std::string("invalid ") + name + ": " + text);
    }
    return static_cast<cellpack::u32>(value);
}

int parse_int(const char *text, const char *name) {
    char *end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < 0 || value > 1000000) {
        throw std::runtime_error(std::string("invalid ") + name + ": " + text);
    }
    return static_cast<int>(value);
}

bench_options parse_args(int argc, char **argv) {
    bench_options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char *name) -> const char * {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (arg == "--scenario") {
            cellpack::oracle_gating_scenario scenario;
            if (!cellpack::parse_oracle_gating_scenario(require_value("--scenario"), &scenario)) {
                throw std::runtime_error("unknown gating scenario");
            }
            options.scenario = scenario;
        } else if (arg == "--rows") options.rows = parse_u32(require_value("--rows"), "rows");
        else if (arg == "--features") options.features = parse_u32(require_value("--features"), "features");
        else if (arg == "--modules") options.modules = parse_u32(require_value("--modules"), "modules");
        else if (arg == "--warmup") options.warmup = parse_int(require_value("--warmup"), "warmup");
        else if (arg == "--repeats") options.repeats = parse_int(require_value("--repeats"), "repeats");
        else if (arg == "--output-dir") options.output_dir = require_value("--output-dir");
        else throw std::runtime_error("unknown argument: " + arg);
    }
    if (options.rows == 0u || options.features == 0u || options.modules == 0u) {
        throw std::runtime_error("rows, features, and modules must be nonzero");
    }
    if (options.repeats <= 0) throw std::runtime_error("repeats must be nonzero");
    options.modules = std::min(options.modules, options.features);
    return options;
}

std::vector<cellpack::u32> make_row_features(
    const bench_options &options,
    cellpack::u32 row,
    cellpack::u32 module_width,
    cellpack::u32 residual_begin) {
    std::vector<cellpack::u32> features;
    const cellpack::u32 module = row % options.modules;
    const cellpack::u32 begin = module * module_width;
    const cellpack::u32 end = std::min(residual_begin, begin + module_width);
    cellpack::u32 active = std::min<cellpack::u32>(module_width, 16u);
    if (options.scenario == cellpack::oracle_gating_scenario::dense_tile_only) {
        active = end > begin ? std::max<cellpack::u32>(1u, ((end - begin) * 7u) / 8u) : 0u;
    } else if (options.scenario == cellpack::oracle_gating_scenario::conditional_only) {
        active = std::min<cellpack::u32>(module_width, 8u);
    }
    for (cellpack::u32 i = 0; i < active && begin + i < end; ++i) {
        features.push_back(begin + i);
    }
    if (options.scenario == cellpack::oracle_gating_scenario::high_residual_skip && residual_begin < options.features) {
        const cellpack::u32 residual_count = options.features - residual_begin;
        for (cellpack::u32 i = 0; i < std::min<cellpack::u32>(residual_count, 8u); ++i) {
            features.push_back(residual_begin + ((row * 17u + i * 13u) % residual_count));
        }
    }
    std::sort(features.begin(), features.end());
    features.erase(std::unique(features.begin(), features.end()), features.end());
    return features;
}

synthetic_fixture generate_fixture(const bench_options &options) {
    synthetic_fixture fixture;
    const bool residual_heavy = options.scenario == cellpack::oracle_gating_scenario::high_residual_skip;
    const cellpack::u32 residual_count = residual_heavy ? std::max<cellpack::u32>(1u, options.features / 8u) : 0u;
    const cellpack::u32 residual_begin = options.features - residual_count;
    const cellpack::u32 module_width = std::max<cellpack::u32>(1u, residual_begin / options.modules);

    fixture.feature_modules.resize(options.features, cellpack::default_residual_module_id);
    for (cellpack::u32 module = 0; module < options.modules; ++module) {
        const cellpack::u32 begin = module * module_width;
        const cellpack::u32 end = std::min(residual_begin, begin + module_width);
        for (cellpack::u32 feature = begin; feature < end; ++feature) {
            fixture.feature_modules[feature] = module;
        }
    }

    fixture.signature_offsets.reserve(static_cast<std::size_t>(options.rows) + 1u);
    fixture.row_offsets.reserve(static_cast<std::size_t>(options.rows) + 1u);
    fixture.signature_offsets.push_back(0u);
    fixture.row_offsets.push_back(0u);
    for (cellpack::u32 row = 0; row < options.rows; ++row) {
        fixture.signature_modules.push_back(row % options.modules);
        if (residual_heavy) fixture.signature_modules.push_back(cellpack::default_residual_module_id);
        fixture.signature_offsets.push_back(static_cast<cellpack::u32>(fixture.signature_modules.size()));

        std::vector<cellpack::u32> row_features = make_row_features(options, row, module_width, residual_begin);
        fixture.feature_ids.insert(fixture.feature_ids.end(), row_features.begin(), row_features.end());
        for (cellpack::u32 feature : row_features) {
            fixture.values.push_back(0.5f + static_cast<float>((row + feature) % 11u) * 0.125f);
        }
        fixture.row_offsets.push_back(static_cast<cellpack::u32>(fixture.feature_ids.size()));
    }
    return fixture;
}

bool route_contains(const cellpack::route_mask &mask, cellpack::u32 region_id) {
    for (cellpack::u32 active : mask.region_ids) {
        if (active == region_id) return true;
    }
    return false;
}

std::vector<float> cpu_forward_reference(
    const cellpack::packed_coordinate_plan &packed,
    const cellpack::route_mask &mask,
    const std::vector<float> &x) {
    std::vector<float> y(packed.row_count, 0.0f);
    for (const cellpack::packed_coordinate &coordinate : packed.coordinates) {
        if (route_contains(mask, coordinate.region_id)) {
            y[coordinate.original_row] += coordinate.value * x[coordinate.original_feature];
        }
    }
    return y;
}

std::vector<float> cpu_backward_reference(
    const cellpack::packed_coordinate_plan &packed,
    const cellpack::route_mask &mask,
    const std::vector<float> &grad_y) {
    std::vector<float> grad_x(packed.feature_count, 0.0f);
    for (const cellpack::packed_coordinate &coordinate : packed.coordinates) {
        if (route_contains(mask, coordinate.region_id)) {
            grad_x[coordinate.original_feature] += coordinate.value * grad_y[coordinate.original_row];
        }
    }
    return grad_x;
}

void require_close(const std::vector<float> &actual, const std::vector<float> &expected, const char *message) {
    if (actual.size() != expected.size()) throw std::runtime_error(message);
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (std::fabs(actual[i] - expected[i]) > 1.0e-4f) throw std::runtime_error(message);
    }
}

double time_forward(
    cellpack::device_coordinate_plan_view device_plan,
    cellpack::route_mask_view mask,
    const float *d_x,
    float *d_y,
    int warmup,
    int repeats) {
    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    check_cuda(cudaEventCreate(&begin), "cudaEventCreate begin");
    check_cuda(cudaEventCreate(&end), "cudaEventCreate end");
    for (int i = 0; i < warmup; ++i) {
        check_cuda(cudaMemset(d_y, 0, static_cast<std::size_t>(device_plan.row_count) * sizeof(float)), "cudaMemset warmup y");
        check_cuda(cellpack::launch_route_forward(device_plan, mask, d_x, d_y), "warmup forward");
    }
    check_cuda(cudaDeviceSynchronize(), "sync warmup forward");
    check_cuda(cudaEventRecord(begin), "record begin forward");
    for (int i = 0; i < repeats; ++i) {
        check_cuda(cudaMemset(d_y, 0, static_cast<std::size_t>(device_plan.row_count) * sizeof(float)), "cudaMemset y");
        check_cuda(cellpack::launch_route_forward(device_plan, mask, d_x, d_y), "launch forward");
    }
    check_cuda(cudaEventRecord(end), "record end forward");
    check_cuda(cudaEventSynchronize(end), "sync end forward");
    float ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&ms, begin, end), "elapsed forward");
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    return static_cast<double>(ms) / static_cast<double>(repeats);
}

double time_backward(
    cellpack::device_coordinate_plan_view device_plan,
    cellpack::route_tape_view tape,
    const float *d_grad_y,
    float *d_grad_x,
    int warmup,
    int repeats) {
    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    check_cuda(cudaEventCreate(&begin), "cudaEventCreate begin");
    check_cuda(cudaEventCreate(&end), "cudaEventCreate end");
    for (int i = 0; i < warmup; ++i) {
        check_cuda(cudaMemset(d_grad_x, 0, static_cast<std::size_t>(device_plan.feature_count) * sizeof(float)), "cudaMemset warmup grad_x");
        check_cuda(cellpack::launch_route_backward_replay(device_plan, tape, d_grad_y, d_grad_x), "warmup backward");
    }
    check_cuda(cudaDeviceSynchronize(), "sync warmup backward");
    check_cuda(cudaEventRecord(begin), "record begin backward");
    for (int i = 0; i < repeats; ++i) {
        check_cuda(cudaMemset(d_grad_x, 0, static_cast<std::size_t>(device_plan.feature_count) * sizeof(float)), "cudaMemset grad_x");
        check_cuda(cellpack::launch_route_backward_replay(device_plan, tape, d_grad_y, d_grad_x), "launch backward");
    }
    check_cuda(cudaEventRecord(end), "record end backward");
    check_cuda(cudaEventSynchronize(end), "sync end backward");
    float ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&ms, begin, end), "elapsed backward");
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    return static_cast<double>(ms) / static_cast<double>(repeats);
}

bench_result run_benchmark(const bench_options &options) {
    bench_result result;
    result.options = options;

    clock_type::time_point begin = clock_type::now();
    synthetic_fixture fixture = generate_fixture(options);
    clock_type::time_point after_generate = clock_type::now();

    cellpack::feature_module_assignment_view features;
    features.feature_to_module = fixture.feature_modules.data();
    features.feature_count = options.features;
    features.residual_module_id = cellpack::default_residual_module_id;
    cellpack::row_signature_view rows;
    rows.row_count = options.rows;
    rows.row_offsets = fixture.signature_offsets.data();
    rows.module_ids = fixture.signature_modules.data();
    rows.entry_count = static_cast<cellpack::u32>(fixture.signature_modules.size());

    cellpack::planner_config planner_config;
    planner_config.residual_module_id = cellpack::default_residual_module_id;
    planner_config.min_primary_rows = options.scenario == cellpack::oracle_gating_scenario::conditional_only
        ? options.rows + 1u
        : 2u;
    cellpack::static_plan plan;
    cellpack::validation_result validation = cellpack::build_static_plan(features, rows, planner_config, &plan);
    if (!validation) throw std::runtime_error(validation.message);

    cellpack::csr_view csr;
    csr.row_count = options.rows;
    csr.feature_count = options.features;
    csr.nnz_count = static_cast<cellpack::u32>(fixture.feature_ids.size());
    csr.row_offsets = fixture.row_offsets.data();
    csr.feature_ids = fixture.feature_ids.data();
    csr.values = fixture.values.data();

    cellpack::packed_coordinate_plan packed;
    validation = cellpack::build_packed_coordinate_plan(csr, plan, &packed);
    if (!validation) throw std::runtime_error(validation.message);

    cellpack::layout_metrics_plan metrics;
    validation = cellpack::build_layout_metrics(plan, packed, cellpack::layout_metrics_config{}, &metrics);
    if (!validation) throw std::runtime_error(validation.message);
    cellpack::layout_selector_config selector_config;
    selector_config.min_dense_tile_fill = 0.60;
    selector_config.max_structured_vs_csr_bytes = 1.30;
    cellpack::layout_selection_plan selection;
    validation = cellpack::select_layouts(metrics, selector_config, &selection);
    if (!validation) throw std::runtime_error(validation.message);
    cellpack::static_plan selected_plan;
    validation = cellpack::apply_layout_selection(plan, selection, &selected_plan);
    if (!validation) throw std::runtime_error(validation.message);
    clock_type::time_point after_plan = clock_type::now();

    cellpack::compiled_coordinate_plan compiled;
    validation = cellpack::build_compiled_coordinate_plan(selected_plan, packed, &compiled);
    if (!validation) throw std::runtime_error(validation.message);
    clock_type::time_point after_compile = clock_type::now();

    cellpack::route_mask no_gating;
    validation = cellpack::build_oracle_route_mask(
        selected_plan,
        cellpack::oracle_gating_scenario::all_regions,
        0u,
        &no_gating);
    if (!validation) throw std::runtime_error(validation.message);
    cellpack::route_mask oracle;
    validation = cellpack::build_oracle_route_mask(selected_plan, options.scenario, 0u, &oracle);
    if (!validation) throw std::runtime_error(validation.message);
    validation = cellpack::validate_route_mask_matches_oracle(
        selected_plan,
        options.scenario,
        0u,
        cellpack::view_route_mask(oracle));
    if (!validation) throw std::runtime_error(validation.message);

    cellpack::route_tape no_gating_tape;
    cellpack::route_tape oracle_tape;
    validation = cellpack::record_route_tape(cellpack::view_route_mask(no_gating), &no_gating_tape);
    if (!validation) throw std::runtime_error(validation.message);
    validation = cellpack::record_route_tape(cellpack::view_route_mask(oracle), &oracle_tape);
    if (!validation) throw std::runtime_error(validation.message);

    device_buffer<cellpack::region_coordinate_span> d_spans(compiled.region_spans.size());
    device_buffer<cellpack::u32> d_rows(compiled.row_ids.size());
    device_buffer<cellpack::u32> d_features(compiled.feature_ids.size());
    device_buffer<float> d_values(compiled.values.size());
    d_spans.copy_from_host(compiled.region_spans.data(), compiled.region_spans.size());
    d_rows.copy_from_host(compiled.row_ids.data(), compiled.row_ids.size());
    d_features.copy_from_host(compiled.feature_ids.data(), compiled.feature_ids.size());
    d_values.copy_from_host(compiled.values.data(), compiled.values.size());

    cellpack::device_coordinate_plan_view device_plan;
    device_plan.row_count = compiled.row_count;
    device_plan.feature_count = compiled.feature_count;
    device_plan.region_span_count = static_cast<cellpack::u32>(compiled.region_spans.size());
    device_plan.coordinate_count = static_cast<cellpack::u32>(compiled.values.size());
    device_plan.region_spans = d_spans.get();
    device_plan.row_ids = d_rows.get();
    device_plan.feature_ids = d_features.get();
    device_plan.values = d_values.get();

    device_buffer<cellpack::u32> d_no_mask(no_gating.region_ids.size());
    device_buffer<cellpack::u32> d_oracle_mask(oracle.region_ids.size());
    d_no_mask.copy_from_host(no_gating.region_ids.data(), no_gating.region_ids.size());
    d_oracle_mask.copy_from_host(oracle.region_ids.data(), oracle.region_ids.size());

    std::vector<float> x(options.features);
    for (cellpack::u32 i = 0; i < options.features; ++i) x[i] = 0.25f + static_cast<float>(i % 17u) * 0.0625f;
    std::vector<float> grad_y(options.rows);
    for (cellpack::u32 i = 0; i < options.rows; ++i) grad_y[i] = 0.5f + static_cast<float>(i % 13u) * 0.03125f;

    device_buffer<float> d_x(x.size());
    device_buffer<float> d_y(options.rows);
    device_buffer<float> d_grad_y(grad_y.size());
    device_buffer<float> d_grad_x(options.features);
    d_x.copy_from_host(x.data(), x.size());
    d_grad_y.copy_from_host(grad_y.data(), grad_y.size());

    cellpack::route_mask_view no_mask_view{ d_no_mask.get(), static_cast<cellpack::u32>(no_gating.region_ids.size()) };
    cellpack::route_mask_view oracle_mask_view{ d_oracle_mask.get(), static_cast<cellpack::u32>(oracle.region_ids.size()) };
    cellpack::route_tape_view no_tape_view{ d_no_mask.get(), static_cast<cellpack::u32>(no_gating.region_ids.size()) };
    cellpack::route_tape_view oracle_tape_view{ d_oracle_mask.get(), static_cast<cellpack::u32>(oracle.region_ids.size()) };

    result.no_gating_forward_ms = time_forward(device_plan, no_mask_view, d_x.get(), d_y.get(), options.warmup, options.repeats);
    result.no_gating_backward_replay_ms = time_backward(device_plan, no_tape_view, d_grad_y.get(), d_grad_x.get(), options.warmup, options.repeats);
    result.oracle_forward_ms = time_forward(device_plan, oracle_mask_view, d_x.get(), d_y.get(), options.warmup, options.repeats);
    result.oracle_backward_replay_ms = time_backward(device_plan, oracle_tape_view, d_grad_y.get(), d_grad_x.get(), options.warmup, options.repeats);
    clock_type::time_point after_runtime = clock_type::now();

    d_y.zero();
    check_cuda(cellpack::launch_route_forward(device_plan, no_mask_view, d_x.get(), d_y.get()), "validate no-gating forward");
    check_cuda(cudaDeviceSynchronize(), "sync validate no-gating forward");
    require_close(d_y.copy_to_host(options.rows), cpu_forward_reference(packed, no_gating, x), "no-gating forward validation failed");

    d_y.zero();
    check_cuda(cellpack::launch_route_forward(device_plan, oracle_mask_view, d_x.get(), d_y.get()), "validate oracle forward");
    check_cuda(cudaDeviceSynchronize(), "sync validate oracle forward");
    require_close(d_y.copy_to_host(options.rows), cpu_forward_reference(packed, oracle, x), "oracle forward validation failed");

    d_grad_x.zero();
    check_cuda(cellpack::launch_route_backward_replay(device_plan, oracle_tape_view, d_grad_y.get(), d_grad_x.get()), "validate oracle backward");
    check_cuda(cudaDeviceSynchronize(), "sync validate oracle backward");
    require_close(d_grad_x.copy_to_host(options.features), cpu_backward_reference(packed, oracle, grad_y), "oracle backward validation failed");
    clock_type::time_point after_validate = clock_type::now();

    result.phases.generate_ms = elapsed_ms(begin, after_generate);
    result.phases.plan_select_ms = elapsed_ms(after_generate, after_plan);
    result.phases.compile_runtime_ms = elapsed_ms(after_plan, after_compile);
    result.phases.forward_ms = result.oracle_forward_ms;
    result.phases.backward_replay_ms = result.oracle_backward_replay_ms;
    result.phases.validate_ms = elapsed_ms(after_runtime, after_validate);
    result.row_count = options.rows;
    result.feature_count = options.features;
    result.nnz = static_cast<cellpack::u32>(packed.coordinates.size());
    result.region_count = static_cast<cellpack::u32>(selected_plan.regions.size());
    result.no_gating_region_count = static_cast<cellpack::u32>(no_gating.region_ids.size());
    result.oracle_region_count = static_cast<cellpack::u32>(oracle.region_ids.size());
    result.correctness_passed = true;
    return result;
}

void write_summary_text(const bench_result &result, std::ostream &out) {
    out << "scenario: " << cellpack::oracle_gating_scenario_name(result.options.scenario) << "\n";
    out << "rows: " << result.row_count << "\n";
    out << "features: " << result.feature_count << "\n";
    out << "modules: " << result.options.modules << "\n";
    out << "nnz: " << result.nnz << "\n";
    out << "regions: " << result.region_count << "\n";
    out << "no_gating_regions: " << result.no_gating_region_count << "\n";
    out << "oracle_gating_regions: " << result.oracle_region_count << "\n";
    out << "correctness: " << (result.correctness_passed ? "pass" : "fail") << "\n";
    out << "phases_ms.generate: " << result.phases.generate_ms << "\n";
    out << "phases_ms.plan_select: " << result.phases.plan_select_ms << "\n";
    out << "phases_ms.compile_runtime: " << result.phases.compile_runtime_ms << "\n";
    out << "phases_ms.forward: " << result.phases.forward_ms << "\n";
    out << "phases_ms.backward_replay: " << result.phases.backward_replay_ms << "\n";
    out << "phases_ms.validate: " << result.phases.validate_ms << "\n";
    out << "no_gating.forward_ms: " << result.no_gating_forward_ms << "\n";
    out << "no_gating.backward_replay_ms: " << result.no_gating_backward_replay_ms << "\n";
    out << "oracle_gating.forward_ms: " << result.oracle_forward_ms << "\n";
    out << "oracle_gating.backward_replay_ms: " << result.oracle_backward_replay_ms << "\n";
}

void write_summary_json(const bench_result &result, std::ostream &out) {
    out << "{\n";
    out << "  \"compare_config\": {\n";
    out << "    \"comparison_id\": \"cellpack-m4-oracle-gating\",\n";
    out << "    \"impl_a_name\": \"no_gating\",\n";
    out << "    \"impl_b_name\": \"oracle_gating\",\n";
    out << "    \"scenario_id\": \"" << cellpack::oracle_gating_scenario_name(result.options.scenario) << "\",\n";
    out << "    \"warmup\": " << result.options.warmup << ",\n";
    out << "    \"repeats\": " << result.options.repeats << ",\n";
    out << "    \"mutex_path\": \"COMPARE_BENCHMARK_MUTEX_PATH or CUDA_V100_BENCHMARK_MUTEX_PATH or /tmp/cuda_v100_benchmark.lock\"\n";
    out << "  },\n";
    out << "  \"status\": \"ok\",\n";
    out << "  \"correctness\": " << (result.correctness_passed ? "true" : "false") << ",\n";
    out << "  \"shape\": {\n";
    out << "    \"rows\": " << result.row_count << ",\n";
    out << "    \"features\": " << result.feature_count << ",\n";
    out << "    \"modules\": " << result.options.modules << ",\n";
    out << "    \"nnz\": " << result.nnz << ",\n";
    out << "    \"regions\": " << result.region_count << ",\n";
    out << "    \"no_gating_regions\": " << result.no_gating_region_count << ",\n";
    out << "    \"oracle_gating_regions\": " << result.oracle_region_count << "\n";
    out << "  },\n";
    out << "  \"phases_ms\": {\n";
    out << "    \"generate\": " << result.phases.generate_ms << ",\n";
    out << "    \"plan_select\": " << result.phases.plan_select_ms << ",\n";
    out << "    \"compile_runtime\": " << result.phases.compile_runtime_ms << ",\n";
    out << "    \"forward\": " << result.phases.forward_ms << ",\n";
    out << "    \"backward_replay\": " << result.phases.backward_replay_ms << ",\n";
    out << "    \"validate\": " << result.phases.validate_ms << "\n";
    out << "  },\n";
    out << "  \"implementations\": {\n";
    out << "    \"no_gating\": {\n";
    out << "      \"forward_ms\": " << result.no_gating_forward_ms << ",\n";
    out << "      \"backward_replay_ms\": " << result.no_gating_backward_replay_ms << "\n";
    out << "    },\n";
    out << "    \"oracle_gating\": {\n";
    out << "      \"forward_ms\": " << result.oracle_forward_ms << ",\n";
    out << "      \"backward_replay_ms\": " << result.oracle_backward_replay_ms << "\n";
    out << "    }\n";
    out << "  }\n";
    out << "}\n";
}

void write_outputs(const bench_result &result) {
    write_summary_text(result, std::cout);
    if (result.options.output_dir.empty()) return;
    std::filesystem::create_directories(result.options.output_dir);
    std::ofstream text_out(result.options.output_dir + "/summary.txt");
    std::ofstream json_out(result.options.output_dir + "/summary.json");
    if (!text_out || !json_out) throw std::runtime_error("failed to open benchmark summary output files");
    write_summary_text(result, text_out);
    write_summary_json(result, json_out);
}

} // namespace

int main(int argc, char **argv) {
    try {
        const bench_options options = parse_args(argc, argv);
        cellerator::bench::benchmark_mutex_guard benchmark_mutex("cellPackGatingBench");
        write_outputs(run_benchmark(options));
    } catch (const std::exception &error) {
        std::fprintf(stderr, "cellPackGatingBench: %s\n", error.what());
        return 1;
    }
    return 0;
}
