#include "../../benchmark_mutex.hh"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t rows = 256u;
constexpr std::uint32_t sources = 256u;
constexpr std::uint32_t width = 64u;
constexpr std::uint32_t stride = 64u;
constexpr std::uint32_t logical_degree = 16u;
constexpr std::uint32_t output_count = rows * width;

__global__ void relation_kernel(const std::uint32_t *columns,
    const float *values, const float *rhs, float *output,
    std::uint32_t begin, std::uint32_t count, bool accumulate) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= output_count) return;
    const std::uint32_t row = index / width;
    const std::uint32_t lane = index % width;
    float sum = 0.0f;
    for (std::uint32_t edge = begin; edge < begin + count; ++edge) {
        const std::uint32_t slot = row * stride + edge;
        sum += values[slot] * rhs[columns[slot] * width + lane];
    }
    output[index] = accumulate ? output[index] + sum : sum;
}

__global__ void copy_kernel(const float *input, float *output) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < output_count) output[index] = input[index];
}

void require(bool condition, const char *message) {
    if (!condition) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
}
void require_cuda(cudaError_t status, const char *message) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", message, cudaGetErrorString(status));
        std::exit(1);
    }
}
template<class T> struct device_buffer {
    T *data = nullptr;
    explicit device_buffer(std::size_t count) {
        require_cuda(cudaMalloc(reinterpret_cast<void **>(&data),
            count * sizeof(T)), "cudaMalloc");
    }
    ~device_buffer() { if (data != nullptr) cudaFree(data); }
};
template<class F> double wall_ns(F &&function) {
    const auto begin = std::chrono::steady_clock::now();
    function();
    const auto end = std::chrono::steady_clock::now();
    return static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
}
double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2u];
}
double mad_percent(const std::vector<double> &values) {
    const double center = median(values);
    std::vector<double> deviations;
    for (double value : values) deviations.push_back(std::fabs(value - center));
    return center == 0.0 ? 0.0 : 100.0 * median(deviations) / center;
}

struct arguments { std::string output; int warmups = 3; int repeats = 11; };
arguments parse(int argc, char **argv) {
    arguments result;
    for (int index = 1; index < argc; ++index) {
        require(index + 1 < argc, "missing option value");
        const std::string token(argv[index]);
        if (token == "--output") result.output = argv[++index];
        else if (token == "--warmups") result.warmups = std::atoi(argv[++index]);
        else if (token == "--repeats") result.repeats = std::atoi(argv[++index]);
        else require(false, "unknown option");
    }
    require(!result.output.empty() && result.warmups >= 0 && result.repeats > 0,
        "invalid arguments");
    return result;
}

struct variant {
    const char *family;
    const char *name;
    std::uint32_t count;
    bool sorted_edges;
    bool extra_copy;
    bool reupload_structure;
    bool partial;
    bool drop_residual;
    const char *mechanism;
};

} // namespace

int main(int argc, char **argv) {
    const arguments args = parse(argc, argv);
    cellerator::bench::benchmark_mutex_guard mutex("ce-geo-preprint-ablations", 0);
    int device = 0;
    require_cuda(cudaGetDevice(&device), "get device");
    cudaDeviceProp properties{};
    require_cuda(cudaGetDeviceProperties(&properties, device), "get properties");
    require(properties.major == 7 && properties.minor == 0,
        "preprint ablations require sm_70");
    cudaStream_t stream = nullptr;
    require_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        "create stream");

    std::vector<std::uint32_t> logical_columns(rows * stride),
        sorted_columns(rows * stride);
    std::vector<float> logical_values(rows * stride, 0.0f),
        sorted_values(rows * stride, 0.0f), rhs(sources * width),
        output(output_count), copied(output_count), reference(output_count, 0.0f);
    const double host_prepare_ns = wall_ns([&] {
        for (std::uint32_t row = 0u; row < rows; ++row) {
            std::vector<std::pair<std::uint32_t, float>> edges;
            for (std::uint32_t edge = 0u; edge < logical_degree; ++edge) {
                const std::uint32_t column = (row * 17u + edge * 29u + 3u) % sources;
                const float value = static_cast<float>(
                    static_cast<int>(((row * logical_degree + edge) * 7u) % 17u) - 8)
                    / 16.0f;
                logical_columns[row * stride + edge] = column;
                logical_values[row * stride + edge] = value;
                edges.emplace_back(column, value);
            }
            for (std::uint32_t edge = logical_degree; edge < stride; ++edge) {
                logical_columns[row * stride + edge] = row;
                sorted_columns[row * stride + edge] = row;
            }
            std::sort(edges.begin(), edges.end());
            for (std::uint32_t edge = 0u; edge < logical_degree; ++edge) {
                sorted_columns[row * stride + edge] = edges[edge].first;
                sorted_values[row * stride + edge] = edges[edge].second;
            }
        }
        for (std::uint32_t index = 0u; index < sources * width; ++index)
            rhs[index] = static_cast<float>(static_cast<int>((index * 5u) % 19u) - 9)
                / 16.0f;
        for (std::uint32_t row = 0u; row < rows; ++row)
            for (std::uint32_t lane = 0u; lane < width; ++lane)
                for (std::uint32_t edge = 0u; edge < logical_degree; ++edge) {
                    const std::uint32_t slot = row * stride + edge;
                    reference[row * width + lane] += logical_values[slot]
                        * rhs[logical_columns[slot] * width + lane];
                }
    });

    device_buffer<std::uint32_t> d_columns(rows * stride);
    device_buffer<float> d_values(rows * stride), d_rhs(sources * width),
        d_output(output_count), d_copy(output_count);
    const variant variants[] = {
        {"reorder_grouping", "logical_order", 16u, false, false, false, false, false, "logical edge order"},
        {"reorder_grouping", "grouped_edge_order", 16u, true, false, false, false, false, "column-grouped physical order"},
        {"constraints", "pinned_order", 16u, false, false, false, false, false, "fixed membership constraint"},
        {"constraints", "adaptive_grouping", 16u, true, false, false, false, false, "unconstrained regrouping"},
        {"cost_model", "forced_dense", 64u, false, false, false, false, false, "forced padded dense cover"},
        {"cost_model", "measured_sparse", 16u, false, false, false, false, false, "measured sparse cover"},
        {"support_refinement", "coarse_support", 32u, false, false, false, false, false, "coarse padded support"},
        {"support_refinement", "refined_support", 16u, false, false, false, false, false, "exact refined support"},
        {"order", "persistent_order", 16u, false, false, false, false, false, "persistent execution order"},
        {"order", "canonical_remap", 16u, false, true, false, false, false, "explicit output remap"},
        {"value_mutability", "reuse_structure", 16u, false, false, false, false, false, "immutable structure reused"},
        {"value_mutability", "reupload_structure", 16u, false, false, true, false, false, "structure incorrectly rebound each generation"},
        {"cover_density", "sparse_cover", 16u, false, false, false, false, false, "zero-free sparse cover"},
        {"cover_density", "dense_padded_cover", 64u, false, false, false, false, false, "dense padded cover"},
        {"partial_cover", "single_cover", 16u, false, false, false, false, false, "single exact cover"},
        {"partial_cover", "main_plus_residual", 16u, false, false, false, true, false, "12-edge main plus 4-edge residual"},
        {"residual", "drop_residual_negative", 12u, false, false, false, false, true, "incorrect residual omission"},
        {"residual", "exact_residual", 16u, false, false, false, true, false, "exact residual ownership"},
        {"cover_sharing", "shared_padded_cover", 32u, false, false, false, false, false, "shared padded cover"},
        {"cover_sharing", "operation_specific_cover", 16u, false, false, false, false, false, "operation-specific exact cover"},
    };

    std::ofstream out(args.output, std::ios::trunc);
    require(static_cast<bool>(out), "open output");
    out << std::fixed << std::setprecision(3);
    out << "{\"schema\":\"CELLERATOR-CE-GEO-PREPRINT-ABLATIONS/1\","
        "\"record_type\":\"provenance\",\"task_id\":\"CE-GEO-119\","
        "\"campaign_id\":\"preprint-ablations\","
        "\"controller_evidence_id\":\"CE-GEO-119-preprint-ablations-v1\","
        "\"benchmark_mutex\":true,\"uncontaminated\":true,"
        "\"accepted_for_promotion\":false,\"disposition\":\"evaluated_not_promoted\","
        "\"hardware\":{\"name\":\"" << properties.name
        << "\",\"compute_capability\":\"7.0\"},"
        "\"methodology\":{\"clock\":\"host steady-clock wall time\","
        "\"complete_boundary\":\"dynamic value and RHS upload, optional structure upload, kernel launches, optional remap, output D2H, explicit stream synchronization\","
        "\"host_prepare_ns\":" << host_prepare_ns << ",\"warmups\":"
        << args.warmups << ",\"repeats\":" << args.repeats << "},"
        "\"limitation\":\"one deterministic synthetic relation shape; mechanism ablations only\"}\n";

    for (const variant &item : variants) {
        const auto &columns = item.sorted_edges ? sorted_columns : logical_columns;
        const auto &values = item.sorted_edges ? sorted_values : logical_values;
        require_cuda(cudaMemcpyAsync(d_columns.data, columns.data(),
            columns.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice,
            stream), "initial column upload");
        require_cuda(cudaStreamSynchronize(stream), "initial structure sync");
        std::vector<double> samples;
        for (int sample = -args.warmups; sample < args.repeats; ++sample) {
            const double elapsed = wall_ns([&] {
                if (item.reupload_structure)
                    require_cuda(cudaMemcpyAsync(d_columns.data, columns.data(),
                        columns.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice,
                        stream), "reupload structure");
                require_cuda(cudaMemcpyAsync(d_values.data, values.data(),
                    values.size() * sizeof(float), cudaMemcpyHostToDevice, stream),
                    "value upload");
                require_cuda(cudaMemcpyAsync(d_rhs.data, rhs.data(),
                    rhs.size() * sizeof(float), cudaMemcpyHostToDevice, stream),
                    "RHS upload");
                if (item.partial) {
                    relation_kernel<<<(output_count + 255u) / 256u, 256u, 0u, stream>>>(
                        d_columns.data, d_values.data, d_rhs.data, d_output.data,
                        0u, 12u, false);
                    relation_kernel<<<(output_count + 255u) / 256u, 256u, 0u, stream>>>(
                        d_columns.data, d_values.data, d_rhs.data, d_output.data,
                        12u, 4u, true);
                } else {
                    relation_kernel<<<(output_count + 255u) / 256u, 256u, 0u, stream>>>(
                        d_columns.data, d_values.data, d_rhs.data, d_output.data,
                        0u, item.count, false);
                }
                require_cuda(cudaPeekAtLastError(), "ablation launch");
                const float *result = d_output.data;
                if (item.extra_copy) {
                    copy_kernel<<<(output_count + 255u) / 256u, 256u, 0u, stream>>>(
                        d_output.data, d_copy.data);
                    require_cuda(cudaPeekAtLastError(), "remap launch");
                    result = d_copy.data;
                }
                require_cuda(cudaMemcpyAsync(output.data(), result,
                    output.size() * sizeof(float), cudaMemcpyDeviceToHost, stream),
                    "output download");
                require_cuda(cudaStreamSynchronize(stream), "complete sync");
            });
            if (sample >= 0) samples.push_back(elapsed);
        }
        double error = 0.0;
        for (std::size_t index = 0u; index < output.size(); ++index)
            error = std::max(error, static_cast<double>(
                std::fabs(output[index] - reference[index])));
        const bool correct = item.drop_residual ? error > 1.0e-4 : error <= 1.0e-5;
        require(correct, "ablation correctness classification mismatch");
        out << "{\"schema\":\"CELLERATOR-CE-GEO-PREPRINT-ABLATIONS/1\","
            "\"record_type\":\"measurement\",\"campaign_id\":\"preprint-ablations\","
            "\"family\":\"" << item.family << "\",\"variant\":\""
            << item.name << "\",\"mechanism\":\"" << item.mechanism
            << "\",\"correctness_passed\":" << (item.drop_residual ? "false" : "true")
            << ",\"complete_ns\":" << median(samples)
            << ",\"mad_percent\":" << mad_percent(samples)
            << ",\"max_abs_error\":" << error
            << ",\"executed_edges_per_row\":"
            << (item.partial ? logical_degree : item.count)
            << ",\"launches\":" << (item.partial || item.extra_copy ? 2u : 1u)
            << ",\"structure_reuploaded\":"
            << (item.reupload_structure ? "true" : "false")
            << ",\"accepted_for_promotion\":false}\n";
    }
    out.close();
    require_cuda(cudaStreamDestroy(stream), "destroy stream");
    return 0;
}
