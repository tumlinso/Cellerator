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

constexpr std::uint32_t degree = 8u;
constexpr std::uint32_t block_size = 256u;
constexpr std::uint32_t maximum_dimension = 512u;
constexpr std::uint32_t maximum_width = 512u;

__global__ void row_owned_sparse_width_kernel(
    const std::uint32_t *columns,
    const float *values,
    const float *rhs,
    float *output,
    std::uint32_t dimension,
    std::uint32_t width) {
    const std::uint64_t count = static_cast<std::uint64_t>(dimension) * width;
    for (std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x
             + threadIdx.x;
         index < count;
         index += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
        const std::uint32_t row = static_cast<std::uint32_t>(index / width);
        const std::uint32_t lane = static_cast<std::uint32_t>(index % width);
        float sum = 0.0f;
#pragma unroll
        for (std::uint32_t edge = 0u; edge < degree; ++edge) {
            const std::uint32_t slot = row * degree + edge;
            sum += values[slot] * rhs[columns[slot] * width + lane];
        }
        output[index] = sum;
    }
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
    device_buffer(const device_buffer &) = delete;
    device_buffer &operator=(const device_buffer &) = delete;
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
    deviations.reserve(values.size());
    for (double value : values) deviations.push_back(std::fabs(value - center));
    return center == 0.0 ? 0.0 : 100.0 * median(deviations) / center;
}

struct arguments {
    std::string output;
    int warmups = 1;
    int repeats = 3;
    bool profile_only = false;
};

arguments parse(int argc, char **argv) {
    arguments result;
    for (int index = 1; index < argc; ++index) {
        const std::string token(argv[index]);
        if (token == "--profile-only") result.profile_only = true;
        else {
            require(index + 1 < argc, "missing option value");
            if (token == "--output") result.output = argv[++index];
            else if (token == "--warmups") result.warmups = std::atoi(argv[++index]);
            else if (token == "--repeats") result.repeats = std::atoi(argv[++index]);
            else require(false, "unknown option");
        }
    }
    require(result.profile_only || !result.output.empty(), "output is required");
    require(result.warmups >= 0 && result.repeats > 0, "invalid sample count");
    return result;
}

void build_structure(std::uint32_t dimension,
    std::vector<std::uint32_t> *columns,
    std::vector<float> *values) {
    columns->resize(static_cast<std::size_t>(dimension) * degree);
    values->resize(static_cast<std::size_t>(dimension) * degree);
    for (std::uint32_t row = 0u; row < dimension; ++row)
        for (std::uint32_t edge = 0u; edge < degree; ++edge) {
            const std::uint32_t slot = row * degree + edge;
            (*columns)[slot] = (row * 17u + edge * 29u + 3u) % dimension;
            (*values)[slot] = static_cast<float>(
                static_cast<int>((slot * 7u) % 17u) - 8) / 16.0f;
        }
}

void build_rhs(std::uint32_t dimension, std::uint32_t width,
    std::vector<float> *rhs) {
    rhs->resize(static_cast<std::size_t>(dimension) * width);
    for (std::size_t index = 0u; index < rhs->size(); ++index)
        (*rhs)[index] = static_cast<float>(
            static_cast<int>((index * 5u) % 19u) - 9) / 16.0f;
}

void launch(const std::uint32_t *columns, const float *values,
    const float *rhs, float *output, std::uint32_t dimension,
    std::uint32_t width, cudaStream_t stream) {
    const std::uint64_t count = static_cast<std::uint64_t>(dimension) * width;
    const std::uint32_t grid = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(65535u, (count + block_size - 1u) / block_size));
    row_owned_sparse_width_kernel<<<grid, block_size, 0u, stream>>>(
        columns, values, rhs, output, dimension, width);
    require_cuda(cudaPeekAtLastError(), "launch width sweep kernel");
}

} // namespace

int main(int argc, char **argv) {
    const arguments args = parse(argc, argv);
    cellerator::bench::benchmark_mutex_guard mutex("ce-geo-width-reuse", 0);
    int device = 0;
    require_cuda(cudaGetDevice(&device), "get device");
    cudaDeviceProp properties{};
    require_cuda(cudaGetDeviceProperties(&properties, device), "get properties");
    require(properties.major == 7 && properties.minor == 0,
        "width/reuse sweep requires sm_70");
    cudaStream_t stream = nullptr;
    require_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        "create stream");

    device_buffer<std::uint32_t> d_columns(maximum_dimension * degree);
    device_buffer<float> d_values(maximum_dimension * degree);
    device_buffer<float> d_rhs(maximum_dimension * maximum_width);
    device_buffer<float> d_output(maximum_dimension * maximum_width);

    cudaFuncAttributes attributes{};
    require_cuda(cudaFuncGetAttributes(&attributes, row_owned_sparse_width_kernel),
        "query kernel attributes");
    int active_blocks = 0;
    require_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks, row_owned_sparse_width_kernel, block_size, 0u),
        "query occupancy");
    const double occupancy_percent = 100.0 * active_blocks * block_size
        / properties.maxThreadsPerMultiProcessor;

    if (args.profile_only) {
        std::vector<std::uint32_t> columns;
        std::vector<float> values, rhs;
        build_structure(512u, &columns, &values);
        build_rhs(512u, 512u, &rhs);
        require_cuda(cudaMemcpyAsync(d_columns.data, columns.data(),
            columns.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice,
            stream), "profile columns upload");
        require_cuda(cudaMemcpyAsync(d_values.data, values.data(),
            values.size() * sizeof(float), cudaMemcpyHostToDevice, stream),
            "profile values upload");
        require_cuda(cudaMemcpyAsync(d_rhs.data, rhs.data(),
            rhs.size() * sizeof(float), cudaMemcpyHostToDevice, stream),
            "profile RHS upload");
        launch(d_columns.data, d_values.data, d_rhs.data, d_output.data,
            512u, 512u, stream);
        require_cuda(cudaStreamSynchronize(stream), "profile synchronization");
        require_cuda(cudaStreamDestroy(stream), "destroy stream");
        return 0;
    }

    const std::uint32_t widths[] = {1u, 4u, 8u, 16u, 32u, 64u, 128u, 256u, 512u};
    const std::uint32_t dimensions[] = {16u, 32u, 64u, 128u, 256u, 512u};
    const std::uint32_t reuses[] = {1u, 4u, 16u, 64u, 256u, 1024u};
    std::ofstream out(args.output, std::ios::trunc);
    require(static_cast<bool>(out), "open evidence output");
    out << std::fixed << std::setprecision(3);
    out << "{\"schema\":\"CELLERATOR-CE-GEO-WIDTH-REUSE/1\","
        "\"record_type\":\"provenance\",\"task_id\":\"CE-GEO-114\","
        "\"campaign_id\":\"width-reuse\","
        "\"controller_evidence_id\":\"CE-GEO-114-width-reuse-v1\","
        "\"benchmark_mutex\":true,\"uncontaminated\":true,"
        "\"accepted_for_promotion\":false,\"disposition\":\"evaluated_not_promoted\","
        "\"hardware\":{\"name\":\"" << properties.name
        << "\",\"compute_capability\":\"7.0\"},"
        "\"methodology\":{\"clock\":\"host steady-clock wall time\","
        "\"consumer_complete\":\"dynamic value and RHS upload, one kernel launch, output D2H, explicit cudaStreamSynchronize\","
        "\"preparation\":\"deterministic synthetic row-owned degree-8 structure construction and structure upload amortized by reuse\","
        "\"reuse_1000_plus\":1024,\"warmups\":" << args.warmups
        << ",\"repeats\":" << args.repeats << "},"
        "\"kernel_resources\":{\"registers_per_thread\":" << attributes.numRegs
        << ",\"static_shared_bytes\":" << attributes.sharedSizeBytes
        << ",\"maximum_dynamic_shared_bytes\":" << attributes.maxDynamicSharedSizeBytes
        << ",\"theoretical_occupancy_percent\":" << occupancy_percent
        << "},\"limitations\":[\"synthetic row-owned residual microkernel, not an end-to-end biological promotion campaign\","
        "\"cache and stall counters are supplied only by the controller profiler record when Nsight Compute permits collection\"]}\n";

    std::vector<std::uint32_t> columns;
    std::vector<float> values, rhs, output, reference;
    for (std::uint32_t dimension : dimensions) {
        const double host_structure_ns = wall_ns([&] {
            build_structure(dimension, &columns, &values);
        });
        const double structure_upload_ns = wall_ns([&] {
            require_cuda(cudaMemcpyAsync(d_columns.data, columns.data(),
                columns.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice,
                stream), "columns upload");
            require_cuda(cudaStreamSynchronize(stream), "structure upload sync");
        });
        for (std::uint32_t width : widths) {
            build_rhs(dimension, width, &rhs);
            output.resize(static_cast<std::size_t>(dimension) * width);
            reference.assign(output.size(), 0.0f);
            for (std::uint32_t row = 0u; row < dimension; ++row)
                for (std::uint32_t lane = 0u; lane < width; ++lane)
                    for (std::uint32_t edge = 0u; edge < degree; ++edge) {
                        const std::uint32_t slot = row * degree + edge;
                        reference[static_cast<std::size_t>(row) * width + lane]
                            += values[slot] * rhs[columns[slot] * width + lane];
                    }
            std::vector<double> samples;
            for (int sample = -args.warmups; sample < args.repeats; ++sample) {
                const double complete = wall_ns([&] {
                    require_cuda(cudaMemcpyAsync(d_values.data, values.data(),
                        values.size() * sizeof(float), cudaMemcpyHostToDevice,
                        stream), "values upload");
                    require_cuda(cudaMemcpyAsync(d_rhs.data, rhs.data(),
                        rhs.size() * sizeof(float), cudaMemcpyHostToDevice,
                        stream), "RHS upload");
                    launch(d_columns.data, d_values.data, d_rhs.data,
                        d_output.data, dimension, width, stream);
                    require_cuda(cudaMemcpyAsync(output.data(), d_output.data,
                        output.size() * sizeof(float), cudaMemcpyDeviceToHost,
                        stream), "output download");
                    require_cuda(cudaStreamSynchronize(stream),
                        "consumer-complete synchronization");
                });
                if (sample >= 0) samples.push_back(complete);
            }
            double max_error = 0.0;
            for (std::size_t index = 0u; index < output.size(); ++index)
                max_error = std::max(max_error,
                    static_cast<double>(std::fabs(output[index] - reference[index])));
            require(max_error <= 1.0e-6, "independent reference mismatch");
            const double steady_ns = median(samples);
            const std::uint64_t interactions = static_cast<std::uint64_t>(dimension)
                * width * degree;
            const std::uint64_t estimated_bytes =
                static_cast<std::uint64_t>(dimension) * degree
                    * (sizeof(std::uint32_t) + sizeof(float))
                + static_cast<std::uint64_t>(dimension) * width * sizeof(float) * 2u
                + interactions * sizeof(float);
            for (std::uint32_t reuse : reuses) {
                const double complete_ns = steady_ns
                    + (host_structure_ns + structure_upload_ns) / reuse;
                out << "{\"schema\":\"CELLERATOR-CE-GEO-WIDTH-REUSE/1\","
                    "\"record_type\":\"measurement\",\"campaign_id\":\"width-reuse\","
                    "\"N\":" << width << ",\"D\":" << dimension
                    << ",\"reuse\":" << reuse << ",\"correctness_passed\":true,"
                    "\"complete_ns\":" << complete_ns << ",\"steady_wall_ns\":" << steady_ns
                    << ",\"host_structure_ns\":" << host_structure_ns
                    << ",\"structure_upload_ns\":" << structure_upload_ns
                    << ",\"launches\":1,\"useful_interactions\":" << interactions
                    << ",\"executed_interactions\":" << interactions
                    << ",\"residual_edges\":" << dimension * degree
                    << ",\"useful_fraction\":1.0,\"estimated_bytes\":" << estimated_bytes
                    << ",\"estimated_effective_bandwidth_gbps\":"
                    << estimated_bytes / steady_ns << ",\"max_abs_error\":" << max_error
                    << ",\"mad_percent\":" << mad_percent(samples)
                    << ",\"accepted_for_promotion\":false}\n";
            }
        }
    }
    out.close();
    require_cuda(cudaStreamDestroy(stream), "destroy stream");
    return 0;
}
