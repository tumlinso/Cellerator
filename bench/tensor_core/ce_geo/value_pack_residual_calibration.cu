#include "../../../src/compute/architecture/providers/nvidia/sm70/value_pack.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace sm70 =
    cellerator::compute::architecture::providers::nvidia::sm70;
namespace projection = cellerator::compute::projection;

namespace {

constexpr int threads_per_block = 256;

void require_cuda(cudaError_t status, const char *what) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(status));
        std::exit(1);
    }
}

void require(bool condition, const char *what) {
    if (!condition) {
        std::fprintf(stderr, "%s\n", what);
        std::exit(1);
    }
}

template<typename T>
struct device_buffer {
    T *data = nullptr;
    std::size_t count = 0u;
    explicit device_buffer(std::size_t size) : count(size) {
        if (count != 0u)
            require_cuda(cudaMalloc(&data, count * sizeof(T)), "cudaMalloc");
    }
    ~device_buffer() { if (data != nullptr) cudaFree(data); }
    device_buffer(const device_buffer &) = delete;
    device_buffer &operator=(const device_buffer &) = delete;
};

__global__ void dense_input_layout_kernel(
    const float *input, float *packed, std::uint64_t count) {
    for (std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x)
             * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<std::uint64_t>(gridDim.x) * blockDim.x)
        packed[index] = input[count - 1u - index];
}

// Calibration-only stand-in. Production sm70/residual.cu is intentionally not
// emulated here: this measures one bounded residual-shaped memory/convert/add
// phase and is never evidence for the absent production kernel.
__global__ void residual_standin_kernel(const float *packed_input,
    const __half *residual_values, float *output,
    std::uint64_t edge_count, std::uint64_t residual_count) {
    for (std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x)
             * blockDim.x + threadIdx.x;
         index < edge_count;
         index += static_cast<std::uint64_t>(gridDim.x) * blockDim.x)
        output[index] = packed_input[index]
            + (index < residual_count
                ? __half2float(residual_values[index]) : 0.0f);
}

__global__ void epilogue_standin_kernel(
    float *values, std::uint64_t count) {
    for (std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x)
             * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<std::uint64_t>(gridDim.x) * blockDim.x)
        values[index] *= 0.5f;
}

__global__ void output_remap_standin_kernel(
    const float *physical, float *logical, std::uint64_t count) {
    for (std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x)
             * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<std::uint64_t>(gridDim.x) * blockDim.x)
        logical[count - 1u - index] = physical[index];
}

template<typename Function>
double cuda_time_ns(cudaStream_t stream, Function &&function) {
    cudaEvent_t begin = nullptr, end = nullptr;
    require_cuda(cudaEventCreate(&begin), "create begin event");
    require_cuda(cudaEventCreate(&end), "create end event");
    require_cuda(cudaEventRecord(begin, stream), "record begin event");
    function();
    require_cuda(cudaEventRecord(end, stream), "record end event");
    require_cuda(cudaEventSynchronize(end), "synchronize end event");
    float milliseconds = 0.0f;
    require_cuda(cudaEventElapsedTime(&milliseconds, begin, end),
        "elapsed event time");
    require_cuda(cudaEventDestroy(end), "destroy end event");
    require_cuda(cudaEventDestroy(begin), "destroy begin event");
    return static_cast<double>(milliseconds) * 1.0e6;
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

struct phase_samples {
    std::vector<double> dynamic_h2d;
    std::vector<double> value_pack;
    std::vector<double> dense_input_layout;
    std::vector<double> residual;
    std::vector<double> epilogue;
    std::vector<double> output_remap;
    std::vector<double> d2h;
    std::vector<double> consumer_complete;
};

struct scenario_result {
    std::uint64_t edges = 0u;
    std::uint32_t reuse = 0u;
    double host_preparation_ns = 0.0;
    double projection_construction_ns = 0.0;
    double persistent_upload_ns = 0.0;
    phase_samples phases{};
    double median_complete_ns = 0.0;
    double complete_mad_percent = 0.0;
    double max_abs_error = 0.0;
};

std::uint32_t blocks(std::uint64_t count) {
    return static_cast<std::uint32_t>(std::min<std::uint64_t>(65535u,
        (count + threads_per_block - 1u) / threads_per_block));
}

scenario_result run_scenario(std::uint64_t edges, std::uint32_t reuse,
    int warmups, int repeats, cudaStream_t stream) {
    scenario_result result{};
    result.edges = edges;
    result.reuse = reuse;
    const std::uint64_t mma_count = (edges + 1u) / 2u;
    const std::uint64_t residual_count = edges / 2u;

    const auto host_begin = std::chrono::steady_clock::now();
    std::vector<__half> host_logical(edges);
    std::vector<float> host_dense(edges);
    for (std::uint64_t edge = 0u; edge < edges; ++edge) {
        host_logical[edge] = __float2half(
            static_cast<float>((edge % 97u) + 1u) / 16.0f);
        host_dense[edge] = static_cast<float>(
            static_cast<int>(edge % 31u) - 15);
    }
    const auto host_end = std::chrono::steady_clock::now();
    result.host_preparation_ns = std::chrono::duration<double, std::nano>(
        host_end - host_begin).count();

    const auto projection_begin = std::chrono::steady_clock::now();
    std::vector<projection::projection_value_map_v1> host_map(edges);
    for (std::uint64_t edge = 0u; edge < edges; ++edge) {
        host_map[edge].logical_edge_id.value = edge;
        host_map[edge].logical_edge_id.width =
            projection::logical_edge_id_width_v1::u32;
        host_map[edge].region_kind = (edge & 1u) == 0u
            ? projection::physical_region_kind_v1::mma
            : projection::physical_region_kind_v1::residual;
        host_map[edge].region_index = 0u;
        host_map[edge].projection_slot = static_cast<std::uint32_t>(edge / 2u);
    }
    const auto projection_end = std::chrono::steady_clock::now();
    result.projection_construction_ns =
        std::chrono::duration<double, std::nano>(
            projection_end - projection_begin).count();

    device_buffer<projection::projection_value_map_v1> map(edges);
    device_buffer<__half> logical(edges), mma(mma_count), residual(residual_count);
    device_buffer<std::uint64_t> mma_offsets(1u), residual_offsets(1u);
    device_buffer<float> dense(edges), packed(edges), physical(edges), remapped(edges);
    const std::uint64_t zero = 0u;
    result.persistent_upload_ns = cuda_time_ns(stream, [&] {
        require_cuda(cudaMemcpyAsync(map.data, host_map.data(),
            host_map.size() * sizeof(host_map[0]), cudaMemcpyHostToDevice, stream),
            "upload value map");
        require_cuda(cudaMemcpyAsync(mma_offsets.data, &zero, sizeof(zero),
            cudaMemcpyHostToDevice, stream), "upload MMA offset");
        require_cuda(cudaMemcpyAsync(residual_offsets.data, &zero, sizeof(zero),
            cudaMemcpyHostToDevice, stream), "upload residual offset");
    });
    require_cuda(cudaMemcpyAsync(logical.data, host_logical.data(),
        host_logical.size() * sizeof(host_logical[0]), cudaMemcpyHostToDevice,
        stream), "upload logical values");
    require_cuda(cudaMemcpyAsync(dense.data, host_dense.data(),
        host_dense.size() * sizeof(host_dense[0]), cudaMemcpyHostToDevice,
        stream), "upload dense values");
    require_cuda(cudaStreamSynchronize(stream), "finish scenario upload");

    sm70::value_pack_request_v1 pack{};
    pack.value_map = map.data;
    pack.value_map_count = edges;
    pack.logical_edge_values = logical.data;
    pack.logical_edge_count = edges;
    pack.mma_region_offsets = mma_offsets.data;
    pack.mma_region_count = 1u;
    pack.residual_region_offsets = residual_offsets.data;
    pack.residual_region_count = 1u;
    pack.mma_values = mma.data;
    pack.mma_value_count = mma_count;
    pack.residual_values = residual.data;
    pack.residual_value_count = residual_count;
    pack.source_generation = {1u};
    pack.stream = stream;

    std::vector<float> host_result(edges);
    for (int sample = -warmups; sample < repeats; ++sample) {
        const double dynamic_h2d_ns = cuda_time_ns(stream, [&] {
            require_cuda(cudaMemcpyAsync(logical.data, host_logical.data(),
                host_logical.size() * sizeof(host_logical[0]),
                cudaMemcpyHostToDevice, stream),
                "upload dynamic logical values");
            require_cuda(cudaMemcpyAsync(dense.data, host_dense.data(),
                host_dense.size() * sizeof(host_dense[0]),
                cudaMemcpyHostToDevice, stream),
                "upload dynamic dense values");
        });
        sm70::value_pack_state_v1 state{};
        const double value_pack_ns = cuda_time_ns(stream, [&] {
            require(sm70::enqueue_value_pack_v1(pack, &state)
                    == sm70::value_pack_status_v1::success,
                "value pack launch failed");
        });
        const double dense_ns = cuda_time_ns(stream, [&] {
            dense_input_layout_kernel<<<blocks(edges), threads_per_block,
                0u, stream>>>(dense.data, packed.data, edges);
            require_cuda(cudaPeekAtLastError(), "dense layout launch");
        });
        const double residual_ns = cuda_time_ns(stream, [&] {
            residual_standin_kernel<<<blocks(edges), threads_per_block,
                0u, stream>>>(packed.data, residual.data, physical.data,
                    edges, residual_count);
            require_cuda(cudaPeekAtLastError(), "residual stand-in launch");
        });
        const double epilogue_ns = cuda_time_ns(stream, [&] {
            epilogue_standin_kernel<<<blocks(edges), threads_per_block,
                0u, stream>>>(physical.data, edges);
            require_cuda(cudaPeekAtLastError(), "epilogue stand-in launch");
        });
        const double remap_ns = cuda_time_ns(stream, [&] {
            output_remap_standin_kernel<<<blocks(edges), threads_per_block,
                0u, stream>>>(physical.data, remapped.data, edges);
            require_cuda(cudaPeekAtLastError(), "output remap stand-in launch");
        });
        const double d2h_ns = cuda_time_ns(stream, [&] {
            require_cuda(cudaMemcpyAsync(host_result.data(), remapped.data,
                edges * sizeof(float), cudaMemcpyDeviceToHost, stream),
                "download remapped output");
        });
        if (sample >= 0) {
            result.phases.dynamic_h2d.push_back(dynamic_h2d_ns);
            result.phases.value_pack.push_back(value_pack_ns);
            result.phases.dense_input_layout.push_back(dense_ns);
            result.phases.residual.push_back(residual_ns);
            result.phases.epilogue.push_back(epilogue_ns);
            result.phases.output_remap.push_back(remap_ns);
            result.phases.d2h.push_back(d2h_ns);
            result.phases.consumer_complete.push_back(
                dynamic_h2d_ns + value_pack_ns / reuse + dense_ns + residual_ns
                + epilogue_ns + remap_ns + d2h_ns);
        }
    }

    double max_error = 0.0;
    for (std::uint64_t logical_index = 0u; logical_index < edges;
         ++logical_index) {
        const std::uint64_t physical_index = edges - 1u - logical_index;
        const float packed_value = host_dense[logical_index];
        const float residual_value = physical_index < residual_count
            ? __half2float(host_logical[physical_index * 2u + 1u]) : 0.0f;
        const float expected = 0.5f * (packed_value + residual_value);
        max_error = std::max(max_error,
            static_cast<double>(std::fabs(host_result[logical_index] - expected)));
    }
    result.max_abs_error = max_error;
    require(max_error <= 1.0e-6, "calibration pipeline correctness failed");
    result.median_complete_ns = median(result.phases.consumer_complete);
    result.complete_mad_percent = mad_percent(result.phases.consumer_complete);
    return result;
}

void print_samples(const char *name, const std::vector<double> &values) {
    std::printf("\"%s\":[", name);
    for (std::size_t index = 0u; index < values.size(); ++index)
        std::printf("%s%.3f", index == 0u ? "" : ",", values[index]);
    std::printf("]");
}

void print_zero_samples(const char *name, int count) {
    std::printf("\"%s\":[", name);
    for (int index = 0; index < count; ++index)
        std::printf("%s0.0", index == 0 ? "" : ",");
    std::printf("]");
}

} // namespace

int main(int argc, char **argv) {
    int warmups = 2;
    int repeats = 5;
    bool correctness_only = false;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--warmups" && index + 1 < argc)
            warmups = std::atoi(argv[++index]);
        else if (argument == "--repeats" && index + 1 < argc)
            repeats = std::atoi(argv[++index]);
        else if (argument == "--correctness-only")
            correctness_only = true;
        else {
            std::fprintf(stderr, "unknown argument: %s\n", argument.c_str());
            return 2;
        }
    }
    require(warmups >= 1 && repeats >= 5 && (repeats & 1) != 0,
        "warmups/repeats contract is invalid");
    int device = 0;
    require_cuda(cudaGetDevice(&device), "get CUDA device");
    cudaDeviceProp properties{};
    require_cuda(cudaGetDeviceProperties(&properties, device),
        "get CUDA device properties");
    require(properties.major == 7 && properties.minor == 0,
        "value-pack calibration requires sm_70");
    cudaStream_t stream = nullptr;
    require_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        "create calibration stream");

    const std::uint64_t sizes[] = {1024u, 65536u, 1048576u};
    const std::uint32_t reuses[] = {1u, 16u, 128u};
    std::vector<scenario_result> scenarios;
    if (correctness_only) {
        scenarios.push_back(run_scenario(4097u, 3u, warmups, repeats, stream));
    } else {
        for (std::uint64_t size : sizes)
            for (std::uint32_t reuse : reuses)
                scenarios.push_back(run_scenario(
                    size, reuse, warmups, repeats, stream));
    }
    require_cuda(cudaStreamDestroy(stream), "destroy calibration stream");

    double aggregate = 0.0;
    for (const auto &scenario : scenarios)
        aggregate += scenario.median_complete_ns;
    aggregate /= scenarios.size();
    std::printf("{\"schema\":\"CELLERATOR-CE-GEO-MICROCAL/1\",");
    std::printf("\"campaign_id\":\"CE-GEO-111-value-pack-residual\",");
    std::printf("\"complete_ns\":%.3f,\"correctness_passed\":true,",
        aggregate);
    std::printf("\"production\":{\"value_pack\":true,\"residual\":false,"
        "\"dense_input_layout\":false,\"epilogue\":false,"
        "\"output_remap\":false},");
    std::printf("\"limitations\":[\"production sm70/residual.cu absent\","
        "\"non-value-pack phases are calibration-only stand-ins\","
        "\"no main relation kernel or communication phase measured\","
        "\"phases are isolated with CUDA event synchronization\","
        "\"consumer complete is an algebraic phase sum, not a continuous pipeline wall time\"],");
    std::printf("\"methodology\":{\"clock\":\"cuda_event\","
        "\"warmups\":%d,\"repeats\":%d,"
        "\"complete_formula\":\"dynamic_h2d+value_pack/reuse+dense_input_layout+residual+epilogue+output_remap+d2h\"},",
        warmups, repeats);
    std::printf("\"scenarios\":[");
    for (std::size_t index = 0u; index < scenarios.size(); ++index) {
        const auto &scenario = scenarios[index];
        std::printf("%s{\"logical_edges\":%llu,\"reuse\":%u,",
            index == 0u ? "" : ",",
            static_cast<unsigned long long>(scenario.edges), scenario.reuse);
        std::printf("\"cold_ns\":{\"host_preparation\":%.3f,"
            "\"semantic_packing\":0.0,\"projection_construction\":%.3f,"
            "\"backend_prepare\":0.0,\"persistent_upload\":%.3f},",
            scenario.host_preparation_ns,
            scenario.projection_construction_ns,
            scenario.persistent_upload_ns);
        std::printf("\"phase_samples_ns\":{");
        print_samples("dynamic_h2d", scenario.phases.dynamic_h2d);
        std::printf(",");
        print_samples("value_pack", scenario.phases.value_pack); std::printf(",");
        print_samples("dense_input_layout", scenario.phases.dense_input_layout);
        std::printf(","); print_samples("residual", scenario.phases.residual);
        std::printf(","); print_samples("epilogue", scenario.phases.epilogue);
        std::printf(","); print_samples("output_remap", scenario.phases.output_remap);
        std::printf(","); print_zero_samples("kernel", repeats);
        std::printf(","); print_zero_samples("communication", repeats);
        std::printf(",");
        print_samples("d2h", scenario.phases.d2h); std::printf(",");
        print_samples("consumer_complete", scenario.phases.consumer_complete);
        std::printf("},\"summary\":{\"median_complete_ns\":%.3f,"
            "\"mad_percent\":%.6f,\"max_abs_error\":%.9f}}",
            scenario.median_complete_ns, scenario.complete_mad_percent,
            scenario.max_abs_error);
    }
    std::printf("]}\n");
    return 0;
}
