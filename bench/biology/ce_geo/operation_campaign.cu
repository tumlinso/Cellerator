#include "../../../src/compute/architecture/providers/nvidia/sm70/relation_apply_n64.cuh"
#include "../../../src/compute/architecture/providers/nvidia/sm70/exchange_cover_native_normalize.cu"
#include "../../benchmark_mutex.hh"

#include <cuda_fp16.h>
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

namespace sm70 = cellerator::compute::architecture::providers::nvidia::sm70;
namespace projection = cellerator::compute::projection;

namespace {

constexpr std::uint32_t relation_rows = 16u;
constexpr std::uint32_t relation_sources = 16u;
constexpr std::uint32_t relation_width = 64u;
constexpr std::uint32_t relation_output_count = relation_rows * relation_width;
constexpr std::uint32_t segment_count = 64u;
constexpr std::uint32_t segment_size = 16u;
constexpr std::uint32_t segment_values = segment_count * segment_size;

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
    for (double value : values) deviations.push_back(std::fabs(value - center));
    return center == 0.0 ? 0.0 : 100.0 * median(deviations) / center;
}

__global__ void hierarchy_pool_broadcast_kernel(const float *members,
    float *pooled, float *broadcast) {
    const std::uint32_t segment = blockIdx.x;
    if (segment >= segment_count || threadIdx.x != 0u) return;
    float sum = 0.0f;
    const std::uint32_t begin = segment * segment_size;
    for (std::uint32_t local = 0u; local < segment_size; ++local)
        sum += members[begin + local];
    pooled[segment] = sum;
    for (std::uint32_t local = 0u; local < segment_size; ++local)
        broadcast[begin + local] = sum;
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

struct measured_case {
    double steady_ns = 0.0;
    double mad = 0.0;
    double preparation_ns = 0.0;
    double error = 0.0;
};

void write_record(std::ofstream &out, const char *operation,
    const char *biological_scope, const char *implementation,
    const measured_case &measured, const char *limitation) {
    out << "{\"schema\":\"CELLERATOR-CE-GEO-BIOLOGY-OPERATIONS/1\","
        "\"record_type\":\"measurement\",\"campaign_id\":\"biology-operations\","
        "\"operation\":\"" << operation << "\",\"biological_scope\":\""
        << biological_scope << "\",\"implementation_class\":\"" << implementation
        << "\",\"correctness_passed\":true,\"complete_ns\":"
        << measured.preparation_ns + measured.steady_ns
        << ",\"steady_wall_ns\":" << measured.steady_ns
        << ",\"preparation_ns\":" << measured.preparation_ns
        << ",\"mad_percent\":" << measured.mad
        << ",\"max_abs_error\":" << measured.error
        << ",\"accepted_for_promotion\":false,\"limitation\":\""
        << limitation << "\"}\n";
}

} // namespace

int main(int argc, char **argv) {
    const arguments args = parse(argc, argv);
    cellerator::bench::benchmark_mutex_guard mutex("ce-geo-biology-operations", 0);
    int device = 0;
    require_cuda(cudaGetDevice(&device), "get device");
    cudaDeviceProp properties{};
    require_cuda(cudaGetDeviceProperties(&properties, device), "get properties");
    require(properties.major == 7 && properties.minor == 0,
        "biology operation campaign requires sm_70");
    cudaStream_t stream = nullptr;
    require_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        "create stream");

    std::vector<__half> relation(256u), transpose(256u), rhs(relation_output_count);
    std::vector<float> relation_output(relation_output_count),
        transpose_output(relation_output_count), relation_reference(relation_output_count),
        transpose_reference(relation_output_count);
    const double relation_prepare_ns = wall_ns([&] {
        for (std::uint32_t row = 0u; row < relation_rows; ++row)
            for (std::uint32_t source = 0u; source < relation_sources; ++source) {
                const std::uint32_t index = row * relation_sources + source;
                relation[index] = __float2half(static_cast<float>(
                    static_cast<int>((index * 7u) % 17u) - 8) / 16.0f);
                transpose[source * relation_rows + row] = relation[index];
            }
        for (std::uint32_t index = 0u; index < relation_output_count; ++index)
            rhs[index] = __float2half(static_cast<float>(
                static_cast<int>((index * 5u) % 19u) - 9) / 16.0f);
        for (std::uint32_t row = 0u; row < relation_rows; ++row)
            for (std::uint32_t column = 0u; column < relation_width; ++column)
                for (std::uint32_t source = 0u; source < relation_sources; ++source) {
                    relation_reference[row * relation_width + column]
                        += __half2float(relation[row * relation_sources + source])
                            * __half2float(rhs[source * relation_width + column]);
                    transpose_reference[row * relation_width + column]
                        += __half2float(transpose[row * relation_sources + source])
                            * __half2float(rhs[source * relation_width + column]);
                }
    });
    device_buffer<__half> d_relation(256u), d_rhs(relation_output_count);
    device_buffer<std::uint32_t> d_destination_offsets(2u), d_source_bases(1u);
    device_buffer<float> d_relation_output(relation_output_count);
    const std::uint32_t destination_offsets[] = {0u, 1u};
    const std::uint32_t source_bases[] = {0u};
    const double relation_structure_upload_ns = wall_ns([&] {
        require_cuda(cudaMemcpyAsync(d_destination_offsets.data, destination_offsets,
            sizeof(destination_offsets), cudaMemcpyHostToDevice, stream),
            "destination offset upload");
        require_cuda(cudaMemcpyAsync(d_source_bases.data, source_bases,
            sizeof(source_bases), cudaMemcpyHostToDevice, stream),
            "source base upload");
        require_cuda(cudaStreamSynchronize(stream), "relation structure sync");
    });
    sm70::relation_apply_n64_request_v1 relation_request{};
    relation_request.relation_tiles = d_relation.data;
    relation_request.tile_count = 1u;
    relation_request.destination_tile_offsets = d_destination_offsets.data;
    relation_request.destination_group_count = 1u;
    relation_request.tile_source_bases = d_source_bases.data;
    relation_request.dense_rhs = d_rhs.data;
    relation_request.source_count = relation_sources;
    relation_request.output = d_relation_output.data;
    relation_request.stream = stream;
    auto measure_relation = [&](const std::vector<__half> &matrix,
        const std::vector<float> &reference, std::vector<float> *output) {
        std::vector<double> samples;
        for (int sample = -args.warmups; sample < args.repeats; ++sample) {
            const double elapsed = wall_ns([&] {
                require_cuda(cudaMemcpyAsync(d_relation.data, matrix.data(),
                    256u * sizeof(__half), cudaMemcpyHostToDevice, stream),
                    "relation value upload");
                require_cuda(cudaMemcpyAsync(d_rhs.data, rhs.data(),
                    relation_output_count * sizeof(__half), cudaMemcpyHostToDevice,
                    stream), "relation RHS upload");
                require(sm70::enqueue_relation_apply_n64_v1(relation_request)
                    == sm70::relation_apply_n64_status_v1::success,
                    "relation apply launch");
                require_cuda(cudaMemcpyAsync(output->data(), d_relation_output.data,
                    relation_output_count * sizeof(float), cudaMemcpyDeviceToHost,
                    stream), "relation output download");
                require_cuda(cudaStreamSynchronize(stream), "relation complete sync");
            });
            if (sample >= 0) samples.push_back(elapsed);
        }
        double error = 0.0;
        for (std::size_t index = 0u; index < output->size(); ++index)
            error = std::max(error, static_cast<double>(
                std::fabs((*output)[index] - reference[index])));
        require(error <= 1.0e-5, "relation reference mismatch");
        return measured_case{median(samples), mad_percent(samples),
            relation_prepare_ns + relation_structure_upload_ns, error};
    };
    const measured_case forward = measure_relation(
        relation, relation_reference, &relation_output);
    const measured_case backward = measure_relation(
        transpose, transpose_reference, &transpose_output);

    std::vector<sm70::support_logical_edge_v1> logical_edges(256u);
    std::vector<projection::projection_value_map_v1> maps(256u);
    std::vector<std::uint8_t> source_support(16u, 1u), destination_support(16u, 1u);
    std::vector<sm70::support_projection_edge_v1> selected(256u);
    sm70::contract_projection_result_v1 contract_result{};
    const double contract_prepare_ns = wall_ns([&] {
        for (std::uint32_t index = 0u; index < 256u; ++index) {
            logical_edges[index].logical_edge_id.value = index + 1u;
            logical_edges[index].source_index = index % 16u;
            logical_edges[index].destination_index = index / 16u;
            maps[index].logical_edge_id = logical_edges[index].logical_edge_id;
            maps[index].region_kind = index < 192u
                ? projection::physical_region_kind_v1::mma
                : projection::physical_region_kind_v1::residual;
            maps[index].region_index = 0u;
            maps[index].projection_slot = index;
        }
        sm70::contract_projection_request_v1 request{};
        request.logical_edges = logical_edges.data();
        request.physical_value_map = maps.data();
        request.logical_edge_count = logical_edges.size();
        request.source_support = source_support.data();
        request.source_count = source_support.size();
        request.destination_support = destination_support.data();
        request.destination_count = destination_support.size();
        request.selected_edges = selected.data();
        request.selected_capacity = selected.size();
        require(sm70::prepare_contract_projection_v1(request, &contract_result)
            == sm70::contract_projection_status_v1::success,
            "support contraction preparation");
    });
    require(contract_result.selected_edge_count == 256u
        && contract_result.mma_edge_count == 192u
        && contract_result.residual_edge_count == 64u,
        "support contraction reference mismatch");
    const measured_case contraction{contract_prepare_ns, 0.0, 0.0, 0.0};

    std::vector<sm70::support_projection_edge_v1> segment_edges(segment_values);
    std::vector<sm70::cover_native_partition_v1> partitions(segment_count);
    std::vector<float> segment_input(segment_values), segment_output(segment_values),
        segment_reference(segment_values);
    const double segment_prepare_ns = wall_ns([&] {
        for (std::uint32_t segment = 0u; segment < segment_count; ++segment) {
            partitions[segment] = {projection::physical_region_kind_v1::residual,
                segment * segment_size, segment_size};
            float maximum = -1.0e30f;
            for (std::uint32_t local = 0u; local < segment_size; ++local) {
                const std::uint32_t index = segment * segment_size + local;
                segment_edges[index].region_kind =
                    projection::physical_region_kind_v1::residual;
                segment_edges[index].stable_output_index = index;
                segment_input[index] = static_cast<float>(
                    static_cast<int>((index * 11u) % 23u) - 11) / 8.0f;
                maximum = std::max(maximum, segment_input[index]);
            }
            float denominator = 0.0f;
            for (std::uint32_t local = 0u; local < segment_size; ++local)
                denominator += std::exp(segment_input[segment * segment_size + local] - maximum);
            for (std::uint32_t local = 0u; local < segment_size; ++local) {
                const std::uint32_t index = segment * segment_size + local;
                segment_reference[index] = std::exp(segment_input[index] - maximum)
                    / denominator;
            }
        }
    });
    device_buffer<sm70::support_projection_edge_v1> d_segment_edges(segment_values);
    device_buffer<sm70::cover_native_partition_v1> d_partitions(segment_count);
    device_buffer<float> d_segment_input(segment_values), d_segment_output(segment_values);
    const double segment_upload_ns = wall_ns([&] {
        require_cuda(cudaMemcpyAsync(d_segment_edges.data, segment_edges.data(),
            segment_values * sizeof(segment_edges[0]), cudaMemcpyHostToDevice,
            stream), "segment edge upload");
        require_cuda(cudaMemcpyAsync(d_partitions.data, partitions.data(),
            segment_count * sizeof(partitions[0]), cudaMemcpyHostToDevice,
            stream), "segment partition upload");
        require_cuda(cudaStreamSynchronize(stream), "segment structure sync");
    });
    sm70::cover_native_normalize_request_v1 normalize{};
    normalize.selected_edges = d_segment_edges.data;
    normalize.selected_edge_count = segment_values;
    normalize.partitions = d_partitions.data;
    normalize.partition_count = segment_count;
    normalize.logical_edge_values = d_segment_input.data;
    normalize.logical_edge_count = segment_values;
    normalize.logical_edge_output = d_segment_output.data;
    normalize.stream = stream;
    std::vector<double> normalize_samples;
    for (int sample = -args.warmups; sample < args.repeats; ++sample) {
        const double elapsed = wall_ns([&] {
            require_cuda(cudaMemcpyAsync(d_segment_input.data, segment_input.data(),
                segment_values * sizeof(float), cudaMemcpyHostToDevice, stream),
                "segment input upload");
            require(sm70::enqueue_cover_native_normalize_v1(normalize)
                == sm70::cover_native_normalize_status_v1::success,
                "cover native normalize");
            require_cuda(cudaMemcpyAsync(segment_output.data(), d_segment_output.data,
                segment_values * sizeof(float), cudaMemcpyDeviceToHost, stream),
                "segment output download");
            require_cuda(cudaStreamSynchronize(stream), "segment complete sync");
        });
        if (sample >= 0) normalize_samples.push_back(elapsed);
    }
    double segment_error = 0.0;
    for (std::uint32_t index = 0u; index < segment_values; ++index)
        segment_error = std::max(segment_error, static_cast<double>(
            std::fabs(segment_output[index] - segment_reference[index])));
    require(segment_error <= 1.0e-6, "segment reference mismatch");
    const measured_case segment_normalize{median(normalize_samples),
        mad_percent(normalize_samples), segment_prepare_ns + segment_upload_ns,
        segment_error};

    std::vector<float> members(segment_values), pooled(segment_count),
        broadcast(segment_values), hierarchy_reference(segment_values);
    for (std::uint32_t index = 0u; index < segment_values; ++index)
        members[index] = static_cast<float>((index * 3u) % 17u) / 16.0f;
    for (std::uint32_t segment = 0u; segment < segment_count; ++segment) {
        float sum = 0.0f;
        for (std::uint32_t local = 0u; local < segment_size; ++local)
            sum += members[segment * segment_size + local];
        for (std::uint32_t local = 0u; local < segment_size; ++local)
            hierarchy_reference[segment * segment_size + local] = sum;
    }
    device_buffer<float> d_members(segment_values), d_pooled(segment_count),
        d_broadcast(segment_values);
    std::vector<double> hierarchy_samples;
    for (int sample = -args.warmups; sample < args.repeats; ++sample) {
        const double elapsed = wall_ns([&] {
            require_cuda(cudaMemcpyAsync(d_members.data, members.data(),
                segment_values * sizeof(float), cudaMemcpyHostToDevice, stream),
                "hierarchy input upload");
            hierarchy_pool_broadcast_kernel<<<segment_count, 1u, 0u, stream>>>(
                d_members.data, d_pooled.data, d_broadcast.data);
            require_cuda(cudaPeekAtLastError(), "hierarchy launch");
            require_cuda(cudaMemcpyAsync(broadcast.data(), d_broadcast.data,
                segment_values * sizeof(float), cudaMemcpyDeviceToHost, stream),
                "hierarchy output download");
            require_cuda(cudaStreamSynchronize(stream), "hierarchy complete sync");
        });
        if (sample >= 0) hierarchy_samples.push_back(elapsed);
    }
    double hierarchy_error = 0.0;
    for (std::uint32_t index = 0u; index < segment_values; ++index)
        hierarchy_error = std::max(hierarchy_error, static_cast<double>(
            std::fabs(broadcast[index] - hierarchy_reference[index])));
    require(hierarchy_error <= 1.0e-6, "hierarchy reference mismatch");
    const measured_case hierarchy{median(hierarchy_samples),
        mad_percent(hierarchy_samples), 0.0, hierarchy_error};

    std::ofstream out(args.output, std::ios::trunc);
    require(static_cast<bool>(out), "open evidence output");
    out << std::fixed << std::setprecision(3);
    out << "{\"schema\":\"CELLERATOR-CE-GEO-BIOLOGY-OPERATIONS/1\","
        "\"record_type\":\"provenance\",\"task_id\":\"CE-GEO-118\","
        "\"campaign_id\":\"biology-operations\","
        "\"controller_evidence_id\":\"CE-GEO-118-biology-operations-v1\","
        "\"benchmark_mutex\":true,\"uncontaminated\":true,"
        "\"accepted_for_promotion\":false,\"disposition\":\"evaluated_not_promoted\","
        "\"hardware\":{\"name\":\"" << properties.name
        << "\",\"compute_capability\":\"7.0\"},"
        "\"methodology\":{\"clock\":\"host steady-clock wall time\","
        "\"consumer_complete\":\"dynamic upload through output D2H and explicit cudaStreamSynchronize\","
        "\"warmups\":" << args.warmups << ",\"repeats\":" << args.repeats
        << "},\"limitations\":[\"deterministic synthetic biological shapes, not real-data performance claims\","
        "\"relation cases share one N64 source-linked provider shape\","
        "\"hierarchy pool/broadcast is a benchmark-local calibration helper\","
        "\"gradient evidence covers input-state transpose propagation, not relation-value gradients\"]}\n";
    write_record(out, "state_embedding", "gene_to_state_embedding",
        "production_sm70_relation_apply_n64", forward,
        "N64 synthetic relation shape only");
    write_record(out, "regulatory_apply", "regulator_to_target_gene",
        "production_sm70_relation_apply_n64", forward,
        "shares calibrated relation geometry with other forward cases");
    write_record(out, "transition_apply", "state_t_to_state_t_plus_one",
        "production_sm70_relation_apply_n64", forward,
        "no temporal workflow policy is benchmarked");
    write_record(out, "support_contraction", "supported_relation_edges",
        "production_sm70_contract_projection_host", contraction,
        "host preparation measurement; no GPU kernel is claimed");
    write_record(out, "segment_normalize", "relation_edge_segments",
        "production_sm70_cover_native_normalize", segment_normalize,
        "cover-native physical partitions, not generic public segment API");
    write_record(out, "exchange", "contract_gate_normalize_exchange_stage",
        "production_sm70_cover_native_normalize", segment_normalize,
        "normalization stage only; no fused four-step exchange claim");
    write_record(out, "hierarchy_pool_broadcast", "cell_to_module_incidence",
        "benchmark_local_calibration_kernel", hierarchy,
        "not a promoted production hierarchy kernel");
    write_record(out, "perturbation_propagation", "perturbed_gene_to_response_gene",
        "production_sm70_relation_apply_n64", forward,
        "delta-generation lifecycle is outside the timed kernel");
    write_record(out, "transpose_apply", "target_to_source_transpose_projection",
        "production_sm70_relation_apply_n64_transposed_projection", backward,
        "transpose projection constructed on host before timing");
    write_record(out, "input_gradient", "relation_apply_input_state_gradient",
        "production_sm70_relation_apply_n64_transposed_projection", backward,
        "input-state gradient only; no relation-value gradient claim");
    out.close();
    require_cuda(cudaStreamDestroy(stream), "destroy stream");
    return 0;
}
