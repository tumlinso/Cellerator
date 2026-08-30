#include "../../../src/compute/architecture/providers/nvidia/sm70/relation_apply_hybrid.cuh"
#include "../../benchmark_mutex.hh"

#include <Cellerator/compute/projection/physical_mma_hybrid.hh>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <string>
#include <vector>

namespace sm70 = cellerator::compute::architecture::providers::nvidia::sm70;
namespace projection = cellerator::compute::projection;

namespace cellerator::compute::projection {
bool build_row_owned_mma_residual_v1(
    std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t,
    const std::uint32_t *, const std::uint32_t *,
    const width_tagged_logical_edge_id_v1 *, std::uint32_t, std::uint32_t,
    std::uint32_t *, std::uint32_t, std::uint32_t *, std::uint32_t,
    projection_value_map_v1 *, std::uint32_t, residual_region_v1 *) noexcept;
}

namespace {

constexpr std::uint32_t rows = 64u;
constexpr std::uint32_t sources = 64u;
constexpr std::uint32_t width = 64u;
constexpr std::uint32_t source_tiles = 2u;
constexpr std::uint32_t destination_groups = rows / 16u;
constexpr std::uint32_t tile_count = destination_groups * source_tiles;
constexpr std::uint32_t mma_edges = rows * 32u;
constexpr std::uint32_t residual_edges = rows * 2u;
constexpr std::uint32_t logical_edges = mma_edges + residual_edges;
constexpr std::uint32_t output_count = rows * width;

void require(bool condition, const char *message) {
    if (!condition) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
}
void require_cuda(cudaError_t status, const char *message) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", message, cudaGetErrorString(status));
        std::exit(1);
    }
}
void require_sparse(cusparseStatus_t status, const char *message) {
    if (status != CUSPARSE_STATUS_SUCCESS) {
        std::fprintf(stderr, "%s: cuSPARSE status %d\n", message,
            static_cast<int>(status));
        std::exit(1);
    }
}

template<class T> struct device_buffer {
    T *data = nullptr;
    explicit device_buffer(std::size_t count) {
        if (count != 0u) require_cuda(cudaMalloc(reinterpret_cast<void **>(&data),
            count * sizeof(T)), "cudaMalloc");
    }
    ~device_buffer() { if (data != nullptr) cudaFree(data); }
    device_buffer(const device_buffer &) = delete;
    device_buffer &operator=(const device_buffer &) = delete;
};

template<class T> void upload(T *target, const std::vector<T> &source) {
    require_cuda(cudaMemcpy(target, source.data(), source.size() * sizeof(T),
        cudaMemcpyHostToDevice), "structure upload");
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
    return center == 0.0 ? 0.0 : median(deviations) * 100.0 / center;
}

template<class F> double host_ns(F &&function) {
    const auto begin = std::chrono::steady_clock::now();
    function();
    const auto end = std::chrono::steady_clock::now();
    return static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
}

template<class F> double consumer_complete_wall_ns(
    cudaStream_t stream, F &&function) {
    const auto begin = std::chrono::steady_clock::now();
    function();
    require_cuda(cudaStreamSynchronize(stream), "consumer-complete synchronization");
    const auto end = std::chrono::steady_clock::now();
    return static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
}

bool sparse_fallback(void *context, cudaStream_t) noexcept {
    return context != nullptr;
}

cellerator::compute::math::feature_order_identity order(
    std::uint64_t identity, std::uint32_t count) {
    cellerator::compute::math::feature_order_identity result{};
    result.kind = cellerator::compute::math::feature_order_kind::packed;
    result.feature_count = count;
    result.feature_axis_identity_version = 1u;
    result.feature_axis_identity = identity;
    result.packing_geometry_identity = 0xCE113u;
    return result;
}

struct arguments { std::string output; int warmups = 3; int repeats = 11; };
arguments parse(int argc, char **argv) {
    arguments result;
    for (int i = 1; i < argc; ++i) {
        const std::string token(argv[i]);
        require(i + 1 < argc, "missing option value");
        if (token == "--output") result.output = argv[++i];
        else if (token == "--warmups") result.warmups = std::atoi(argv[++i]);
        else if (token == "--repeats") result.repeats = std::atoi(argv[++i]);
        else require(false, "unknown option");
    }
    require(!result.output.empty() && result.warmups >= 0 && result.repeats > 0,
        "invalid arguments");
    return result;
}

} // namespace

int main(int argc, char **argv) {
    const arguments args = parse(argc, argv);
    cellerator::bench::benchmark_mutex_guard mutex("ce-geo-sm70-hybrid-forward", 0);
    int device = 0;
    require_cuda(cudaGetDevice(&device), "get CUDA device");
    cudaDeviceProp properties{};
    require_cuda(cudaGetDeviceProperties(&properties, device), "get device properties");
    require(properties.major == 7 && properties.minor == 0,
        "CE-GEO-113 requires sm_70 hardware");

    std::vector<std::uint32_t> edge_rows, edge_columns;
    std::vector<projection::width_tagged_logical_edge_id_v1> edge_ids;
    std::vector<__half> logical_values(logical_edges);
    std::vector<__half> rhs(sources * width);
    std::vector<float> sparse_values(logical_edges), sparse_rhs(sources * width);
    std::vector<std::uint32_t> csr_offsets(rows + 1u), csr_columns(logical_edges);
    std::vector<projection::projection_value_map_v1> value_maps(logical_edges);
    std::vector<std::uint32_t> destination_offsets(destination_groups + 1u);
    std::vector<std::uint32_t> tile_sources(tile_count);

    double semantic_search_ns = host_ns([&] {
        for (std::uint32_t group = 0; group < destination_groups; ++group) {
            destination_offsets[group] = group * source_tiles;
            tile_sources[group * source_tiles] = 0u;
            tile_sources[group * source_tiles + 1u] = 16u;
        }
        destination_offsets[destination_groups] = tile_count;
    });
    double refinement_ns = host_ns([&] {
        edge_rows.reserve(residual_edges); edge_columns.reserve(residual_edges);
        edge_ids.resize(residual_edges);
        for (std::uint32_t row = 0; row < rows; ++row) {
            for (std::uint32_t local = 0; local < 2u; ++local) {
                const std::uint32_t index = row * 2u + local;
                edge_rows.push_back(row);
                edge_columns.push_back(32u + (row * 3u + local * 13u) % 32u);
                edge_ids[index].value = mma_edges + index;
            }
        }
    });

    std::vector<std::uint32_t> residual_offsets(rows + 1u), residual_columns(residual_edges);
    projection::residual_region_v1 residual_region{};
    double projection_ns = host_ns([&] {
        for (std::uint32_t group = 0; group < destination_groups; ++group)
            for (std::uint32_t tile = 0; tile < source_tiles; ++tile)
                for (std::uint32_t row_local = 0; row_local < 16u; ++row_local)
                    for (std::uint32_t source_local = 0; source_local < 16u; ++source_local) {
                        const std::uint32_t row = group * 16u + row_local;
                        const std::uint32_t source = tile * 16u + source_local;
                        const std::uint32_t id = row * 32u + source;
                        auto &map = value_maps[id];
                        map.logical_edge_id.value = id;
                        map.region_kind = projection::physical_region_kind_v1::mma;
                        map.region_index = 0u;
                        map.projection_slot = (group * source_tiles + tile) * 256u
                            + row_local * 16u + source_local;
                    }
        require(projection::build_row_owned_mma_residual_v1(
            0u, 1u, 0u, rows, edge_rows.data(), edge_columns.data(),
            edge_ids.data(), residual_edges, mma_edges, residual_offsets.data(),
            residual_offsets.size(), residual_columns.data(), residual_columns.size(),
            value_maps.data() + mma_edges, residual_edges, &residual_region),
            "residual projection construction failed");
    });

    for (std::uint32_t row = 0; row < rows; ++row) {
        csr_offsets[row] = row * 34u;
        for (std::uint32_t source = 0; source < 32u; ++source) {
            const std::uint32_t id = row * 32u + source;
            csr_columns[row * 34u + source] = source;
            const float value = static_cast<float>(static_cast<int>((id * 7u) % 17u) - 8) / 16.0f;
            logical_values[id] = __float2half(value);
            sparse_values[row * 34u + source] = __half2float(logical_values[id]);
        }
        for (std::uint32_t local = 0; local < 2u; ++local) {
            const std::uint32_t residual = row * 2u + local;
            const std::uint32_t id = mma_edges + residual;
            csr_columns[row * 34u + 32u + local] = edge_columns[residual];
            const float value = static_cast<float>(static_cast<int>((id * 11u) % 13u) - 6) / 16.0f;
            logical_values[id] = __float2half(value);
            sparse_values[row * 34u + 32u + local] = __half2float(logical_values[id]);
        }
    }
    csr_offsets[rows] = logical_edges;
    for (std::uint32_t i = 0; i < sources * width; ++i) {
        rhs[i] = __float2half(static_cast<float>(static_cast<int>((i * 5u) % 19u) - 9) / 16.0f);
        sparse_rhs[i] = __half2float(rhs[i]);
    }

    device_buffer<projection::projection_value_map_v1> d_maps(value_maps.size());
    device_buffer<__half> d_logical(logical_values.size()), d_mma(tile_count * 256u),
        d_residual(residual_edges), d_rhs(rhs.size());
    device_buffer<std::uint64_t> d_region_offsets(1u);
    device_buffer<std::uint32_t> d_destination_offsets(destination_offsets.size()),
        d_tile_sources(tile_sources.size()), d_residual_offsets(residual_offsets.size()),
        d_residual_columns(residual_columns.size()), d_csr_offsets(csr_offsets.size()),
        d_csr_columns(csr_columns.size());
    device_buffer<float> d_accumulation(output_count), d_hybrid_output(output_count),
        d_sparse_values(sparse_values.size()), d_sparse_rhs(sparse_rhs.size()),
        d_sparse_output(output_count);
    double structure_upload_ns = host_ns([&] {
        upload(d_maps.data, value_maps); upload(d_destination_offsets.data, destination_offsets);
        upload(d_tile_sources.data, tile_sources); upload(d_residual_offsets.data, residual_offsets);
        upload(d_residual_columns.data, residual_columns); upload(d_csr_offsets.data, csr_offsets);
        upload(d_csr_columns.data, csr_columns); const std::uint64_t zero = 0u;
        require_cuda(cudaMemcpy(d_region_offsets.data, &zero, sizeof(zero), cudaMemcpyHostToDevice), "region upload");
    });

    cudaStream_t stream = nullptr;
    require_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
    sm70::value_pack_state_v1 pack_state{};
    sm70::relation_apply_hybrid_request_v1 hybrid{};
    hybrid.value_pack = {d_maps.data, logical_edges, d_logical.data, logical_edges,
        d_region_offsets.data, 1u, d_region_offsets.data, 1u, d_mma.data,
        tile_count * 256u, d_residual.data, residual_edges, {1u}, stream};
    hybrid.value_pack_state = &pack_state;
    hybrid.mma = {d_mma.data, tile_count, d_destination_offsets.data,
        destination_groups, d_tile_sources.data, d_rhs.data, sources,
        d_accumulation.data, stream};
    hybrid.residual = {d_residual_offsets.data, rows, d_residual_columns.data,
        residual_edges, d_residual.data, d_rhs.data, sources, width,
        d_accumulation.data, stream};
    hybrid.output = d_hybrid_output.data; hybrid.output_count = output_count;
    hybrid.source_order = order(0x11301u, sources);
    hybrid.destination_order = order(0x11302u, rows);
    hybrid.pure_sparse_fallback = &sparse_fallback;
    hybrid.pure_sparse_context = &device;
    hybrid.stream = stream;

    cusparseHandle_t sparse_handle = nullptr;
    cusparseSpMatDescr_t matrix = nullptr;
    cusparseDnMatDescr_t dense_rhs = nullptr, dense_output = nullptr;
    std::size_t workspace_bytes = 0u;
    device_buffer<std::byte> *workspace = nullptr;
    double sparse_prepare_ns = host_ns([&] {
        require_sparse(cusparseCreate(&sparse_handle), "create cuSPARSE");
        require_sparse(cusparseSetStream(sparse_handle, stream), "set cuSPARSE stream");
        require_sparse(cusparseCreateCsr(&matrix, rows, sources, logical_edges,
            d_csr_offsets.data, d_csr_columns.data, d_sparse_values.data,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
            CUDA_R_32F), "create CSR");
        require_sparse(cusparseCreateDnMat(&dense_rhs, sources, width, width,
            d_sparse_rhs.data, CUDA_R_32F, CUSPARSE_ORDER_ROW), "create dense RHS");
        require_sparse(cusparseCreateDnMat(&dense_output, rows, width, width,
            d_sparse_output.data, CUDA_R_32F, CUSPARSE_ORDER_ROW), "create dense output");
        const float alpha = 1.0f, beta = 0.0f;
        require_sparse(cusparseSpMM_bufferSize(sparse_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matrix, dense_rhs, &beta, dense_output, CUDA_R_32F,
            CUSPARSE_SPMM_ALG_DEFAULT, &workspace_bytes), "query SpMM workspace");
        workspace = new device_buffer<std::byte>(workspace_bytes);
    });

    std::vector<float> hybrid_output(output_count), sparse_output(output_count), reference(output_count);
    auto launch_hybrid = [&](std::uint64_t generation) {
        require_cuda(cudaMemcpyAsync(d_logical.data, logical_values.data(),
            logical_values.size() * sizeof(__half), cudaMemcpyHostToDevice, stream), "upload hybrid values");
        require_cuda(cudaMemcpyAsync(d_rhs.data, rhs.data(), rhs.size() * sizeof(__half),
            cudaMemcpyHostToDevice, stream), "upload hybrid RHS");
        hybrid.value_pack.source_generation.value = generation;
        require(sm70::enqueue_relation_apply_hybrid_v1(hybrid)
            == sm70::relation_apply_hybrid_status_v1::success, "hybrid launch failed");
        require_cuda(cudaMemcpyAsync(hybrid_output.data(), d_hybrid_output.data,
            hybrid_output.size() * sizeof(float), cudaMemcpyDeviceToHost, stream), "download hybrid output");
    };
    const float alpha = 1.0f, beta = 0.0f;
    auto launch_sparse = [&] {
        require_cuda(cudaMemcpyAsync(d_sparse_values.data, sparse_values.data(),
            sparse_values.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "upload sparse values");
        require_cuda(cudaMemcpyAsync(d_sparse_rhs.data, sparse_rhs.data(),
            sparse_rhs.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "upload sparse RHS");
        require_sparse(cusparseSpMM(sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matrix, dense_rhs, &beta,
            dense_output, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
            workspace->data), "run cuSPARSE SpMM");
        require_cuda(cudaMemcpyAsync(sparse_output.data(), d_sparse_output.data,
            sparse_output.size() * sizeof(float), cudaMemcpyDeviceToHost, stream), "download sparse output");
    };
    launch_hybrid(1u); launch_sparse(); require_cuda(cudaStreamSynchronize(stream), "correctness sync");
    for (std::uint32_t row = 0; row < rows; ++row)
        for (std::uint32_t column = 0; column < width; ++column) {
            float sum = 0.0f;
            for (std::uint32_t edge = csr_offsets[row]; edge < csr_offsets[row + 1u]; ++edge)
                sum += sparse_values[edge] * sparse_rhs[csr_columns[edge] * width + column];
            reference[row * width + column] = sum;
        }
    double hybrid_error = 0.0, sparse_error = 0.0;
    for (std::size_t i = 0; i < reference.size(); ++i) {
        hybrid_error = std::max(hybrid_error, static_cast<double>(std::fabs(hybrid_output[i] - reference[i])));
        sparse_error = std::max(sparse_error, static_cast<double>(std::fabs(sparse_output[i] - reference[i])));
    }
    const bool correctness = hybrid_error <= 1.0e-4 && sparse_error <= 1.0e-4;
    require(correctness, "exact reference validation failed");

    std::vector<double> hybrid_samples, sparse_samples;
    for (int sample = -args.warmups; sample < args.repeats; ++sample) {
        const double h = consumer_complete_wall_ns(stream, [&] {
            launch_hybrid(static_cast<std::uint64_t>(sample + args.warmups + 2));
        });
        const double s = consumer_complete_wall_ns(stream, launch_sparse);
        if (sample >= 0) { hybrid_samples.push_back(h); sparse_samples.push_back(s); }
    }
    const double hybrid_steady = median(hybrid_samples), sparse_steady = median(sparse_samples);
    const double hybrid_cold = semantic_search_ns + refinement_ns + projection_ns + structure_upload_ns;
    const double sparse_cold = semantic_search_ns + refinement_ns + structure_upload_ns + sparse_prepare_ns;
    const double hybrid_r1 = hybrid_cold + hybrid_steady, hybrid_r16 = hybrid_cold / 16.0 + hybrid_steady;
    const double sparse_r1 = sparse_cold + sparse_steady, sparse_r16 = sparse_cold / 16.0 + sparse_steady;
    const bool promoted = correctness && hybrid_r1 < 0.95 * sparse_r1
        && hybrid_r16 < 0.95 * sparse_r16 && mad_percent(hybrid_samples) <= 5.0;

    std::ofstream out(args.output, std::ios::trunc);
    require(static_cast<bool>(out), "open evidence output");
    out << std::fixed << std::setprecision(3);
    out << "{\"schema\":\"CELLERATOR-CE-GEO-HYBRID-FORWARD/1\","
        "\"record_type\":\"provenance\",\"task_id\":\"CE-GEO-113\","
        "\"campaign_id\":\"sm70-hybrid-forward\","
        "\"controller_evidence_id\":\"CE-GEO-113-hybrid-forward-v1\","
        "\"benchmark_mutex\":true,\"uncontaminated\":true,"
        "\"accepted_for_promotion\":" << (promoted ? "true" : "false") << ","
        "\"disposition\":\"" << (promoted ? "validated" : "evaluated_not_promoted") << "\","
        "\"hardware\":{\"name\":\"" << properties.name << "\",\"compute_capability\":\"7.0\"},"
        "\"methodology\":{\"warmups\":" << args.warmups << ",\"repeats\":" << args.repeats
        << ",\"width\":64,\"logical_edges\":" << logical_edges
        << ",\"mma_edges\":" << mma_edges << ",\"residual_edges\":" << residual_edges
        << ",\"strongest_sparse_baseline\":\"cuSPARSE SpMM N64 float32\","
        "\"host_controller\":\"deterministic synthetic search and refinement over one fixed N64 exact cover\","
        "\"timing\":\"host steady-clock wall time around dynamic upload, launch, consumer D2H, and explicit cudaStreamSynchronize; cold phases amortized by reuse\"}}\n";
    auto write_measurement = [&](int reuse, double hybrid_complete, double sparse_complete) {
        out << "{\"schema\":\"CELLERATOR-CE-GEO-HYBRID-FORWARD/1\","
            "\"record_type\":\"measurement\",\"campaign_id\":\"sm70-hybrid-forward\","
            "\"reuse\":" << reuse << ",\"correctness_passed\":true,"
            "\"complete_ns\":" << hybrid_complete << ",\"hybrid_complete_ns\":" << hybrid_complete
            << ",\"sparse_complete_ns\":" << sparse_complete << ","
            "\"accepted_for_promotion\":" << (promoted ? "true" : "false") << ","
            "\"max_abs_error\":" << hybrid_error << ",\"sparse_max_abs_error\":" << sparse_error << ","
            "\"phases_ns\":{\"semantic_search\":" << semantic_search_ns
            << ",\"refinement\":" << refinement_ns << ",\"projection_construction\":" << projection_ns
            << ",\"structure_upload\":" << structure_upload_ns << ",\"sparse_prepare\":" << sparse_prepare_ns
            << ",\"hybrid_dynamic_pack_execute_epilogue_order_sync_d2h\":" << hybrid_steady
            << ",\"sparse_dynamic_upload_execute_sync_d2h\":" << sparse_steady << "},"
            "\"mad_percent\":{\"hybrid\":" << mad_percent(hybrid_samples)
            << ",\"sparse\":" << mad_percent(sparse_samples) << "}}\n";
    };
    write_measurement(1, hybrid_r1, sparse_r1);
    write_measurement(16, hybrid_r16, sparse_r16);
    out.close();

    delete workspace;
    require_sparse(cusparseDestroyDnMat(dense_output), "destroy output descriptor");
    require_sparse(cusparseDestroyDnMat(dense_rhs), "destroy RHS descriptor");
    require_sparse(cusparseDestroySpMat(matrix), "destroy CSR descriptor");
    require_sparse(cusparseDestroy(sparse_handle), "destroy cuSPARSE");
    require_cuda(cudaStreamDestroy(stream), "destroy stream");
    return 0;
}
