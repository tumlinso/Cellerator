#include "../../../src/compute/architecture/providers/nvidia/sm70/relation_apply_n64.cuh"

#include <Cellerator/compute/candidate/csr_fallback_candidate.hh>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace sm70 = cellerator::compute::architecture::providers::nvidia::sm70;
namespace core = cellerator::compute::math::core;
namespace math = cellerator::compute::math;
namespace execution = cellerator::execution;
namespace wmma = nvcuda::wmma;

namespace {

constexpr std::uint32_t width = 64u;

void require(bool condition, const char *message) {
    if (!condition) {
        std::fprintf(stderr, "%s\n", message);
        std::exit(1);
    }
}

void require(core::operation_status status, const char *message) {
    if (!status) {
        std::fprintf(stderr, "%s: code=%u binding=%u detail=%s\n", message,
            static_cast<unsigned>(status.code),
            static_cast<unsigned>(status.binding), status.message);
        std::exit(1);
    }
}

void require_cuda(cudaError_t status, const char *message) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", message, cudaGetErrorString(status));
        std::exit(1);
    }
}

void require_cublas(cublasStatus_t status, const char *message) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "%s: cublas status %d\n", message,
            static_cast<int>(status));
        std::exit(1);
    }
}

template<typename T>
struct device_buffer {
    T *data = nullptr;
    std::size_t count = 0u;
    explicit device_buffer(std::size_t size) : count(size) {
        if (count != 0u)
            require_cuda(cudaMalloc(reinterpret_cast<void **>(&data),
                count * sizeof(T)), "cudaMalloc");
    }
    ~device_buffer() { if (data != nullptr) cudaFree(data); }
    device_buffer(const device_buffer &) = delete;
    device_buffer &operator=(const device_buffer &) = delete;
};

template<typename T>
void upload(device_buffer<T> &device, const std::vector<T> &host) {
    require(device.count >= host.size(), "device upload capacity");
    require_cuda(cudaMemcpy(device.data, host.data(), host.size() * sizeof(T),
        cudaMemcpyHostToDevice), "upload");
}

// Source-faithful reproduction of the retained historical experiment. The
// production candidate's kernel is source-private, so this benchmark cannot
// invoke it directly. Each fragment owns a CTA and atomically accumulates.
__global__ void historical_atomic_fragment_kernel(const __half *tiles,
    const std::uint32_t *destination_bases,
    const std::uint32_t *source_bases,
    std::uint32_t fragment_count,
    const __half *rhs,
    std::uint32_t source_count,
    float *output) {
    const std::uint32_t fragment = blockIdx.x;
    const std::uint32_t column_tile = blockIdx.y;
    if (fragment >= fragment_count || threadIdx.x >= 32u) return;
    const std::uint32_t destination_base = destination_bases[fragment];
    const std::uint32_t source_base = source_bases[fragment];
    const std::uint32_t column_base = column_tile * 16u;
    if (source_base + 16u > source_count) return;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half,
        wmma::row_major> relation;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half,
        wmma::row_major> dense;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
    wmma::fill_fragment(accumulator, 0.0f);
    wmma::load_matrix_sync(relation,
        tiles + static_cast<std::size_t>(fragment) * 256u, 16u);
    wmma::load_matrix_sync(dense,
        rhs + static_cast<std::size_t>(source_base) * width + column_base,
        width);
    wmma::mma_sync(accumulator, relation, dense, accumulator);
    __shared__ float result[256];
    wmma::store_matrix_sync(result, accumulator, 16u, wmma::mem_row_major);
    __syncwarp();
    for (std::uint32_t slot = threadIdx.x; slot < 256u; slot += 32u) {
        const std::uint32_t row = slot / 16u;
        const std::uint32_t column = slot % 16u;
        atomicAdd(output
                + static_cast<std::size_t>(destination_base + row) * width
                + column_base + column,
            result[slot]);
    }
}

__global__ void column_major_to_row_major(const float *input, float *output,
    std::uint32_t rows) {
    for (std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < rows * width; index += blockDim.x * gridDim.x) {
        const std::uint32_t row = index / width;
        const std::uint32_t column = index % width;
        output[index] = input[static_cast<std::size_t>(column) * rows + row];
    }
}

template<typename Function>
double time_ns(cudaStream_t stream, Function &&function) {
    cudaEvent_t begin = nullptr, end = nullptr;
    require_cuda(cudaEventCreate(&begin), "create timer begin");
    require_cuda(cudaEventCreate(&end), "create timer end");
    require_cuda(cudaEventRecord(begin, stream), "record timer begin");
    function();
    require_cuda(cudaEventRecord(end, stream), "record timer end");
    require_cuda(cudaEventSynchronize(end), "synchronize timer end");
    float milliseconds = 0.0f;
    require_cuda(cudaEventElapsedTime(&milliseconds, begin, end),
        "read timer");
    require_cuda(cudaEventDestroy(end), "destroy timer end");
    require_cuda(cudaEventDestroy(begin), "destroy timer begin");
    return static_cast<double>(milliseconds) * 1.0e6;
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

execution::axis_identity axis(std::uint32_t base) {
    return {{base, 1u}, {base + 1u, 1u},
        {base + 2u, 1u}, {base + 3u, 1u}};
}

execution::device_location location(int device) {
    return {execution::residency_kind::device, {}, device, 0u};
}

execution::dense_tensor_view vector_view(void *data,
    execution::axis_identity identity, std::uint64_t size, int device) {
    execution::dense_tensor_view view{};
    view.data = data;
    view.location = location(device);
    view.value_type = execution::numeric_type::f32;
    view.rank = 1u;
    view.axes[0] = identity;
    view.shape[0] = size;
    view.stride[0] = 1;
    return view;
}

core::numeric_policy csr_numeric() {
    core::numeric_policy value{};
    value.sparse_storage = execution::numeric_type::f16;
    value.dense_storage = execution::numeric_type::f32;
    value.output_storage = execution::numeric_type::f32;
    value.multiply = execution::numeric_type::f32;
    value.accumulation = execution::numeric_type::f32;
    value.scalar = execution::numeric_type::u32;
    value.bias = execution::numeric_type::invalid;
    return value;
}

struct result {
    std::uint32_t destination_groups = 0u;
    std::uint32_t source_tiles = 0u;
    std::uint64_t logical_edges = 0u;
    std::vector<double> output_owned;
    std::vector<double> historical_atomic;
    std::vector<double> csr_n1_x64;
    std::vector<double> dense_cublas;
    double max_abs_error = 0.0;
};

result run_scenario(std::uint32_t destination_groups,
    std::uint32_t source_tiles, int warmups, int repeats,
    int device, cudaStream_t stream, cublasHandle_t cublas) {
    const std::uint32_t rows = destination_groups * 16u;
    const std::uint32_t sources = source_tiles * 16u;
    const std::uint32_t fragments = destination_groups * source_tiles;
    const std::uint32_t nnz = rows * sources;
    result measured{};
    measured.destination_groups = destination_groups;
    measured.source_tiles = source_tiles;
    measured.logical_edges = nnz;

    std::vector<__half> tiles(static_cast<std::size_t>(fragments) * 256u);
    std::vector<__half> dense_relation(static_cast<std::size_t>(rows) * sources);
    std::vector<__half> csr_values(nnz);
    std::vector<__half> rhs(static_cast<std::size_t>(sources) * width);
    std::vector<float> rhs_columns(static_cast<std::size_t>(width) * sources);
    std::vector<std::uint32_t> destination_offsets(destination_groups + 1u);
    std::vector<std::uint32_t> destination_bases(fragments), source_bases(fragments);
    std::vector<std::uint32_t> row_offsets(rows + 1u), column_ids(nnz);
    for (std::uint32_t group = 0u; group < destination_groups; ++group) {
        destination_offsets[group] = group * source_tiles;
        for (std::uint32_t tile = 0u; tile < source_tiles; ++tile) {
            const std::uint32_t fragment = group * source_tiles + tile;
            destination_bases[fragment] = group * 16u;
            source_bases[fragment] = tile * 16u;
            for (std::uint32_t row = 0u; row < 16u; ++row)
                for (std::uint32_t source = 0u; source < 16u; ++source) {
                    const std::uint32_t slot = row * 16u + source;
                    const float value = static_cast<float>(
                        static_cast<int>((fragment * 13u + slot * 7u) % 17u) - 8)
                        / 16.0f;
                    tiles[static_cast<std::size_t>(fragment) * 256u + slot]
                        = __float2half(value);
                    dense_relation[static_cast<std::size_t>(group * 16u + row)
                            * sources + tile * 16u + source]
                        = __float2half(value);
                }
        }
    }
    destination_offsets[destination_groups] = fragments;
    for (std::uint32_t row = 0u; row < rows; ++row) {
        row_offsets[row] = row * sources;
        for (std::uint32_t source = 0u; source < sources; ++source) {
            const std::size_t index = static_cast<std::size_t>(row) * sources + source;
            column_ids[index] = source;
            csr_values[index] = dense_relation[index];
        }
    }
    row_offsets[rows] = nnz;
    for (std::uint32_t source = 0u; source < sources; ++source)
        for (std::uint32_t column = 0u; column < width; ++column) {
            const float value = static_cast<float>(
                static_cast<int>((source * 5u + column * 3u) % 19u) - 9)
                / 16.0f;
            rhs[static_cast<std::size_t>(source) * width + column]
                = __float2half(value);
            rhs_columns[static_cast<std::size_t>(column) * sources + source]
                = __half2float(rhs[static_cast<std::size_t>(source) * width + column]);
        }

    device_buffer<__half> d_tiles(tiles.size()), d_relation(dense_relation.size());
    device_buffer<__half> d_csr_values(csr_values.size()), d_rhs(rhs.size());
    device_buffer<float> d_rhs_columns(rhs_columns.size());
    device_buffer<std::uint32_t> d_destination_offsets(destination_offsets.size());
    device_buffer<std::uint32_t> d_destination_bases(destination_bases.size());
    device_buffer<std::uint32_t> d_source_bases(source_bases.size());
    device_buffer<std::uint32_t> d_row_offsets(row_offsets.size());
    device_buffer<std::uint32_t> d_column_ids(column_ids.size());
    device_buffer<float> d_output_owned(static_cast<std::size_t>(rows) * width);
    device_buffer<float> d_historical(static_cast<std::size_t>(rows) * width);
    device_buffer<float> d_csr_columns(static_cast<std::size_t>(rows) * width);
    device_buffer<float> d_csr_output(static_cast<std::size_t>(rows) * width);
    device_buffer<float> d_dense(static_cast<std::size_t>(rows) * width);
    upload(d_tiles, tiles); upload(d_relation, dense_relation);
    upload(d_csr_values, csr_values); upload(d_rhs, rhs);
    upload(d_rhs_columns, rhs_columns); upload(d_destination_offsets, destination_offsets);
    upload(d_destination_bases, destination_bases); upload(d_source_bases, source_bases);
    upload(d_row_offsets, row_offsets); upload(d_column_ids, column_ids);

    sm70::relation_apply_n64_request_v1 output_owned{};
    output_owned.relation_tiles = d_tiles.data;
    output_owned.tile_count = fragments;
    output_owned.destination_tile_offsets = d_destination_offsets.data;
    output_owned.destination_group_count = destination_groups;
    output_owned.tile_source_bases = d_source_bases.data;
    output_owned.dense_rhs = d_rhs.data;
    output_owned.source_count = sources;
    output_owned.output = d_output_owned.data;
    output_owned.stream = stream;

    const execution::axis_identity source_axis = axis(10u);
    const execution::axis_identity row_axis = axis(20u);
    core::structure_set_key structures{};
    structures.count = 1u;
    structures.structures[0] = {{11u, 12u}, {21u, 1u}, {1u}};
    const core::projection_key projection{{31u, 32u}, {42u, 1u},
        core::projection_kind::csr, math::execution_csr_schema_version, 1u};
    math::execution_csr_view csr{};
    csr.full_row_count = rows; csr.row_count = rows; csr.feature_count = sources;
    csr.nnz_count = nnz; csr.value_size_bytes = sizeof(__half);
    csr.row_domain_identity = 0x3003u;
    csr.structure.identity_version = math::execution_csr_structure_identity_version;
    csr.structure.value = 0x7072u;
    csr.feature_order.kind = math::feature_order_kind::packed;
    csr.feature_order.feature_count = sources;
    csr.feature_order.feature_axis_identity_version = 1u;
    csr.feature_order.feature_axis_identity = 0x5005u;
    csr.feature_order.packing_geometry_identity = 0x1001u;
    csr.row_offsets = d_row_offsets.data;
    csr.execution_feature_ids = d_column_ids.data;
    csr.values = d_csr_values.data;
    const core::operation_problem problem{core::operation_core_schema_version,
        core::operation_kind::weighted_relation_reduce, 0u, {72u, 1u},
        1u, 1u, nnz};
    core::csr_fallback_prepared_state state{};
    core::prepared_operation prepared{};
    const core::prepare_policy policy{true, false, true, true, 8u, 0u, 0u};
    require(core::prepare_csr_fallback_operation(problem, structures,
        projection, csr_numeric(), policy, csr, device, source_axis, row_axis,
        &state, &prepared), "prepare production CSR fallback");
    execution::relation_structure relation{};
    relation.identity = structures.structures[0].runtime;
    relation.epoch = structures.structures[0].epoch;
    relation.source_axis = source_axis;
    relation.destination_axis = row_axis;
    relation.projections = {1u, 1u};
    relation.logical_edge_count = nnz;
    execution::value_plane plane{};
    plane.structure = relation.identity; plane.structure_epoch_value = relation.epoch;
    plane.values = d_csr_values.data; plane.location = location(device);
    plane.numeric = {execution::numeric_type::f16, execution::numeric_type::f32,
        execution::numeric_type::f32, 0u};
    plane.quantization.kind = execution::quantization_kind::none;
    plane.layout = execution::value_layout_kind::projection_local_order;
    plane.generation = {1u}; plane.element_count = nnz;
    plane.value_bytes = static_cast<std::uint64_t>(nnz) * sizeof(__half);
    execution::value_binding binding{&plane, plane.generation};
    execution::biological_operand_view input{}, output{};
    input.kind = execution::operand_kind::dense_tensor;
    output.kind = execution::operand_kind::dense_tensor;
    execution::launch_bindings launch{};
    launch.structures = &relation; launch.inputs = &input; launch.outputs = &output;
    launch.values = &binding; launch.input_count = 1u; launch.output_count = 1u;
    launch.value_count = 1u; launch.structure_count = 1u;
    launch.stream = {stream, device, 0u};
    launch.workspace = {nullptr, 0u, location(device)};

    const float alpha = 1.0f, beta = 0.0f;
    for (int sample = -warmups; sample < repeats; ++sample) {
        const double owned_ns = time_ns(stream, [&] {
            require(sm70::enqueue_relation_apply_n64_v1(output_owned)
                    == sm70::relation_apply_n64_status_v1::success,
                "launch production output-owned N64");
        });
        const double historical_ns = time_ns(stream, [&] {
            require_cuda(cudaMemsetAsync(d_historical.data, 0,
                d_historical.count * sizeof(float), stream), "zero historical output");
            historical_atomic_fragment_kernel<<<dim3(fragments, 4u), 32u, 0u,
                stream>>>(d_tiles.data, d_destination_bases.data,
                d_source_bases.data, fragments, d_rhs.data, sources,
                d_historical.data);
            require_cuda(cudaPeekAtLastError(), "launch historical atomic experiment");
        });
        const double csr_ns = time_ns(stream, [&] {
            for (std::uint32_t column = 0u; column < width; ++column) {
                input.storage.dense = vector_view(
                    d_rhs_columns.data + static_cast<std::size_t>(column) * sources,
                    source_axis, sources, device);
                output.storage.dense = vector_view(
                    d_csr_columns.data + static_cast<std::size_t>(column) * rows,
                    row_axis, rows, device);
                require(core::run_prepared_operation(prepared, launch),
                    "run production CSR N=1 fallback");
            }
            column_major_to_row_major<<<std::min(65535u,
                (rows * width + 255u) / 256u), 256u, 0u, stream>>>(
                    d_csr_columns.data, d_csr_output.data, rows);
            require_cuda(cudaPeekAtLastError(), "launch CSR output remap");
        });
        const double dense_ns = time_ns(stream, [&] {
            require_cublas(cublasGemmEx(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                width, rows, sources, &alpha, d_rhs.data, CUDA_R_16F, width,
                d_relation.data, CUDA_R_16F, sources, &beta, d_dense.data,
                CUDA_R_32F, width, CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP), "run legal cuBLAS dense baseline");
        });
        if (sample >= 0) {
            measured.output_owned.push_back(owned_ns);
            measured.historical_atomic.push_back(historical_ns);
            measured.csr_n1_x64.push_back(csr_ns);
            measured.dense_cublas.push_back(dense_ns);
        }
    }

    std::vector<float> reference(static_cast<std::size_t>(rows) * width, 0.0f);
    for (std::uint32_t row = 0u; row < rows; ++row)
        for (std::uint32_t source = 0u; source < sources; ++source)
            for (std::uint32_t column = 0u; column < width; ++column)
                reference[static_cast<std::size_t>(row) * width + column]
                    += __half2float(dense_relation[
                        static_cast<std::size_t>(row) * sources + source])
                        * __half2float(rhs[
                            static_cast<std::size_t>(source) * width + column]);
    const float *outputs[] = {d_output_owned.data, d_historical.data,
        d_csr_output.data, d_dense.data};
    std::vector<float> actual(reference.size());
    for (const float *device_output : outputs) {
        require_cuda(cudaMemcpy(actual.data(), device_output,
            actual.size() * sizeof(float), cudaMemcpyDeviceToHost),
            "download candidate output");
        for (std::size_t index = 0u; index < actual.size(); ++index)
            measured.max_abs_error = std::max(measured.max_abs_error,
                static_cast<double>(std::fabs(actual[index] - reference[index])));
    }
    require(measured.max_abs_error <= 0.03,
        "N64 independent reference tolerance exceeded");
    return measured;
}

void print_samples(const char *name, const std::vector<double> &values) {
    std::printf("\"%s\":[", name);
    for (std::size_t index = 0u; index < values.size(); ++index)
        std::printf("%s%.3f", index == 0u ? "" : ",", values[index]);
    std::printf("]");
}

} // namespace

int main(int argc, char **argv) {
    int warmups = 2, repeats = 5;
    bool correctness_only = false;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--warmups" && index + 1 < argc)
            warmups = std::atoi(argv[++index]);
        else if (argument == "--repeats" && index + 1 < argc)
            repeats = std::atoi(argv[++index]);
        else if (argument == "--correctness-only") correctness_only = true;
        else { std::fprintf(stderr, "unknown argument: %s\n", argv[index]); return 2; }
    }
    require(warmups >= 1 && repeats >= 5 && (repeats & 1) != 0,
        "warmup/repeat contract invalid");
    int device = -1;
    require_cuda(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp properties{};
    require_cuda(cudaGetDeviceProperties(&properties, device),
        "cudaGetDeviceProperties");
    require(properties.major == 7 && properties.minor == 0,
        "N64 comparison requires sm_70");
    cudaStream_t stream = nullptr;
    require_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        "create stream");
    cublasHandle_t cublas = nullptr;
    require_cublas(cublasCreate(&cublas), "create cuBLAS handle");
    require_cublas(cublasSetStream(cublas, stream), "bind cuBLAS stream");
    require_cublas(cublasSetMathMode(cublas, CUBLAS_TENSOR_OP_MATH),
        "enable V100 tensor operations");

    std::vector<result> results;
    if (correctness_only) results.push_back(run_scenario(
        2u, 3u, warmups, repeats, device, stream, cublas));
    else
        for (std::uint32_t groups : {1u, 16u, 64u})
            for (std::uint32_t tiles : {1u, 4u, 16u})
                results.push_back(run_scenario(
                    groups, tiles, warmups, repeats, device, stream, cublas));
    require_cublas(cublasDestroy(cublas), "destroy cuBLAS handle");
    require_cuda(cudaStreamDestroy(stream), "destroy stream");

    double aggregate = 0.0;
    for (const result &value : results) aggregate += median(value.output_owned);
    aggregate /= results.size();
    std::printf("{\"schema\":\"CELLERATOR-CE-GEO-N64-CAL/1\","
        "\"campaign_id\":\"n64-output-owned\",\"complete_ns\":%.3f,"
        "\"correctness_passed\":true,", aggregate);
    std::printf("\"paths\":{\"output_owned\":\"production relation_apply_n64\","
        "\"historical_atomic\":\"source-faithful benchmark reproduction; production entrypoint private\","
        "\"csr_n1_x64\":\"production CSR fallback repeated 64 times plus explicit remap\","
        "\"dense_cublas\":\"legal cuBLAS tensor-op dense baseline\"},");
    std::printf("\"limitations\":[\"historical candidate kernel has no public direct benchmark entrypoint\","
        "\"CSR fallback is N=1 and therefore requires 64 launches\","
        "\"fully populated 16x16 fragments are a dense-favorable structural regime\"],");
    std::printf("\"methodology\":{\"clock\":\"cuda_event\",\"warmups\":%d,"
        "\"repeats\":%d,\"width\":64,\"consumer_complete\":true},"
        "\"scenarios\":[", warmups, repeats);
    for (std::size_t index = 0u; index < results.size(); ++index) {
        const result &value = results[index];
        std::printf("%s{\"destination_groups\":%u,\"source_tiles\":%u,"
            "\"logical_edges\":%llu,\"samples_ns\":{",
            index == 0u ? "" : ",", value.destination_groups,
            value.source_tiles,
            static_cast<unsigned long long>(value.logical_edges));
        print_samples("output_owned", value.output_owned); std::printf(",");
        print_samples("historical_atomic", value.historical_atomic); std::printf(",");
        print_samples("csr_n1_x64", value.csr_n1_x64); std::printf(",");
        print_samples("dense_cublas", value.dense_cublas);
        std::printf("},\"median_ns\":{\"output_owned\":%.3f,"
            "\"historical_atomic\":%.3f,\"csr_n1_x64\":%.3f,"
            "\"dense_cublas\":%.3f},\"mad_percent\":{"
            "\"output_owned\":%.6f,\"historical_atomic\":%.6f,"
            "\"csr_n1_x64\":%.6f,\"dense_cublas\":%.6f},"
            "\"max_abs_error\":%.9f}", median(value.output_owned),
            median(value.historical_atomic), median(value.csr_n1_x64),
            median(value.dense_cublas), mad_percent(value.output_owned),
            mad_percent(value.historical_atomic), mad_percent(value.csr_n1_x64),
            mad_percent(value.dense_cublas), value.max_abs_error);
    }
    std::printf("]}\n");
    return 0;
}
