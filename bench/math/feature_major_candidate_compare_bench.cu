/*
CE-ARCH-76 V100 evidence harness, 2026-08-25.
This file contains only benchmark packing/interleave kernels. They make the
rank-1 row-masked and CSR candidates consume and produce the same row-major
KxN/MxN operands as the native feature-major candidate. Their time is recorded
as dynamic input-pack and output-order work, never hidden in kernel timing.
The production candidate kernels and layouts are unchanged by this harness.
*/

#include <Cellerator/compute/math/operation_core/csr_fallback_candidate.hh>
#include <Cellerator/compute/math/operation_core/feature_major_small_n_candidate.hh>
#include <Cellerator/compute/math/operation_core/row_masked_n1_candidate.hh>
#include <Cellerator/compute/math/physical_csr.hh>

#include <CellPack/persistent_packing_payload.hh>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>

namespace core = cellerator::compute::math::core;
namespace cm = cellerator::compute::math;
namespace execution = cellerator::execution;
namespace cp = cellpack;

namespace {

using clock_type = std::chrono::steady_clock;
using storage_t = cellerator::real::storage_t;
using compute_t = cellerator::real::compute_t;
using accum_t = cellerator::real::accum_t;

constexpr cp::u32 full_rows = 65536u;
constexpr cp::u32 quick_rows = 256u;
constexpr cp::u32 full_features = 32768u;
constexpr cp::u32 quick_features = 2048u;
constexpr cp::u32 block_width = 16u;
constexpr cp::u32 tile_width = 32u;
constexpr cp::u32 row_blocks = 2u;
constexpr cp::u32 nnz_per_row = block_width * row_blocks;
constexpr std::uint32_t full_warmups = 3u;
constexpr std::uint32_t full_repeats = 11u;
constexpr std::uint32_t quick_warmups = 1u;
constexpr std::uint32_t quick_repeats = 3u;
constexpr std::uint64_t expected_reuse = 8u;
constexpr std::uint32_t small_dense_widths[] = {1u, 2u, 4u, 8u, 16u};
constexpr std::uint32_t medium_dense_widths[] = {17u, 32u, 64u};

[[noreturn]] void fail(const char *message) {
    std::fprintf(stderr, "celleratorCeArch76CandidateBench: %s\n", message);
    std::exit(1);
}

void require(bool condition, const char *message) {
    if (!condition) fail(message);
}

void require(core::operation_status status, const char *message) {
    if (status) return;
    std::fprintf(stderr,
        "celleratorCeArch76CandidateBench: %s (code=%u detail=%s)\n",
        message, static_cast<unsigned>(status.code), status.message);
    std::exit(1);
}

void require(cm::physical_view_status status, const char *message) {
    if (status) return;
    std::fprintf(stderr,
        "celleratorCeArch76CandidateBench: %s (code=%u detail=%s)\n",
        message, static_cast<unsigned>(status.code), status.message);
    std::exit(1);
}

void require_cuda(cudaError_t status, const char *message) {
    if (status == cudaSuccess) return;
    std::fprintf(stderr, "celleratorCeArch76CandidateBench: %s: %s\n",
        message, cudaGetErrorString(status));
    std::exit(1);
}

template<typename T>
struct device_array {
    T *data = nullptr;
    std::size_t size = 0u;

    explicit device_array(std::size_t count = 0u) : size(count) {
        if (count != 0u)
            require_cuda(cudaMalloc(reinterpret_cast<void **>(&data),
                count * sizeof(T)), "cudaMalloc");
    }
    ~device_array() { if (data != nullptr) cudaFree(data); }
    device_array(const device_array &) = delete;
    device_array &operator=(const device_array &) = delete;
};

template<typename T>
void upload(device_array<T> &device, const std::vector<T> &host) {
    require(device.size >= host.size(), "device upload capacity");
    if (!host.empty())
        require_cuda(cudaMemcpy(device.data, host.data(),
            host.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
}

double milliseconds(clock_type::time_point begin, clock_type::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

execution::axis_identity axis(std::uint32_t base) {
    return {{base, 1u}, {base + 1u, 1u},
        {base + 2u, 1u}, {base + 3u, 1u}};
}

execution::device_location device_location(int ordinal) {
    return {execution::residency_kind::device, {}, ordinal, 0u};
}

execution::dense_tensor_view dense_vector(void *pointer,
    execution::axis_identity value_axis, std::uint64_t count, int ordinal) {
    execution::dense_tensor_view view{};
    view.data = pointer;
    view.location = device_location(ordinal);
    view.value_type = execution::numeric_type::f32;
    view.rank = 1u;
    view.axes[0] = value_axis;
    view.shape[0] = count;
    view.stride[0] = 1;
    return view;
}

execution::dense_tensor_view dense_matrix(void *pointer,
    execution::axis_identity major_axis,
    execution::axis_identity minor_axis,
    std::uint64_t rows,
    std::uint64_t columns,
    int ordinal) {
    execution::dense_tensor_view view{};
    view.data = pointer;
    view.location = device_location(ordinal);
    view.value_type = execution::numeric_type::f32;
    view.rank = 2u;
    view.axes[0] = major_axis;
    view.axes[1] = minor_axis;
    view.shape[0] = rows;
    view.shape[1] = columns;
    view.stride[0] = static_cast<std::int64_t>(columns);
    view.stride[1] = 1;
    return view;
}

core::numeric_policy vector_numeric() {
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

core::numeric_policy matrix_numeric() {
    core::numeric_policy value = vector_numeric();
    value.scalar = execution::numeric_type::f32;
    return value;
}

storage_t stored(float value) { return static_cast<storage_t>(value); }

float value_for(cp::u32 row, cp::u32 feature) {
    const int code = static_cast<int>((row * 17u + feature * 13u) % 31u) - 15;
    return static_cast<float>(code) * 0.0625f;
}

struct host_case {
    const char *name = nullptr;
    cp::u32 rows = 0u;
    cp::u32 features = 0u;
    cp::u32 sharing_groups = 0u;
    std::vector<cp::u32> feature_offsets;
    std::vector<cp::u32> feature_permutation;
    std::vector<cp::u32> row_permutation;
    std::vector<cp::u32> csr_offsets;
    std::vector<cp::u32> csr_columns;
    std::vector<storage_t> csr_values;
    std::vector<cp::u32> tile_offsets;
    std::vector<cp::u32> tile_blocks;
    std::vector<cp::u32> tile_cell_masks;
    std::vector<cp::u32> entry_offsets;
    std::vector<cp::u32> gene_masks;
    std::vector<cp::u32> value_offsets;
    std::vector<storage_t> tile_values;
    unsigned char image_byte = 0u;
    cp::persistent_packing_payload_view payload{};
};

host_case make_case(const char *name, cp::u32 rows,
    cp::u32 features, cp::u32 sharing_groups) {
    require(rows % tile_width == 0u && features % block_width == 0u,
        "benchmark shape must align to the frozen tile grammar");
    host_case result;
    result.name = name;
    result.rows = rows;
    result.features = features;
    result.sharing_groups = sharing_groups;
    const cp::u32 block_count = features / block_width;
    const cp::u32 descriptor_span = sharing_groups * row_blocks;
    require(descriptor_span <= block_count,
        "sharing regime exceeds feature block count");
    result.feature_offsets.resize(static_cast<std::size_t>(block_count) + 1u);
    result.feature_permutation.resize(features);
    result.row_permutation.resize(rows);
    result.csr_offsets.resize(static_cast<std::size_t>(rows) + 1u);
    std::iota(result.feature_permutation.begin(),
        result.feature_permutation.end(), 0u);
    std::iota(result.row_permutation.begin(), result.row_permutation.end(), 0u);
    for (cp::u32 block = 0u; block <= block_count; ++block)
        result.feature_offsets[block] = block * block_width;
    result.tile_offsets.push_back(0u);
    result.entry_offsets.push_back(0u);
    result.value_offsets.push_back(0u);
    result.csr_columns.reserve(static_cast<std::size_t>(rows) * nnz_per_row);
    result.csr_values.reserve(static_cast<std::size_t>(rows) * nnz_per_row);
    result.tile_values.reserve(static_cast<std::size_t>(rows) * nnz_per_row);

    const cp::u32 tile_count = rows / tile_width;
    const cp::u32 base_count = block_count - descriptor_span + 1u;
    for (cp::u32 tile = 0u; tile < tile_count; ++tile) {
        std::map<cp::u32, cp::u32> masks;
        std::vector<std::vector<cp::u32>> lane_blocks(tile_width);
        const cp::u32 base = (tile * descriptor_span) % base_count;
        for (cp::u32 lane = 0u; lane < tile_width; ++lane) {
            const cp::u32 group = lane % sharing_groups;
            lane_blocks[lane] = {base + group * row_blocks,
                base + group * row_blocks + 1u};
            for (cp::u32 block : lane_blocks[lane]) masks[block] |= 1u << lane;
        }
        for (const auto &descriptor : masks) {
            result.tile_blocks.push_back(descriptor.first);
            result.tile_cell_masks.push_back(descriptor.second);
            for (cp::u32 lane = 0u; lane < tile_width; ++lane) {
                if ((descriptor.second & (1u << lane)) == 0u) continue;
                result.gene_masks.push_back(0xffffu);
                const cp::u32 row = tile * tile_width + lane;
                const cp::u32 feature_begin = descriptor.first * block_width;
                for (cp::u32 local = 0u; local < block_width; ++local)
                    result.tile_values.push_back(stored(
                        value_for(row, feature_begin + local)));
                result.value_offsets.push_back(
                    static_cast<cp::u32>(result.tile_values.size()));
            }
            result.entry_offsets.push_back(
                static_cast<cp::u32>(result.gene_masks.size()));
        }
        result.tile_offsets.push_back(
            static_cast<cp::u32>(result.tile_blocks.size()));
        for (cp::u32 lane = 0u; lane < tile_width; ++lane) {
            const cp::u32 row = tile * tile_width + lane;
            for (cp::u32 block : lane_blocks[lane]) {
                const cp::u32 feature_begin = block * block_width;
                for (cp::u32 local = 0u; local < block_width; ++local) {
                    const cp::u32 feature = feature_begin + local;
                    result.csr_columns.push_back(feature);
                    result.csr_values.push_back(stored(value_for(row, feature)));
                }
            }
            result.csr_offsets[row + 1u] =
                static_cast<cp::u32>(result.csr_columns.size());
        }
    }

    result.payload.payload_schema_version =
        cp::persistent_packing_payload_schema_version;
    result.payload.payload_kind = cp::persistent_packing_payload_kind;
    result.payload.payload_identity = 0x43504b760000ull + sharing_groups;
    result.payload.image_base = &result.image_byte;
    result.payload.image_bytes = 1u;
    result.payload.plan.semantic_plan_schema_version =
        cp::packing_plan_semantic_schema_version;
    result.payload.plan.geometry_identity_version =
        cp::feature_block_geometry_identity_version;
    result.payload.plan.feature_count = features;
    result.payload.plan.feature_block_count = block_count;
    result.payload.plan.feature_block_geometry_identity = 0x76001000ull;
    result.payload.plan.feature_block_offsets = result.feature_offsets.data();
    result.payload.plan.feature_permutation = result.feature_permutation.data();
    result.payload.order.order_schema_version = cp::local_cell_order_schema_version;
    result.payload.order.signature_algorithm_version =
        cp::local_cell_signature_algorithm_version;
    result.payload.order.kind = cp::local_cell_order_kind::original;
    result.payload.order.window_size = 1024u;
    result.payload.order.group_width = tile_width;
    result.payload.order.ordering_identity = 0x76002000ull + sharing_groups;
    result.payload.order.full_row_count = rows;
    result.payload.order.row_count = rows;
    result.payload.order.feature_block_count = block_count;
    result.payload.order.feature_block_geometry_identity = 0x76001000ull;
    result.payload.order.row_domain_identity = 0x76003000ull;
    result.payload.order.row_permutation = result.row_permutation.data();
    auto &tiles = result.payload.tiles;
    tiles.tile_schema_version = cp::warp_tile_schema_version;
    tiles.record_schema_version = cp::cell_block_record_schema_version;
    tiles.semantic_plan_schema_version = cp::packing_plan_semantic_schema_version;
    tiles.geometry_identity_version = cp::feature_block_geometry_identity_version;
    tiles.order_schema_version = cp::local_cell_order_schema_version;
    tiles.tile_identity = 0x76004000ull + sharing_groups;
    tiles.feature_block_geometry_identity = 0x76001000ull;
    tiles.ordering_identity = result.payload.order.ordering_identity;
    tiles.full_row_count = rows;
    tiles.row_count = rows;
    tiles.feature_count = features;
    tiles.feature_block_count = block_count;
    tiles.tile_row_width = tile_width;
    tiles.tile_count = tile_count;
    tiles.nnz_count = static_cast<cp::u32>(result.tile_values.size());
    tiles.tile_block_count = static_cast<cp::u32>(result.tile_blocks.size());
    tiles.row_block_entry_count = static_cast<cp::u32>(result.gene_masks.size());
    tiles.value_size_bytes = sizeof(storage_t);
    tiles.feature_axis_fingerprint = 0x76005000ull;
    tiles.feature_axis_fingerprint_version = 1u;
    tiles.row_domain_identity = result.payload.order.row_domain_identity;
    tiles.tile_block_offsets = result.tile_offsets.data();
    tiles.tile_block_ids = result.tile_blocks.data();
    tiles.tile_block_cell_masks = result.tile_cell_masks.data();
    tiles.block_row_entry_offsets = result.entry_offsets.data();
    tiles.row_block_gene_masks = result.gene_masks.data();
    tiles.row_block_value_offsets = result.value_offsets.data();
    tiles.values = result.tile_values.data();
    return result;
}

void refresh_payload(host_case &source) {
    source.payload.image_base = &source.image_byte;
    source.payload.plan.feature_block_offsets = source.feature_offsets.data();
    source.payload.plan.feature_permutation = source.feature_permutation.data();
    source.payload.order.row_permutation = source.row_permutation.data();
    source.payload.tiles.tile_block_offsets = source.tile_offsets.data();
    source.payload.tiles.tile_block_ids = source.tile_blocks.data();
    source.payload.tiles.tile_block_cell_masks = source.tile_cell_masks.data();
    source.payload.tiles.block_row_entry_offsets = source.entry_offsets.data();
    source.payload.tiles.row_block_gene_masks = source.gene_masks.data();
    source.payload.tiles.row_block_value_offsets = source.value_offsets.data();
    source.payload.tiles.values = source.tile_values.data();
}

struct host_projections {
    cm::lazy_execution_csr_requirements csr_requirements{};
    std::vector<cp::u32> csr_rows;
    std::vector<cp::u32> csr_features;
    std::vector<cp::u32> csr_cursors;
    std::vector<unsigned char> csr_values;
    cm::execution_csr_view csr{};
    cm::feature_major_projection_requirements feature_requirements{};
    std::vector<unsigned char> feature_payload;
    cm::feature_major_projection_view feature{};
    std::vector<unsigned char> feature_values;
    double csr_query_ms = 0.0;
    double csr_build_ms = 0.0;
    double feature_query_ms = 0.0;
    double feature_build_ms = 0.0;
    double feature_value_pack_ms = 0.0;
};

host_projections build_projections(const host_case &source,
    execution::structure_id structure_id,
    execution::structure_handle structure_handle,
    execution::structure_epoch epoch,
    execution::projection_id feature_id,
    execution::projection_handle feature_handle) {
    host_projections result;
    auto begin = clock_type::now();
    require(cm::query_lazy_execution_csr_requirements(
        source.payload, &result.csr_requirements), "query CSR projection");
    auto end = clock_type::now();
    result.csr_query_ms = milliseconds(begin, end);
    result.csr_rows.resize(result.csr_requirements.row_offset_count);
    result.csr_features.resize(result.csr_requirements.execution_feature_count);
    result.csr_cursors.resize(result.csr_requirements.row_cursor_count);
    result.csr_values.resize(result.csr_requirements.value_bytes);
    cm::lazy_execution_csr_buffers csr_buffers{
        result.csr_rows.size(), result.csr_features.size(),
        result.csr_values.size(), result.csr_cursors.size(),
        result.csr_rows.data(), result.csr_features.data(),
        result.csr_values.data(), result.csr_cursors.data()};
    begin = clock_type::now();
    require(cm::materialize_execution_csr_from_cpk1_host(
        source.payload, csr_buffers, &result.csr), "build CSR projection");
    end = clock_type::now();
    result.csr_build_ms = milliseconds(begin, end);

    cm::feature_major_projection_build_request request{};
    request.structure_identity = structure_id;
    request.runtime_structure = structure_handle;
    request.structure_epoch_value = epoch;
    request.projection_identity = feature_id;
    request.runtime_projection = feature_handle;
    request.source = source.payload;
    begin = clock_type::now();
    require(cm::query_feature_major_projection_requirements_host(
        request, &result.feature_requirements), "query feature projection");
    end = clock_type::now();
    result.feature_query_ms = milliseconds(begin, end);
    result.feature_payload.resize(result.feature_requirements.payload_bytes);
    begin = clock_type::now();
    require(cm::build_feature_major_projection_host(request,
        {result.feature_payload.data(), result.feature_payload.size()},
        &result.feature), "build feature projection");
    end = clock_type::now();
    result.feature_build_ms = milliseconds(begin, end);
    result.feature_values.resize(static_cast<std::size_t>(
        source.payload.tiles.nnz_count) * source.payload.tiles.value_size_bytes);
    begin = clock_type::now();
    require(cm::pack_feature_major_values_host(result.feature,
        source.tile_values.data(), source.tile_values.size() * sizeof(storage_t),
        {result.feature_values.data(), result.feature_values.size()}),
        "pack feature projection values");
    end = clock_type::now();
    result.feature_value_pack_ms = milliseconds(begin, end);
    return result;
}

__global__ void pack_rhs_columns_kernel(const float *row_major,
    float *column_major, std::uint32_t features, std::uint32_t columns) {
    const std::uint64_t count = static_cast<std::uint64_t>(features) * columns;
    for (std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x)
             * blockDim.x + threadIdx.x;
         index < count; index += static_cast<std::uint64_t>(blockDim.x)
             * gridDim.x) {
        const std::uint32_t feature = static_cast<std::uint32_t>(index / columns);
        const std::uint32_t column = static_cast<std::uint32_t>(
            index - static_cast<std::uint64_t>(feature) * columns);
        column_major[static_cast<std::size_t>(column) * features + feature]
            = row_major[index];
    }
}

__global__ void interleave_output_columns_kernel(const float *column_major,
    float *row_major, std::uint32_t rows, std::uint32_t columns) {
    const std::uint64_t count = static_cast<std::uint64_t>(rows) * columns;
    for (std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x)
             * blockDim.x + threadIdx.x;
         index < count; index += static_cast<std::uint64_t>(blockDim.x)
             * gridDim.x) {
        const std::uint32_t row = static_cast<std::uint32_t>(index / columns);
        const std::uint32_t column = static_cast<std::uint32_t>(
            index - static_cast<std::uint64_t>(row) * columns);
        row_major[index] = column_major[
            static_cast<std::size_t>(column) * rows + row];
    }
}

void launch_pack_rhs(const float *source, float *target,
    cp::u32 features, cp::u32 columns, cudaStream_t stream) {
    const std::uint64_t count = static_cast<std::uint64_t>(features) * columns;
    const std::uint32_t threads = 256u;
    const std::uint32_t blocks = static_cast<std::uint32_t>(
        std::min<std::uint64_t>((count + threads - 1u) / threads, 65535u));
    pack_rhs_columns_kernel<<<blocks, threads, 0u, stream>>>(
        source, target, features, columns);
    require_cuda(cudaPeekAtLastError(), "launch RHS pack");
}

void launch_interleave(const float *source, float *target,
    cp::u32 rows, cp::u32 columns, cudaStream_t stream) {
    const std::uint64_t count = static_cast<std::uint64_t>(rows) * columns;
    const std::uint32_t threads = 256u;
    const std::uint32_t blocks = static_cast<std::uint32_t>(
        std::min<std::uint64_t>((count + threads - 1u) / threads, 65535u));
    interleave_output_columns_kernel<<<blocks, threads, 0u, stream>>>(
        source, target, rows, columns);
    require_cuda(cudaPeekAtLastError(), "launch output interleave");
}

enum class measured_kind { row_masked, csr, feature_major };

struct pipeline {
    measured_kind kind{};
    const core::prepared_operation *prepared = nullptr;
    std::vector<execution::launch_bindings> launches;
    const float *common_rhs = nullptr;
    float *packed_rhs = nullptr;
    float *column_outputs = nullptr;
    float *common_output = nullptr;
    cp::u32 rows = 0u;
    cp::u32 features = 0u;
    cp::u32 columns = 0u;
    cudaStream_t stream = nullptr;
};

bool enqueue_pipeline(const pipeline &value) {
    if (value.kind != measured_kind::feature_major)
        launch_pack_rhs(value.common_rhs, value.packed_rhs,
            value.features, value.columns, value.stream);
    for (const execution::launch_bindings &launch : value.launches)
        if (!core::run_prepared_operation(*value.prepared, launch)) return false;
    if (value.kind != measured_kind::feature_major)
        launch_interleave(value.column_outputs, value.common_output,
            value.rows, value.columns, value.stream);
    return true;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2u];
}

struct timing_result {
    double pack_ms = 0.0;
    double kernel_ms = 0.0;
    double order_ms = 0.0;
    double total_ms = 0.0;
    double mad_percent = 0.0;
};

timing_result measure_pipeline(const pipeline &value,
    std::uint32_t warmups, std::uint32_t repeats) {
    for (std::uint32_t warmup = 0u; warmup < warmups; ++warmup)
        require(enqueue_pipeline(value), "warmup pipeline failed");
    require_cuda(cudaStreamSynchronize(value.stream), "warmup synchronize");
    cudaEvent_t events[4]{};
    for (cudaEvent_t &event : events)
        require_cuda(cudaEventCreate(&event), "create timing event");
    std::vector<double> packs, kernels, orders, totals;
    packs.reserve(repeats); kernels.reserve(repeats);
    orders.reserve(repeats); totals.reserve(repeats);
    for (std::uint32_t sample = 0u; sample < repeats; ++sample) {
        require_cuda(cudaEventRecord(events[0], value.stream), "record begin");
        if (value.kind != measured_kind::feature_major)
            launch_pack_rhs(value.common_rhs, value.packed_rhs,
                value.features, value.columns, value.stream);
        require_cuda(cudaEventRecord(events[1], value.stream), "record after pack");
        for (const execution::launch_bindings &launch : value.launches)
            require(core::run_prepared_operation(*value.prepared, launch),
                "timed candidate execution");
        require_cuda(cudaEventRecord(events[2], value.stream), "record after kernel");
        if (value.kind != measured_kind::feature_major)
            launch_interleave(value.column_outputs, value.common_output,
                value.rows, value.columns, value.stream);
        require_cuda(cudaEventRecord(events[3], value.stream), "record pipeline end");
        require_cuda(cudaEventSynchronize(events[3]), "sample synchronize");
        float pack = 0.0f, kernel = 0.0f, order = 0.0f, total = 0.0f;
        require_cuda(cudaEventElapsedTime(&pack, events[0], events[1]),
            "measure pack");
        require_cuda(cudaEventElapsedTime(&kernel, events[1], events[2]),
            "measure kernel");
        require_cuda(cudaEventElapsedTime(&order, events[2], events[3]),
            "measure order");
        require_cuda(cudaEventElapsedTime(&total, events[0], events[3]),
            "measure total");
        packs.push_back(value.kind == measured_kind::feature_major ? 0.0 : pack);
        kernels.push_back(kernel);
        orders.push_back(value.kind == measured_kind::feature_major ? 0.0 : order);
        totals.push_back(total);
    }
    for (cudaEvent_t event : events) cudaEventDestroy(event);
    timing_result result;
    result.pack_ms = median(packs);
    result.kernel_ms = median(kernels);
    result.order_ms = median(orders);
    result.total_ms = median(totals);
    std::vector<double> deviations;
    deviations.reserve(totals.size());
    for (double sample : totals)
        deviations.push_back(std::fabs(sample - result.total_ms));
    result.mad_percent = result.total_ms == 0.0 ? 0.0
        : median(deviations) * 100.0 / result.total_ms;
    return result;
}

std::vector<double> reference(const host_case &source,
    const std::vector<float> &rhs, cp::u32 columns) {
    std::vector<double> output(static_cast<std::size_t>(source.rows) * columns);
    for (cp::u32 row = 0u; row < source.rows; ++row)
        for (cp::u32 entry = source.csr_offsets[row];
             entry < source.csr_offsets[row + 1u]; ++entry) {
            const cp::u32 feature = source.csr_columns[entry];
            const double sparse = __half2float(source.csr_values[entry]);
            for (cp::u32 column = 0u; column < columns; ++column)
                output[static_cast<std::size_t>(row) * columns + column]
                    += sparse * rhs[static_cast<std::size_t>(feature)
                        * columns + column];
        }
    return output;
}

void validate_output(const std::vector<double> &expected,
    const std::vector<float> &actual) {
    require(expected.size() == actual.size(), "reference shape mismatch");
    for (std::size_t index = 0u; index < expected.size(); ++index) {
        const double error = std::fabs(expected[index] - actual[index]);
        if (error > 1.0e-5 + 1.0e-5 * std::fabs(expected[index]))
            fail("candidate numerical mismatch");
    }
}

std::uint64_t row_metadata_bytes(const host_case &source) {
    return (source.feature_offsets.size() + source.feature_permutation.size()
        + source.row_permutation.size() + source.tile_offsets.size()
        + source.tile_blocks.size() + source.tile_cell_masks.size()
        + source.entry_offsets.size() + source.gene_masks.size()
        + source.value_offsets.size()) * sizeof(cp::u32);
}

struct evidence_record {
    const char *schema = nullptr;
    const char *candidate = nullptr;
    const char *regime = nullptr;
    cp::u32 rows = 0u;
    cp::u32 features = 0u;
    cp::u32 nnz = 0u;
    cp::u32 sharing_groups = 0u;
    cp::u32 columns = 0u;
    std::uint32_t warmups = 0u;
    std::uint32_t repeats = 0u;
    std::uint64_t metadata_bytes = 0u;
    std::uint64_t value_bytes = 0u;
    std::uint64_t output_bytes = 0u;
    double query_ms = 0.0;
    double projection_build_ms = 0.0;
    double value_pack_ms = 0.0;
    double backend_prepare_ms = 0.0;
    timing_result timing{};
};

void emit_record(std::FILE *output, const evidence_record &record,
    const char *device_name, int sm, int driver, int runtime) {
    const double amortized = record.timing.total_ms
        + (record.query_ms + record.projection_build_ms
            + record.value_pack_ms + record.backend_prepare_ms)
            / static_cast<double>(expected_reuse);
    const double metadata_per_nnz = record.nnz == 0u ? 0.0
        : static_cast<double>(record.metadata_bytes) / record.nnz;
    std::fprintf(output,
        "{\"schema\":\"%s\",\"candidate\":\"%s\","
        "\"regime\":\"%s\",\"rows\":%u,\"features\":%u,\"nnz\":%u,"
        "\"sharing_groups\":%u,\"n\":%u,\"warmups\":%u,\"repeats\":%u,"
        "\"device\":\"%s\",\"sm\":%d,\"cuda_driver\":%d,"
        "\"cuda_runtime\":%d,\"expected_reuse\":%llu,"
        "\"correct\":true,\"output_effect\":\"overwrite\","
        "\"input_order\":\"packed-row-major\","
        "\"output_order\":\"execution-row-major\","
        "\"query_ms\":%.9g,\"projection_build_ms\":%.9g,"
        "\"value_pack_ms\":%.9g,\"backend_prepare_ms\":%.9g,"
        "\"dynamic_input_pack_ms\":%.9g,\"kernel_ms\":%.9g,"
        "\"output_order_ms\":%.9g,\"median_total_ms\":%.9g,"
        "\"mad_percent\":%.9g,\"amortized_total_ms\":%.9g,"
        "\"metadata_bytes\":%llu,\"value_bytes\":%llu,"
        "\"output_bytes\":%llu,\"metadata_bytes_per_nnz\":%.9g}\n",
        record.schema, record.candidate, record.regime, record.rows, record.features,
        record.nnz, record.sharing_groups, record.columns, record.warmups,
        record.repeats, device_name, sm, driver, runtime,
        static_cast<unsigned long long>(expected_reuse), record.query_ms,
        record.projection_build_ms, record.value_pack_ms,
        record.backend_prepare_ms, record.timing.pack_ms,
        record.timing.kernel_ms, record.timing.order_ms,
        record.timing.total_ms, record.timing.mad_percent, amortized,
        static_cast<unsigned long long>(record.metadata_bytes),
        static_cast<unsigned long long>(record.value_bytes),
        static_cast<unsigned long long>(record.output_bytes), metadata_per_nnz);
}

void benchmark_case(const host_case &source,
    const host_projections &host_views,
    cp::u32 columns,
    std::uint32_t warmups,
    std::uint32_t repeats,
    int device,
    cudaStream_t stream,
    std::FILE *artifact,
    const char *device_name,
    int sm,
    int driver,
    int runtime) {
    const execution::structure_id structure_id{0x7611u, 0x7612u};
    const execution::structure_handle structure_handle{76u, 1u};
    const execution::structure_epoch epoch{1u};
    const execution::projection_id row_id{0x7621u, source.sharing_groups};
    const execution::projection_handle row_handle{77u, source.sharing_groups};
    const execution::projection_id csr_id{0x7631u, source.sharing_groups};
    const execution::projection_handle csr_handle{78u, source.sharing_groups};
    const execution::projection_id feature_id{0x7641u, source.sharing_groups};
    const execution::projection_handle feature_handle{79u, source.sharing_groups};
    const execution::axis_identity feature_axis = axis(10u);
    const execution::axis_identity row_axis = axis(20u);
    const execution::axis_identity dense_axis = axis(30u);

    device_array<unsigned char> d_image(1u);
    device_array<cp::u32> d_feature_offsets(source.feature_offsets.size());
    device_array<cp::u32> d_feature_permutation(source.feature_permutation.size());
    device_array<cp::u32> d_row_permutation(source.row_permutation.size());
    device_array<cp::u32> d_tile_offsets(source.tile_offsets.size());
    device_array<cp::u32> d_tile_blocks(source.tile_blocks.size());
    device_array<cp::u32> d_tile_masks(source.tile_cell_masks.size());
    device_array<cp::u32> d_entry_offsets(source.entry_offsets.size());
    device_array<cp::u32> d_gene_masks(source.gene_masks.size());
    device_array<cp::u32> d_value_offsets(source.value_offsets.size());
    device_array<storage_t> d_row_values(source.tile_values.size());
    device_array<cp::u32> d_csr_rows(host_views.csr_rows.size());
    device_array<cp::u32> d_csr_features(host_views.csr_features.size());
    device_array<storage_t> d_csr_values(host_views.csr_values.size()
        / sizeof(storage_t));
    device_array<unsigned char> d_feature_payload(host_views.feature_payload.size());
    device_array<storage_t> d_feature_values(host_views.feature_values.size()
        / sizeof(storage_t));
    upload(d_feature_offsets, source.feature_offsets);
    upload(d_feature_permutation, source.feature_permutation);
    upload(d_row_permutation, source.row_permutation);
    upload(d_tile_offsets, source.tile_offsets);
    upload(d_tile_blocks, source.tile_blocks);
    upload(d_tile_masks, source.tile_cell_masks);
    upload(d_entry_offsets, source.entry_offsets);
    upload(d_gene_masks, source.gene_masks);
    upload(d_value_offsets, source.value_offsets);
    upload(d_row_values, source.tile_values);
    upload(d_csr_rows, host_views.csr_rows);
    upload(d_csr_features, host_views.csr_features);
    require_cuda(cudaMemcpy(d_csr_values.data, host_views.csr_values.data(),
        host_views.csr_values.size(), cudaMemcpyHostToDevice), "upload CSR values");
    require_cuda(cudaMemcpy(d_feature_payload.data, host_views.feature_payload.data(),
        host_views.feature_payload.size(), cudaMemcpyHostToDevice),
        "upload feature payload");
    require_cuda(cudaMemcpy(d_feature_values.data, host_views.feature_values.data(),
        host_views.feature_values.size(), cudaMemcpyHostToDevice),
        "upload feature values");

    cp::persistent_packing_payload_view row_view = source.payload;
    row_view.image_base = d_image.data;
    row_view.plan.feature_block_offsets = d_feature_offsets.data;
    row_view.plan.feature_permutation = d_feature_permutation.data;
    row_view.order.row_permutation = d_row_permutation.data;
    row_view.tiles.tile_block_offsets = d_tile_offsets.data;
    row_view.tiles.tile_block_ids = d_tile_blocks.data;
    row_view.tiles.tile_block_cell_masks = d_tile_masks.data;
    row_view.tiles.block_row_entry_offsets = d_entry_offsets.data;
    row_view.tiles.row_block_gene_masks = d_gene_masks.data;
    row_view.tiles.row_block_value_offsets = d_value_offsets.data;
    row_view.tiles.values = d_row_values.data;
    cm::execution_csr_view csr_view = host_views.csr;
    csr_view.row_offsets = d_csr_rows.data;
    csr_view.execution_feature_ids = d_csr_features.data;
    csr_view.values = d_csr_values.data;
    cm::feature_major_projection_view feature_view{};
    require(cm::rebind_feature_major_projection(host_views.feature,
        d_feature_payload.data, host_views.feature_payload.size(), &feature_view),
        "rebind feature projection");

    core::structure_set_key structures{};
    structures.count = 1u;
    structures.structures[0] = {structure_id, structure_handle, epoch};
    const core::operation_problem vector_problem{core::operation_core_schema_version,
        core::operation_kind::weighted_relation_reduce, 0u,
        {7601u, columns}, 1u, 1u, source.payload.tiles.nnz_count};
    const core::operation_problem matrix_problem{core::operation_core_schema_version,
        core::operation_kind::sparse_dense_multiply, 0u,
        {7602u, columns}, 1u, 1u,
        static_cast<std::uint64_t>(source.payload.tiles.nnz_count) * columns};
    const core::projection_key row_key{row_id, row_handle,
        core::projection_kind::native_row_masked,
        cp::persistent_packing_payload_schema_version, 1u};
    const core::projection_key csr_key{csr_id, csr_handle,
        core::projection_kind::csr, cm::execution_csr_schema_version, 1u};
    const core::projection_key feature_key{feature_id, feature_handle,
        core::projection_kind::native_feature_major,
        cm::feature_major_projection_schema_version,
        cm::feature_major_projection_variant};
    const core::prepare_policy prepare_policy{true, false, true, true,
        static_cast<std::uint32_t>(expected_reuse), 0u, 0u};
    core::row_masked_n1_prepared_state row_state{};
    core::csr_fallback_prepared_state csr_state{};
    core::feature_major_small_n_prepared_state feature_state{};
    core::prepared_operation row_prepared{}, csr_prepared{}, feature_prepared{};
    auto begin = clock_type::now();
    require(core::prepare_row_masked_n1_operation(vector_problem, structures,
        row_key, vector_numeric(), prepare_policy, row_view,
        feature_axis, row_axis, &row_state, &row_prepared), "prepare row candidate");
    auto end = clock_type::now();
    const double row_prepare_ms = milliseconds(begin, end);
    begin = clock_type::now();
    require(core::prepare_csr_fallback_operation(vector_problem, structures,
        csr_key, vector_numeric(), prepare_policy, csr_view, device,
        feature_axis, row_axis, &csr_state, &csr_prepared), "prepare CSR candidate");
    end = clock_type::now();
    const double csr_prepare_ms = milliseconds(begin, end);
    begin = clock_type::now();
    const bool medium_n = columns
        >= core::feature_major_cta_medium_n_minimum;
    const core::operation_status feature_prepare_status = medium_n
        ? core::prepare_feature_major_cta_medium_n_operation(matrix_problem,
            structures, feature_key, matrix_numeric(), prepare_policy,
            feature_view, device, columns, feature_axis, row_axis, dense_axis,
            &feature_state, &feature_prepared)
        : core::prepare_feature_major_small_n_operation(matrix_problem,
            structures, feature_key, matrix_numeric(), prepare_policy,
            feature_view, device, columns, feature_axis, row_axis, dense_axis,
            &feature_state, &feature_prepared);
    require(feature_prepare_status, "prepare feature candidate");
    end = clock_type::now();
    const double feature_prepare_ms = milliseconds(begin, end);

    execution::relation_structure relation{};
    relation.identity = structure_handle;
    relation.epoch = epoch;
    relation.source_axis = feature_axis;
    relation.destination_axis = row_axis;
    relation.projections = {1u, 1u};
    relation.logical_edge_count = source.payload.tiles.nnz_count;
    execution::value_plane row_plane{};
    row_plane.structure = structure_handle;
    row_plane.structure_epoch_value = epoch;
    row_plane.values = d_row_values.data;
    row_plane.location = device_location(device);
    row_plane.numeric = {execution::numeric_type::f16,
        execution::numeric_type::f32, execution::numeric_type::f32, 0u};
    row_plane.quantization.kind = execution::quantization_kind::none;
    row_plane.layout = execution::value_layout_kind::projection_local_order;
    row_plane.generation = {1u};
    row_plane.element_count = source.payload.tiles.nnz_count;
    row_plane.value_bytes = source.tile_values.size() * sizeof(storage_t);
    execution::value_plane csr_plane = row_plane;
    csr_plane.values = d_csr_values.data;
    execution::value_plane feature_plane = row_plane;
    feature_plane.values = d_feature_values.data;
    execution::value_binding row_binding{&row_plane, row_plane.generation};
    execution::value_binding csr_binding{&csr_plane, csr_plane.generation};
    execution::value_binding feature_binding{&feature_plane,
        feature_plane.generation};

    std::vector<float> rhs(static_cast<std::size_t>(source.features) * columns);
    for (cp::u32 feature = 0u; feature < source.features; ++feature)
        for (cp::u32 column = 0u; column < columns; ++column)
            rhs[static_cast<std::size_t>(feature) * columns + column]
                = 0.25f + static_cast<float>((feature * 7u + column * 11u) % 97u)
                    * 0.0078125f;
    const std::vector<double> expected = reference(source, rhs, columns);
    device_array<float> d_rhs(rhs.size());
    device_array<float> d_packed_rhs(rhs.size());
    device_array<float> d_column_outputs(
        static_cast<std::size_t>(source.rows) * columns);
    device_array<float> d_row_output(
        static_cast<std::size_t>(source.rows) * columns);
    device_array<float> d_csr_output(
        static_cast<std::size_t>(source.rows) * columns);
    device_array<float> d_feature_output(
        static_cast<std::size_t>(source.rows) * columns);
    upload(d_rhs, rhs);

    std::vector<execution::biological_operand_view> row_inputs(columns),
        row_outputs(columns), csr_inputs(columns), csr_outputs(columns);
    std::vector<execution::launch_bindings> row_launches(columns),
        csr_launches(columns);
    for (cp::u32 column = 0u; column < columns; ++column) {
        row_inputs[column].kind = execution::operand_kind::dense_tensor;
        row_inputs[column].storage.dense = dense_vector(
            d_packed_rhs.data + static_cast<std::size_t>(column) * source.features,
            feature_axis, source.features, device);
        csr_inputs[column] = row_inputs[column];
        row_outputs[column].kind = execution::operand_kind::dense_tensor;
        row_outputs[column].storage.dense = dense_vector(
            d_column_outputs.data + static_cast<std::size_t>(column) * source.rows,
            row_axis, source.rows, device);
        csr_outputs[column] = row_outputs[column];
        row_launches[column].structures = &relation;
        row_launches[column].inputs = &row_inputs[column];
        row_launches[column].outputs = &row_outputs[column];
        row_launches[column].values = &row_binding;
        row_launches[column].input_count = 1u;
        row_launches[column].output_count = 1u;
        row_launches[column].value_count = 1u;
        row_launches[column].structure_count = 1u;
        row_launches[column].scalars.values[0] = {
            core::row_masked_n1_feature_weight_generation_binding,
            execution::numeric_type::u32, {}, 1u};
        row_launches[column].scalars.count = 1u;
        row_launches[column].stream = {stream, device, 0u};
        row_launches[column].workspace = {nullptr, 0u, device_location(device)};
        csr_launches[column] = row_launches[column];
        csr_launches[column].inputs = &csr_inputs[column];
        csr_launches[column].outputs = &csr_outputs[column];
        csr_launches[column].values = &csr_binding;
    }
    execution::biological_operand_view feature_input{}, feature_output{};
    feature_input.kind = execution::operand_kind::dense_tensor;
    feature_input.storage.dense = dense_matrix(d_rhs.data, feature_axis,
        dense_axis, source.features, columns, device);
    feature_output.kind = execution::operand_kind::dense_tensor;
    feature_output.storage.dense = dense_matrix(d_feature_output.data, row_axis,
        dense_axis, source.rows, columns, device);
    execution::launch_bindings feature_launch{};
    feature_launch.structures = &relation;
    feature_launch.inputs = &feature_input;
    feature_launch.outputs = &feature_output;
    feature_launch.values = &feature_binding;
    feature_launch.input_count = 1u;
    feature_launch.output_count = 1u;
    feature_launch.value_count = 1u;
    feature_launch.structure_count = 1u;
    feature_launch.stream = {stream, device, 0u};
    feature_launch.workspace = {nullptr, 0u, device_location(device)};

    pipeline row_pipeline{measured_kind::row_masked, &row_prepared,
        row_launches, d_rhs.data, d_packed_rhs.data, d_column_outputs.data,
        d_row_output.data, source.rows, source.features, columns, stream};
    pipeline csr_pipeline{measured_kind::csr, &csr_prepared,
        csr_launches, d_rhs.data, d_packed_rhs.data, d_column_outputs.data,
        d_csr_output.data, source.rows, source.features, columns, stream};
    pipeline feature_pipeline{measured_kind::feature_major, &feature_prepared,
        {feature_launch}, d_rhs.data, nullptr, nullptr, d_feature_output.data,
        source.rows, source.features, columns, stream};

    std::vector<float> actual(static_cast<std::size_t>(source.rows) * columns);
    for (const pipeline *candidate : {&row_pipeline, &csr_pipeline,
             &feature_pipeline}) {
        require(enqueue_pipeline(*candidate), "correctness pipeline failed");
        require_cuda(cudaStreamSynchronize(stream), "correctness synchronize");
        require_cuda(cudaMemcpy(actual.data(), candidate->common_output,
            actual.size() * sizeof(float), cudaMemcpyDeviceToHost),
            "download candidate output");
        validate_output(expected, actual);
    }

    const timing_result row_timing = measure_pipeline(
        row_pipeline, warmups, repeats);
    const timing_result csr_timing = measure_pipeline(
        csr_pipeline, warmups, repeats);
    const timing_result feature_timing = measure_pipeline(
        feature_pipeline, warmups, repeats);
    const std::uint64_t value_bytes = source.tile_values.size() * sizeof(storage_t);
    const std::uint64_t output_bytes = static_cast<std::uint64_t>(source.rows)
        * columns * sizeof(float);
    const std::uint64_t csr_metadata = (host_views.csr_rows.size()
        + host_views.csr_features.size()) * sizeof(cp::u32);
    const char *const schema = medium_n
        ? "CE-ARCH-84-EVIDENCE/1" : "CE-ARCH-76-EVIDENCE/1";
    const evidence_record records[3] = {
        {schema, "row_masked", source.name, source.rows, source.features,
            source.payload.tiles.nnz_count, source.sharing_groups, columns,
            warmups, repeats, row_metadata_bytes(source), value_bytes,
            output_bytes, 0.0, 0.0, 0.0, row_prepare_ms, row_timing},
        {schema, "csr", source.name, source.rows, source.features,
            source.payload.tiles.nnz_count, source.sharing_groups, columns,
            warmups, repeats, csr_metadata, value_bytes, output_bytes,
            host_views.csr_query_ms, host_views.csr_build_ms, 0.0,
            csr_prepare_ms, csr_timing},
        {schema, medium_n ? "feature_major_cta" : "feature_major",
            source.name, source.rows, source.features,
            source.payload.tiles.nnz_count, source.sharing_groups, columns,
            warmups, repeats, host_views.feature_payload.size(), value_bytes,
            output_bytes, host_views.feature_query_ms,
            host_views.feature_build_ms, host_views.feature_value_pack_ms,
            feature_prepare_ms, feature_timing}
    };
    for (const evidence_record &record : records) {
        emit_record(stdout, record, device_name, sm, driver, runtime);
        if (artifact != nullptr)
            emit_record(artifact, record, device_name, sm, driver, runtime);
    }
}

struct options {
    bool quick = false;
    bool ce_arch_84 = false;
    const char *output_path = nullptr;
};

options parse_options(int argc, char **argv) {
    options result;
    for (int index = 1; index < argc; ++index) {
        if (std::strcmp(argv[index], "--quick") == 0) result.quick = true;
        else if (std::strcmp(argv[index], "--ce-arch-84") == 0)
            result.ce_arch_84 = true;
        else if (std::strcmp(argv[index], "--output") == 0
            && index + 1 < argc) result.output_path = argv[++index];
        else fail("usage: [--quick] [--ce-arch-84] [--output path]");
    }
    return result;
}

} // namespace

int main(int argc, char **argv) {
    if (!(std::is_same<storage_t, __half>::value
            && std::is_same<compute_t, float>::value
            && std::is_same<accum_t, float>::value)) {
        fail("CE-ARCH-76 evidence requires configured f16/f32/f32 precision");
    }
    const options option = parse_options(argc, argv);
    int device = -1;
    require_cuda(cudaGetDevice(&device), "get controller-selected device");
    cudaDeviceProp properties{};
    require_cuda(cudaGetDeviceProperties(&properties, device),
        "get device properties");
    int driver = 0, runtime = 0;
    require_cuda(cudaDriverGetVersion(&driver), "get CUDA driver version");
    require_cuda(cudaRuntimeGetVersion(&runtime), "get CUDA runtime version");
    require(properties.major == 7 && properties.minor == 0,
        "CE-ARCH-76 live contract requires a V100 sm_70 device");
    cudaStream_t stream = nullptr;
    require_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        "create benchmark stream");
    std::FILE *artifact = nullptr;
    if (option.output_path != nullptr) {
        artifact = std::fopen(option.output_path, "w");
        if (artifact == nullptr) fail("open evidence output");
    }
    const cp::u32 rows = option.quick ? quick_rows : full_rows;
    const cp::u32 features = option.quick ? quick_features : full_features;
    const std::uint32_t warmups = option.quick ? quick_warmups : full_warmups;
    const std::uint32_t repeats = option.quick ? quick_repeats : full_repeats;
    const struct { const char *name; cp::u32 sharing; } regimes[] = {
        {"high_sharing", 1u}, {"medium_sharing", 8u}, {"low_sharing", 32u}};
    for (const auto &regime : regimes) {
        host_case source = make_case(
            regime.name, rows, features, regime.sharing);
        refresh_payload(source);
        const execution::structure_id structure_id{0x7611u, 0x7612u};
        const execution::structure_handle structure_handle{76u, 1u};
        const execution::structure_epoch epoch{1u};
        const execution::projection_id feature_id{0x7641u, regime.sharing};
        const execution::projection_handle feature_handle{79u, regime.sharing};
        const host_projections projections = build_projections(source,
            structure_id, structure_handle, epoch, feature_id, feature_handle);
        if (option.ce_arch_84) {
            for (std::uint32_t columns : medium_dense_widths)
                benchmark_case(source, projections, columns, warmups, repeats,
                    device, stream, artifact, properties.name,
                    properties.major * 10 + properties.minor, driver, runtime);
        } else {
            for (std::uint32_t columns : small_dense_widths)
                benchmark_case(source, projections, columns, warmups, repeats,
                    device, stream, artifact, properties.name,
                    properties.major * 10 + properties.minor, driver, runtime);
        }
    }
    if (artifact != nullptr && std::fclose(artifact) != 0)
        fail("close evidence output");
    require_cuda(cudaStreamDestroy(stream), "destroy benchmark stream");
    return 0;
}
