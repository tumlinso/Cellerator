#include "Cellerator/geometry/persistent_packing_payload.hh"
#include "Cellerator/geometry/feature_weighted_row_reduction_cuda.hh"

#include <CellShard/io/pack/execution_payload.cuh>

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

namespace cp = cellpack;
namespace cs = cellshard;
using storage_t = cellerator::real::storage_t;
using compute_t = cellerator::real::compute_t;
using accum_t = cellerator::real::accum_t;

void require(bool condition, const char *message) {
    if (!condition) {
        std::fprintf(stderr, "cellPackPersistentPackingPayloadTest: %s\n", message);
        std::exit(1);
    }
}

void require_status(cp::validation_result status, const char *message) {
    if (!status) {
        std::fprintf(stderr, "cellPackPersistentPackingPayloadTest: %s: %s\n",
            message, status.message);
        std::exit(1);
    }
}

void require_cuda(cudaError_t status, const char *message) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "cellPackPersistentPackingPayloadTest: %s: %s\n",
            message, cudaGetErrorString(status));
        std::exit(1);
    }
}

template<typename T>
struct device_array {
    T *data = nullptr;
    explicit device_array(std::size_t count) {
        if (count != 0u) require_cuda(cudaMalloc(reinterpret_cast<void **>(&data),
            count * sizeof(T)), "cudaMalloc");
    }
    ~device_array() { if (data != nullptr) cudaFree(data); }
    device_array(const device_array &) = delete;
    device_array &operator=(const device_array &) = delete;
};

cp::frozen_packing_plan make_plan() {
    const cp::u32 permutation[] = {2u, 0u, 4u, 5u, 1u, 3u};
    cp::u32 inverse[6]{}, to_block[6]{}, to_local[6]{};
    const cp::u32 block_offsets[] = {0u, 3u, 6u};
    for (cp::u32 block = 0u; block < 2u; ++block) {
        for (cp::u32 execution = block_offsets[block];
             execution < block_offsets[block + 1u]; ++execution) {
            const cp::u32 canonical = permutation[execution];
            inverse[canonical] = execution;
            to_block[canonical] = block;
            to_local[canonical] = execution - block_offsets[block];
        }
    }
    const cp::u32 row_groups[] = {0u, 4u, 8u};
    cp::frozen_packing_plan_build_view build;
    build.row_count = 8u;
    build.feature_count = 6u;
    build.feature_permutation = permutation;
    build.inverse_feature_permutation = inverse;
    build.feature_block_count = 2u;
    build.feature_block_offsets = block_offsets;
    build.feature_to_block = to_block;
    build.feature_to_local = to_local;
    build.row_group_count = 2u;
    build.row_group_offsets = row_groups;
    build.maximum_feature_block_width = 3u;
    build.row_group_width = 4u;
    build.identity.feature_axis_fingerprint = 0x1020304050607080ull;
    build.identity.feature_axis_fingerprint_version = 1u;
    build.identity.row_domain_kind = cp::packing_row_domain_kind::full_dataset_identity;
    build.identity.row_domain_identity = 0x8877665544332211ull;
    build.identity.evaluation_source_identity = 0x1234u;
    build.cost_policy_identity = 0x5678u;
    cp::frozen_packing_plan plan;
    require_status(cp::freeze_packing_plan(build, &plan), "freeze plan");
    return plan;
}

struct source_fixture {
    std::vector<cp::u32> offsets{0u, 3u, 5u, 6u, 9u};
    std::vector<cp::u32> features{0u, 2u, 5u, 1u, 4u, 3u, 0u, 1u, 5u};
    std::vector<storage_t> values;
    source_fixture() {
        for (cp::u32 index = 0u; index < features.size(); ++index)
            values.push_back(static_cast<storage_t>(0.25f * static_cast<float>(index + 1u)));
    }
};

cp::plan_application_context application_context() {
    cp::plan_application_context result;
    result.full_row_count = 8u;
    result.feature_count = 6u;
    result.feature_axis_fingerprint = 0x1020304050607080ull;
    result.feature_axis_fingerprint_version = 1u;
    result.row_domain_identity = 0x8877665544332211ull;
    return result;
}

cp::plan_application_source_view source_view(const source_fixture &source) {
    cp::plan_application_source_view result;
    result.global_row_begin = 2u;
    result.row_count = 4u;
    result.feature_count = 6u;
    result.nnz_count = static_cast<cp::u32>(source.features.size());
    result.value_size_bytes = sizeof(storage_t);
    result.row_offsets = source.offsets.data();
    result.canonical_feature_ids = source.features.data();
    result.values = source.values.data();
    return result;
}

struct packed_fixture {
    std::vector<cp::u32> ordered_offsets, blocks, locals, canonical;
    std::vector<storage_t> ordered_values;
    cp::ordered_plan_partition_view ordered{};
    std::vector<cp::u32> record_offsets, record_blocks, record_masks,
        record_value_offsets;
    std::vector<unsigned char> record_values;
    cp::cell_block_record_view records{};
    std::vector<cp::u64> primary;
    std::vector<cp::u32> secondary, active, nnz, permutation, inverse;
    cp::local_cell_order_view order{};
    std::vector<cp::u32> tile_offsets, tile_blocks, tile_masks, entry_offsets,
        gene_masks, value_offsets;
    std::vector<unsigned char> tile_values;
    cp::warp_tile_view tiles{};
};

packed_fixture pack(const cp::frozen_packing_plan &plan,
    const source_fixture &source) {
    packed_fixture result;
    const cp::u32 nnz = static_cast<cp::u32>(source.features.size());
    result.ordered_offsets.resize(source.offsets.size());
    result.blocks.resize(nnz);
    result.locals.resize(nnz);
    result.canonical.resize(nnz);
    result.ordered_values.resize(nnz);
    std::vector<cp::u64> keys(nnz);
    std::vector<cp::u32> source_order(nnz);
    cp::plan_application_host_workspace_view workspace{nnz, keys.data(), source_order.data()};
    cp::plan_application_buffers buffers{result.ordered_offsets.size(), nnz,
        nnz * sizeof(storage_t), result.ordered_offsets.data(), result.blocks.data(),
        result.locals.data(), result.canonical.data(), result.ordered_values.data()};
    require_status(cp::apply_frozen_plan_host(plan, application_context(),
        source_view(source), workspace, buffers, &result.ordered), "apply plan");

    cp::cell_block_record_requirements record_required;
    require_status(cp::query_cell_block_record_requirements_host(plan,
        result.ordered, &record_required), "query records");
    result.record_offsets.resize(record_required.row_record_offset_count);
    result.record_blocks.resize(record_required.record_count);
    result.record_masks.resize(record_required.record_count);
    result.record_value_offsets.resize(record_required.record_value_offset_count);
    result.record_values.resize(record_required.value_bytes);
    cp::cell_block_record_buffers record_buffers{result.record_offsets.size(),
        result.record_blocks.size(), result.record_value_offsets.size(),
        result.record_values.size(), result.record_offsets.data(),
        result.record_blocks.data(), result.record_masks.data(),
        result.record_value_offsets.data(), result.record_values.data()};
    require_status(cp::build_cell_block_records_host(plan, result.ordered,
        record_buffers, &result.records), "build records");

    result.primary.resize(4u);
    result.secondary.resize(4u);
    result.active.resize(4u);
    result.nnz.resize(4u);
    result.permutation.resize(4u);
    result.inverse.resize(4u);
    cp::local_cell_order_buffers order_buffers{4u, result.primary.data(),
        result.secondary.data(), result.active.data(), result.nnz.data(),
        result.permutation.data(), result.inverse.data()};
    cp::local_cell_order_config config;
    config.kind = cp::local_cell_order_kind::row_nnz_descending;
    config.window_size = 4u;
    config.group_width = 4u;
    require_status(cp::build_local_cell_order_host(result.records, config,
        order_buffers, &result.order), "build local order");
    require(result.permutation != std::vector<cp::u32>({0u, 1u, 2u, 3u}),
        "fixture must exercise nonidentity row order");

    cp::warp_tile_requirements tile_required;
    require_status(cp::query_warp_tile_requirements_host(plan, result.records,
        result.order, &tile_required), "query tiles");
    result.tile_offsets.resize(tile_required.tile_block_offset_count);
    result.tile_blocks.resize(tile_required.tile_block_count);
    result.tile_masks.resize(tile_required.tile_block_count);
    result.entry_offsets.resize(tile_required.block_row_entry_offset_count);
    result.gene_masks.resize(tile_required.row_block_entry_count);
    result.value_offsets.resize(tile_required.row_block_value_offset_count);
    result.tile_values.resize(tile_required.value_bytes);
    cp::warp_tile_buffers tile_buffers{result.tile_offsets.size(),
        result.tile_blocks.size(), result.entry_offsets.size(), result.gene_masks.size(),
        result.value_offsets.size(), result.tile_values.size(), result.tile_offsets.data(),
        result.tile_blocks.data(), result.tile_masks.data(), result.entry_offsets.data(),
        result.gene_masks.data(), result.value_offsets.data(), result.tile_values.data()};
    require_status(cp::build_warp_tiles_host(plan, result.records, result.order,
        tile_buffers, &result.tiles), "build tiles");
    return result;
}

std::string temporary_path() {
    std::string path = "/tmp/cellpack_persistent_payloadXXXXXX";
    const int descriptor = ::mkstemp(path.data());
    require(descriptor >= 0, "mkstemp");
    ::close(descriptor);
    ::unlink(path.c_str());
    return path + ".cspack";
}

} // namespace

int main() {
    const auto plan = make_plan();
    const source_fixture source;
    const auto packed = pack(plan, source);
    cp::persistent_packing_payload_requirements required;
    require_status(cp::query_persistent_packing_payload_requirements_host(plan,
        packed.records, packed.order, packed.tiles, &required), "query persistent image");
    std::vector<unsigned char> image(required.image_bytes);
    cp::persistent_packing_payload_view built;
    require_status(cp::build_persistent_packing_payload_host(plan, packed.records,
        packed.order, packed.tiles, {image.size(), image.data()}, &built),
        "build persistent image");
    require(built.payload_identity != 0u && built.tiles.full_row_count == 8u
        && built.tiles.row_count == 4u, "persistent image identity/domain");

    cs::execution_payload_identity identity;
    identity.dataset_identity = 0x9001u;
    identity.generation = {1u, 2u, 3u, 4u};
    identity.partition_identity = 0x9010u;
    identity.global_row_begin = built.tiles.global_row_begin;
    identity.row_count = built.tiles.row_count;
    identity.feature_count = built.tiles.feature_count;
    identity.feature_axis_fingerprint = built.tiles.feature_axis_fingerprint;
    identity.feature_axis_fingerprint_version = built.tiles.feature_axis_fingerprint_version;
    identity.payload_kind = built.payload_kind;
    identity.payload_schema_version = built.payload_schema_version;
    identity.row_domain_identity = built.tiles.row_domain_identity;
    identity.payload_identity = built.payload_identity;
    const cs::execution_payload_source archived{identity, image.data(), image.size()};
    const std::string path = temporary_path();
    require(cs::store_execution_cspack(path.c_str(), 0x77u, &archived, 1u) != 0,
        "publish CSPACK execution payload");

    std::fill(image.begin(), image.end(), 0u);
    cs::execution_payload_host loaded;
    require(cs::load_execution_cspack_partition(path.c_str(), 0x77u, 0u,
        identity, &loaded) != 0, "load CSPACK execution payload");
    cp::persistent_packing_payload_compatibility compatibility{identity.global_row_begin,
        identity.row_count, identity.feature_count, identity.feature_axis_fingerprint,
        identity.feature_axis_fingerprint_version, identity.row_domain_identity,
        identity.payload_identity};
    cp::persistent_packing_payload_view host_view;
    require_status(cp::validate_persistent_packing_payload_host(loaded.payload,
        loaded.payload_bytes, compatibility, &host_view), "validate loaded image");
    auto *loaded_bytes = static_cast<unsigned char *>(loaded.storage);
    loaded_bytes[loaded.payload_bytes - 1u] ^= 0x80u;
    cp::persistent_packing_payload_view rejected;
    require(!cp::validate_persistent_packing_payload_host(loaded.payload,
        loaded.payload_bytes, compatibility, &rejected), "reject inner image tamper");
    loaded_bytes[loaded.payload_bytes - 1u] ^= 0x80u;
    auto wrong = compatibility;
    wrong.feature_axis_fingerprint ^= 1u;
    require(!cp::validate_persistent_packing_payload_host(loaded.payload,
        loaded.payload_bytes, wrong, &rejected), "reject image compatibility mismatch");

    std::vector<compute_t> weights(plan.feature_count());
    for (cp::u32 feature = 0u; feature < weights.size(); ++feature)
        weights[feature] = static_cast<compute_t>(0.5f + 0.125f * feature);
    const auto host_input = cp::make_feature_weighted_row_reduction_view(plan,
        packed.tiles, 0xabcdefu, weights.size(), weights.data());
    std::vector<accum_t> reference(packed.tiles.row_count);
    cp::feature_weighted_row_reduction_result_view reference_result;
    require_status(cp::evaluate_feature_weighted_row_reduction_canonical_host(plan,
        application_context(), source_view(source), host_input,
        {reference.size(), reference.data()}, &reference_result), "canonical reference");

    cudaStream_t stream = nullptr;
    require_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
    cs::execution_payload_device device_payload;
    require_cuda(cs::upload_execution_payload_async(loaded, 0, stream,
        &device_payload), "upload persistent image");
    device_array<compute_t> device_weights(weights.size());
    device_array<accum_t> device_output(reference.size());
    require_cuda(cudaMemcpyAsync(device_weights.data, weights.data(),
        weights.size() * sizeof(compute_t), cudaMemcpyHostToDevice, stream),
        "upload weights");
    cp::persistent_packing_payload_view device_view;
    require_status(cp::rebind_persistent_packing_payload(host_view,
        device_payload.payload, device_payload.payload_bytes, &device_view),
        "rebind persistent image");
    const auto device_input = cp::make_persistent_feature_weighted_row_reduction_view(
        device_view, 0xabcdefu, weights.size(), device_weights.data);
    cp::feature_weighted_row_reduction_result_view result;
    require_status(cp::evaluate_feature_weighted_row_reduction_tiles_cuda(device_input,
        device_view.order, {reference.size(), device_output.data}, stream, &result),
        "execute directly from persistent image");
    std::vector<accum_t> actual(reference.size());
    require_cuda(cudaMemcpyAsync(actual.data(), device_output.data,
        actual.size() * sizeof(accum_t), cudaMemcpyDeviceToHost, stream),
        "download output");
    require_cuda(cudaStreamSynchronize(stream), "synchronize execution");
    for (std::size_t row = 0u; row < actual.size(); ++row)
        require(cp::feature_weighted_row_reduction_within_tolerance(reference[row],
            actual[row]), "persistent direct execution mismatch");

    require(result.global_row_begin == 2u && result.row_count == 4u
        && result.row_domain_identity == identity.row_domain_identity,
        "persistent execution result identity");
    require_cuda(cs::clear_execution_payload_device(&device_payload),
        "release device image");
    require_cuda(cudaStreamDestroy(stream), "destroy stream");
    cs::clear_execution_payload_host(&loaded);
    ::unlink(path.c_str());
    std::puts("cellPackPersistentPackingPayloadTest: passed");
    return 0;
}
