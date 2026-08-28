#include "Cellerator/geometry/apply_plan.hh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

namespace {

using cellpack::u32;
using cellpack::u64;

void require(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "cellPackApplyPlanTest: " << message << '\n';
        std::exit(1);
    }
}

void require_cuda(cudaError_t error, const char *message) {
    if (error != cudaSuccess) {
        std::cerr << "cellPackApplyPlanTest: " << message << ": "
                  << cudaGetErrorString(error) << '\n';
        std::exit(1);
    }
}

template<class T>
struct device_buffer {
    T *data = nullptr;
    std::size_t count = 0u;

    device_buffer() = default;
    explicit device_buffer(std::size_t count_) : count(count_) {
        if (count != 0u) require_cuda(cudaMalloc(&data, count * sizeof(T)), "cudaMalloc failed");
    }
    device_buffer(const device_buffer &) = delete;
    device_buffer &operator=(const device_buffer &) = delete;
    device_buffer(device_buffer &&other) noexcept : data(other.data), count(other.count) {
        other.data = nullptr;
        other.count = 0u;
    }
    ~device_buffer() { if (data != nullptr) cudaFree(data); }
};

cellpack::frozen_packing_plan make_plan(cellpack::packing_row_domain_kind row_kind) {
    const std::vector<u32> feature_permutation{3u, 1u, 5u, 0u, 4u, 2u};
    std::vector<u32> inverse(feature_permutation.size());
    for (u32 execution = 0u; execution < feature_permutation.size(); ++execution) {
        inverse[feature_permutation[execution]] = execution;
    }
    const std::vector<u32> block_offsets{0u, 2u, 5u, 6u};
    std::vector<u32> feature_to_block(feature_permutation.size());
    std::vector<u32> feature_to_local(feature_permutation.size());
    for (u32 block = 0u; block + 1u < block_offsets.size(); ++block) {
        for (u32 execution = block_offsets[block]; execution < block_offsets[block + 1u]; ++execution) {
            const u32 canonical = feature_permutation[execution];
            feature_to_block[canonical] = block;
            feature_to_local[canonical] = execution - block_offsets[block];
        }
    }
    const std::vector<u32> row_offsets{0u, 4u, 8u};
    cellpack::frozen_packing_plan_build_view build;
    build.row_count = 8u;
    build.feature_count = static_cast<u32>(feature_permutation.size());
    build.feature_permutation = feature_permutation.data();
    build.inverse_feature_permutation = inverse.data();
    build.feature_block_count = static_cast<u32>(block_offsets.size() - 1u);
    build.feature_block_offsets = block_offsets.data();
    build.feature_to_block = feature_to_block.data();
    build.feature_to_local = feature_to_local.data();
    build.row_group_count = static_cast<u32>(row_offsets.size() - 1u);
    build.row_group_offsets = row_offsets.data();
    build.maximum_feature_block_width = 3u;
    build.row_group_width = 4u;
    build.identity.feature_axis_fingerprint = 0x12345678u;
    build.identity.feature_axis_fingerprint_version = 1u;
    build.identity.row_domain_kind = row_kind;
    build.identity.row_domain_identity = 0xabcdu;
    build.identity.evaluation_source_identity = 0x777u;
    build.identity.sampling_provenance_identity =
        row_kind == cellpack::packing_row_domain_kind::sampled_rows_identity ? 0x888u : 0u;
    build.cost_policy_identity = 0x999u;
    cellpack::frozen_packing_plan result;
    require(static_cast<bool>(cellpack::freeze_packing_plan(build, &result)), "failed to freeze test plan");
    return result;
}

cellpack::plan_application_context make_context() {
    cellpack::plan_application_context context;
    context.full_row_count = 8u;
    context.feature_count = 6u;
    context.feature_axis_fingerprint = 0x12345678u;
    context.feature_axis_fingerprint_version = 1u;
    context.row_domain_identity = 0xabcdu;
    return context;
}

struct source_fixture {
    std::vector<u32> row_offsets{0u, 3u, 3u, 6u, 12u};
    std::vector<u32> features{0u, 3u, 5u, 1u, 2u, 4u, 0u, 1u, 2u, 3u, 4u, 5u};
    std::vector<u64> values{10u, 13u, 15u, 21u, 22u, 24u, 30u, 31u, 32u, 33u, 34u, 35u};

    cellpack::plan_application_source_view view() const {
        cellpack::plan_application_source_view source;
        source.global_row_begin = 2u;
        source.row_count = 4u;
        source.feature_count = 6u;
        source.nnz_count = static_cast<u32>(features.size());
        source.value_size_bytes = sizeof(u64);
        source.row_offsets = row_offsets.data();
        source.canonical_feature_ids = features.data();
        source.values = values.data();
        return source;
    }
};

struct host_output {
    std::vector<u32> row_offsets, blocks, locals, features;
    std::vector<u64> values;
    cellpack::ordered_plan_partition_view view{};
};

host_output run_host(
    const cellpack::frozen_packing_plan &plan,
    const cellpack::plan_application_context &context,
    const cellpack::plan_application_source_view &source) {
    host_output result;
    result.row_offsets.resize(static_cast<std::size_t>(source.row_count) + 1u);
    result.blocks.resize(source.nnz_count);
    result.locals.resize(source.nnz_count);
    result.features.resize(source.nnz_count);
    result.values.resize(source.nnz_count);
    std::vector<u64> keys(source.nnz_count);
    std::vector<u32> order(source.nnz_count);
    cellpack::plan_application_host_workspace_view workspace;
    workspace.entry_capacity = source.nnz_count;
    workspace.keys = keys.data();
    workspace.source_order = order.data();
    cellpack::plan_application_buffers buffers;
    buffers.row_offset_capacity = result.row_offsets.size();
    buffers.entry_capacity = source.nnz_count;
    buffers.value_capacity_bytes = result.values.size() * sizeof(u64);
    buffers.row_offsets = result.row_offsets.data();
    buffers.block_ids = result.blocks.data();
    buffers.local_feature_ids = result.locals.data();
    buffers.canonical_feature_ids = result.features.data();
    buffers.values = result.values.data();
    require(static_cast<bool>(cellpack::apply_frozen_plan_host(
        plan, context, source, workspace, buffers, &result.view)), "host plan application failed");
    return result;
}

void require_round_trip(
    const cellpack::frozen_packing_plan &plan,
    const source_fixture &fixture,
    const host_output &packed) {
    require(packed.row_offsets == fixture.row_offsets, "row offsets changed during plan application");
    for (u32 row = 0u; row + 1u < packed.row_offsets.size(); ++row) {
        std::vector<std::pair<u32, u64>> reconstructed;
        for (u32 entry = packed.row_offsets[row]; entry < packed.row_offsets[row + 1u]; ++entry) {
            require(plan.feature_to_block()[packed.features[entry]] == packed.blocks[entry],
                "block id does not match canonical feature lookup");
            require(plan.feature_to_local()[packed.features[entry]] == packed.locals[entry],
                "local feature id does not match canonical feature lookup");
            reconstructed.emplace_back(packed.features[entry], packed.values[entry]);
        }
        std::sort(reconstructed.begin(), reconstructed.end());
        for (u32 i = 0u; i < reconstructed.size(); ++i) {
            const u32 source_entry = fixture.row_offsets[row] + i;
            require(reconstructed[i].first == fixture.features[source_entry], "canonical feature round trip failed");
            require(reconstructed[i].second == fixture.values[source_entry], "value round trip failed");
        }
    }
}

void test_host_reference_and_validation() {
    cellpack::frozen_packing_plan plan = make_plan(cellpack::packing_row_domain_kind::full_dataset_identity);
    const cellpack::plan_application_context context = make_context();
    const source_fixture fixture;
    const host_output first = run_host(plan, context, fixture.view());
    const host_output second = run_host(plan, context, fixture.view());
    require(first.features == std::vector<u32>({3u, 5u, 0u, 1u, 4u, 2u, 3u, 1u, 5u, 0u, 4u, 2u}),
        "host packed feature order is wrong");
    require(first.blocks == std::vector<u32>({0u, 1u, 1u, 0u, 1u, 2u, 0u, 0u, 1u, 1u, 1u, 2u}),
        "host packed block ids are wrong");
    require(first.locals == std::vector<u32>({0u, 0u, 1u, 1u, 2u, 0u, 0u, 1u, 0u, 1u, 2u, 0u}),
        "host packed local ids are wrong");
    require(first.features == second.features && first.values == second.values,
        "host plan application is not deterministic");
    require(first.view.global_row_begin == 2u && first.view.full_row_count == 8u
        && first.view.row_domain_identity == context.row_domain_identity,
        "partition/provenance metadata was not preserved");
    require_round_trip(plan, fixture, first);

    cellpack::frozen_packing_plan sampled = make_plan(cellpack::packing_row_domain_kind::sampled_rows_identity);
    require(!cellpack::validate_plan_application_source_host(sampled, context, fixture.view()),
        "sample-scoped plan was accepted for a full partition");
    cellpack::plan_application_context wrong_context = context;
    wrong_context.feature_axis_fingerprint ^= 1u;
    require(!cellpack::validate_plan_application_source_host(plan, wrong_context, fixture.view()),
        "feature-axis mismatch was accepted");
    wrong_context = context;
    wrong_context.row_domain_identity ^= 1u;
    require(!cellpack::validate_plan_application_source_host(plan, wrong_context, fixture.view()),
        "row-domain identity mismatch was accepted");
    cellpack::plan_application_source_view invalid = fixture.view();
    invalid.global_row_begin = 7u;
    require(!cellpack::validate_plan_application_source_host(plan, context, invalid),
        "partition row-domain overflow was accepted");
    std::vector<u32> bad_features = fixture.features;
    bad_features[0] = context.feature_count;
    invalid = fixture.view();
    invalid.canonical_feature_ids = bad_features.data();
    require(!cellpack::validate_plan_application_source_host(plan, context, invalid),
        "invalid canonical feature id was accepted");
    std::vector<u32> bad_offsets = fixture.row_offsets;
    bad_offsets[2] = bad_offsets[1] - 1u;
    invalid = fixture.view();
    invalid.row_offsets = bad_offsets.data();
    require(!cellpack::validate_plan_application_source_host(plan, context, invalid),
        "nonmonotonic row offsets were accepted");

    std::vector<u64> keys(fixture.features.size());
    std::vector<u32> order(fixture.features.size());
    std::vector<u32> output_rows(fixture.row_offsets.size()), output_blocks(fixture.features.size());
    std::vector<u32> output_locals(fixture.features.size()), output_features(fixture.features.size());
    std::vector<u64> output_values(fixture.values.size());
    cellpack::plan_application_host_workspace_view short_workspace{
        static_cast<u32>(fixture.features.size() - 1u), keys.data(), order.data()};
    cellpack::plan_application_buffers adequate_buffers{
        output_rows.size(), output_blocks.size(), output_values.size() * sizeof(u64),
        output_rows.data(), output_blocks.data(), output_locals.data(), output_features.data(),
        output_values.data()};
    cellpack::ordered_plan_partition_view ignored;
    require(!cellpack::apply_frozen_plan_host(
        plan, context, fixture.view(), short_workspace, adequate_buffers, &ignored),
        "insufficient host workspace was accepted");
    cellpack::plan_application_host_workspace_view adequate_workspace{
        static_cast<u32>(keys.size()), keys.data(), order.data()};
    cellpack::plan_application_buffers short_buffers = adequate_buffers;
    short_buffers.entry_capacity = fixture.features.size() - 1u;
    require(!cellpack::apply_frozen_plan_host(
        plan, context, fixture.view(), adequate_workspace, short_buffers, &ignored),
        "insufficient output capacity was accepted");

    u32 zero_offset = 0u;
    u32 output_offset = 99u;
    u64 key = 0u;
    u32 empty_order = 0u;
    cellpack::plan_application_source_view empty;
    empty.global_row_begin = 8u;
    empty.row_count = 0u;
    empty.feature_count = context.feature_count;
    empty.value_size_bytes = sizeof(u64);
    empty.row_offsets = &zero_offset;
    cellpack::plan_application_host_workspace_view workspace{0u, &key, &empty_order};
    cellpack::plan_application_buffers buffers;
    buffers.row_offset_capacity = 1u;
    buffers.row_offsets = &output_offset;
    cellpack::ordered_plan_partition_view empty_view;
    require(static_cast<bool>(cellpack::apply_frozen_plan_host(
        plan, context, empty, workspace, buffers, &empty_view)), "empty partition failed");
    require(output_offset == 0u && empty_view.nnz_count == 0u, "empty partition output is wrong");

    const u32 empty_rows[] = {0u, 0u, 0u};
    u32 empty_output_rows[] = {99u, 99u, 99u};
    empty.global_row_begin = 3u;
    empty.row_count = 2u;
    empty.row_offsets = empty_rows;
    buffers.row_offset_capacity = 3u;
    buffers.row_offsets = empty_output_rows;
    require(static_cast<bool>(cellpack::apply_frozen_plan_host(
        plan, context, empty, workspace, buffers, &empty_view)), "all-empty row partition failed");
    require(empty_output_rows[0] == 0u && empty_output_rows[1] == 0u
        && empty_output_rows[2] == 0u, "all-empty row offsets are wrong");

    cellpack::plan_application_cuda_requirements overflow;
    require(!cellpack::query_plan_application_cuda_requirements(
        std::numeric_limits<u32>::max(), 0u, &overflow), "CUB segment-count overflow was accepted");
    require(!cellpack::query_plan_application_cuda_requirements(
        1u, std::numeric_limits<u32>::max(), &overflow), "CUB item-count overflow was accepted");
}

void test_cuda_matches_host() {
    cellpack::frozen_packing_plan plan = make_plan(cellpack::packing_row_domain_kind::full_dataset_identity);
    const cellpack::plan_application_context context = make_context();
    const source_fixture fixture;
    const cellpack::plan_application_source_view source = fixture.view();
    const host_output reference = run_host(plan, context, source);

    device_buffer<u32> d_rows(fixture.row_offsets.size()), d_features(fixture.features.size());
    device_buffer<u64> d_values(fixture.values.size());
    device_buffer<u32> d_feature_to_block(plan.feature_count()), d_feature_to_local(plan.feature_count());
    require_cuda(cudaMemcpy(d_rows.data, fixture.row_offsets.data(), fixture.row_offsets.size() * sizeof(u32), cudaMemcpyHostToDevice), "row upload failed");
    require_cuda(cudaMemcpy(d_features.data, fixture.features.data(), fixture.features.size() * sizeof(u32), cudaMemcpyHostToDevice), "feature upload failed");
    require_cuda(cudaMemcpy(d_values.data, fixture.values.data(), fixture.values.size() * sizeof(u64), cudaMemcpyHostToDevice), "value upload failed");
    require_cuda(cudaMemcpy(d_feature_to_block.data, plan.feature_to_block(), plan.feature_count() * sizeof(u32), cudaMemcpyHostToDevice), "block map upload failed");
    require_cuda(cudaMemcpy(d_feature_to_local.data, plan.feature_to_local(), plan.feature_count() * sizeof(u32), cudaMemcpyHostToDevice), "local map upload failed");

    cellpack::plan_application_cuda_requirements required;
    require(static_cast<bool>(cellpack::query_plan_application_cuda_requirements(
        source.row_count, source.nnz_count, &required)), "CUDA requirements query failed");
    device_buffer<u64> d_keys_in(source.nnz_count), d_keys_out(source.nnz_count);
    device_buffer<u32> d_order_in(source.nnz_count), d_order_out(source.nnz_count);
    device_buffer<unsigned char> d_cub(required.cub_temporary_bytes);
    device_buffer<u32> d_out_rows(fixture.row_offsets.size()), d_out_blocks(source.nnz_count);
    device_buffer<u32> d_out_locals(source.nnz_count), d_out_features(source.nnz_count);
    device_buffer<u64> d_out_values(source.nnz_count);

    cellpack::plan_application_source_view device_source = source;
    device_source.row_offsets = d_rows.data;
    device_source.canonical_feature_ids = d_features.data;
    device_source.values = d_values.data;
    cellpack::plan_application_device_feature_view device_plan;
    device_plan.feature_count = plan.feature_count();
    device_plan.feature_block_count = plan.feature_block_count();
    device_plan.feature_to_block = d_feature_to_block.data;
    device_plan.feature_to_local = d_feature_to_local.data;
    cellpack::plan_application_cuda_workspace_view workspace;
    workspace.entry_capacity = source.nnz_count;
    workspace.keys_in = d_keys_in.data;
    workspace.keys_out = d_keys_out.data;
    workspace.source_order_in = d_order_in.data;
    workspace.source_order_out = d_order_out.data;
    workspace.cub_temporary_storage = d_cub.data;
    workspace.cub_temporary_bytes = required.cub_temporary_bytes;
    cellpack::plan_application_buffers buffers;
    buffers.row_offset_capacity = fixture.row_offsets.size();
    buffers.entry_capacity = source.nnz_count;
    buffers.value_capacity_bytes = fixture.values.size() * sizeof(u64);
    buffers.row_offsets = d_out_rows.data;
    buffers.block_ids = d_out_blocks.data;
    buffers.local_feature_ids = d_out_locals.data;
    buffers.canonical_feature_ids = d_out_features.data;
    buffers.values = d_out_values.data;
    cellpack::ordered_plan_partition_view device_result;
    require(static_cast<bool>(cellpack::apply_frozen_plan_cuda(
        plan, context, device_source, device_plan, workspace, buffers, nullptr, &device_result)),
        "CUDA plan application failed");
    require_cuda(cudaDeviceSynchronize(), "CUDA plan application synchronization failed");

    host_output actual;
    actual.row_offsets.resize(fixture.row_offsets.size());
    actual.blocks.resize(source.nnz_count);
    actual.locals.resize(source.nnz_count);
    actual.features.resize(source.nnz_count);
    actual.values.resize(source.nnz_count);
    require_cuda(cudaMemcpy(actual.row_offsets.data(), d_out_rows.data, actual.row_offsets.size() * sizeof(u32), cudaMemcpyDeviceToHost), "row download failed");
    require_cuda(cudaMemcpy(actual.blocks.data(), d_out_blocks.data, actual.blocks.size() * sizeof(u32), cudaMemcpyDeviceToHost), "block download failed");
    require_cuda(cudaMemcpy(actual.locals.data(), d_out_locals.data, actual.locals.size() * sizeof(u32), cudaMemcpyDeviceToHost), "local download failed");
    require_cuda(cudaMemcpy(actual.features.data(), d_out_features.data, actual.features.size() * sizeof(u32), cudaMemcpyDeviceToHost), "feature download failed");
    require_cuda(cudaMemcpy(actual.values.data(), d_out_values.data, actual.values.size() * sizeof(u64), cudaMemcpyDeviceToHost), "value download failed");
    require(actual.row_offsets == reference.row_offsets && actual.blocks == reference.blocks
        && actual.locals == reference.locals && actual.features == reference.features
        && actual.values == reference.values, "CUDA output differs from CPU reference");
    require(device_result.global_row_begin == source.global_row_begin
        && device_result.values == d_out_values.data, "CUDA result view metadata is wrong");
}

} // namespace

int main() {
    test_host_reference_and_validation();
    test_cuda_matches_host();
    return 0;
}
