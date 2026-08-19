/*
 * CP-MATH-03 device-resident feature-row packing benchmark. The bitwise row-
 * copy reference is correctness-only; timed results report the fused kernel.
 */

#include <Cellerator/compute/math/packed_dense_operand.hh>

#include <bench/benchmark_mutex.hh>
#include <bench/math/benchmark_support.hh>

#include <cuda_runtime_api.h>

#include <cstdio>
#include <cstdlib>
#include <memory>

namespace cm = cellerator::compute::math;
namespace cmb = cellerator::compute::math::bench;

namespace {

void require(bool condition, const char *message) {
    if (!condition) {
        std::fprintf(stderr, "cpMathPackedDenseOperandBench: %s\n", message);
        std::exit(1);
    }
}

} // namespace

int main() {
    constexpr cm::u32 features = 32768u, samples = 15u, repetitions = 64u;
    constexpr cm::u64 columns = 512u;
    cellerator::bench::benchmark_mutex_guard benchmark_mutex(
        "cpMathPackedDenseOperandBench", 0);
    const std::size_t elements = static_cast<std::size_t>(features) * columns;
    const std::size_t bytes = elements * sizeof(cm::u32);
    std::unique_ptr<cm::u32[]> permutation(new cm::u32[features]);
    for (cm::u32 index = 0u; index < features; ++index) {
        permutation[index] = features - 1u - index;
    }

    cm::u32 *device_permutation = nullptr, *source = nullptr;
    cm::u32 *packed = nullptr, *reference = nullptr, *device_offsets = nullptr;
    require(cudaMalloc(&device_permutation, features * sizeof(cm::u32)) == cudaSuccess,
        "permutation allocation failed");
    require(cudaMalloc(&source, bytes) == cudaSuccess
            && cudaMalloc(&packed, bytes) == cudaSuccess
            && cudaMalloc(&reference, bytes) == cudaSuccess,
        "value allocation failed");
    require(cudaMemcpy(device_permutation, permutation.get(), features * sizeof(cm::u32),
        cudaMemcpyHostToDevice) == cudaSuccess, "permutation copy failed");
    require(cudaMemset(source, 0x5a, bytes) == cudaSuccess, "source fill failed");

    const cm::u32 block_offsets[]{0u, features};
    require(cudaMalloc(&device_offsets, sizeof(block_offsets)) == cudaSuccess
            && cudaMemcpy(device_offsets, block_offsets, sizeof(block_offsets),
                cudaMemcpyHostToDevice) == cudaSuccess,
        "block offset setup failed");
    cellpack::feature_weighted_row_reduction_plan_view plan;
    plan.semantic_plan_schema_version = cellpack::packing_plan_semantic_schema_version;
    plan.geometry_identity_version = cellpack::feature_block_geometry_identity_version;
    plan.feature_count = features;
    plan.feature_block_count = 1u;
    plan.feature_block_geometry_identity = 0x303b3e11ull;
    plan.feature_block_offsets = device_offsets;
    plan.feature_permutation = device_permutation;
    cm::canonical_dense_operand_view input;
    input.values = source;
    input.feature_count = features;
    input.column_count = columns;
    input.leading_dimension = columns;
    input.value_size_bytes = sizeof(cm::u32);
    input.feature_order.feature_count = features;
    input.feature_order.feature_axis_identity_version = 1u;
    input.feature_order.feature_axis_identity = 0x303b3e12ull;
    input.operand_identity = 0x303b3e13ull;
    cm::packed_dense_operand_view output;
    require(static_cast<bool>(cm::pack_dense_operand_cuda(
        plan, input, {bytes, packed}, nullptr, &output)), "warmup failed");
    require(cudaDeviceSynchronize() == cudaSuccess, "warmup sync failed");

    cmb::cuda_event_timer timer;
    require(static_cast<bool>(timer.init()), "event timer init failed");
    double elapsed[samples]{}, scratch[samples]{};
    for (cm::u32 sample = 0u; sample < samples; ++sample) {
        require(static_cast<bool>(timer.begin()), "timer start failed");
        for (cm::u32 repeat = 0u; repeat < repetitions; ++repeat) {
            require(static_cast<bool>(cm::pack_dense_operand_cuda(
                plan, input, {bytes, packed}, nullptr, &output)), "pack failed");
        }
        require(static_cast<bool>(timer.end(&elapsed[sample])), "timer end failed");
        elapsed[sample] /= repetitions;
    }

    for (cm::u32 execution = 0u; execution < features; ++execution) {
        require(cudaMemcpyAsync(reference + static_cast<std::size_t>(execution) * columns,
            source + static_cast<std::size_t>(permutation[execution]) * columns,
            columns * sizeof(cm::u32), cudaMemcpyDeviceToDevice) == cudaSuccess,
            "row-copy reference failed");
    }
    require(cudaDeviceSynchronize() == cudaSuccess, "reference sync failed");
    std::unique_ptr<cm::u32[]> host_packed(new cm::u32[elements]);
    std::unique_ptr<cm::u32[]> host_reference(new cm::u32[elements]);
    require(cudaMemcpy(host_packed.get(), packed, bytes, cudaMemcpyDeviceToHost) == cudaSuccess
            && cudaMemcpy(host_reference.get(), reference, bytes,
                cudaMemcpyDeviceToHost) == cudaSuccess,
        "correctness copies failed");
    for (std::size_t index = 0u; index < elements; ++index) {
        require(host_packed[index] == host_reference[index], "outputs differ");
    }

    cmb::timing_summary timing;
    require(static_cast<bool>(cmb::summarize_timing_samples(
        elapsed, samples, scratch, samples, &timing)), "timing summary failed");
    std::printf("{\"schema_version\":1,\"benchmark\":\"packed_dense_operand\"," 
        "\"median_ms\":%.9g,\"p05_ms\":%.9g,\"p95_ms\":%.9g,"
        "\"bytes\":%zu,\"features\":%u,\"columns\":%llu,\"exact_match\":true}\n",
        timing.median_ms, timing.p05_ms, timing.p95_ms, bytes, features,
        static_cast<unsigned long long>(columns));

    cudaFree(device_offsets);
    cudaFree(reference);
    cudaFree(packed);
    cudaFree(source);
    cudaFree(device_permutation);
    return 0;
}
