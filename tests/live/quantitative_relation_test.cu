#include <bench/ce_live/runtime_fixture/quantitative_fixture.hh>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace fixture = cellerator::ce_live;
namespace execution = cellerator::execution;

namespace {

constexpr std::uint32_t destination_count = 3u;
constexpr std::uint32_t source_count = 4u;
constexpr std::uint64_t edge_count = 7u;
constexpr std::uint64_t offsets[]{0u, 2u, 4u, 7u};
constexpr std::uint32_t sources[]{0u, 2u, 1u, 3u, 0u, 2u, 3u};
float generation_1[]{2.0f, -1.0f, 3.0f, 4.0f, 5.0f, 6.0f, -2.0f};
float generation_2[]{-0.5f, 1.25f, 2.0f, -3.0f, 0.75f, -1.5f, 4.0f};

void require(bool condition, const char *message) {
    if (condition) return;
    std::cerr << "quantitative_relation_test: " << message << '\n';
    std::exit(1);
}

void require_cuda(cudaError_t status, const char *message) {
    if (status == cudaSuccess) return;
    std::cerr << "quantitative_relation_test: " << message << ": "
              << cudaGetErrorString(status) << '\n';
    std::exit(1);
}

template<typename Identity>
Identity digest_identity(const char *hex) {
    Identity result{};
    require(fixture::identity_from_sha256(hex, &result.low, &result.high)
            == fixture::quantitative_fixture_status::ok,
        "fixture digest identity is invalid");
    return result;
}

fixture::quantitative_fixture_identities fixture_identities() {
    fixture::quantitative_fixture_identities ids{};
    ids.observation_domain = digest_identity<execution::domain_id>(
        "44ea28be6b4a74a8a64e9cbe7cd5a916ac00a6f722e3c3aa58106aaddb9a5723");
    ids.feature_domain = digest_identity<execution::domain_id>(
        "8420274d629c50bfd3f9afe1d87ea84e2284c14deb62f9cf97fc806a7711410f");
    ids.observation_order = digest_identity<execution::order_id>(
        "8f983e4ce7db7e74a051d86afb000c99a011ec9280875714babc7a6e52439179");
    ids.feature_order = digest_identity<execution::order_id>(
        "9e94b86f247b755cb856accf23c0fa667ccd16e2fbe45ca2bdb8d4ae109c0cde");
    ids.geometry = digest_identity<execution::geometry_id>(
        "0d63d9dd8fd3fd8e61e3e62ef58e75ea9cba861550bc318db0f1b430aabf75f1");
    ids.partition = digest_identity<execution::partition_id>(
        "5afe26a676eab57a8b77485d2880f609a03a4650c649fffc215fef6ac5095f43");
    ids.structure = digest_identity<execution::structure_id>(
        "154872f3501eebdf01b7dcb7d021ffae46bb7ac36c4cfac628af8daa3022d8ce");
    // Tagged SHA-256 of the CE-LIVE destination-row CSR projection name.
    ids.destination_row_csr_projection = digest_identity<execution::projection_id>(
        "1efb5340f3ed9e74baad47cfb63aeef46393c9e2eb0ed89585ccfa430f3ac98d");
    return ids;
}

__global__ void destination_row_spmm(const std::uint64_t *row_offsets,
    const std::uint32_t *source_indices, const float *edge_values,
    const float *dense, float *output, std::uint32_t rows,
    std::uint32_t width) {
    const std::uint32_t lane = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t row = blockIdx.y;
    if (row >= rows || lane >= width) return;
    float sum = 0.0f;
    for (std::uint64_t edge = row_offsets[row];
         edge < row_offsets[row + 1u]; ++edge)
        sum += edge_values[edge]
            * dense[static_cast<std::size_t>(source_indices[edge]) * width + lane];
    output[static_cast<std::size_t>(row) * width + lane] = sum;
}

std::vector<double> independent_coordinate_referee(const float *edge_values,
    const std::vector<float> &dense, std::uint32_t width) {
    std::vector<double> result(
        static_cast<std::size_t>(destination_count) * width, 0.0);
    // This coordinate expansion is intentionally independent of the device
    // kernel's destination-row traversal.
    for (std::uint32_t row = 0u; row < destination_count; ++row)
        for (std::uint64_t edge = offsets[row]; edge < offsets[row + 1u]; ++edge)
            for (std::uint32_t lane = 0u; lane < width; ++lane)
                result[static_cast<std::size_t>(row) * width + lane]
                    += static_cast<double>(edge_values[edge])
                    * static_cast<double>(dense[
                        static_cast<std::size_t>(sources[edge]) * width + lane]);
    return result;
}

void verify_generation(const float *host_values, float *device_values,
    const std::vector<float> &dense, std::uint32_t width,
    const std::uint64_t *device_offsets,
    const std::uint32_t *device_sources, float *device_dense,
    float *device_output) {
    require_cuda(cudaMemcpy(device_dense, dense.data(),
        dense.size() * sizeof(float), cudaMemcpyHostToDevice),
        "dense operand copy failed");
    require_cuda(cudaMemcpy(device_values, host_values,
        edge_count * sizeof(float), cudaMemcpyHostToDevice),
        "value generation copy failed");
    const dim3 block(32u, 1u, 1u);
    const dim3 grid((width + block.x - 1u) / block.x, destination_count, 1u);
    destination_row_spmm<<<grid, block>>>(device_offsets, device_sources,
        device_values, device_dense, device_output, destination_count, width);
    require_cuda(cudaGetLastError(), "quantitative relation launch failed");
    std::vector<float> actual(
        static_cast<std::size_t>(destination_count) * width);
    require_cuda(cudaMemcpy(actual.data(), device_output,
        actual.size() * sizeof(float), cudaMemcpyDeviceToHost),
        "quantitative output copy failed");
    const std::vector<double> expected =
        independent_coordinate_referee(host_values, dense, width);
    for (std::size_t i = 0u; i < actual.size(); ++i) {
        const double tolerance = 2.0e-5 * std::max(1.0, std::fabs(expected[i]));
        require(std::fabs(static_cast<double>(actual[i]) - expected[i]) <= tolerance,
            "device output disagrees with independent referee");
    }
}

} // namespace

int main() {
    const fixture::quantitative_fixture_identities pbmc_ids =
        fixture::pbmc3k_quantitative_v1_identities();
    require(execution::same_identity(pbmc_ids.feature_domain,
                digest_identity<execution::domain_id>(
                    "46c0b8e197efcc3099e90064f068b973261c39b25708879910b9395aa19903fd"))
            && execution::same_identity(pbmc_ids.structure,
                digest_identity<execution::structure_id>(
                    "5ec566e0bd56b468e9025ffe7c75fc54a4cf0eae2bc93107ae570fae188a7ccb")),
        "PBMC3K manifest identities drifted");
    execution::identity_registry registry{};
    fixture::native_quantitative_relation relation{};
    const fixture::quantitative_fixture_arrays arrays{
        {offsets, sources, destination_count, source_count, edge_count},
        generation_1, generation_2};
    const fixture::quantitative_fixture_identities ids = fixture_identities();
    require(fixture::bind_quantitative_fixture(
        arrays, ids, &registry, {1u, 1u}, &relation)
            == fixture::quantitative_fixture_status::ok,
        "native relation binding failed");
    require(execution::validate_relation_structure(relation.structure)
            == execution::lifetime_validation_code::ok,
        "relation structure is invalid");
    require(execution::same_axis_identity(
                relation.structure.source_axis, relation.operand.source_axis)
            && execution::same_axis_identity(relation.structure.destination_axis,
                relation.operand.destination_axis)
            && relation.projection.source_count == source_count
            && relation.projection.destination_count == destination_count,
        "feature-source to cell-destination orientation changed");
    execution::domain_id resolved_source{}, resolved_destination{};
    require(execution::resolve_identity(registry,
                relation.structure.source_axis.domain, &resolved_source)
            == execution::identity_registry_status::ok
            && execution::resolve_identity(registry,
                relation.structure.destination_axis.domain, &resolved_destination)
            == execution::identity_registry_status::ok
            && execution::same_identity(resolved_source, ids.feature_domain)
            && execution::same_identity(resolved_destination, ids.observation_domain),
        "persistent feature/cell identities were not preserved");
    const execution::value_binding binding_1{&relation.generations[0], {1u}};
    const execution::value_binding binding_2{&relation.generations[1], {2u}};
    require(execution::validate_value_binding(relation.structure, binding_1)
            == execution::lifetime_validation_code::ok
            && execution::validate_value_binding(relation.structure, binding_2)
            == execution::lifetime_validation_code::ok
            && relation.generations[0].structure.slot
                == relation.generations[1].structure.slot,
        "mutable generations did not share immutable topology");
    const execution::value_binding stale{&relation.generations[1], {1u}};
    require(execution::validate_value_binding(relation.structure, stale)
            == execution::lifetime_validation_code::stale_value_generation,
        "stale generation was accepted");

    std::uint64_t *device_offsets = nullptr;
    std::uint32_t *device_sources = nullptr;
    float *device_values = nullptr;
    float *device_dense = nullptr;
    float *device_output = nullptr;
    require_cuda(cudaMalloc(&device_offsets, sizeof(offsets)),
        "offset allocation failed");
    require_cuda(cudaMalloc(&device_sources, sizeof(sources)),
        "source allocation failed");
    require_cuda(cudaMalloc(&device_values, edge_count * sizeof(float)),
        "value allocation failed");
    constexpr std::uint32_t maximum_width = 64u;
    require_cuda(cudaMalloc(&device_dense,
        source_count * maximum_width * sizeof(float)),
        "dense allocation failed");
    require_cuda(cudaMalloc(&device_output,
        destination_count * maximum_width * sizeof(float)),
        "output allocation failed");
    require_cuda(cudaMemcpy(device_offsets, offsets, sizeof(offsets),
        cudaMemcpyHostToDevice), "offset copy failed");
    require_cuda(cudaMemcpy(device_sources, sources, sizeof(sources),
        cudaMemcpyHostToDevice), "source copy failed");

    constexpr std::uint32_t widths[]{1u, 16u, 17u, 31u, 32u, 48u, 64u};
    for (std::uint32_t width : widths) {
        std::vector<float> dense(static_cast<std::size_t>(source_count) * width);
        fixture::fill_deterministic_dense_operand(
            dense.data(), source_count, width);
        verify_generation(generation_1, device_values, dense, width,
            device_offsets, device_sources, device_dense, device_output);
        verify_generation(generation_2, device_values, dense, width,
            device_offsets, device_sources, device_dense, device_output);
    }

    require_cuda(cudaFree(device_output), "output release failed");
    require_cuda(cudaFree(device_dense), "dense release failed");
    require_cuda(cudaFree(device_values), "value release failed");
    require_cuda(cudaFree(device_sources), "source release failed");
    require_cuda(cudaFree(device_offsets), "offset release failed");
    std::cout << "quantitative_relation_test passed widths=7 generations=2\n";
    return 0;
}
