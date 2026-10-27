#include <Cellerator/compute/projection/physical_mma_hybrid.hh>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

namespace projection = cellerator::compute::projection;
namespace math = cellerator::compute::math;

namespace cellerator::compute::projection {
bool validate_physical_mma_hybrid_image_v1_reference(
    const void*, std::size_t) noexcept;
}

namespace {

using clock_type = std::chrono::steady_clock;

double elapsed_ms(clock_type::time_point start, clock_type::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

double quantile(const std::vector<double>& sorted, double fraction) {
    const auto index = static_cast<std::size_t>(
        fraction * static_cast<double>(sorted.size() - 1u));
    return sorted[index];
}

int run_case(std::uint64_t edge_count) {
    constexpr std::uint32_t entities = 4096u;
    std::vector<std::uint32_t> sources(static_cast<std::size_t>(edge_count));
    std::vector<std::uint32_t> destinations(static_cast<std::size_t>(edge_count));
    for (std::uint64_t edge = 0u; edge < edge_count; ++edge) {
        sources[edge] = static_cast<std::uint32_t>(edge / entities);
        destinations[edge] = static_cast<std::uint32_t>(edge % entities);
    }

    projection::physical_mma_hybrid_build_request_v1 request{};
    request.structure_identity.identity_version = 1u;
    request.structure_identity.value = 0x56414c4944415445ull;
    request.source_order.kind = math::feature_order_kind::canonical;
    request.source_order.feature_count = entities;
    request.source_order.feature_axis_identity_version = 1u;
    request.source_order.feature_axis_identity = 0x101u;
    request.destination_order.kind = math::feature_order_kind::canonical;
    request.destination_order.feature_count = entities;
    request.destination_order.feature_axis_identity_version = 1u;
    request.destination_order.feature_axis_identity = 0x202u;
    request.provider_identity_low = 0x301u;
    request.capability_identity_low = 0x401u;
    request.source_indices = sources.data();
    request.destination_indices = destinations.data();
    request.edge_count = edge_count;
    request.source_count = entities;
    request.destination_count = entities;
    request.dense_width = 64u;

    std::vector<std::uint8_t> image;
    const auto build_start = clock_type::now();
    const auto build_status = projection::build_physical_mma_hybrid_image_v1(
        request, &image);
    const auto build_end = clock_type::now();
    if (build_status != projection::physical_image_build_status_v1::success)
        return 2;

    for (unsigned warmup = 0u; warmup < 3u; ++warmup)
        if (!projection::validate_physical_mma_hybrid_image_v1(
                image.data(), image.size()))
            return 3;
    constexpr std::size_t repeats = 15u;
    std::vector<double> samples;
    samples.reserve(repeats);
    for (std::size_t repeat = 0u; repeat < repeats; ++repeat) {
        const auto start = clock_type::now();
        const bool valid = projection::validate_physical_mma_hybrid_image_v1(
            image.data(), image.size());
        const auto end = clock_type::now();
        if (!valid) return 4;
        samples.push_back(elapsed_ms(start, end));
    }
    std::sort(samples.begin(), samples.end());
    double mean = 0.0;
    for (const auto sample : samples) mean += sample;
    mean /= static_cast<double>(samples.size());
    double squared = 0.0;
    for (const auto sample : samples) {
        const double difference = sample - mean;
        squared += difference * difference;
    }
    const double stdev = samples.size() > 1u
        ? std::sqrt(squared / static_cast<double>(samples.size() - 1u)) : 0.0;
    projection::physical_mma_hybrid_header_v1 header{};
    std::memcpy(&header, image.data(), sizeof(header));
    std::vector<double> reference_samples;
    if (edge_count <= 10000u) {
        for (unsigned warmup = 0u; warmup < 1u; ++warmup)
            if (!projection::validate_physical_mma_hybrid_image_v1_reference(
                    image.data(), image.size()))
                return 5;
        for (unsigned repeat = 0u; repeat < 5u; ++repeat) {
            const auto start = clock_type::now();
            const bool valid = projection::validate_physical_mma_hybrid_image_v1_reference(
                image.data(), image.size());
            const auto end = clock_type::now();
            if (!valid) return 6;
            reference_samples.push_back(elapsed_ms(start, end));
        }
        std::sort(reference_samples.begin(), reference_samples.end());
    }
    const std::uint64_t scratch_bytes =
        header.value_map_count * 2u * sizeof(std::uint8_t)
        + header.compact_slot_count * sizeof(std::uint64_t)
        + static_cast<std::uint64_t>(header.residual_region_count) * 3u
            * 2u * sizeof(std::uint64_t);
    std::cout << "{\"edges\":" << edge_count
              << ",\"entities\":" << entities
              << ",\"warmup\":3,\"repeats\":" << repeats
              << ",\"validation_ms\":{\"median\":" << samples[samples.size() / 2u]
              << ",\"mean\":" << mean
              << ",\"stdev\":" << stdev
              << ",\"p10\":" << quantile(samples, 0.10)
              << ",\"p90\":" << quantile(samples, 0.90) << "}"
              << ",\"frozen_original_validation_ms\":";
    if (reference_samples.empty()) std::cout << "null";
    else std::cout << reference_samples[reference_samples.size() / 2u];
    std::cout << ",\"new_over_original_speedup\":";
    if (reference_samples.empty()) std::cout << "null";
    else std::cout << reference_samples[reference_samples.size() / 2u]
        / samples[samples.size() / 2u];
    std::cout << ",\"builder_and_validation_ms\":"
              << elapsed_ms(build_start, build_end)
              << ",\"image_bytes\":" << image.size()
              << ",\"tile_count\":" << header.tile_count
              << ",\"residual_regions\":" << header.residual_region_count
              << ",\"primary_validation_arrays_bytes\":" << scratch_bytes
              << "}\n" << std::flush;
    return 0;
}

} // namespace

int main() {
    for (const auto edge_count : {1000ull, 10000ull, 262144ull}) {
        if (run_case(edge_count) != 0) return 1;
    }
    return 0;
}
