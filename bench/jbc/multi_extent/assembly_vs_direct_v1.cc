#include <Cellerator/execution/object_binding/relation_apply_candidate_v1.hh>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <vector>

namespace binding = cellerator::execution::object_binding;
namespace candidate = cellerator::compute::candidate::jbc_multi_extent;

int main() {
    constexpr std::uint64_t extent_count = 8u;
    constexpr std::uint64_t elements_per_extent = 256u;
    constexpr std::uint64_t repeat_count = 20000u;

    std::vector<std::vector<float>> storage(extent_count,
        std::vector<float>(elements_per_extent));
    std::vector<binding::physical_extent_binding_v1> extents(extent_count);
    std::vector<float> scales(extent_count, 1.0f);
    for (std::uint64_t extent = 0u; extent < extent_count; ++extent) {
        for (std::uint64_t element = 0u; element < elements_per_extent;
             ++element) {
            storage[extent][element] =
                static_cast<float>((extent + element) % 17u) * 0.125f;
        }
        extents[extent] = {{extent + 1u, 1u}, storage[extent].data(),
            elements_per_extent * sizeof(float), elements_per_extent,
            sizeof(float), alignof(float), 1u,
            binding::extent_residency_v1::host, {}};
    }
    const binding::multi_extent_physical_binding_v1 input{
        {100u, 1u}, extents.data(), extent_count};
    const candidate::relation_apply_state_v1 state{
        scales.data(), extent_count, 0.0f};
    const auto direct = candidate::make_experimental_relation_apply_candidate_v1(
        &state, {101u, 1u});
    if (!binding::validate_direct_multi_extent_candidate_v1(direct, input)) {
        return 2;
    }

    std::vector<binding::contiguous_assembly_segment_v1> segments(extent_count);
    binding::contiguous_assembly_plan_v1 assembly{};
    if (!binding::compile_contiguous_assembly_v1(input, alignof(float),
            segments.data(), segments.size(), &assembly)) {
        return 3;
    }
    std::vector<float> contiguous(extent_count * elements_per_extent);

    float direct_result = 0.0f;
    if (!direct.launch(direct.prepared_state, input, &direct_result,
            sizeof(direct_result), nullptr) ||
        !binding::execute_contiguous_assembly_v1(
            assembly, contiguous.data(), contiguous.size() * sizeof(float))) {
        return 4;
    }
    float assembly_result = 0.0f;
    for (const auto value : contiguous) {
        assembly_result += value;
    }
    if (direct_result != assembly_result) {
        return 5;
    }

    volatile float sink = 0.0f;
    const auto direct_begin = std::chrono::steady_clock::now();
    for (std::uint64_t repeat = 0u; repeat < repeat_count; ++repeat) {
        float output = 0.0f;
        const auto status = direct.launch(direct.prepared_state, input, &output,
            sizeof(output), nullptr);
        if (!status) {
            return 6;
        }
        sink = sink + output;
    }
    const auto direct_end = std::chrono::steady_clock::now();

    const auto assembly_begin = std::chrono::steady_clock::now();
    for (std::uint64_t repeat = 0u; repeat < repeat_count; ++repeat) {
        const auto status = binding::execute_contiguous_assembly_v1(
            assembly, contiguous.data(), contiguous.size() * sizeof(float));
        if (!status) {
            return 7;
        }
        float output = 0.0f;
        for (const auto value : contiguous) {
            output += value;
        }
        sink = sink + output;
    }
    const auto assembly_end = std::chrono::steady_clock::now();

    const auto direct_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        direct_end - direct_begin).count();
    const auto assembly_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            assembly_end - assembly_begin).count();
    std::cout << "path,extent_count,elements_per_extent,repeats,total_ns,ns_per_repeat,setup_included\n";
    std::cout << "direct," << extent_count << ',' << elements_per_extent << ','
              << repeat_count << ',' << direct_ns << ','
              << static_cast<double>(direct_ns) / repeat_count << ",false\n";
    std::cout << "assembly_then_apply," << extent_count << ','
              << elements_per_extent << ',' << repeat_count << ','
              << assembly_ns << ','
              << static_cast<double>(assembly_ns) / repeat_count
              << ",false\n";
    return sink == 0.0f ? 1 : 0;
}
