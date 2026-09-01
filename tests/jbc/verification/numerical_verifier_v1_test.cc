#include "numerical_verifier_v1.hh"

#include <cassert>
#include <cstdint>
#include <vector>

namespace verify = cellerator::jbc::verification;

int main() {
    constexpr std::uint64_t fragment_count = 4096u;
    constexpr std::uint64_t values_per_fragment = 4u;
    constexpr std::uint64_t canonical_count =
        fragment_count * values_per_fragment;
    std::vector<std::uint64_t> maps(canonical_count);
    std::vector<double> values(canonical_count);
    std::vector<verify::numerical_fragment_v1> fragments(fragment_count);
    std::vector<double> expected(canonical_count);
    for (std::uint64_t fragment = 0u; fragment < fragment_count; ++fragment) {
        const auto begin = fragment * values_per_fragment;
        for (std::uint64_t local = 0u; local < values_per_fragment; ++local) {
            const auto canonical = begin + values_per_fragment - local - 1u;
            maps[begin + local] = canonical;
            values[begin + local] = static_cast<double>(canonical) * 0.25;
            expected[canonical] = values[begin + local];
        }
        fragments[fragment] = {maps.data() + begin, values.data() + begin,
            values_per_fragment};
    }
    std::vector<double> reconstructed(canonical_count);
    std::vector<std::uint8_t> written(canonical_count);
    assert(verify::reconstruct_canonical_values_v1(fragments.data(),
        fragments.size(), reconstructed.data(), written.data(),
        reconstructed.size()));
    assert(verify::verify_numerical_values_v1(expected.data(),
        reconstructed.data(), canonical_count, 0.0, 0.0));

    reconstructed[canonical_count - 1u] += 0.5;
    const auto mismatch = verify::verify_numerical_values_v1(expected.data(),
        reconstructed.data(), canonical_count, 0.01, 0.0001);
    assert(mismatch.code == verify::numerical_code_v1::tolerance_exceeded);
    assert(mismatch.index == canonical_count - 1u);

    fragments[fragment_count - 1u].local_to_canonical = maps.data();
    assert(verify::reconstruct_canonical_values_v1(fragments.data(),
               fragments.size(), reconstructed.data(), written.data(),
               reconstructed.size()).code ==
        verify::numerical_code_v1::duplicate_value);
}
