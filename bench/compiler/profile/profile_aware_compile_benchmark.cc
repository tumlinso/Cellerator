#include <Cellerator/compiler/profile/deliver_the_first_profile_aware_compile_benchmark_v1.hh>

#include <chrono>
#include <cstdint>
#include <iostream>

namespace profile = cellerator::compiler::profile::v1;

int main() {
    profile::relation_field_source_v1 source{{1u, 2u}, {3u, 4u}, {5u, 6u},
                                             {7u, 8u}, 0xa95f02f7cc11813bull, 4u, 0u};
    profile::profile_compile_state_v1 states[2]{};
    for (std::uint32_t index = 0u; index != 2u; ++index) {
        auto &state = states[index];
        state.state = {10u + index, 20u + index};
        state.structure.evidence = {30u + index, 40u + index};
        state.structure.relation = source.relation;
        state.structure.source_axis.domain = source.source_domain;
        state.structure.destination_axis.domain = source.destination_domain;
        state.structure.destination_axis.extent = 4096u;
        state.structure.support_count = index == 0u ? 16384u : 262144u;
        state.structure.degree.mean = index == 0u ? 4.0 : 64.0;
        state.structure.ordering_stability = index == 0u ? 0.4 : 0.95;
        state.values.evidence = {50u + index, 60u + index};
        state.values.observation_count = 4096u;
        state.values.zero_count = index == 0u ? 64u : 3072u;
        state.values.approximation_risk = index == 0u ? 0.2 : 0.01;
        state.reuse.evidence = {70u + index, 80u + index};
        state.reuse.reuse_horizon = index == 0u ? 1.0 : 32.0;
        state.reuse.structure_change.upper_95 = index == 0u ? 0.6 : 0.05;
    }

    constexpr std::uint32_t repetitions = 100000u;
    for (std::uint32_t state_index = 0u; state_index != 2u; ++state_index) {
        profile::profile_aware_compile_result_v1 result{};
        std::uint64_t load_nanoseconds = 0u;
        std::uint64_t propagation_nanoseconds = 0u;
        const auto begin = std::chrono::steady_clock::now();
        for (std::uint32_t repetition = 0u; repetition != repetitions; ++repetition) {
            if (profile::compile_profile_aware_relation_v1(source, states[state_index], &result) !=
                profile::profile_aware_compile_status_v1::ok) {
                return 1;
            }
            load_nanoseconds += result.profile_load_nanoseconds;
            propagation_nanoseconds += result.propagation_nanoseconds;
        }
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - begin).count();
        std::cout << "state=" << state_index
                  << " source=" << result.source_semantics_fingerprint
                  << " profile=" << result.profile_fingerprint
                  << " search=" << result.search_fingerprint
                  << " candidates=" << result.candidate_count
                  << " memory_bytes=" << result.compiler_memory_bytes
                  << " mean_profile_load_ns=" << load_nanoseconds / repetitions
                  << " mean_propagation_ns=" << propagation_nanoseconds / repetitions
                  << " mean_compile_ns=" << elapsed / repetitions << '\n';
    }
}
