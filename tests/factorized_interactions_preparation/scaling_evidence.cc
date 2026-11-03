// Manual CPU-only preparation scaling evidence. This is intentionally not a
// timing assertion or part of the regular unit-test target.
#include <Cellerator/layout/factorized_interaction.hh>
#include "../../src/layout/factorized_interaction_test_support.hh"

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <utility>
#include <vector>

namespace layout = cellerator::layout;

layout::FactorizedInteractionOptions options() {
    layout::FactorizedInteractionOptions value;
    value.max_mechanisms = 32;
    value.max_inputs = 64;
    value.max_contexts = 64;
    value.pattern_id = 101;
    value.structure_id = 202;
    value.structure_epoch = 303;
    return value;
}

void run_case(const char* shape, std::uint64_t count, bool shared) {
    std::vector<layout::FactorizedInteractionMechanism> mechanisms;
    mechanisms.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t m = 0; m < count; ++m) {
        layout::FactorizedInteractionMechanism mechanism;
        mechanism.logical_id = m + 1;
        mechanism.inputs[0] = shared ? 0 : 2 * m;
        mechanism.inputs[1] = shared ? 1 : 2 * m + 1;
        mechanism.target = shared ? 0 : m;
        mechanism.alpha = 0;
        mechanism.bias = 0;
        mechanism.arity = 2;
        mechanisms.push_back(std::move(mechanism));
    }
    const layout::FactorizedInteractionExtents extents{
        shared ? 2 : 2 * count, 0, shared ? 1 : count, 1, 1, 0};
    layout::test_support::FactorizedInteractionPreparationWork work;
    const auto start = std::chrono::steady_clock::now();
    const auto prepared = layout::test_support::prepare_factorized_interactions_counted(
        mechanisms, extents, options(), work);
    const auto stop = std::chrono::steady_clock::now();
    const auto milliseconds = std::chrono::duration<double, std::milli>(stop - start).count();
    std::cout << "shape=" << shape
              << " mechanisms=" << count
              << " packets=" << prepared.packets.size()
              << " affinity_entries=" << work.affinity_entries_visited
              << " affinity_lookups=" << work.affinity_key_lookups
              << " candidates_scored=" << work.candidates_scored
              << " fallback_entries=" << work.fallback_entries_visited
              << " prepare_ms=" << std::fixed << std::setprecision(2) << milliseconds
              << '\n';
}

int main() {
    for (const std::uint64_t count : {16000, 32000, 64000, 128000}) {
        run_case("disjoint", count, false);
        run_case("shared_two_input_target", count, true);
    }
    run_case("shared_two_input_target", 100000, true);
    run_case("shared_two_input_target", 1000000, true);
    return 0;
}
