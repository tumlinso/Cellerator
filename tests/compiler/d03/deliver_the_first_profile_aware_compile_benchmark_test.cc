#include <Cellerator/compiler/profile/deliver_the_first_profile_aware_compile_benchmark_v1.hh>

#include <cassert>

namespace profile = cellerator::compiler::profile::v1;

profile::profile_compile_state_v1 make_state(std::uint64_t state_id,
                                             std::uint64_t support_count,
                                             double degree,
                                             double stability,
                                             double reuse,
                                             double change_upper,
                                             double risk,
                                             std::uint64_t zero_count) {
    profile::profile_compile_state_v1 state{};
    state.state = {state_id, state_id + 1u};
    state.structure.evidence = {state_id + 10u, state_id + 11u};
    state.structure.relation = {30u, 31u};
    state.structure.source_axis.domain = {40u, 41u};
    state.structure.destination_axis.domain = {50u, 51u};
    state.structure.destination_axis.extent = 128u;
    state.structure.support_count = support_count;
    state.structure.degree.mean = degree;
    state.structure.ordering_stability = stability;
    state.values.evidence = {state_id + 20u, state_id + 21u};
    state.values.observation_count = 128u;
    state.values.zero_count = zero_count;
    state.values.approximation_risk = risk;
    state.reuse.evidence = {state_id + 30u, state_id + 31u};
    state.reuse.reuse_horizon = reuse;
    state.reuse.structure_change.upper_95 = change_upper;
    return state;
}

int main() {
    const profile::relation_field_source_v1 source{
        {10u, 11u}, {30u, 31u}, {40u, 41u}, {50u, 51u},
        0x83ba28a25d7f71e3ull, 3u, 0u};
    const auto baseline = make_state(100u, 512u, 4.0, 0.5, 1.0, 0.5, 0.2, 8u);
    const auto recurrent = make_state(200u, 4096u, 36.0, 0.95, 16.0, 0.1, 0.01, 96u);

    profile::profile_aware_compile_result_v1 first{};
    profile::profile_aware_compile_result_v1 second{};
    assert(profile::compile_profile_aware_relation_v1(source, baseline, &first) ==
           profile::profile_aware_compile_status_v1::ok);
    assert(profile::compile_profile_aware_relation_v1(source, recurrent, &second) ==
           profile::profile_aware_compile_status_v1::ok);

    assert(first.source_semantics_fingerprint == source.source_semantics_fingerprint);
    assert(second.source_semantics_fingerprint == source.source_semantics_fingerprint);
    assert(first.profile_fingerprint != second.profile_fingerprint);
    assert(first.search_fingerprint != second.search_fingerprint);
    assert(first.search.support_count != second.search.support_count);
    assert(first.search.preferred_rows_per_group != second.search.preferred_rows_per_group);
    assert(first.search.candidate_flags != second.search.candidate_flags);
    assert(first.candidate_count < second.candidate_count);
    assert(first.profile_load_nanoseconds > 0u && second.profile_load_nanoseconds > 0u);
    assert(first.propagation_nanoseconds > 0u && second.propagation_nanoseconds > 0u);
    assert(first.compiler_memory_bytes == second.compiler_memory_bytes);

    profile::profile_aware_compile_result_v1 repeated{};
    assert(profile::compile_profile_aware_relation_v1(source, recurrent, &repeated) ==
           profile::profile_aware_compile_status_v1::ok);
    assert(repeated.source_semantics_fingerprint == second.source_semantics_fingerprint);
    assert(repeated.profile_fingerprint == second.profile_fingerprint);
    assert(repeated.search_fingerprint == second.search_fingerprint);

    auto mismatched = recurrent;
    mismatched.structure.relation = {99u, 100u};
    assert(profile::compile_profile_aware_relation_v1(source, mismatched, &repeated) ==
           profile::profile_aware_compile_status_v1::identity_mismatch);
}
